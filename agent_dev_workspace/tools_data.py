from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def check_missing_tool(column_name: str = "text") -> dict:
        """
        Checks for missing, null, or empty string values in a specified column of session.dataframe.

        Args:
            column_name: Name of the column to check for missing/empty values. Defaults to 'text'.

        Returns:
            Dict containing result_id, column, total_rows, missing_count,
            has_missing, and a capped list of missing_indices (up to 20).
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        if column_name not in session.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in session.dataframe.")

        series = session.dataframe[column_name]
        is_missing = series.isna() | (series.astype(str).str.strip() == "")

        missing_df = session.dataframe[is_missing]
        all_missing_indices = missing_df.index.tolist()
        missing_count = len(all_missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "column": column_name,
            "total_rows": len(session.dataframe),
            "missing_count": missing_count,
            "has_missing": missing_count > 0,
            "missing_indices": all_missing_indices[:20],
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"column_name": column_name},
            summary=summary,
        )
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        When drop=True, removes every copy of a duplicated row (keep=False),
        matching Master's drop_duplicates(keep=False, inplace=True) behavior.

        Args:
            drop: If True, drops all copies of duplicate rows from session.dataframe.

        Returns:
            Dict containing result_id, total_rows_before, n_duplicates,
            duplicate_indices (capped at 20), dropped, rows_removed,
            dtm_reset, and total_rows_after.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        df = session.dataframe
        total_rows_before = len(df)

        # Select hashable columns (exclude list or dict columns like 'unigrams')
        check_cols = [
            col for col in df.columns
            if not df[col].apply(lambda x: isinstance(x, (list, dict))).any()
        ]

        # Identify duplicate extra copies (keep='first') for counting
        extra_dups_mask = df.duplicated(subset=check_cols, keep="first")
        all_dup_indices = df[extra_dups_mask].index.tolist()
        n_duplicates = len(all_dup_indices)

        rows_removed = 0
        dtm_reset = False

        if drop and n_duplicates > 0:
            # Drop ALL copies of duplicated rows matching Master's keep=False
            all_copies_mask = df.duplicated(subset=check_cols, keep=False)
            session.dataframe = df[~all_copies_mask].copy()
            rows_removed = total_rows_before - len(session.dataframe)

            # If DTM was previously built, reset it since row alignment was changed
            if session.feature_matrix is not None or session.feature_names is not None:
                session.feature_matrix = None
                session.feature_names = None
                dtm_reset = True

            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].tolist())

        total_rows_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows_before": total_rows_before,
            "n_duplicates": n_duplicates,
            "duplicate_indices": all_dup_indices[:20],
            "dropped": rows_removed > 0,
            "rows_removed": rows_removed,
            "dtm_reset": dtm_reset,
            "total_rows_after": total_rows_after,
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
        )
        return summary

    @tool
    def tokenize_tool(text_column: str = "text", lowercase: bool = True) -> dict:
        """
        Tokenizes text into unigrams using nltk.word_tokenize and stores result in session.dataframe['unigrams'].

        Args:
            text_column: Name of text column to tokenize. Defaults to 'text'.
            lowercase: Whether to convert text to lowercase before tokenizing. Defaults to True.

        Returns:
            Dict containing result_id, text_column, total_documents,
            total_tokens, avg_tokens_per_doc, and sample_tokens preview.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        # Ensure NLTK tokenizer resources are present before tokenizing
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                dl_ok = nltk.download("punkt_tab", quiet=True) or nltk.download("punkt", quiet=True)
                if not dl_ok:
                    raise RuntimeError("NLTK punkt dataset is missing and download failed. Run `nltk.download('punkt')`.")

        def tokenize_text(val):
            if pd.isna(val) or not str(val).strip():
                return []
            text_str = str(val)
            if lowercase:
                text_str = text_str.lower()
            try:
                return nltk.word_tokenize(text_str)
            except LookupError as e:
                raise RuntimeError(f"NLTK word_tokenize failed: {e}. Please ensure NLTK 'punkt' data is installed.")

        # Store token lists directly on dataframe['unigrams']
        unigrams_series = session.dataframe[text_column].apply(tokenize_text)
        session.dataframe["unigrams"] = unigrams_series

        total_documents = len(session.dataframe)
        doc_lengths = unigrams_series.apply(len)
        total_tokens = int(doc_lengths.sum())
        avg_tokens_per_doc = (
            round(float(total_tokens / total_documents), 2)
            if total_documents > 0
            else 0.0
        )

        # Capped sample preview: first 3 documents, max 10 tokens per document
        sample_tokens = [
            tokens[:10] for tokens in unigrams_series.head(3).tolist()
        ]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "sample_tokens": sample_tokens,
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={"text_column": text_column, "lowercase": lowercase},
            summary=summary,
        )
        return summary

    @tool
    def sample_data_tool(n: int = 1000, random_state: int = 42) -> dict:
        """
        Subsamples session.dataframe down to n rows using pandas.DataFrame.sample.

        Args:
            n: Target number of rows to sample. Defaults to 1000.
            random_state: Random seed for reproducible sampling. Defaults to 42.

        Returns:
            Dict containing result_id, original_rows, sampled_rows,
            sampled_indices (capped at 20), random_state, and dtm_reset.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        df = session.dataframe
        original_rows = len(df)

        dtm_reset = False
        if n < original_rows:
            sampled_df = df.sample(n=n, random_state=random_state)
            session.dataframe = sampled_df.copy()

            if session.feature_matrix is not None or session.feature_names is not None:
                session.feature_matrix = None
                session.feature_names = None
                dtm_reset = True

            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].tolist())

        sampled_rows = len(session.dataframe)
        sampled_indices = session.dataframe.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "original_rows": original_rows,
            "sampled_rows": sampled_rows,
            "sampled_indices": sampled_indices[:20],
            "random_state": random_state,
            "dtm_reset": dtm_reset,
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
        )
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """
        Lists files and directories in a workspace subdirectory.

        Args:
            subdirectory: Path relative to workspace root. Defaults to '.'.

        Returns:
            Dict containing result_id, subdirectory, n_directories, n_files,
            directories (capped at 50), and files (capped at 50).
        """
        repo_root = Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        if not target_path.is_relative_to(repo_root):
            raise ValueError(f"Access denied: Path '{subdirectory}' is outside workspace boundary.")

        if not target_path.exists():
            raise ValueError(f"Path '{subdirectory}' does not exist.")

        if not target_path.is_dir():
            raise ValueError(f"Path '{subdirectory}' is a file, not a directory.")

        all_dirs = []
        all_files = []

        for item in target_path.iterdir():
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                all_dirs.append(item.name)
            elif item.is_file():
                all_files.append(item.name)

        all_dirs.sort()
        all_files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "n_directories": len(all_dirs),
            "n_files": len(all_files),
            "directories": all_dirs[:50],
            "files": all_files[:50],
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
        )
        return summary

    @tool
    def describe_data_tool() -> dict:
        """
        Computes summary statistics on text_length and generates a box plot grouped by category.

        Returns:
            Dict containing result_id, overall_stats, and by_category_stats.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or does not contain a 'text' column.")

        # Compute character length of text column
        session.dataframe["text_length"] = session.dataframe["text"].fillna("").apply(len)

        def clean_describe_stats(series):
            desc = series.describe().to_dict()
            cleaned = {}
            for k, v in desc.items():
                if k == "count":
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = round(float(v), 4)
            return cleaned

        overall_stats = clean_describe_stats(session.dataframe["text_length"])

        group_col = None
        if "category_name" in session.dataframe.columns:
            group_col = "category_name"
        elif "category" in session.dataframe.columns:
            group_col = "category"

        by_category_stats = {}
        if group_col is not None:
            for cat_name, group_df in session.dataframe.groupby(group_col):
                by_category_stats[str(cat_name)] = clean_describe_stats(group_df["text_length"])

        # Generate box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if group_col is not None:
            sns.boxplot(data=session.dataframe, x=group_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=session.dataframe["text_length"], ax=ax)
            ax.set_xlabel("All Documents")

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Distribution of Document Text Length")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
        )
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """
        Returns a sample preview of session.dataframe rows and metadata.

        Args:
            n: Number of sample rows to return (default 5).

        Returns:
            Dict containing result_id, shape, columns, and sample_rows.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        df = session.dataframe
        sample_df = df.head(n)

        sample_rows = []
        for idx, row in sample_df.iterrows():
            row_dict = {"_index": int(idx)}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (list, tuple, dict, set)):
                    row_dict[col] = f"[list with {len(val)} items]"
                elif isinstance(val, str):
                    if len(val) > 200:
                        row_dict[col] = val[:200] + "..."
                    else:
                        row_dict[col] = val
                elif pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (int, float, bool)):
                    # Convert numpy ints/floats to standard Python primitives
                    if hasattr(val, "item"):
                        row_dict[col] = val.item()
                    else:
                        row_dict[col] = val
                else:
                    row_dict[col] = str(val)
            sample_rows.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "sample_rows": sample_rows,
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
        )
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        list_files_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
