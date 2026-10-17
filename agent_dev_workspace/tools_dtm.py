from typing import List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> dict:
        """
        Builds a Document-Term Matrix (DTM) from text in session.dataframe.

        Args:
            max_features: Maximum number of terms to keep in vocabulary.
            min_df: Minimum document frequency threshold (count or proportion).
            max_df: Maximum document frequency threshold (count or proportion).
            ngram_range: Lower and upper boundary of n-grams as [min_n, max_n].

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column.")

        texts = session.dataframe["text"].fillna("")
        ngram_tuple = (ngram_range[0], ngram_range[1])

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        dtm_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = dtm_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_documents, n_terms = dtm_matrix.shape
        non_zero = int(dtm_matrix.nnz)
        total_elements = int(n_documents * n_terms)
        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements)
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "min_df": min_df, "max_df": max_df, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> dict:
        """
        Aggregates term frequencies across all documents in the document-term matrix.

        Args:
            top_n: Number of top terms to include in the summary dictionary.

        Returns:
            Dict containing result_id, total_terms, and top_terms dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix is missing. Run build_dtm_tool first.")

        counts = session.feature_matrix.sum(axis=0).A1

        term_counts = [
            (term, int(count))
            for term, count in zip(session.feature_names, counts)
        ]

        term_counts.sort(key=lambda x: (-x[1], x[0]))

        full_report = pd.DataFrame(term_counts, columns=["term", "frequency"])

        top_terms = dict(term_counts[:top_n])

        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "top_terms": top_terms,
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_report,
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """
        Generates a heatmap of a positional slice of the Document-Term Matrix (DTM).

        Args:
            n_terms: Number of initial vocabulary terms (columns) to include.
            n_documents: Number of initial documents (rows) to include.

        Returns:
            Dict containing result_id, n_terms, n_documents, terms, document_indices,
            and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix is missing. Run build_dtm_tool first.")

        # Positional slice of DTM
        sub_matrix = session.feature_matrix[:n_documents, :n_terms].toarray()
        sliced_n_docs, sliced_n_terms = sub_matrix.shape

        terms = session.feature_names[:sliced_n_terms]
        doc_indices = list(range(sliced_n_docs))

        # Build heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, sliced_n_terms * 0.8), max(4, sliced_n_docs * 0.5)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms,
            yticklabels=doc_indices,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap (First {sliced_n_docs} Docs x {sliced_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": sliced_n_docs,
            "n_terms": sliced_n_terms,
            "terms": terms,
            "document_indices": doc_indices,
            "matrix_slice": sub_matrix.tolist(),
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
