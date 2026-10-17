"""Tools for frequent pattern mining.

This module provides tools for mining frequent itemsets/patterns from per-category
document-term matrices using PAMI algorithms (FPGrowth, Top-K, MaxFPGrowth).
"""

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Build and return pattern mining tools bound to the given session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns for a specific category using PAMI algorithms.

        Constructs a fresh per-category DTM with English stopwords removed, filters
        vocabulary based on variance, tfidf, or term frequency, converts surviving
        data to a transactional database, and runs FPGrowth, Top-K, or MaxFPGrowth.

        Args:
            category_name: Which category to mine.
            filtering_method: Vocabulary filtering approach ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support count (required for fpgrowth and maxfpgrowth).
            k: Top-K patterns count (required for topk).

        Returns:
            Dict summarizing mined patterns (terms and support), category, and count.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        df = session.dataframe
        if "category_name" not in df.columns:
            raise ValueError("Dataset lacks 'category_name' column.")

        available_categories = sorted(list(df["category_name"].dropna().unique()))
        if category_name not in available_categories:
            raise ValueError(
                f"Category '{category_name}' not found. Available categories: {available_categories}"
            )

        # Validate algorithm parameter dependencies strictly
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            raise ValueError(f"min_sup is required for algorithm '{algorithm}'.")
        if algorithm == "topk" and k is None:
            raise ValueError("k is required for algorithm 'topk'.")

        args_dict = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        # Step 1: Filter to category and build fresh per-category CountVectorizer(stop_words="english")
        cat_df = df[df["category_name"] == category_name]
        cat_texts = cat_df["text"].fillna("").values

        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(cat_texts)
        except ValueError:
            # Handle empty vocabulary gracefully
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "Empty vocabulary after stopword removal.",
            }
            session.store_result("mine_patterns_tool", args_dict, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        n_terms = len(feature_names)

        if n_terms == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "No terms survived initial vectorization.",
            }
            session.store_result("mine_patterns_tool", args_dict, summary)
            return summary

        # Step 2: Calculate filtering metric and keep percentile bands (inclusive lower <= x <= upper)
        X_dense = X.toarray()

        if filtering_method == "variance":
            scores = np.var(X_dense, axis=0)
            p_low, p_high = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= p_low) & (scores <= p_high)
        elif filtering_method == "term_frequency":
            scores = np.sum(X_dense, axis=0)
            p_low, p_high = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= p_low) & (scores <= p_high)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(X)
            scores = np.array(tfidf_mat.mean(axis=0)).ravel()
            p_low = np.percentile(scores, 20)
            keep_mask = scores >= p_low
        else:
            raise ValueError(f"Unsupported filtering_method: {filtering_method}")

        surviving_terms = feature_names[keep_mask]
        X_surviving = X_dense[:, keep_mask]

        result_id = session.next_result_id("patterns")

        # Step 3: Handle zero surviving terms cleanly
        if len(surviving_terms) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args_dict, summary)
            return summary

        # Step 4: Convert surviving terms to transactional DB DataFrame
        # Filter out rows with zero surviving terms
        row_sums = X_surviving.sum(axis=1)
        valid_rows = row_sums > 0
        X_trans = X_surviving[valid_rows]

        if X_trans.shape[0] == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no documents containing surviving terms",
            }
            session.store_result("mine_patterns_tool", args_dict, summary)
            return summary

        df_trans = pd.DataFrame(X_trans, columns=surviving_terms)

        # Prepare temporary transactional database file safely
        temp_db = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        db_path = temp_db.name
        temp_db.close()

        parsed_patterns: List[Dict[str, Any]] = []

        try:
            from PAMI.extras.convert import DF2DB
            df2db = DF2DB.DF2DB(df_trans)
            df2db.convert2TransactionalDatabase(db_path, ">=", 1)

            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                # PAMI returns columns like 'Patterns' and 'Support'
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"])
                    sup_val = int(row["Support"])
                    terms = pat_str.strip().split("\t")
                    parsed_patterns.append({"terms": terms, "support": sup_val})

                # Sort by support descending
                parsed_patterns.sort(key=lambda p: p["support"], reverse=True)

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        # Store in artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = parsed_patterns

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(parsed_patterns),
            "patterns": parsed_patterns[:20],  # cap top-20 in summary
        }

        session.store_result("mine_patterns_tool", args_dict, summary, full_report=parsed_patterns)
        return summary

    return [mine_patterns_tool]
