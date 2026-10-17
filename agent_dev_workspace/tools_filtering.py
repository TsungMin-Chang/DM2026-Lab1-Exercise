from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool
from sklearn.feature_selection import VarianceThreshold


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> dict:
        """
        Flags terms in session.feature_matrix with population variance strictly greater than threshold.

        Does not modify session.feature_matrix directly, ensuring fitted vectorizer and feature alignment remain intact.
        Stores kept terms in session.artifacts['variance_filtered_terms'].

        Args:
            threshold: Minimum population variance threshold (default 0.15).

        Returns:
            Dict containing result_id, threshold, total_terms, n_kept, n_removed,
            kept_terms (capped at top 20 by variance), and removed_terms (capped at 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or feature_names missing. Run build_dtm_tool first.")

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Efficiently compute population variance on sparse matrix via VarianceThreshold
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(matrix)
        variances = vt.variances_

        kept_terms = []
        removed_terms = []
        term_variance_pairs = []

        for term, var in zip(feature_names, variances):
            var_val = float(var)
            term_variance_pairs.append((term, var_val))
            if var_val > threshold:
                kept_terms.append((term, var_val))
            else:
                removed_terms.append(term)

        # Sort kept_terms by variance descending so top 20 by variance are shown in summary
        kept_terms_sorted = sorted(kept_terms, key=lambda x: x[1], reverse=True)
        kept_terms_names = [t[0] for t in kept_terms_sorted]

        # Store kept terms in session.artifacts for downstream use
        session.artifacts["variance_filtered_terms"] = kept_terms_names

        # Build full_report DataFrame sorted descending by variance
        report_df = pd.DataFrame(term_variance_pairs, columns=["term", "variance"])
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "n_kept": len(kept_terms_names),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms_names[:20],
            "removed_terms": removed_terms[:20],
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str, min_abs_r: float = 0.0) -> dict:
        """
        Computes Pearson correlation between each term's frequency and a one-vs-rest binary target class.

        Args:
            target_class: The category label to evaluate correlation against.
            min_abs_r: Minimum absolute correlation threshold for keeping terms (default 0.0).

        Returns:
            Dict containing result_id, target_class, min_abs_r, total_terms, n_kept, n_removed,
            top_terms (top 20 by absolute r with term name and r value), and kept_terms list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or feature_names missing. Run build_dtm_tool first.")

        if session.labels is None or len(session.labels) == 0:
            raise ValueError("session.labels missing or empty. Load dataset and build DTM first.")

        n_docs, n_terms = session.feature_matrix.shape
        if len(session.labels) != n_docs:
            raise ValueError(f"Mismatch: len(session.labels)={len(session.labels)} does not match feature_matrix row count={n_docs}.")

        categories = session.categories if session.categories else list(set(session.labels))
        if target_class not in categories and target_class not in session.labels:
            raise ValueError(f"target_class '{target_class}' not found in dataset categories: {categories}.")

        # Binary one-vs-rest target vector
        y = (np.array(session.labels) == target_class).astype(float)
        std_y = np.std(y, ddof=0)

        if std_y == 0:
            raise ValueError(f"target_class '{target_class}' is present in all or no documents (std_y = 0).")

        matrix = session.feature_matrix
        y_centered = y - np.mean(y)

        # Covariance via sparse matrix-vector product
        cov = np.ravel(matrix.T.dot(y_centered)) / n_docs

        # Column standard deviations from population variance
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(matrix)
        std_X = np.sqrt(vt.variances_)

        # Vectorized correlation r = cov / (std_X * std_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_vec = cov / (std_X * std_y)
            r_vec = np.nan_to_num(r_vec, nan=0.0)
            r_vec = np.clip(r_vec, -1.0, 1.0)

        records = []
        kept_terms = []
        removed_terms = []

        for term, r_val in zip(session.feature_names, r_vec):
            r_float = round(float(r_val), 4)
            abs_r = abs(r_float)
            records.append((term, r_float, abs_r))

            if abs_r >= min_abs_r:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Sort records by absolute r descending
        records_sorted = sorted(records, key=lambda x: x[2], reverse=True)

        # Build full_report DataFrame with term and pearson_r sorted by |r|
        report_df = pd.DataFrame([(r[0], r[1]) for r in records_sorted], columns=["term", "pearson_r"])

        # Store kept terms in session.artifacts
        session.artifacts[f"pearson_filtered_{target_class}"] = kept_terms

        top_terms = [{"term": r[0], "r": r[1]} for r in records_sorted[:20]]

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "min_abs_r": min_abs_r,
            "total_terms": n_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_terms": top_terms,
            "kept_terms": kept_terms[:20],
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "min_abs_r": min_abs_r},
            summary=summary,
            full_report=report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        min_abs_r: float = 0.0,
        max_terms: int = 1500,
    ) -> dict:
        """
        Computes Spearman rank correlation between term frequencies and a one-vs-rest target class.

        Narrows terms to top max_terms by variance first if total_terms > max_terms.

        Args:
            target_class: The category label to evaluate correlation against.
            min_abs_r: Minimum absolute correlation threshold for keeping terms (default 0.0).
            max_terms: Pre-filtering variance cap to limit rank computations (default 1500).

        Returns:
            Dict containing result_id, target_class, min_abs_r, total_terms, terms_evaluated,
            narrowed, n_kept, n_removed, top_terms, and kept_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or feature_names missing. Run build_dtm_tool first.")

        if session.labels is None or len(session.labels) == 0:
            raise ValueError("session.labels missing or empty. Load dataset and build DTM first.")

        n_docs, n_terms = session.feature_matrix.shape
        if len(session.labels) != n_docs:
            raise ValueError(f"Mismatch: len(session.labels)={len(session.labels)} does not match feature_matrix row count={n_docs}.")

        categories = session.categories if session.categories else list(set(session.labels))
        if target_class not in categories and target_class not in session.labels:
            raise ValueError(f"target_class '{target_class}' not found in dataset categories: {categories}.")

        # Binary one-vs-rest target vector
        y = (np.array(session.labels) == target_class).astype(float)
        if np.std(y, ddof=0) == 0:
            raise ValueError(f"target_class '{target_class}' is present in all or no documents (std_y = 0).")

        # Rank target vector using average ranks for ties
        ry = rankdata(y, method="average")
        std_ry = np.std(ry, ddof=0)
        ry_centered = ry - np.mean(ry)

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Narrow to max_terms by variance if dataset has more features
        narrowed = False
        if n_terms > max_terms:
            vt = VarianceThreshold(threshold=0.0)
            vt.fit(matrix)
            top_indices = np.argsort(vt.variances_)[::-1][:max_terms]
            matrix_eval = matrix[:, top_indices]
            feature_names_eval = [feature_names[i] for i in top_indices]
            narrowed = True
        else:
            matrix_eval = matrix
            feature_names_eval = feature_names

        terms_evaluated = len(feature_names_eval)

        # Convert evaluated matrix columns to dense arrays for rankdata (average ties)
        if hasattr(matrix_eval, "toarray"):
            dense_eval = matrix_eval.toarray()
        else:
            dense_eval = np.asarray(matrix_eval)

        records = []
        kept_terms = []
        removed_terms = []

        for idx, term in enumerate(feature_names_eval):
            x_col = dense_eval[:, idx]
            std_x = np.std(x_col, ddof=0)
            if std_x == 0:
                r_val = 0.0
            else:
                rx = rankdata(x_col, method="average")
                std_rx = np.std(rx, ddof=0)
                if std_rx == 0:
                    r_val = 0.0
                else:
                    rx_centered = rx - np.mean(rx)
                    r_val = np.dot(rx_centered, ry_centered) / (n_docs * std_rx * std_ry)
                    r_val = np.clip(r_val, -1.0, 1.0)

            r_float = round(float(r_val), 4)
            abs_r = abs(r_float)
            records.append((term, r_float, abs_r))

            if abs_r >= min_abs_r:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Sort records by |r| descending
        records_sorted = sorted(records, key=lambda x: x[2], reverse=True)

        report_df = pd.DataFrame([(r[0], r[1]) for r in records_sorted], columns=["term", "spearman_r"])

        session.artifacts[f"spearman_filtered_{target_class}"] = kept_terms

        top_terms = [{"term": r[0], "r": r[1]} for r in records_sorted[:20]]

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "min_abs_r": min_abs_r,
            "total_terms": n_terms,
            "terms_evaluated": terms_evaluated,
            "narrowed": narrowed,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_terms": top_terms,
            "kept_terms": kept_terms[:20],
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "min_abs_r": min_abs_r, "max_terms": max_terms},
            summary=summary,
            full_report=report_df,
        )
        return summary

    return [
        variance_filter_tool,
        pearson_filter_tool,
        spearman_filter_tool,
    ]
