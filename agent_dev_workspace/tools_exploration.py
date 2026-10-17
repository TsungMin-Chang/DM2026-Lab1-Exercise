from typing import Optional, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc1_index: Optional[int] = None,
        doc2_index: Optional[int] = None,
        doc1_text: Optional[str] = None,
        doc2_text: Optional[str] = None,
    ) -> dict:
        """
        Computes cosine similarity between two documents specified by DataFrame index label or raw text string.

        Args:
            doc1_index: DataFrame index label for document 1.
            doc2_index: DataFrame index label for document 2.
            doc1_text: Raw text string for document 1.
            doc2_text: Raw text string for document 2.

        Returns:
            Dict containing result_id, doc1_identifier, doc2_identifier, similarity, and shared_terms (up to 10).
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is missing. Load a dataset first.")

        # Ensure exactly one of doc1_index or doc1_text is provided
        if (doc1_index is None and doc1_text is None) or (doc1_index is not None and doc1_text is not None):
            raise ValueError("Must provide exactly one of doc1_index or doc1_text.")

        # Ensure exactly one of doc2_index or doc2_text is provided
        if (doc2_index is None and doc2_text is None) or (doc2_index is not None and doc2_text is not None):
            raise ValueError("Must provide exactly one of doc2_index or doc2_text.")

        # Helper to get vector and display identifier for doc 1 / doc 2
        def get_doc_vector_and_id(doc_index: Optional[int], doc_text: Optional[str]):
            if doc_index is not None:
                if doc_index not in session.dataframe.index:
                    raise ValueError(f"Document index label {doc_index} not found in session.dataframe.")
                pos = session.dataframe.index.get_loc(doc_index)
                if session.feature_matrix is None:
                    raise ValueError("session.feature_matrix missing. Run build_dtm_tool first.")
                vec = session.feature_matrix[pos]
                doc_id = f"index_{doc_index}"
            else:
                vectorizer = session.artifacts.get("count_vectorizer")
                if vectorizer is None:
                    raise ValueError("count_vectorizer missing from session.artifacts. Run build_dtm_tool first.")
                vec = vectorizer.transform([doc_text])
                doc_id = f"text_({doc_text[:20]}...)" if len(doc_text) > 20 else f"text_({doc_text})"
            return vec, doc_id

        vec1, id1 = get_doc_vector_and_id(doc1_index, doc1_text)
        vec2, id2 = get_doc_vector_and_id(doc2_index, doc2_text)

        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim_val = float(sim_matrix[0, 0])
            sim_val = np.nan_to_num(sim_val, nan=0.0)
            sim_val = float(np.clip(sim_val, 0.0, 1.0))

        # Identify shared terms (up to 10) if feature_names exists
        shared_terms = []
        if session.feature_names is not None:
            dense1 = np.ravel(vec1.toarray())
            dense2 = np.ravel(vec2.toarray())
            for feat_idx, (c1, c2) in enumerate(zip(dense1, dense2)):
                if c1 > 0 and c2 > 0:
                    shared_terms.append(session.feature_names[feat_idx])
                if len(shared_terms) >= 10:
                    break

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc1_identifier": id1,
            "doc2_identifier": id2,
            "similarity": round(sim_val, 4),
            "shared_terms": shared_terms,
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={
                "doc1_index": doc1_index,
                "doc2_index": doc2_index,
                "doc1_text": doc1_text,
                "doc2_text": doc2_text,
            },
            summary=summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """
        Computes feature-vs-feature Pearson correlation matrix for the top 20 terms by variance and generates a heatmap.

        Returns:
            Dict containing result_id, terms, and correlation_matrix (nested list rounded to 4 decimals).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or feature_names missing. Run build_dtm_tool first.")

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute population variance on sparse matrix
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(matrix)
        variances = vt.variances_

        # Sort indices by variance descending, take top 20
        top_indices = np.argsort(variances)[::-1][:20]
        top_terms = [feature_names[i] for i in top_indices]

        # Slice sparse matrix to top columns and densify only those columns
        sliced_sparse = matrix[:, top_indices]
        if hasattr(sliced_sparse, "toarray"):
            dense_top = sliced_sparse.toarray()
        else:
            dense_top = np.asarray(sliced_sparse)

        # Compute Pearson correlation with rowvar=False (columns are variables/terms)
        corr_matrix = np.corrcoef(dense_top, rowvar=False)

        # Handle scalar case if single feature
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0]])

        # Handle NaNs from zero-variance columns while keeping diagonal == 1.0
        nan_mask = np.isnan(corr_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Round to 4 decimal places for summary / assertion readability
        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        # Build heatmap plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Top-20 Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_matrix_rounded,
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={},
            summary=summary,
        )
        return summary

    return [
        cosine_similarity_tool,
        feature_correlation_matrix_tool,
    ]
