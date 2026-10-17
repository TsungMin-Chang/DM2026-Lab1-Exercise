"""Tools for dimensionality reduction and label binarization.

This module provides tools for reducing document-term matrices to 2D
coordinates using PCA, t-SNE, or UMAP, and binarizing category labels.
"""

from typing import Any, Dict, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Build and return dimensionality reduction and label binarization tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        max_features: int = 500,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduce document-term matrix dimensions to 2D using PCA, t-SNE, or UMAP.

        Generates 2D coordinates for all documents in session.feature_matrix and
        builds a scatter plot colored by category set on session.pending_figure.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap").
            perplexity: Perplexity for t-SNE (must be < n_samples).
            n_neighbors: Number of neighbors for UMAP (must be < n_samples).
            max_features: Maximum top-variance features to retain before densifying.
            random_state: Random seed for reproducible output.

        Returns:
            Dict summarizing the dimensional reduction result.
        """
        if session.feature_matrix is None:
            raise ValueError("No feature matrix found in session. Run build_dtm_tool first.")

        X = session.feature_matrix
        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("Feature matrix is empty.")

        # Check parameter bounds strictly
        if method == "tsne" and perplexity >= n_samples:
            raise ValueError(
                f"perplexity ({perplexity}) must be strictly less than number of samples ({n_samples})."
            )
        if method == "umap" and n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors ({n_neighbors}) must be strictly less than number of samples ({n_samples})."
            )

        # Sparse-friendly top-N variance filtering to avoid huge dense matrices
        narrowed_features = False
        if n_features > max_features:
            narrowed_features = True
            if issparse(X):
                # E[X^2] - (E[X])^2 computed without full densification
                mean = np.array(X.mean(axis=0)).ravel()
                mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
                variances = mean_sq - (mean ** 2)
            else:
                variances = np.var(X, axis=0)

            top_indices = np.argsort(variances)[-max_features:]
            top_indices = np.sort(top_indices)  # preserve original order
            X_proc = X[:, top_indices]
        else:
            X_proc = X

        # Densify only after narrowing down features
        if issparse(X_proc):
            X_dense = X_proc.toarray()
        else:
            X_dense = np.asarray(X_proc)

        # Apply reduction algorithm (n_components fixed at 2)
        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = [
                round(float(v), 4) for v in reducer.explained_variance_ratio_
            ]
        elif method == "tsne":
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Extract category labels for coloring
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels
        else:
            labels = ["Uncategorized"] * n_samples

        # Create scatter plot and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=str(label),
                alpha=0.7,
                edgecolors="none",
            )

        ax.set_title(f"{method.upper()} 2D Projection")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(title="Category")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Build coords DataFrame using session.dataframe.index if available
        doc_index = session.dataframe.index if session.dataframe is not None else range(n_samples)
        coords_df = pd.DataFrame(
            coords, columns=["Dim 1", "Dim 2"], index=doc_index
        )
        coords_df["category"] = labels

        session.artifacts["reduced_coordinates"] = coords
        session.artifacts[f"reduced_{method}"] = coords_df

        preview = [
            [round(float(row[0]), 4), round(float(row[1]), 4)]
            for row in coords[:5]
        ]

        result_id = session.next_result_id("reduce_dim")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_documents": n_samples,
            "n_components": 2,
            "features_used": X_proc.shape[1],
            "narrowed_features": narrowed_features,
            "coordinates_preview": preview,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args_dict = {
            "method": method,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "max_features": max_features,
            "random_state": random_state,
        }
        session.store_result("reduce_dimensions_tool", args_dict, summary, full_report=coords_df)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode document category labels into a binary matrix with one column per category.

        Extracts category labels from session state, sorts unique classes alphabetically,
        and constructs a binary matrix ensuring one column per class (even for 2 classes).

        Returns:
            Dict summarizing class names, shape, per-class counts, and a 5-row preview.
        """
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels

        if labels is None or len(labels) == 0:
            raise ValueError("No category labels found in session.")

        classes = sorted(list(np.unique(labels)))
        n_samples = len(labels)

        # Build 1-hot matrix ensuring one column per class
        one_hot = np.zeros((n_samples, len(classes)), dtype=int)
        for i, label in enumerate(labels):
            col_idx = classes.index(label)
            one_hot[i, col_idx] = 1

        doc_index = session.dataframe.index if session.dataframe is not None else range(n_samples)
        df_binarized = pd.DataFrame(one_hot, columns=classes, index=doc_index)

        # Store in artifacts
        session.artifacts["binarized_labels"] = df_binarized

        class_counts = {cls: int((df_binarized[cls] == 1).sum()) for cls in classes}

        preview = [
            {"label": str(labels[i]), "one_hot": [int(x) for x in one_hot[i]]}
            for i in range(min(5, n_samples))
        ]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(one_hot.shape),
            "class_counts": class_counts,
            "preview": preview,
        }

        session.store_result("binarize_labels_tool", {}, summary, full_report=df_binarized)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
