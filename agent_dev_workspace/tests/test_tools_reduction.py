"""Tests for tools_reduction.py (reduce_dimensions_tool and binarize_labels_tool).

Tests known answers 7, 8, and 9 from TEST_FIXTURE.md:
- PCA 2D coordinates and explained_variance_ratio
- t-SNE 2D coordinates structural assertions with perplexity=2, random_state=42
- UMAP 2D coordinates with n_neighbors=3, random_state=42
- Verification of session.pending_figure set on every run
- ValueError raised when perplexity >= n_samples
- Binarize labels tool: 1-hot encoding verification on fixture and 3-class dynamic column growth
"""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_dtm_session():
    """Helper to initialize SessionState and populate feature_matrix via load_dataset_tool and build_dtm_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    build_dtm_tool.invoke({})
    return session


def test_reduce_dimensions_pca_known_answer_7():
    """Test PCA 2D coordinates and explained_variance_ratio against known answer #7."""
    session = _setup_dtm_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_var_ratio = [0.7532, 0.1965]

    session.pending_figure = None
    summary = reduce_dimensions_tool.invoke({"method": "pca"})

    assert session.pending_figure is not None
    assert summary["method"] == "pca"
    assert summary["n_components"] == 2

    # Verify explained variance ratio
    assert np.allclose(summary["explained_variance_ratio"], expected_var_ratio, atol=1e-3)

    # Verify full 8x2 coordinates from full_report
    full_report = session.results_store[summary["result_id"]]["full_report"]
    actual_coords = full_report[["Dim 1", "Dim 2"]].to_numpy()

    # Exact comparison against deterministic PCA output
    assert np.allclose(actual_coords, expected_pca, atol=1e-3)


def test_reduce_dimensions_tsne_structural_checks():
    """Test t-SNE structural properties: duplicate rows get identical coordinates, pending_figure set."""
    session = _setup_dtm_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    session.pending_figure = None
    summary = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42,
    })

    assert session.pending_figure is not None
    assert summary["method"] == "tsne"

    full_report = session.results_store[summary["result_id"]]["full_report"]
    actual_coords = full_report[["Dim 1", "Dim 2"]].to_numpy()

    # Structural check 1: Exact duplicate rows 5 and 6 must have identical reduced coordinates
    assert np.allclose(actual_coords[5], actual_coords[6], atol=1e-3)

    # Structural check 2: max distance among catA rows 0-2 < min distance between any catA row (0-2) and any catB row (3-6)
    catA_indices = [0, 1, 2]
    catB_indices = [3, 4, 5, 6]

    intra_catA_distances = [
        np.linalg.norm(actual_coords[i] - actual_coords[j])
        for i in catA_indices for j in catA_indices if i < j
    ]
    inter_catA_catB_distances = [
        np.linalg.norm(actual_coords[i] - actual_coords[j])
        for i in catA_indices for j in catB_indices
    ]

    max_intra_A = max(intra_catA_distances)
    min_inter_A_B = min(inter_catA_catB_distances)

    assert max_intra_A < min_inter_A_B


def test_reduce_dimensions_umap_known_answer_9():
    """Test UMAP 2D coordinates against known answer #9 with n_neighbors=3, random_state=42."""
    session = _setup_dtm_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    session.pending_figure = None
    summary = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })

    assert session.pending_figure is not None
    assert summary["method"] == "umap"

    full_report = session.results_store[summary["result_id"]]["full_report"]
    actual_coords = full_report[["Dim 1", "Dim 2"]].to_numpy()

    assert np.allclose(actual_coords, expected_umap, atol=1e-1)


def test_reduce_dimensions_invalid_perplexity():
    """Test that t-SNE raises ValueError when perplexity >= n_samples."""
    session = _setup_dtm_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    with pytest.raises(ValueError, match="perplexity"):
        reduce_dimensions_tool.invoke({
            "method": "tsne",
            "perplexity": 8,
            "random_state": 42,
        })


def test_binarize_labels_tool():
    """Test binarize_labels_tool one-hot encoding on fixture data and 3-class expansion."""
    session = _setup_dtm_session()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    summary = binarize_labels_tool.invoke({})

    assert summary["classes"] == ["catA", "catB"]
    assert summary["shape"] == [8, 2]
    assert summary["class_counts"] == {"catA": 4, "catB": 4}

    full_report = session.results_store[summary["result_id"]]["full_report"]
    matrix = full_report[["catA", "catB"]].to_numpy()

    # Check row 0 (catA -> [1, 0]) and row 3 (catB -> [0, 1])
    assert np.array_equal(matrix[0], [1, 0])
    assert np.array_equal(matrix[3], [0, 1])

    # Check column sums [4, 4] and row sums equal to 1 for all rows
    assert np.array_equal(matrix.sum(axis=0), [4, 4])
    assert np.all(matrix.sum(axis=1) == 1)

    # Store in session.artifacts check
    assert "binarized_labels" in session.artifacts

    # Dynamic column growth check: modify row 7 category_name to "catC"
    session.dataframe.loc[7, "category_name"] = "catC"
    summary_3class = binarize_labels_tool.invoke({})

    assert summary_3class["classes"] == ["catA", "catB", "catC"]
    assert summary_3class["shape"] == [8, 3]


def test_binarize_labels_tool_missing_labels():
    """Test binarize_labels_tool raises ValueError when no labels exist in session."""
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    with pytest.raises(ValueError, match="No category labels found"):
        binarize_labels_tool.invoke({})
