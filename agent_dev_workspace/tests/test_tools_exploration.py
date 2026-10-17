import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student-built build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer 16 assertions:
    # 0 vs 1 = 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 3 vs 4 = 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc1_index": 3, "doc2_index": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 0 vs 3 = 0.0909, shared_terms == ["always"]
    res_0_3 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)
    assert res_0_3["shared_terms"] == ["always"]

    # 0 vs 7 = 0.0 exactly
    res_0_7 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 7})
    assert res_0_7["similarity"] == 0.0

    # Text path: doc1_text vs doc2_index=1
    res_text = cosine_similarity_tool.invoke({
        "doc1_text": "always alpha alpha alpha beta",
        "doc2_index": 1
    })
    assert res_text["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Invalid index label raises ValueError
    with pytest.raises(ValueError, match="Document index label 99 not found"):
        cosine_similarity_tool.invoke({"doc1_index": 99, "doc2_index": 0})


def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student-built build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({})

    # Assert term order descending by variance
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Full 5x5 expected correlation matrix from Known Answer 19
    expected_matrix = [
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746,  0.3780],
        [0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [0.2601,  0.3140,  0.3780,  0.2928,  1.0000],
    ]

    actual_matrix = res["correlation_matrix"]
    assert len(actual_matrix) == 5
    for row in actual_matrix:
        assert len(row) == 5

    for i in range(5):
        for j in range(5):
            assert actual_matrix[i][j] == pytest.approx(expected_matrix[i][j], abs=1e-4)

    # Spot-checks:
    # alpha (idx 0) vs beta (idx 3) = 0.8885
    assert actual_matrix[0][3] == pytest.approx(0.8885, abs=1e-4)
    # gamma (idx 1) vs delta (idx 2) = 0.8307
    assert actual_matrix[1][2] == pytest.approx(0.8307, abs=1e-4)

    # Confirm diagonal is all 1.0
    for i in range(5):
        assert actual_matrix[i][i] == pytest.approx(1.0, abs=1e-4)

    # Confirm symmetry
    np_matrix = np.array(actual_matrix)
    assert np.allclose(np_matrix, np_matrix.T, atol=1e-4)

    # Confirm heatmap plot figure assigned to session
    assert session.pending_figure is not None
