from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student-built build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert known answer 4 values
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert session.feature_matrix.shape == (8, 5)
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == 47.5

    # Assert artifact storage
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get tools for this session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first, then run term_frequency_tool
    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({"top_n": 5})

    # Assert returned summary top_terms exactly matches expected counts and key ordering
    expected_top_terms = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    assert res["top_terms"] == expected_top_terms
    assert list(res["top_terms"].keys()) == list(expected_top_terms.keys())

    # Check every count in top_terms is a plain int
    for term, count in res["top_terms"].items():
        assert type(count) is int

    # Retrieve full_report from session.results_store via result_id
    result_id = res["result_id"]
    assert result_id in session.results_store
    full_report = session.results_store[result_id]["full_report"]

    # Verify full_report's first two columns are term and frequency
    assert list(full_report.columns[:2]) == ["term", "frequency"]


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get tools for this session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # (1) Test with default parameters (n_terms=20, n_documents=20)
    res_default = dtm_heatmap_tool.invoke({})

    assert res_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert res_default["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Clear pending figure explicitly before second call to verify it produces a new plot
    session.pending_figure = None

    # (2) Test slicing with specific dimensions (n_terms=2, n_documents=3)
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    expected_corner = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    assert res_sliced["matrix_slice"] == expected_corner
    assert res_sliced["n_documents"] == 3
    assert res_sliced["n_terms"] == 2
    assert res_sliced["terms"] == ["alpha", "always"]
    assert session.pending_figure is not None
