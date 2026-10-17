import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
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

    # Run variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert summary fields
    assert res["n_kept"] == 4
    assert res["removed_terms"] == ["always"]
    assert session.artifacts["variance_filtered_terms"] == ["alpha", "gamma", "delta", "beta"]

    # Assert feature_matrix shape untouched
    assert session.feature_matrix.shape == (8, 5)

    # Retrieve full_report DataFrame stored in session.results_store
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]

    assert list(full_report.columns) == ["term", "variance"]

    # Expected order and population variances from Known Answer 5
    expected_order = [
        ("alpha", 1.1875),
        ("gamma", 1.109375),
        ("delta", 0.25),
        ("beta", 0.234375),
        ("always", 0.109375),
    ]

    for idx, (term, expected_var) in enumerate(expected_order):
        row_term = full_report.loc[idx, "term"]
        row_var = full_report.loc[idx, "variance"]
        assert row_term == term
        assert row_var == pytest.approx(expected_var, abs=1e-4)


def test_pearson_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Test invalid target_class raises ValueError
    with pytest.raises(ValueError, match="target_class 'catZ' not found"):
        pearson_filter_tool.invoke({"target_class": "catZ"})

    # Run pearson_filter_tool for target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB"})

    # Retrieve full_report DataFrame from results_store
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]

    assert list(full_report.columns) == ["term", "pearson_r"]

    # Expected order by |r| descending and signed Pearson r values from Known Answer 6
    expected_order = [
        ("delta", 1.0),
        ("gamma", 0.8307),
        ("beta", -0.7746),
        ("alpha", -0.6882),
        ("always", 0.3780),
    ]

    for idx, (term, expected_r) in enumerate(expected_order):
        row_term = full_report.loc[idx, "term"]
        row_r = full_report.loc[idx, "pearson_r"]
        assert row_term == term
        assert row_r == pytest.approx(expected_r, abs=1e-4)


def test_spearman_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Part 1: Default max_terms (no narrowing on 5 terms)
    res1 = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res1["narrowed"] is False
    assert res1["terms_evaluated"] == 5

    stored1 = session.results_store[res1["result_id"]]
    full_report1 = stored1["full_report"]

    assert list(full_report1.columns) == ["term", "spearman_r"]

    # Expected order by |r| descending and signed Spearman r values from Known Answer 6
    expected_order = [
        ("delta", 1.0),
        ("gamma", 0.9363),
        ("beta", -0.7746),
        ("alpha", -0.7500),
        ("always", 0.3780),
    ]

    for idx, (term, expected_r) in enumerate(expected_order):
        row_term = full_report1.loc[idx, "term"]
        row_r = full_report1.loc[idx, "spearman_r"]
        assert row_term == term
        assert row_r == pytest.approx(expected_r, abs=1e-4)

    # Part 2: Explicit narrowing with max_terms=3
    res2 = spearman_filter_tool.invoke({"target_class": "catB", "max_terms": 3})
    assert res2["narrowed"] is True
    assert res2["terms_evaluated"] == 3

    stored2 = session.results_store[res2["result_id"]]
    full_report2 = stored2["full_report"]
    evaluated_terms = set(full_report2["term"].tolist())
    assert evaluated_terms == {"alpha", "gamma", "delta"}
