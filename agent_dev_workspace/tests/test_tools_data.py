import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def test_check_missing_tool_known_answer_2():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student-built check_missing_tool
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Run check_missing_tool
    result = check_missing_tool.invoke({})

    # Assert known answer 2 values
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert result["has_missing"] is True
    assert result["total_rows"] == 8

    # Confirm the dataframe still has all 8 rows (tool reports only, no dropping)
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_known_answer_3():
    # Part 1: drop=False check
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_data_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    res_no_drop = check_duplicates_tool1.invoke({"drop": False})
    assert res_no_drop["n_duplicates"] == 1
    assert res_no_drop["duplicate_indices"] == [6]
    assert len(session1.dataframe) == 8
    assert res_no_drop["dropped"] is False
    assert res_no_drop["rows_removed"] == 0

    # Part 2: drop=True with pre-built DTM on fresh session
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools2 = make_dtm_tools(session2)
    build_dtm_tool2 = next(t for t in dtm_tools2 if t.name == "build_dtm_tool")
    build_dtm_tool2.invoke({})
    assert session2.feature_matrix is not None

    data_tools2 = make_data_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    res_drop = check_duplicates_tool2.invoke({"drop": True})
    assert res_drop["rows_removed"] == 2
    assert res_drop["total_rows_after"] == 6
    assert list(session2.dataframe.index) == [0, 1, 2, 3, 4, 7]
    assert res_drop["dtm_reset"] is True
    assert session2.feature_matrix is None
    assert session2.feature_names is None


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student-built tokenize_tool
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    res = tokenize_tool.invoke({})

    # Assert total_tokens and sample_tokens constraints
    assert res["total_tokens"] == 27
    assert len(res["sample_tokens"]) == 3
    for sample in res["sample_tokens"]:
        assert len(sample) <= 10

    # Expected token lists from Known Answer 10
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_unigrams


def test_sample_data_tool_known_answer_13():
    # Part 1: n=4, random_state=42 (Known Answer 13)
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_data_tools(session1)
    sample_data_tool1 = next(t for t in data_tools1 if t.name == "sample_data_tool")

    res1 = sample_data_tool1.invoke({"n": 4, "random_state": 42})
    assert res1["sampled_indices"] == [1, 5, 0, 7]
    assert len(session1.dataframe) == 4
    assert list(session1.dataframe.index) == [1, 5, 0, 7]

    # Part 2: n=100 (larger than total rows) with pre-built DTM on fresh session
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools2 = make_dtm_tools(session2)
    build_dtm_tool2 = next(t for t in dtm_tools2 if t.name == "build_dtm_tool")
    build_dtm_tool2.invoke({})
    assert session2.feature_matrix is not None

    data_tools2 = make_data_tools(session2)
    sample_data_tool2 = next(t for t in data_tools2 if t.name == "sample_data_tool")

    res2 = sample_data_tool2.invoke({"n": 100})
    assert res2["sampled_rows"] == 8
    assert res2["dtm_reset"] is False
    assert list(session2.dataframe.index) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert session2.feature_matrix is not None


def test_list_files_tool_known_answer_11():
    session = SessionState()

    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer 11: subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["n_directories"] == 0
    assert res["n_files"] == 1

    # Test path traversal protection: subdirectory="../.." must raise ValueError
    with pytest.raises(ValueError, match="outside workspace boundary"):
        list_files_tool.invoke({"subdirectory": "../.."})


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student-built describe_data_tool
    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Run describe_data_tool
    res = describe_data_tool.invoke({})

    # Check text_length column added to session.dataframe
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert session.dataframe["text_length"].tolist() == expected_lengths

    # Assert overall stats matching Known Answer 18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-4)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-4)
    assert overall["min"] == pytest.approx(0.0, abs=1e-4)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-4)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-4)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-4)
    assert overall["max"] == pytest.approx(30.0, abs=1e-4)

    # Assert per-category stats
    by_cat = res["by_category_stats"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-4)
    assert catA["std"] == pytest.approx(12.5, abs=1e-4)
    assert catA["min"] == pytest.approx(0.0, abs=1e-4)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-4)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-4)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-4)
    assert catA["max"] == pytest.approx(29.0, abs=1e-4)

    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-4)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-4)
    assert catB["min"] == pytest.approx(18.0, abs=1e-4)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-4)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-4)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-4)
    assert catB["max"] == pytest.approx(30.0, abs=1e-4)

    # Assert box plot figure assigned to session
    assert session.pending_figure is not None


def test_inspect_data_tool_custom_design():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # (1) Right after loading: shape == [8, 3] and columns include text, category_name, category
    res1 = inspect_data_tool.invoke({"n": 8})
    assert res1["shape"] == [8, 3]
    for col in ["text", "category_name", "category"]:
        assert col in res1["columns"]

    # (2) With n=8: 8 sample_rows, row 7 text is "", category is int (type(...) is int)
    assert len(res1["sample_rows"]) == 8
    assert res1["sample_rows"][7]["text"] == ""
    assert type(res1["sample_rows"][0]["category"]) is int

    # (3) After tokenize_tool: row 0 unigrams shows "[list with 5 items]"
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    tokenize_tool.invoke({})

    res2 = inspect_data_tool.invoke({"n": 5})
    assert res2["sample_rows"][0]["unigrams"] == "[list with 5 items]"

    # (4) Overwrite row 0 text with "x" * 300, re-inspect, expect length 203 ending with "..."
    session.dataframe.loc[0, "text"] = "x" * 300
    res3 = inspect_data_tool.invoke({"n": 5})
    truncated_text = res3["sample_rows"][0]["text"]
    assert len(truncated_text) == 203
    assert truncated_text.endswith("...")
