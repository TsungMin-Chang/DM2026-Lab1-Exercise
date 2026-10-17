"""Tests for tools_patterns.py (mine_patterns_tool).

Tests known answers #17 and #20 from TEST_FIXTURE.md:
- FPGrowth pattern mining on tfidf-filtered DTM for catA and catB (known answer #17)
- Zero terms surviving filtering handling for variance/term_frequency methods
- TopK pattern mining on catA and catB for k=1, 2, 3 (known answer #20)
- MaxFPGrowth pattern mining returning 0 patterns (known answer #20)
- Validation errors for missing category, missing min_sup, and missing k
"""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    """Helper to initialize SessionState and load sample fixture."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    return session


def test_mine_patterns_fpgrowth_known_answer_17():
    """Test mine_patterns_tool with FPGrowth and tfidf filtering against known answer #17."""
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # catA, tfidf, fpgrowth, min_sup=1 -> {alpha} support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0] == {"terms": ["alpha"], "support": 3}

    # catB, tfidf, fpgrowth, min_sup=1 -> {gamma} support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0] == {"terms": ["gamma"], "support": 4}

    # catA, min_sup=4 -> 0 patterns
    res_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catA_sup4["n_patterns"] == 0

    # catB, min_sup=4 -> {gamma} support 4
    res_catB_sup4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catB_sup4["n_patterns"] == 1
    assert res_catB_sup4["patterns"][0] == {"terms": ["gamma"], "support": 4}


def test_mine_patterns_zero_surviving_terms_known_answer_17():
    """Test variance and term_frequency filtering return zero surviving terms cleanly."""
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    for filtering in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": filtering,
                "algorithm": "fpgrowth",
                "min_sup": 1,
            })
            assert res["n_patterns"] == 0
            assert res["note"] == "no terms survived filtering"


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """Test TopK and MaxFPGrowth algorithms against known answer #20."""
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # TopK for catA at k=1, 2, 3 -> {alpha} support 3
    for k_val in [1, 2, 3]:
        res_topk_A = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert res_topk_A["n_patterns"] == 1
        assert res_topk_A["patterns"][0] == {"terms": ["alpha"], "support": 3}

    # TopK for catB at k=1, 2, 3 -> {gamma} support 4
    for k_val in [1, 2, 3]:
        res_topk_B = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert res_topk_B["n_patterns"] == 1
        assert res_topk_B["patterns"][0] == {"terms": ["gamma"], "support": 4}

    # MaxFPGrowth for catA and catB at min_sup=1 and min_sup=4 -> 0 patterns
    for cat in ["catA", "catB"]:
        for min_sup_val in [1, 4]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup_val,
            })
            assert res_max["n_patterns"] == 0


def test_mine_patterns_validation_errors():
    """Test mine_patterns_tool raises ValueError on invalid inputs."""
    session = _setup_session()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Invalid category
    with pytest.raises(ValueError, match="Category 'nonexistent' not found"):
        mine_patterns_tool.invoke({
            "category_name": "nonexistent",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        })

    # Missing min_sup for fpgrowth
    with pytest.raises(ValueError, match="min_sup is required"):
        mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
        })

    # Missing k for topk
    with pytest.raises(ValueError, match="k is required"):
        mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
        })
