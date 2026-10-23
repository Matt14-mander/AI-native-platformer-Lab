import copy

import pytest
from pydantic import ValidationError

from ai_platformer.content.generation_suite import (
    GenerationCase,
    GenerationSuite,
    compare_expectations,
    load_suite,
)

SUITE = "game_content/generation_suites/small_v1.json"


def test_frozen_small_suite_has_18_distinct_map_ids_and_three_coin_layouts():
    suite = load_suite(SUITE)
    assert len(suite.cases) == 6
    assert len(suite.generation_seeds) == 3
    ids = [
        case.expected_request(seed).level_id
        for case in suite.cases
        for seed in suite.generation_seeds
    ]
    assert len(ids) == len(set(ids)) == 18
    assert {case.expected["coin_layout"] for case in suite.cases} == {"none", "ground", "low_jump"}


@pytest.mark.parametrize(
    "expected",
    [
        {"seed": 1},
        {"level_id": "override"},
        {"schema_version": 1},
        {"obstacle_count": True},
        {"obstacle_count": 9},
        {"obstacle_count": 8, "gap_count": 8},
        {"coin_layout": "none"},
        {"gap_width": {"minimum": 176, "maximum": 96}},
        {"code": "print(1)"},
    ],
)
def test_expected_design_is_strict_and_cannot_override_pins(expected):
    with pytest.raises((ValueError, ValidationError)):
        GenerationCase(case_id="test", prompt="test", expected=expected)


def test_suite_rejects_duplicate_seeds_and_ids():
    data = load_suite(SUITE).model_dump()
    duplicate_seeds = copy.deepcopy(data)
    duplicate_seeds["generation_seeds"] = [1, 1]
    with pytest.raises(ValidationError):
        GenerationSuite.model_validate(duplicate_seeds)
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValidationError):
        GenerationSuite.model_validate(data)


def test_semantics_compare_nested_explicit_fields_without_changing_request():
    case = load_suite(SUITE).cases[1]
    request = case.expected_request(42)
    assert compare_expectations(request, case.expected)["passed"]
    changed = request.model_copy(update={"obstacle_count": 2})
    report = compare_expectations(changed, case.expected)
    assert not report["passed"]
    assert report["mismatches"] == [{"field": "obstacle_count", "expected": 3, "actual": 2}]
    assert changed.obstacle_count == 2


def test_suite_version_is_not_a_boolean_and_budget_is_bounded():
    data = load_suite(SUITE).model_dump()
    data["schema_version"] = True
    with pytest.raises(ValidationError):
        GenerationSuite.model_validate(data)
    data["schema_version"] = 1
    data["generation_seeds"] = list(range(8))
    template = data["cases"][0]
    data["cases"] = [{**template, "case_id": f"case_{index}"} for index in range(9)]
    with pytest.raises(ValidationError, match="64 candidates"):
        GenerationSuite.model_validate(data)
