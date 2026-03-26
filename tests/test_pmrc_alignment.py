from __future__ import annotations

from collections.abc import Callable
import math

import pytest

from tests.gold_fixtures import (
    FIXTURES,
    GoldRow,
    ScenarioFixture,
    find_duplicate_age_rows,
    fixture_initial_row,
    format_failure,
    load_gold_rows,
    relative_error,
    simulate_pmrc_rows,
)


STRICT_COLUMNS: dict[str, Callable[[GoldRow], float]] = {
    "age": lambda row: row.age,
    "tpa": lambda row: row.tpa,
    "hd": lambda row: row.hd,
    "ba": lambda row: row.ba,
    "volume_tvob": lambda row: row.volume_tvob,
    "yield_dwib": lambda row: row.yield_dwib,
    "dq": lambda row: row.dq,
    "volume_pulp": lambda row: row.volume_pulp,
    "volume_cns": lambda row: row.volume_cns,
    "volume_saw": lambda row: row.volume_saw,
    "green_pulp": lambda row: row.green_pulp,
    "green_cns": lambda row: row.green_cns,
    "green_saw": lambda row: row.green_saw,
    "remove_pulp": lambda row: row.remove_pulp,
    "remove_cns": lambda row: row.remove_cns,
    "remove_saw": lambda row: row.remove_saw,
}


def _assert_close(actual: float, expected: float, fixture_name: str, age: float, column: str) -> None:
    abs_tol = 1e-6
    rel_tol = 1e-6
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"age={age} "
        + format_failure(
            fixture_name=fixture_name,
            label=column,
            actual=actual,
            expected=expected,
        )
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_fixture_metadata_matches_scenario_definition(fixture: ScenarioFixture) -> None:
    initial_row = fixture_initial_row(fixture)
    assert fixture.region == "pucp", f"fixture={fixture.name} field=region failed: expected='pucp', actual={fixture.region!r}"
    assert fixture.age0 == 5.0, f"fixture={fixture.name} field=age0 failed: expected=5.0, actual={fixture.age0}"
    assert fixture.end_age == 35.0, f"fixture={fixture.name} field=end_age failed: expected=35.0, actual={fixture.end_age}"
    assert fixture.tpa0 == initial_row.tpa, format_failure(
        fixture_name=fixture.name,
        label="fixture_tpa0_vs_gold_initial_tpa",
        actual=fixture.tpa0,
        expected=initial_row.tpa,
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_pmrc_initial_row_matches_gold_fixture(fixture: ScenarioFixture) -> None:
    gold_initial = fixture_initial_row(fixture)
    model_initial = simulate_pmrc_rows(fixture)[0]

    for column, accessor in STRICT_COLUMNS.items():
        _assert_close(
            accessor(model_initial),
            accessor(gold_initial),
            fixture.name,
            gold_initial.age,
            f"initial_{column}",
        )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_pmrc_matches_gold_csv_rows(fixture: ScenarioFixture) -> None:
    gold_rows = load_gold_rows(fixture)
    model_rows = simulate_pmrc_rows(fixture)

    assert len(model_rows) == len(gold_rows), (
        f"fixture={fixture.name} field=row_count failed: expected={len(gold_rows)}, actual={len(model_rows)}, "
        f"abs_err={abs(len(model_rows) - len(gold_rows))}, rel_err={relative_error(len(model_rows), len(gold_rows)):.12%}"
    )

    for gold_row, model_row in zip(gold_rows, model_rows, strict=True):
        for column, accessor in STRICT_COLUMNS.items():
            _assert_close(
                accessor(model_row),
                accessor(gold_row),
                fixture.name,
                gold_row.age,
                column,
            )


@pytest.mark.parametrize(
    "fixture",
    [fixture for fixture in FIXTURES if fixture.thin_age is not None],
    ids=lambda fixture: fixture.name,
)
def test_pmrc_duplicate_age_rows_assert_thinning_behavior(fixture: ScenarioFixture) -> None:
    gold_duplicates = find_duplicate_age_rows(load_gold_rows(fixture))
    model_duplicates = find_duplicate_age_rows(simulate_pmrc_rows(fixture))

    assert len(model_duplicates) == len(gold_duplicates) == 1, (
        f"fixture={fixture.name} field=duplicate_age_thinning_event_count failed: "
        f"expected={len(gold_duplicates)}, actual={len(model_duplicates)}, "
        f"abs_err={abs(len(model_duplicates) - len(gold_duplicates))}, "
        f"rel_err={relative_error(len(model_duplicates), len(gold_duplicates)):.12%}"
    )

    gold_pre, gold_post = gold_duplicates[0]
    model_pre, model_post = model_duplicates[0]

    _assert_close(model_pre.age, gold_pre.age, fixture.name, gold_pre.age, "pre_thin_age")
    _assert_close(model_post.age, gold_post.age, fixture.name, gold_post.age, "post_thin_age")

    assert math.isclose(model_pre.tpa, gold_pre.tpa, rel_tol=1e-6, abs_tol=1e-6), (
        f"age={gold_pre.age} "
        + format_failure(
            fixture_name=fixture.name,
            label="pre_thin_tpa",
            actual=model_pre.tpa,
            expected=gold_pre.tpa,
        )
    )
    assert math.isclose(model_post.tpa, gold_post.tpa, rel_tol=1e-6, abs_tol=1e-6), (
        f"age={gold_post.age} "
        + format_failure(
            fixture_name=fixture.name,
            label="post_thin_tpa",
            actual=model_post.tpa,
            expected=gold_post.tpa,
        )
    )

    for column, accessor in STRICT_COLUMNS.items():
        _assert_close(accessor(model_pre), accessor(gold_pre), fixture.name, gold_pre.age, f"pre_{column}")
        _assert_close(accessor(model_post), accessor(gold_post), fixture.name, gold_post.age, f"post_{column}")
