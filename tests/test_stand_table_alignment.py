from __future__ import annotations

import warnings

import pytest

from tests.gold_fixtures import FIXTURES, load_gold_rows, max_relative_error, simulate_stand_table_rows


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_stand_table_reports_closeness_to_gold_fixture(fixture):
    gold_rows = load_gold_rows(fixture)
    stand_rows = simulate_stand_table_rows(fixture)

    assert len(stand_rows) == len({row.age for row in gold_rows}), (
        f"{fixture.name}: stand-table projection should produce one row per distinct age"
    )

    comparable_gold_rows = []
    seen_ages: set[float] = set()
    for row in gold_rows:
        if row.age in seen_ages:
            continue
        seen_ages.add(row.age)
        comparable_gold_rows.append(row)

    metrics = {
        "tpa": max_relative_error(stand_rows, comparable_gold_rows, lambda row: row.tpa),
        "hd": max_relative_error(stand_rows, comparable_gold_rows, lambda row: row.hd),
        "ba": max_relative_error(stand_rows, comparable_gold_rows, lambda row: row.ba),
        "volume_tvob": max_relative_error(stand_rows, comparable_gold_rows, lambda row: row.volume_tvob),
        "dq": max_relative_error(stand_rows, comparable_gold_rows, lambda row: row.dq),
    }

    report = ", ".join(f"{name}={value:.6%}" for name, value in metrics.items())
    warnings.warn(f"{fixture.name} stand-table closeness: {report}", stacklevel=2)
