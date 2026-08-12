from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import csv
from pathlib import Path

from core.actions import ActionModel
from core.pmrc_model import DEFAULT_DBH_BOUNDS
from core.config import ThinningParams
from core.pmrc_model import PMRCModel
from core.stand_table_system import StandTable, project_stand_table
from core.state import Region, StandState


CSV_DIR = Path(__file__).resolve().parents[1] / "test_csvs"


@dataclass(frozen=True)
class ScenarioFixture:
    name: str
    csv_name: str
    region: Region
    si25: float
    tpa0: float
    age0: float
    end_age: float
    thin_age: float | None = None

    @property
    def csv_path(self) -> Path:
        return CSV_DIR / self.csv_name

    @property
    def rotation_length(self) -> int:
        return int(self.end_age - self.age0)


@dataclass(frozen=True)
class GoldRow:
    age: float
    tpa: float
    hd: float
    ba: float
    volume_tvob: float
    yield_dwib: float
    dq: float
    volume_pulp: float
    volume_cns: float
    volume_saw: float
    green_pulp: float
    green_cns: float
    green_saw: float
    remove_pulp: float
    remove_cns: float
    remove_saw: float


def relative_error(actual: float, expected: float) -> float:
    scale = max(abs(actual), abs(expected), 1.0)
    return abs(actual - expected) / scale


def format_failure(*, fixture_name: str, label: str, actual: float, expected: float) -> str:
    return (
        f"fixture={fixture_name} field={label} failed: "
        f"expected={expected:.12g}, actual={actual:.12g}, "
        f"abs_err={abs(actual - expected):.12g}, rel_err={relative_error(actual, expected):.12%}"
    )


FIXTURES = [
    ScenarioFixture(
        name="scenario_1_nothin",
        csv_name="scenario_1_nothin.csv",
        region="pucp",
        si25=75.0,
        tpa0=800.0,
        age0=5.0,
        end_age=35.0,
    ),
    ScenarioFixture(
        name="scenario_2_nothin",
        csv_name="scenario_2_nothin.csv",
        region="pucp",
        si25=60.0,
        tpa0=600.0,
        age0=5.0,
        end_age=35.0,
    ),
    ScenarioFixture(
        name="scenario_3_nothin",
        csv_name="scenario_3_nothin.csv",
        region="pucp",
        si25=90.0,
        tpa0=550.0,
        age0=5.0,
        end_age=35.0,
    ),
    ScenarioFixture(
        name="scenario_1_thin",
        csv_name="scenario_1_thin.csv",
        region="pucp",
        si25=75.0,
        tpa0=800.0,
        age0=5.0,
        end_age=35.0,
        thin_age=15.0,
    ),
    ScenarioFixture(
        name="scenario_2_thin",
        csv_name="scenario_2_thin.csv",
        region="pucp",
        si25=60.0,
        tpa0=600.0,
        age0=5.0,
        end_age=35.0,
        thin_age=18.0,
    ),
]


def load_gold_rows(fixture: ScenarioFixture) -> list[GoldRow]:
    with fixture.csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            GoldRow(
                age=float(row["A"]),
                tpa=float(row["N"]),
                hd=float(row["H"]),
                ba=float(row["G"]),
                volume_tvob=float(row["V"]),
                yield_dwib=float(row["DW"]),
                dq=float(row["Dq"]),
                volume_pulp=float(row["Vpulp"]),
                volume_cns=float(row["Vchns"]),
                volume_saw=float(row["Vsaw"]),
                green_pulp=float(row["Gpulp"]),
                green_cns=float(row["Gchns"]),
                green_saw=float(row["Gsaw"]),
                remove_pulp=float(row["Remove.Plp"]),
                remove_cns=float(row["Remove.ChnS"]),
                remove_saw=float(row["Remove.Saw"]),
            )
            for row in reader
        ]


def fixture_initial_row(fixture: ScenarioFixture) -> GoldRow:
    return load_gold_rows(fixture)[0]


def build_initial_state(fixture: ScenarioFixture, pmrc: PMRCModel) -> StandState:
    initial_row = load_gold_rows(fixture)[0]
    return StandState(
        age=initial_row.age,
        hd=initial_row.hd,
        tpa=initial_row.tpa,
        ba=initial_row.ba,
        si25=fixture.si25,
        region=fixture.region,
    )


def stand_to_gold_row(age: float, hd: float, tpa: float, ba: float, pmrc: PMRCModel, region: Region) -> GoldRow:
    product_tvob = pmrc.product_yields(age=age, tpa=tpa, hd=hd, ba=ba, unit="TVOB", region=region)
    product_gwob = pmrc.product_yields(age=age, tpa=tpa, hd=hd, ba=ba, unit="GWOB", region=region)
    return GoldRow(
        age=age,
        tpa=tpa,
        hd=hd,
        ba=ba,
        volume_tvob=pmrc.yield_predict(age=age, tpa=tpa, hd=hd, ba=ba, unit="TVOB", region=region),
        yield_dwib=pmrc.yield_predict(age=age, tpa=tpa, hd=hd, ba=ba, unit="DWIB", region=region),
        dq=pmrc.qmd(tpa=tpa, ba=ba),
        volume_pulp=product_tvob.pulpwood,
        volume_cns=product_tvob.chip_n_saw,
        volume_saw=product_tvob.sawtimber,
        green_pulp=product_gwob.pulpwood,
        green_cns=product_gwob.chip_n_saw,
        green_saw=product_gwob.sawtimber,
        remove_pulp=0.0,
        remove_cns=0.0,
        remove_saw=0.0,
    )


def simulate_pmrc_rows(fixture: ScenarioFixture) -> list[GoldRow]:
    pmrc = PMRCModel(region=fixture.region)
    state = build_initial_state(fixture, pmrc)
    rows = [stand_to_gold_row(state.age, state.hd, state.tpa, state.ba, pmrc, fixture.region)]
    action_model = ActionModel(pmrc, _infer_thinning_params(fixture))

    while state.age < fixture.end_age:
        age2 = state.age + 1.0
        hd2 = pmrc.hd_project(state.age, state.hd, age2)
        tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba2 = pmrc.ba_project(state.age, state.tpa, tpa2, state.ba, state.hd, hd2, age2, state.region)
        state = StandState(
            age=age2,
            hd=hd2,
            tpa=tpa2,
            ba=ba2,
            si25=state.si25,
            region=state.region,
            phwd=state.phwd,
        )
        rows.append(stand_to_gold_row(state.age, state.hd, state.tpa, state.ba, pmrc, fixture.region))
        if action_model.should_thin(state):
            result = action_model.apply_thinning(state)
            if result.post_thin_state is None:
                raise AssertionError(f"Expected thinning for fixture {fixture.name} at age {state.age}")
            state = result.post_thin_state
            rows.append(stand_to_gold_row(state.age, state.hd, state.tpa, state.ba, pmrc, fixture.region))
    return rows


def _infer_thinning_params(fixture: ScenarioFixture) -> ThinningParams | None:
    gold_rows = load_gold_rows(fixture)
    for idx in range(1, len(gold_rows)):
        current = gold_rows[idx]
        previous = gold_rows[idx - 1]
        if current.age == previous.age:
            return ThinningParams(
                trigger_age=current.age,
                ba_threshold=previous.ba,
                residual_ba=current.ba,
            )
    return None


def simulate_stand_table_rows(fixture: ScenarioFixture) -> list[GoldRow]:
    pmrc = PMRCModel(region=fixture.region)
    state = build_initial_state(fixture, pmrc)
    dist = pmrc.diameter_class_distribution(
        ba=state.ba,
        tpa=state.tpa,
        dbh_bounds=DEFAULT_DBH_BOUNDS,
        region=fixture.region,
    )
    dbh_midpoints = [0.5 * (lo + hi) for lo, hi in zip(DEFAULT_DBH_BOUNDS[:-1], DEFAULT_DBH_BOUNDS[1:], strict=True)]
    stand = StandTable.from_arrays(
        age=state.age,
        hd=state.hd,
        region=fixture.region,
        dbh_midpoints=dbh_midpoints,
        tpa_per_class=dist.tpa_per_class.tolist(),
    )
    rows = [stand_to_gold_row(stand.age, stand.hd, stand.tpa, stand.ba, pmrc, fixture.region)]

    while stand.age < fixture.end_age:
        stand = project_stand_table(stand, stand.age + 1.0)
        rows.append(stand_to_gold_row(stand.age, stand.hd, stand.tpa, stand.ba, pmrc, fixture.region))
    return rows


def max_relative_error(rows_a: list[GoldRow], rows_b: list[GoldRow], accessor: Callable[[GoldRow], float]) -> float:
    max_error = 0.0
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        a = accessor(row_a)
        b = accessor(row_b)
        scale = max(abs(a), abs(b), 1.0)
        max_error = max(max_error, abs(a - b) / scale)
    return max_error


def find_duplicate_age_rows(rows: list[GoldRow]) -> list[tuple[GoldRow, GoldRow]]:
    duplicates: list[tuple[GoldRow, GoldRow]] = []
    for idx in range(1, len(rows)):
        if rows[idx].age == rows[idx - 1].age:
            duplicates.append((rows[idx - 1], rows[idx]))
    return duplicates
