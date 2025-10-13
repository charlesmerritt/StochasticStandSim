import math

import pytest

from core.growth import PMRC1996, PMRCGrowth, StandState
from core.growth import pmrc_coeffs as coeffs
from core.growth.types import Region


@pytest.fixture(scope="module")
def pmrc() -> PMRC1996:
    return PMRC1996()


def test_site_index_round_trip(pmrc: PMRC1996) -> None:
    age = 18.0
    hd = 47.5
    si = pmrc.site_index_from_hd(age, hd)
    recon = pmrc.hd_from_site_index(age, si)
    assert pytest.approx(hd, rel=1e-6) == recon


def test_height_projection_matches_report_equation(pmrc: PMRC1996) -> None:
    age1, hd1, age2 = 12.0, 34.2, 22.0
    alpha = coeffs.HEIGHT_SITE_PARAMETERS["alpha"]
    m = coeffs.HEIGHT_SITE_PARAMETERS["m"]
    g1 = 1 - math.exp(-alpha * age1)
    g2 = 1 - math.exp(-alpha * age2)
    expected = hd1 * (g2 / g1) ** m
    projected = pmrc.project_hd(age1, hd1, age2, pmrc.site_index_from_hd(age1, hd1))
    assert pytest.approx(expected, rel=1e-9) == projected


def test_project_tpa(pmrc: PMRC1996) -> None:
    params = coeffs.TPA_PARAMETERS
    age1, tpa1, age2, si = 10.0, 550.0, 25.0, 70.0
    term = (tpa1 - params["asymptote"]) ** (-params["b"]) + (params["c"] ** 2) * si * (
        age2 ** params["d"] - age1 ** params["d"]
    )
    expected = params["asymptote"] + term ** (-1.0 / params["b"])
    projected = pmrc.project_tpa(age1, tpa1, age2, si)
    assert pytest.approx(expected, rel=1e-9) == projected


def test_predict_ba_piedmont_with_hardwood(pmrc: PMRC1996) -> None:
    coeff = coeffs.BA_PREDICT[Region.PIEDMONT]
    age, tpa, hd, phwd = 22.0, 430.0, 63.0, 18.0
    ln_tpa = math.log(tpa)
    ln_hd = math.log(hd)
    ln_ba = (
        coeff.intercept
        + coeff.inv_age / age
        + coeff.ln_tpa * ln_tpa
        + coeff.ln_hd * ln_hd
        + coeff.ln_tpa_over_age * (ln_tpa / age)
        + coeff.ln_hd_over_age * (ln_hd / age)
        + coeffs.HARDWOOD_ADJUSTMENT["coeff"] * phwd
    )
    expected = math.exp(ln_ba)
    predicted = pmrc.predict_ba(age, tpa, hd, Region.PIEDMONT, phwd)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_project_ba_lcp(pmrc: PMRC1996) -> None:
    coeff = coeffs.BA_PROJECT[Region.LOWER_COASTAL_PLAIN]
    age1, age2 = 12.0, 22.0
    ba1, tpa1, hd1 = 60.0, 600.0, 42.0
    tpa2, hd2 = 500.0, 63.0

    ln_ba1 = math.log(ba1)
    ln_tpa1 = math.log(tpa1)
    ln_hd1 = math.log(hd1)
    ln_tpa2 = math.log(tpa2)
    ln_hd2 = math.log(hd2)

    ln_ba2 = (
        ln_ba1
        + coeff.inv_age * ((1 / age2) - (1 / age1))
        + coeff.ln_tpa * (ln_tpa2 - ln_tpa1)
        + coeff.ln_hd * (ln_hd2 - ln_hd1)
        + coeff.ln_tpa_over_age * ((ln_tpa2 / age2) - (ln_tpa1 / age1))
        + coeff.ln_hd_over_age * ((ln_hd2 / age2) - (ln_hd1 / age1))
    )
    expected = math.exp(ln_ba2)
    projected = pmrc.project_ba(age1, ba1, tpa1, hd1, age2, tpa2, hd2, Region.LOWER_COASTAL_PLAIN)
    assert pytest.approx(expected, rel=1e-9) == projected


@pytest.mark.parametrize(
    "region,unit",
    [
        (Region.UPPER_COASTAL_PLAIN, "TVOB"),
        (Region.UPPER_COASTAL_PLAIN, "GWOB"),
        (Region.LOWER_COASTAL_PLAIN, "DWIB"),
    ],
)
def test_predict_yield(pmrc: PMRC1996, region: Region, unit: str) -> None:
    coeff = coeffs.YIELD_COEFFICIENTS[(region, unit)]
    age, tpa, hd, ba = 25.0, 550.0, 70.0, 110.0
    ln_tpa = math.log(tpa)
    ln_hd = math.log(hd)
    ln_ba = math.log(ba)

    intercept, b_hd, b_ba, b_tpa_over_age, b_hd_over_age, b_ba_over_age, b_tpa = coeff
    expected = math.exp(
        intercept
        + b_hd * ln_hd
        + b_ba * ln_ba
        + b_tpa_over_age * (ln_tpa / age)
        + b_hd_over_age * (ln_hd / age)
        + b_ba_over_age * (ln_ba / age)
        + b_tpa * ln_tpa
    )
    predicted = pmrc.predict_yield(age, tpa, hd, ba, region, unit)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_merchantable_fraction(pmrc: PMRC1996) -> None:
    region = Region.LOWER_COASTAL_PLAIN
    unit = "GWOB"
    coeff = coeffs.MERCHANTABLE_COEFFICIENTS[(region, unit)]
    total, dmin, ttop, tpa, hd, ba = 120.0, 6.5, 4.0, 480.0, 65.0, 105.0
    qmd = math.sqrt((ba / tpa) / 0.005454154)
    b1, b2, b3, b4, b5 = coeff
    expected = total * math.exp(b1 * (ttop / qmd) ** b5 + b2 * (tpa ** b3) * ((dmin / qmd) ** b4))
    predicted = pmrc.merchantable_fraction(total, dmin, ttop, tpa, hd, ba, region, unit)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_diameter_percentiles_monotonic(pmrc: PMRC1996) -> None:
    ba, tpa = 120.0, 550.0
    result = pmrc.diameter_percentiles(ba, tpa, Region.PIEDMONT, percent_hardwood_ba=15.0)
    assert set(result.keys()) == {25, 50, 75}
    assert result[25] < result[50] < result[75]


def test_project_relative_size(pmrc: PMRC1996) -> None:
    value = pmrc.project_relative_size(0.8, 1.2, 15.0, 25.0, Region.UPPER_COASTAL_PLAIN)
    assert value < 1.2
    assert value > 0.8


def test_height_given_dbh(pmrc: PMRC1996) -> None:
    region = Region.LOWER_COASTAL_PLAIN
    intercept, b_hd, b_dq, b_dbh, b_ratio_dq, b_ratio_hd = coeffs.HEIGHT_DBH_COEFFICIENTS[region]
    hd, dq, dbh = 70.0, 8.5, 12.0
    ln_height = (
        intercept
        + b_hd * math.log(hd)
        + b_dq * math.log(dq)
        + b_dbh * math.log(dbh)
        + b_ratio_dq * math.log(dbh / dq)
        + b_ratio_hd * math.log(dbh / hd)
    )
    expected = math.exp(ln_height)
    predicted = pmrc.height_given_dbh(hd, dq, dbh, region)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_competition_index(pmrc: PMRC1996) -> None:
    value = pmrc.competition_index(ba_thinned=90.0, ba_unthinned=120.0, region=Region.UPPER_COASTAL_PLAIN)
    assert pytest.approx(0.25, rel=1e-9) == value


def test_estimate_ba_removed(pmrc: PMRC1996) -> None:
    ba_before, tpa_before = 120.0, 500.0
    row, select = 125.0, 80.0
    remaining = tpa_before - row
    row_fraction = row / tpa_before
    select_fraction = (select / remaining) ** 1.2345
    expected = ba_before * (row_fraction + (1 - row_fraction) * select_fraction)
    predicted = pmrc.estimate_ba_removed(ba_before, tpa_before, row, select)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_project_thin_response(pmrc: PMRC1996) -> None:
    ba_future, ci1 = 140.0, 0.25
    age1, age2 = 15.0, 25.0
    decay = coeffs.RELATIVE_SIZE_DECAY[Region.LOWER_COASTAL_PLAIN]
    expected = ba_future * (1 - ci1 * math.exp(-decay * (age2 - age1)))
    predicted = pmrc.project_thin_response_ba(ba_future, ci1, age1, age2, Region.LOWER_COASTAL_PLAIN)
    assert pytest.approx(expected, rel=1e-9) == predicted


def test_fertilisation_responses(pmrc: PMRC1996) -> None:
    years = 3.0
    n_lbs = 150.0
    hd_expected = (
        (coeffs.FERT_HEIGHT["N"] * n_lbs + coeffs.FERT_HEIGHT["P"]) * years * math.exp(-coeffs.FERT_HEIGHT["k"] * years)
    )
    ba_expected = (
        (coeffs.FERT_BA["N"] * n_lbs + coeffs.FERT_BA["P"]) * years * math.exp(-coeffs.FERT_BA["k"] * years)
    )
    assert pytest.approx(hd_expected, rel=1e-9) == pmrc.fert_response_hd(years, n_lbs, True)
    assert pytest.approx(ba_expected, rel=1e-9) == pmrc.fert_response_ba(years, n_lbs, True)


def test_growth_facade_projects_consistently() -> None:
    pmrc = PMRC1996()
    engine = PMRCGrowth(pmrc, default_region=Region.UPPER_COASTAL_PLAIN)
    stand = StandState(age=15.0, tpa=600.0, hd=45.0, region=Region.UPPER_COASTAL_PLAIN)

    # Determine derived state first
    ba_now = engine.ba(stand)
    assert ba_now > 0

    age2 = 25.0
    hd2 = engine.project_height(stand, age2)
    tpa2 = engine.project_tpa(stand, age2)
    ba2 = engine.project_ba(stand, age2, tpa2, hd2)

    assert hd2 > stand.hd
    assert tpa2 < stand.tpa
    assert ba2 > ba_now

    total_yield = engine.yield_total(StandState(age=age2, tpa=tpa2, hd=hd2, ba=ba2, region=Region.UPPER_COASTAL_PLAIN))
    assert total_yield > 0
