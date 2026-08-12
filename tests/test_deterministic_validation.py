"""Deterministic growth model validation tests.

Validates the PMRC deterministic backbone across six areas:
1. SI25 ↔ HD consistency (multi-region, multi-SI, round-trip)
2. Growth trajectory shape (monotonicity, concavity, cross-scenario ordering)
3. Product distribution reasonableness (age-appropriate assertions)
4. Thinning from below (QMD increase, HD unchanged, product shift)
5. QMD / BA / TPA internal consistency
6. Yield equation cross-checks (DWIB < TVIB < TVOB, magnitude sanity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from core.pmrc_model import DEFAULT_DBH_BOUNDS, PMRCModel
from core.products import ProductDistribution, estimate_product_distribution
from core.state import Region, StandState, hd_from_si25_at_age, si25_from_hd_at_age


# ---------------------------------------------------------------------------
# Helper: run a deterministic trajectory and return per-year records
# ---------------------------------------------------------------------------

@dataclass
class YearSnapshot:
    """One year of a deterministic trajectory."""
    age: float
    hd: float
    tpa: float
    ba: float
    vol_tvob: float
    vol_tvib: float
    vol_gwob: float
    vol_dwib: float
    qmd: float
    products: ProductDistribution


def run_deterministic_trajectory(
    si25: float,
    tpa0: float,
    age0: float,
    rotation: int,
    region: Region = "ucp",
) -> list[YearSnapshot]:
    """Run a deterministic PMRC trajectory and return yearly snapshots."""
    pmrc = PMRCModel(region=region)
    hd0 = hd_from_si25_at_age(si25, age0)
    ba0 = pmrc.ba_predict(age=age0, tpa=tpa0, hd=hd0, region=region)

    state = StandState(
        age=age0, hd=hd0, tpa=tpa0, ba=ba0,
        si25=si25, region=region,
    )

    def _snapshot(s: StandState) -> YearSnapshot:
        prods = estimate_product_distribution(
            pmrc=pmrc, age=s.age, ba=s.ba, tpa=s.tpa, hd=s.hd, region=s.region,
        )
        return YearSnapshot(
            age=s.age, hd=s.hd, tpa=s.tpa, ba=s.ba,
            vol_tvob=pmrc.yield_predict(s.age, s.tpa, s.hd, s.ba, "TVOB", s.region),
            vol_tvib=pmrc.yield_predict(s.age, s.tpa, s.hd, s.ba, "TVIB", s.region),
            vol_gwob=pmrc.yield_predict(s.age, s.tpa, s.hd, s.ba, "GWOB", s.region),
            vol_dwib=pmrc.yield_predict(s.age, s.tpa, s.hd, s.ba, "DWIB", s.region),
            qmd=pmrc.qmd(s.tpa, s.ba),
            products=prods,
        )

    snapshots = [_snapshot(state)]

    for _ in range(rotation):
        age2 = state.age + 1.0
        hd2 = pmrc.hd_project(state.age, state.hd, age2)
        tpa2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
        ba2 = pmrc.ba_project(
            state.age, state.tpa, tpa2, state.ba, state.hd, hd2, age2, state.region,
        )
        state = StandState(
            age=age2, hd=hd2, tpa=tpa2, ba=ba2,
            si25=si25, region=region,
        )
        snapshots.append(_snapshot(state))

    return snapshots


# =========================================================================
# 1. SI25 ↔ HD Consistency
# =========================================================================

class TestSI25Consistency:
    """HD at age 25 must equal SI25; round-trip must be exact."""

    @pytest.mark.parametrize("si25", [50.0, 60.0, 70.0, 75.0, 80.0, 90.0])
    @pytest.mark.parametrize("region", ["ucp", "pucp", "lcp"])
    @pytest.mark.parametrize("age0", [3.0, 5.0, 10.0])
    def test_hd_at_age25_equals_si25(self, si25: float, region: Region, age0: float):
        """Project HD from age0 to 25 — must equal SI25."""
        pmrc = PMRCModel(region=region)
        hd0 = hd_from_si25_at_age(si25, age0)
        hd_at_25 = pmrc.hd_project(age0, hd0, 25.0)
        assert abs(hd_at_25 - si25) < 0.01, (
            f"HD at 25 = {hd_at_25:.4f}, expected SI25 = {si25} "
            f"(region={region}, age0={age0})"
        )

    @pytest.mark.parametrize("si25", [50.0, 60.0, 75.0, 90.0])
    @pytest.mark.parametrize("age", [5.0, 10.0, 15.0, 20.0, 30.0, 35.0])
    def test_round_trip_si25_hd(self, si25: float, age: float):
        """si25 → hd → si25 round-trip within tolerance."""
        hd = hd_from_si25_at_age(si25, age)
        recovered = si25_from_hd_at_age(hd, age)
        assert abs(recovered - si25) < 1e-6, (
            f"Round-trip failed: si25={si25}, age={age}, recovered={recovered:.8f}"
        )

    @pytest.mark.parametrize("region", ["ucp", "pucp", "lcp"])
    def test_hd_projection_matches_si_curve_all_ages(self, region: Region):
        """Step-by-step projection matches direct SI curve evaluation."""
        pmrc = PMRCModel(region=region)
        si25 = 70.0
        age = 5.0
        hd = hd_from_si25_at_age(si25, age)

        for target in range(6, 36):
            hd = pmrc.hd_project(age, hd, float(target))
            expected = hd_from_si25_at_age(si25, float(target))
            assert abs(hd - expected) < 0.02, (
                f"region={region}, age={target}: projected={hd:.4f}, expected={expected:.4f}"
            )
            age = float(target)


# =========================================================================
# 2. Growth Trajectory Shape
# =========================================================================

# Standard scenarios for trajectory tests
_TRAJECTORY_SCENARIOS = [
    (60.0, 600.0, 5.0, 30, "pucp"),
    (75.0, 800.0, 5.0, 30, "pucp"),
    (90.0, 550.0, 5.0, 30, "pucp"),
    (80.0, 850.0, 5.0, 30, "ucp"),
    (70.0, 700.0, 5.0, 30, "lcp"),
]


class TestGrowthTrajectoryShape:
    """Deterministic trajectory must be biologically reasonable."""

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_hd_monotonically_increasing(self, si25, tpa0, age0, rotation, region):
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for i in range(1, len(snaps)):
            assert snaps[i].hd >= snaps[i - 1].hd, (
                f"HD decreased at age {snaps[i].age}: "
                f"{snaps[i].hd:.4f} < {snaps[i-1].hd:.4f}"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_hd_increments_decreasing(self, si25, tpa0, age0, rotation, region):
        """HD annual increment should decrease with age (concavity)."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        increments = [snaps[i].hd - snaps[i - 1].hd for i in range(1, len(snaps))]
        for i in range(1, len(increments)):
            assert increments[i] <= increments[i - 1] + 1e-6, (
                f"HD increment increased at age {snaps[i+1].age}: "
                f"{increments[i]:.4f} > {increments[i-1]:.4f}"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_tpa_monotonically_decreasing(self, si25, tpa0, age0, rotation, region):
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for i in range(1, len(snaps)):
            assert snaps[i].tpa <= snaps[i - 1].tpa + 1e-6, (
                f"TPA increased at age {snaps[i].age}: "
                f"{snaps[i].tpa:.4f} > {snaps[i-1].tpa:.4f}"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_tpa_never_below_asymptote(self, si25, tpa0, age0, rotation, region):
        """TPA must never drop below the PMRC 100-tree asymptote."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            assert s.tpa >= 100.0, f"TPA below asymptote at age {s.age}: {s.tpa:.2f}"

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_ba_increasing(self, si25, tpa0, age0, rotation, region):
        """BA should increase over a standard rotation (growth > mortality loss)."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for i in range(1, len(snaps)):
            assert snaps[i].ba >= snaps[i - 1].ba - 1e-6, (
                f"BA decreased at age {snaps[i].age}: "
                f"{snaps[i].ba:.4f} < {snaps[i-1].ba:.4f}"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_volume_monotonically_increasing(self, si25, tpa0, age0, rotation, region):
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for i in range(1, len(snaps)):
            assert snaps[i].vol_tvob >= snaps[i - 1].vol_tvob - 1e-6, (
                f"TVOB decreased at age {snaps[i].age}: "
                f"{snaps[i].vol_tvob:.2f} < {snaps[i-1].vol_tvob:.2f}"
            )

    def test_higher_si_produces_higher_hd_and_volume(self):
        """Cross-scenario: higher SI25 → higher HD and volume at every age."""
        traj_lo = run_deterministic_trajectory(60.0, 600.0, 5.0, 30, "pucp")
        traj_hi = run_deterministic_trajectory(90.0, 550.0, 5.0, 30, "pucp")
        for lo, hi in zip(traj_lo, traj_hi, strict=True):
            assert hi.hd > lo.hd, (
                f"At age {lo.age}: SI90 HD ({hi.hd:.2f}) <= SI60 HD ({lo.hd:.2f})"
            )
            assert hi.vol_tvob > lo.vol_tvob, (
                f"At age {lo.age}: SI90 TVOB ({hi.vol_tvob:.2f}) <= SI60 TVOB ({lo.vol_tvob:.2f})"
            )


# =========================================================================
# 3. Product Distribution Reasonableness
# =========================================================================

class TestProductDistribution:
    """Product volumes must be age-appropriate and internally consistent."""

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_non_negative_product_volumes(self, si25, tpa0, age0, rotation, region):
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            assert s.products.vol_pulp >= -1e-6, f"Negative pulp at age {s.age}"
            assert s.products.vol_cns >= -1e-6, f"Negative CNS at age {s.age}"
            assert s.products.vol_saw >= -1e-6, f"Negative saw at age {s.age}"

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_product_sum_le_total_tvob(self, si25, tpa0, age0, rotation, region):
        """Merchantable products must not exceed total stand volume."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            merch = s.products.vol_pulp + s.products.vol_cns + s.products.vol_saw
            assert merch <= s.vol_tvob + 1e-3, (
                f"At age {s.age}: merch ({merch:.2f}) > TVOB ({s.vol_tvob:.2f})"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_total_merchantable_volume_increasing(self, si25, tpa0, age0, rotation, region):
        """Total merchantable volume (pulp+CNS+saw) should increase with age."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        prev_total = 0.0
        for s in snaps:
            total = s.products.vol_pulp + s.products.vol_cns + s.products.vol_saw
            assert total >= prev_total - 1e-3, (
                f"Merchantable vol decreased at age {s.age}: {total:.2f} < {prev_total:.2f}"
            )
            prev_total = total

    def test_young_stand_no_sawtimber(self):
        """At age ≤ 10, sawtimber should be negligible (QMD far below 12\")."""
        snaps = run_deterministic_trajectory(75.0, 800.0, 5.0, 30, "pucp")
        for s in snaps:
            if s.age <= 10:
                total = s.products.vol_pulp + s.products.vol_cns + s.products.vol_saw
                if total > 1.0:
                    saw_frac = s.products.vol_saw / total
                    assert saw_frac < 0.01, (
                        f"At age {s.age}: sawtimber fraction = {saw_frac:.4f} (should be ~0)"
                    )

    def test_mature_stand_has_sawtimber(self):
        """At age 30+, high-SI stands should have meaningful sawtimber volume."""
        snaps = run_deterministic_trajectory(90.0, 550.0, 5.0, 30, "pucp")
        final = snaps[-1]
        total = final.products.vol_pulp + final.products.vol_cns + final.products.vol_saw
        assert total > 0, "No merchantable volume at rotation end"
        saw_frac = final.products.vol_saw / total
        assert saw_frac > 0.05, (
            f"At age {final.age}, SI90: sawtimber fraction = {saw_frac:.4f} (expected > 5%)"
        )

    def test_product_mix_shifts_toward_larger_classes_with_age(self):
        """As QMD increases, the sawtimber+CNS share should increase."""
        snaps = run_deterministic_trajectory(75.0, 800.0, 5.0, 30, "pucp")
        # Compare age 15 vs age 35
        s15 = next(s for s in snaps if s.age == 15)
        s35 = next(s for s in snaps if s.age == 35)
        total_15 = s15.products.vol_pulp + s15.products.vol_cns + s15.products.vol_saw
        total_35 = s35.products.vol_pulp + s35.products.vol_cns + s35.products.vol_saw
        if total_15 > 1.0 and total_35 > 1.0:
            large_frac_15 = (s15.products.vol_cns + s15.products.vol_saw) / total_15
            large_frac_35 = (s35.products.vol_cns + s35.products.vol_saw) / total_35
            assert large_frac_35 > large_frac_15, (
                f"CNS+saw fraction did not increase: age15={large_frac_15:.4f}, age35={large_frac_35:.4f}"
            )


# =========================================================================
# 4. Thinning From Below
# =========================================================================

class TestThinningFromBelow:
    """Thinning must remove small trees, raise QMD, and preserve HD."""

    @staticmethod
    def _get_pre_post_thin(
        fixture_csv: str,
    ) -> tuple[dict, dict]:
        """Read gold CSV and return the pre/post thin rows (duplicate age)."""
        import csv
        from pathlib import Path

        csv_path = Path(__file__).resolve().parents[1] / "test_csvs" / fixture_csv
        rows: list[dict] = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        for i in range(1, len(rows)):
            if float(rows[i]["A"]) == float(rows[i - 1]["A"]):
                return rows[i - 1], rows[i]
        raise ValueError(f"No duplicate-age (thin) rows found in {fixture_csv}")

    @pytest.mark.parametrize("csv_name", ["scenario_1_thin.csv", "scenario_2_thin.csv"])
    def test_post_thin_qmd_greater_than_pre_thin(self, csv_name: str):
        """Thinning from below must increase QMD."""
        pre, post = self._get_pre_post_thin(csv_name)
        qmd_pre = float(pre["Dq"])
        qmd_post = float(post["Dq"])
        assert qmd_post > qmd_pre, (
            f"{csv_name}: post-thin QMD ({qmd_post:.3f}) <= pre-thin QMD ({qmd_pre:.3f})"
        )

    @pytest.mark.parametrize("csv_name", ["scenario_1_thin.csv", "scenario_2_thin.csv"])
    def test_hd_unchanged_by_thinning(self, csv_name: str):
        """Dominant height must not change during thinning."""
        pre, post = self._get_pre_post_thin(csv_name)
        hd_pre = float(pre["H"])
        hd_post = float(post["H"])
        assert abs(hd_post - hd_pre) < 1e-6, (
            f"{csv_name}: HD changed during thin: {hd_pre:.6f} → {hd_post:.6f}"
        )

    @pytest.mark.parametrize("csv_name", ["scenario_1_thin.csv", "scenario_2_thin.csv"])
    def test_ba_reduced_by_thinning(self, csv_name: str):
        """BA must decrease after thinning."""
        pre, post = self._get_pre_post_thin(csv_name)
        ba_pre = float(pre["G"])
        ba_post = float(post["G"])
        assert ba_post < ba_pre, (
            f"{csv_name}: BA not reduced: {ba_pre:.3f} → {ba_post:.3f}"
        )

    @pytest.mark.parametrize("csv_name", ["scenario_1_thin.csv", "scenario_2_thin.csv"])
    def test_tpa_reduced_by_thinning(self, csv_name: str):
        """TPA must decrease after thinning."""
        pre, post = self._get_pre_post_thin(csv_name)
        tpa_pre = float(pre["N"])
        tpa_post = float(post["N"])
        assert tpa_post < tpa_pre, (
            f"{csv_name}: TPA not reduced: {tpa_pre:.1f} → {tpa_post:.1f}"
        )

    @pytest.mark.parametrize("csv_name", ["scenario_1_thin.csv", "scenario_2_thin.csv"])
    def test_post_thin_higher_cns_saw_fraction(self, csv_name: str):
        """Post-thin product mix should shift toward larger classes."""
        pre, post = self._get_pre_post_thin(csv_name)
        for label, row in [("pre", pre), ("post", post)]:
            vpulp = float(row["Vpulp"])
            vcns = float(row["Vchns"])
            vsaw = float(row["Vsaw"])
            total = vpulp + vcns + vsaw
            if total < 1.0:
                pytest.skip(f"Total merchantable volume too small ({label})")

        pre_total = float(pre["Vpulp"]) + float(pre["Vchns"]) + float(pre["Vsaw"])
        post_total = float(post["Vpulp"]) + float(post["Vchns"]) + float(post["Vsaw"])
        pre_large = (float(pre["Vchns"]) + float(pre["Vsaw"])) / pre_total
        post_large = (float(post["Vchns"]) + float(post["Vsaw"])) / post_total
        assert post_large > pre_large, (
            f"{csv_name}: CNS+saw fraction did not increase after thin: "
            f"{pre_large:.4f} → {post_large:.4f}"
        )


# =========================================================================
# 5. QMD / BA / TPA Internal Consistency
# =========================================================================

class TestInternalConsistency:
    """Derived quantities must be internally consistent at every step."""

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_qmd_identity(self, si25, tpa0, age0, rotation, region):
        """QMD must satisfy sqrt(BA / TPA / 0.005454154)."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            expected_qmd = math.sqrt(s.ba / s.tpa / 0.005454154)
            assert abs(s.qmd - expected_qmd) < 1e-6, (
                f"At age {s.age}: QMD={s.qmd:.6f}, expected={expected_qmd:.6f}"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_qmd_monotonically_increasing(self, si25, tpa0, age0, rotation, region):
        """QMD should increase over a standard no-thin rotation."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for i in range(1, len(snaps)):
            assert snaps[i].qmd >= snaps[i - 1].qmd - 1e-6, (
                f"QMD decreased at age {snaps[i].age}: "
                f"{snaps[i].qmd:.4f} < {snaps[i-1].qmd:.4f}"
            )

    def test_weibull_class_totals_match_stand(self):
        """Weibull-derived sum(tpa_per_class) ≈ TPA and sum(ba_per_class) ≈ BA."""
        pmrc = PMRCModel(region="pucp")
        snaps = run_deterministic_trajectory(75.0, 800.0, 5.0, 30, "pucp")
        for s in snaps:
            dist = pmrc.diameter_class_distribution(
                ba=s.ba, tpa=s.tpa,
                dbh_bounds=DEFAULT_DBH_BOUNDS,
                region="pucp",
            )
            assert abs(dist.total_tpa - s.tpa) / s.tpa < 0.05, (
                f"At age {s.age}: Weibull TPA sum={dist.total_tpa:.1f} vs stand TPA={s.tpa:.1f}"
            )
            assert abs(dist.total_ba - s.ba) / s.ba < 0.05, (
                f"At age {s.age}: Weibull BA sum={dist.total_ba:.2f} vs stand BA={s.ba:.2f}"
            )

    def test_ba_predict_close_to_projected_ba(self):
        """Prediction-form BA should be close to projection-form BA along the trajectory."""
        pmrc = PMRCModel(region="pucp")
        snaps = run_deterministic_trajectory(75.0, 800.0, 5.0, 30, "pucp")
        for s in snaps:
            ba_pred = pmrc.ba_predict(s.age, s.tpa, s.hd, "pucp")
            rel_err = abs(ba_pred - s.ba) / max(s.ba, 1.0)
            assert rel_err < 0.10, (
                f"At age {s.age}: BA predict={ba_pred:.2f} vs projected={s.ba:.2f} "
                f"(rel_err={rel_err:.4f})"
            )


# =========================================================================
# 6. Yield Equation Cross-Checks
# =========================================================================

class TestYieldCrossChecks:
    """Volume units must be ordered correctly and magnitudes realistic."""

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_yield_unit_ordering(self, si25, tpa0, age0, rotation, region):
        """DWIB < TVIB < TVOB must hold at every age with positive volume."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            if s.vol_tvob < 1.0:
                continue
            assert s.vol_dwib <= s.vol_tvib + 1e-3, (
                f"At age {s.age}: DWIB ({s.vol_dwib:.2f}) > TVIB ({s.vol_tvib:.2f})"
            )
            assert s.vol_tvib <= s.vol_tvob + 1e-3, (
                f"At age {s.age}: TVIB ({s.vol_tvib:.2f}) > TVOB ({s.vol_tvob:.2f})"
            )

    @pytest.mark.parametrize(
        "si25,tpa0,age0,rotation,region", _TRAJECTORY_SCENARIOS,
        ids=lambda v: str(v) if not isinstance(v, str) else v,
    )
    def test_gwob_between_tvib_and_tvob(self, si25, tpa0, age0, rotation, region):
        """GWOB should be between TVIB and TVOB for pine."""
        snaps = run_deterministic_trajectory(si25, tpa0, age0, rotation, region)
        for s in snaps:
            if s.vol_tvob < 1.0:
                continue
            assert s.vol_gwob <= s.vol_tvob + 1e-3, (
                f"At age {s.age}: GWOB ({s.vol_gwob:.2f}) > TVOB ({s.vol_tvob:.2f})"
            )

    def test_tvob_magnitude_at_age25_si75(self):
        """TVOB at age 25, SI75 should be in plausible loblolly range (~2000–6000 cuft/ac)."""
        snaps = run_deterministic_trajectory(75.0, 800.0, 5.0, 30, "pucp")
        s25 = next(s for s in snaps if s.age == 25)
        assert 2000 < s25.vol_tvob < 7000, (
            f"TVOB at age 25, SI75 = {s25.vol_tvob:.0f} cuft/ac — outside plausible range"
        )

    def test_tvob_magnitude_at_age35_si90(self):
        """TVOB at age 35, SI90 should be in plausible range (~5000–12000 cuft/ac)."""
        snaps = run_deterministic_trajectory(90.0, 550.0, 5.0, 30, "pucp")
        s35 = next(s for s in snaps if s.age == 35)
        assert 4000 < s35.vol_tvob < 12000, (
            f"TVOB at age 35, SI90 = {s35.vol_tvob:.0f} cuft/ac — outside plausible range"
        )
