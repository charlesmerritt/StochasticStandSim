"""
main.py — Comprehensive model walkthrough and validation script.

Exercises every primary decision point of the PMRC-based stochastic forest
simulation: deterministic growth, Weibull size classes, thinning from below,
product distribution, process noise, disturbances, recruitment, salvage,
replanting, feasibility constraints, and Monte Carlo experimentation.

Run:  uv run python main.py

Output:
  - Console text with section-by-section validation
  - Diagnostic plots saved to output/walkthrough/
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from core.state import StandState, hd_from_si25_at_age, si25_from_hd_at_age
from core.pmrc_model import PMRCModel
from core.process_noise import NoiseParams, ProcessNoiseModel
from core.disturbances import DisturbanceParams, DisturbanceModel
from core.config import ScenarioConfig, ThinningParams
from core.stochastic_model import StochasticPMRC
from core.products import (
    ProductPrices,
    HarvestCosts,
    CUFT_TO_TON,
    estimate_product_distribution,
)
from core.simulate import (
    run_scenario,
    run_batch,
    compute_npv,
    compute_lev,
)
from core.metrics import summarize_distribution, compare_scenarios

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join("output", "walkthrough")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global constants — every parameter printed once, used everywhere
# ---------------------------------------------------------------------------
SEED = 42
AGE0 = 5.0
TPA0 = 850.0
SI25 = 80.0
REGION = "ucp"
ROTATION = 35
THIN_PARAMS = ThinningParams(
    trigger_age=15.0,
    ba_threshold=150.0,
    residual_ba=100.0,
    thin_cost=87.34,
)
PRICES = ProductPrices(pulpwood=9.51, chip_n_saw=23.51, sawtimber=27.82)
COSTS = HarvestCosts(logging=150.0, replanting=150.80)
DISCOUNT_RATE = 0.05
N_MC = 1000


def banner(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def sub_banner(title: str) -> None:
    print(f"\n--- {title} ---")


# ###########################################################################
#  PART A — Deterministic Baseline & Components
# ###########################################################################

def part_a():
    banner("PART A — Deterministic Baseline & Components")
    pmrc = PMRCModel(region=REGION)

    # ------------------------------------------------------------------
    # Block 1: Site index & initial state
    # ------------------------------------------------------------------
    sub_banner("1. Site Index & Initial State")
    print(f"  age0    = {AGE0} yr")
    print(f"  TPA0    = {TPA0}")
    print(f"  SI25    = {SI25} ft")
    print(f"  region  = {REGION}")

    hd0 = hd_from_si25_at_age(SI25, AGE0)
    si25_check = si25_from_hd_at_age(hd0, AGE0)
    ba0 = pmrc.ba_predict(age=AGE0, tpa=TPA0, hd=hd0, region=REGION)
    qmd0 = pmrc.qmd(tpa=TPA0, ba=ba0)

    print(f"\n  Derived HD at age {AGE0}: {hd0:.2f} ft")
    print(f"  Round-trip SI25:         {si25_check:.4f} (should be {SI25})")
    print(f"  Predicted initial BA:    {ba0:.2f} ft²/ac")
    print(f"  Initial QMD:             {qmd0:.2f} in")

    state0 = StandState.from_si25(
        age=AGE0, si25=SI25, tpa=TPA0, ba=ba0, region=REGION
    )

    # ------------------------------------------------------------------
    # Block 2: One-step PMRC projection
    # ------------------------------------------------------------------
    sub_banner("2. One-Step PMRC Projection (age 5 → 6)")
    hd1 = pmrc.hd_project(state0.age, state0.hd, state0.age + 1)
    tpa1 = pmrc.tpa_project(state0.tpa, state0.si25, state0.age, state0.age + 1)
    ba1 = pmrc.ba_project(
        state0.age, state0.tpa, tpa1, state0.ba, state0.hd, hd1,
        state0.age + 1, REGION,
    )
    print(f"  HD:  {state0.hd:.2f} → {hd1:.2f}  (Δ = {hd1 - state0.hd:+.2f} ft)")
    print(f"  TPA: {state0.tpa:.0f} → {tpa1:.0f}  (Δ = {tpa1 - state0.tpa:+.0f})")
    print(f"  BA:  {state0.ba:.2f} → {ba1:.2f}  (Δ = {ba1 - state0.ba:+.2f} ft²/ac)")

    # ------------------------------------------------------------------
    # Block 3: Full deterministic rotation (no thinning)
    # ------------------------------------------------------------------
    sub_banner("3. Deterministic Rotation — 35 yr, No Thinning")
    det_ages, det_hd, det_tpa, det_ba, det_vol = [], [], [], [], []
    state = StandState.from_si25(age=AGE0, si25=SI25, tpa=TPA0, ba=ba0, region=REGION)

    for yr in range(ROTATION + 1):
        if yr > 0:
            age2 = state.age + 1
            h2 = pmrc.hd_project(state.age, state.hd, age2)
            t2 = pmrc.tpa_project(state.tpa, state.si25, state.age, age2)
            b2 = pmrc.ba_project(
                state.age, state.tpa, t2, state.ba, state.hd, h2, age2, REGION
            )
            state = StandState(
                age=age2, hd=h2, tpa=t2, ba=b2,
                si25=SI25, region=REGION,
            )
        vol = pmrc.yield_predict(
            age=state.age, tpa=state.tpa, hd=state.hd, ba=state.ba,
            unit="TVOB", region=REGION,
        )
        det_ages.append(state.age)
        det_hd.append(state.hd)
        det_tpa.append(state.tpa)
        det_ba.append(state.ba)
        det_vol.append(vol)

    print(f"  {'Age':>5} {'HD':>8} {'TPA':>8} {'BA':>8} {'Vol':>10}")
    for i in range(0, len(det_ages), 5):
        print(
            f"  {det_ages[i]:5.0f} {det_hd[i]:8.1f} {det_tpa[i]:8.0f} "
            f"{det_ba[i]:8.1f} {det_vol[i]:10.1f}"
        )
    # Always print final year
    if len(det_ages) % 5 != 1:
        i = len(det_ages) - 1
        print(
            f"  {det_ages[i]:5.0f} {det_hd[i]:8.1f} {det_tpa[i]:8.0f} "
            f"{det_ba[i]:8.1f} {det_vol[i]:10.1f}"
        )

    det_final_nothin = StandState(
        age=det_ages[-1], hd=det_hd[-1], tpa=det_tpa[-1], ba=det_ba[-1],
        si25=SI25, region=REGION,
    )

    # ------------------------------------------------------------------
    # Block 4: Weibull size classes at mid-rotation
    # ------------------------------------------------------------------
    sub_banner("4. Weibull Size Classes at Age 20")
    # Find state at age 20
    idx20 = int(20 - AGE0)
    state20 = StandState(
        age=det_ages[idx20], hd=det_hd[idx20], tpa=det_tpa[idx20],
        ba=det_ba[idx20], si25=SI25, region=REGION,
    )
    dist20 = pmrc.diameter_class_distribution(
        ba=state20.ba, tpa=state20.tpa, region=REGION,
    )
    class_labels = ["0-6\"", "6-9\"", "9-12\"", "12-24\""]
    print(f"  Stand at age 20: HD={state20.hd:.1f}, TPA={state20.tpa:.0f}, BA={state20.ba:.1f}")
    print(f"  Weibull params: a={dist20.weibull_params.a:.3f}, "
          f"b={dist20.weibull_params.b:.3f}, c={dist20.weibull_params.c:.3f}")
    print(f"  Percentiles: {dist20.percentiles}")
    print(f"\n  {'Class':>8} {'TPA':>8} {'BA':>8}")
    for lbl, t, b in zip(class_labels, dist20.tpa_per_class, dist20.ba_per_class, strict=False):
        print(f"  {lbl:>8} {t:8.1f} {b:8.1f}")
    print(f"  {'Total':>8} {dist20.total_tpa:8.1f} {dist20.total_ba:8.1f}")

    # ------------------------------------------------------------------
    # Block 5: Product yields & valuation at ages 20 and 35
    # ------------------------------------------------------------------
    sub_banner("5. Product Yields & Valuation")
    for label, st in [("Age 20", state20), ("Age 40 (terminal)", det_final_nothin)]:
        py = pmrc.product_yields(
            age=st.age, tpa=st.tpa, hd=st.hd, ba=st.ba, unit="TVOB", region=REGION,
        )
        tons_p = py.pulpwood * CUFT_TO_TON
        tons_c = py.chip_n_saw * CUFT_TO_TON
        tons_s = py.sawtimber * CUFT_TO_TON
        rev = tons_p * PRICES.pulpwood + tons_c * PRICES.chip_n_saw + tons_s * PRICES.sawtimber
        print(f"\n  {label}: age={st.age:.0f}, TPA={st.tpa:.0f}, BA={st.ba:.1f}, HD={st.hd:.1f}")
        print(f"    Pulpwood:  {py.pulpwood:8.1f} cuft  ({tons_p:.2f} tons, ${tons_p * PRICES.pulpwood:.2f}/ac)")
        print(f"    CNS:       {py.chip_n_saw:8.1f} cuft  ({tons_c:.2f} tons, ${tons_c * PRICES.chip_n_saw:.2f}/ac)")
        print(f"    Sawtimber: {py.sawtimber:8.1f} cuft  ({tons_s:.2f} tons, ${tons_s * PRICES.sawtimber:.2f}/ac)")
        print(f"    Gross revenue: ${rev:.2f}/ac")

    # ------------------------------------------------------------------
    # Block 6: Deterministic rotation WITH BAT thinning
    # ------------------------------------------------------------------
    sub_banner("6. Deterministic Rotation — 35 yr, BAT Thinning")
    det_thin_config = ScenarioConfig(
        name="det_thin",
        scenario_type="deterministic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        discount_rate=DISCOUNT_RATE,
        prices=PRICES,
        costs=COSTS,
    )
    det_thin_result = run_scenario(det_thin_config)

    print(f"  Thinning occurred: {det_thin_result.thin_occurred}")
    if det_thin_result.thin_occurred:
        print(f"  Thinning year:     {det_thin_result.thin_year}")
        # Find pre-thin and post-thin states
        thin_yr = det_thin_result.thin_year
        rec_pre = det_thin_result.trajectory[thin_yr - 1]
        rec_post = det_thin_result.trajectory[thin_yr]
        print(f"  Pre-thin:  age={rec_pre.age:.0f}, TPA={rec_pre.tpa:.0f}, BA={rec_pre.ba:.1f}")
        print(f"  Post-thin: age={rec_post.age:.0f}, TPA={rec_post.tpa:.0f}, BA={rec_post.ba:.1f}")
        print(f"  BA removed:       {rec_post.thin_ba_removed:.1f} ft²/ac")
        print(f"  Thin revenue:     ${det_thin_result.thin_revenue:.2f}/ac")

    final_thin = det_thin_result.trajectory[-1]
    print(f"\n  Final state: age={final_thin.age:.0f}, HD={final_thin.hd:.1f}, "
          f"TPA={final_thin.tpa:.0f}, BA={final_thin.ba:.1f}")

    # ------------------------------------------------------------------
    # Block 7: Terminal harvest, NPV, LEV comparison
    # ------------------------------------------------------------------
    sub_banner("7. Terminal Harvest, NPV, LEV — Thin vs No-Thin")

    # No-thin NPV/LEV
    prod_nt = estimate_product_distribution(
        pmrc, age=det_final_nothin.age, ba=det_final_nothin.ba,
        tpa=det_final_nothin.tpa, hd=det_final_nothin.hd, region=REGION,
    )
    tons_nt = {
        "pulp": prod_nt.vol_pulp * CUFT_TO_TON,
        "cns": prod_nt.vol_cns * CUFT_TO_TON,
        "saw": prod_nt.vol_saw * CUFT_TO_TON,
    }
    gross_nt = (
        tons_nt["pulp"] * PRICES.pulpwood
        + tons_nt["cns"] * PRICES.chip_n_saw
        + tons_nt["saw"] * PRICES.sawtimber
    )
    net_nt = gross_nt - COSTS.total
    npv_nt = compute_npv(
        thin_revenue=0, thin_year=None,
        harvest_revenue=net_nt,
        rotation_length=ROTATION,
        discount_rate=DISCOUNT_RATE,
        establishment_cost=COSTS.replanting,
    )
    lev_nt = compute_lev(npv_nt, ROTATION, DISCOUNT_RATE)

    # Thin
    ty = det_thin_result.terminal_yield
    npv_th = det_thin_result.npv
    lev_th = det_thin_result.lev

    print(f"  {'':>18} {'No-Thin':>12} {'BAT Thin':>12}")
    print(f"  {'Gross harvest':>18} ${gross_nt:>10.2f}  ${ty.gross_revenue:>10.2f}")
    print(f"  {'Net harvest':>18} ${net_nt:>10.2f}  ${ty.net_revenue:>10.2f}")
    print(f"  {'Thin revenue':>18} ${'0.00':>10}  ${det_thin_result.thin_revenue:>10.2f}")
    print(f"  {'NPV':>18} ${npv_nt:>10.2f}  ${npv_th:>10.2f}")
    print(f"  {'LEV':>18} ${lev_nt:>10.2f}  ${lev_th:>10.2f}")

    # Extract thinned deterministic BA trajectory for plotting
    det_thin_ba = [rec.ba for rec in det_thin_result.trajectory]

    return det_ages, det_hd, det_tpa, det_ba, det_vol, npv_nt, det_final_nothin, det_thin_ba


# ###########################################################################
#  PART B — Stochastic Component Validation
# ###########################################################################

def part_b(det_ages, det_hd, det_tpa, det_ba, det_final_nothin):
    banner("PART B — Stochastic Component Validation")
    pmrc = PMRCModel(region=REGION)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Block 8: Process noise sampling
    # ------------------------------------------------------------------
    sub_banner("8. Process Noise Sampling (10k draws)")
    noise_params = NoiseParams(sigma_log_ba=0.14, sigma_log_hd=None, sigma_tpa=30.0,
                               use_binomial_tpa=True, lambda_proc=1.0)
    noise_model = ProcessNoiseModel(noise_params)

    ba_mults = np.array([noise_model.sample_ba_multiplier(rng) for _ in range(10_000)])
    print(f"  BA multiplier: mean={ba_mults.mean():.4f} (expect 1.0), "
          f"std={ba_mults.std():.4f}, range=[{ba_mults.min():.3f}, {ba_mults.max():.3f}]")

    # TPA noise (binomial): use reference TPA=500, expected mortality=20
    tpa_deltas = np.array([noise_model.sample_tpa_noise(500, 20, rng) for _ in range(10_000)])
    print(f"  TPA noise (binomial, n=500, p_die=20/500): mean={tpa_deltas.mean():.2f}, "
          f"std={tpa_deltas.std():.2f}")

    # Recruitment
    recruits = np.array([noise_model.sample_recruitment(ba=100, si25=SI25,
                         rng=rng) for _ in range(10_000)])
    lam_expected = max(0, 1.0 + (-0.005) * 100 + 0.02 * SI25)
    print(f"  Recruitment (BA=100, SI25={SI25}): λ_expected={lam_expected:.2f}, "
          f"empirical mean={recruits.mean():.2f}, std={recruits.std():.2f}")

    # ------------------------------------------------------------------
    # Block 9: Disturbance model sampling
    # ------------------------------------------------------------------
    sub_banner("9. Disturbance Model Sampling")
    dist_params = DisturbanceParams(p_dist=1/20, severity_mean=0.30, severity_kappa=12.0)
    dist_model = DisturbanceModel(dist_params)

    occurrences = np.array([dist_model.sample_occurrence(rng) for _ in range(10_000)])
    print(f"  Occurrence (p=1/20=0.05): empirical={occurrences.mean():.4f}")

    severities = np.array([dist_model.sample_severity(rng) for _ in range(1_000)])
    alpha_b = dist_params.severity_mean * dist_params.severity_kappa
    beta_b = (1 - dist_params.severity_mean) * dist_params.severity_kappa
    print(f"  Severity Beta(α={alpha_b:.1f}, β={beta_b:.1f}): "
          f"mean={severities.mean():.4f} (expect 0.30), "
          f"std={severities.std():.4f}, P5={np.percentile(severities, 5):.3f}, "
          f"P95={np.percentile(severities, 95):.3f}")

    # Apply one shock
    ref_state = StandState.from_si25(age=20, si25=SI25, tpa=400, ba=120, region=REGION)
    shocked_state, event = dist_model.apply_shock(ref_state, severity=0.40)
    print(f"\n  Shock example (severity=0.40):")
    print(f"    Before: TPA={ref_state.tpa:.0f}, BA={ref_state.ba:.1f}, HD={ref_state.hd:.1f}")
    print(f"    After:  TPA={shocked_state.tpa:.0f}, BA={shocked_state.ba:.1f}, HD={shocked_state.hd:.1f}")
    print(f"    Losses: TPA={event.tpa_loss:.0f}, BA={event.ba_loss:.1f}, HD={event.hd_loss:.1f}")

    # ------------------------------------------------------------------
    # Block 10: Recruitment across BA gradient
    # ------------------------------------------------------------------
    sub_banner("10. Recruitment Across BA Gradient")
    alpha = (1.0, -0.005, 0.02)
    print(f"  Recruitment λ = max(0, {alpha[0]} + {alpha[1]}·BA + {alpha[2]}·SI25)")
    print(f"  SI25 = {SI25}")
    print(f"  {'BA':>6} {'λ':>8} {'mean(1k)':>10}")
    for ba_val in [20, 60, 100, 140, 180]:
        lam = max(0, alpha[0] + alpha[1] * ba_val + alpha[2] * SI25)
        draws = np.array([rng.poisson(lam) for _ in range(1000)])
        print(f"  {ba_val:6d} {lam:8.3f} {draws.mean():10.3f}")

    # ------------------------------------------------------------------
    # Block 11: Feasibility constraints
    # ------------------------------------------------------------------
    sub_banner("11. Feasibility Constraints")
    print("  Rules enforced by StochasticPMRC._project_feasible:")
    print("    TPA >= 100  (PMRC lower bound)")
    print("    BA  >= 0")
    print("    HD  >= prev_hd  (height non-decreasing)")
    print("    age > 0")

    # Construct extreme post-disturbance state
    extreme = StandState(age=15, hd=30.0, tpa=50, ba=-5.0, si25=SI25, region=REGION)
    stoch_test = StochasticPMRC(pmrc, noise_params=NoiseParams(lambda_proc=0))
    clamped = stoch_test._project_feasible(extreme, prev_hd=35.0)
    print(f"\n  Extreme input:  age={extreme.age}, HD={extreme.hd}, TPA={extreme.tpa}, BA={extreme.ba}")
    print(f"  Clamped output: age={clamped.age}, HD={clamped.hd}, TPA={clamped.tpa}, BA={clamped.ba}")
    print(f"    HD clamped to prev_hd=35.0: {clamped.hd == 35.0}")
    print(f"    TPA clamped to 100:         {clamped.tpa == 100.0}")
    print(f"    BA clamped to 0:            {clamped.ba == 0.0}")

    # ------------------------------------------------------------------
    # Block 12: Zero-noise recovery test
    # ------------------------------------------------------------------
    sub_banner("12. Zero-Noise Recovery Test (stochastic must match deterministic)")
    zero_stoch = StochasticPMRC(
        pmrc,
        noise_params=NoiseParams(lambda_proc=0.0),
        disturbance_params=DisturbanceParams(p_dist=0.0),
    )
    ba0 = pmrc.ba_predict(age=AGE0, tpa=TPA0, hd=hd_from_si25_at_age(SI25, AGE0), region=REGION)
    state = StandState.from_si25(age=AGE0, si25=SI25, tpa=TPA0, ba=ba0, region=REGION)
    rng_zero = np.random.default_rng(999)

    for _ in range(ROTATION):
        state, _ = zero_stoch.sample_next_state(state, dt=1.0, rng=rng_zero)

    tol = 0.01
    hd_match = abs(state.hd - det_hd[-1]) < tol
    ba_match = abs(state.ba - det_ba[-1]) < tol
    # TPA may differ slightly due to feasibility floor (100 vs asymptote)
    tpa_match = abs(state.tpa - det_tpa[-1]) < 1.0

    print(f"  Deterministic final: HD={det_hd[-1]:.2f}, TPA={det_tpa[-1]:.0f}, BA={det_ba[-1]:.2f}")
    print(f"  Zero-noise final:   HD={state.hd:.2f}, TPA={state.tpa:.0f}, BA={state.ba:.2f}")
    print(f"  Match (tol={tol}): HD={hd_match}, TPA={tpa_match}, BA={ba_match}")
    if not (hd_match and ba_match and tpa_match):
        print("  *** WARNING: Zero-noise recovery FAILED — stochastic does not reduce to PMRC ***")
    else:
        print("  PASSED: Stochastic model reduces exactly to deterministic PMRC.")

    return severities


# ###########################################################################
#  PART C — Monte Carlo Experiments (1000 runs each)
# ###########################################################################

def part_c():
    banner("PART C — Monte Carlo Experiments (1000 runs each)")

    # Build the 4 configs
    det_config = ScenarioConfig(
        name="DET",
        scenario_type="deterministic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
    )

    s1_config = ScenarioConfig(
        name="S1_noise",
        scenario_type="stochastic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        noise_params=NoiseParams(lambda_proc=1.0),
        disturbance_params=DisturbanceParams(p_dist=0.0),
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
        n_trajectories=N_MC, seed=SEED,
    )

    s2_config = ScenarioConfig(
        name="S2_dist",
        scenario_type="stochastic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        noise_params=NoiseParams(lambda_proc=0.0),
        disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=0.30, severity_kappa=12.0),
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
        n_trajectories=N_MC, seed=SEED,
    )

    s3_config = ScenarioConfig(
        name="S3_full",
        scenario_type="stochastic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        noise_params=NoiseParams(lambda_proc=1.0),
        disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=0.30, severity_kappa=12.0),
        salvage_enabled=True,
        salvage_severity_threshold=0.3832,
        salvage_price_fraction=0.50,
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
        n_trajectories=N_MC, seed=SEED,
    )

    # Run deterministic baseline
    print("\n  Running DET (deterministic baseline)...")
    det_result = run_scenario(det_config)

    # ------------------------------------------------------------------
    # Block 13: S1 — Noise-only batch
    # ------------------------------------------------------------------
    sub_banner("13. S1: Noise-Only Batch (λ=1.0, p_dist=0)")
    print(f"  Running {N_MC} trajectories...")
    s1_batch = run_batch(s1_config, n_trajectories=N_MC, seed=SEED,
                         store_trajectories=False)
    s1_npv = summarize_distribution(s1_batch.npvs)
    print(f"  NPV: mean=${s1_npv.mean:.2f}, std=${s1_npv.std:.2f}, "
          f"P5=${s1_npv.p5:.2f}, P95=${s1_npv.p95:.2f}, CVaR5=${s1_npv.cvar_5:.2f}")

    # Thinning occurrence across runs
    thin_count = int(np.sum(s1_batch.thin_revenues > 0))
    print(f"  Thinning occurred in {thin_count}/{N_MC} runs "
          f"({100 * thin_count / N_MC:.1f}%)")
    if thin_count > 0:
        thin_revs = s1_batch.thin_revenues[s1_batch.thin_revenues > 0]
        print(f"  Thin revenue (when occurred): mean=${thin_revs.mean():.2f}, "
              f"std=${thin_revs.std():.2f}")

    # ------------------------------------------------------------------
    # Block 14: S2 — Disturbance-only batch
    # ------------------------------------------------------------------
    sub_banner("14. S2: Disturbance-Only Batch (λ=0, p_dist=1/20)")
    print(f"  Running {N_MC} trajectories...")
    s2_batch = run_batch(s2_config, n_trajectories=N_MC, seed=SEED + 1,
                         store_trajectories=True)
    s2_npv = summarize_distribution(s2_batch.npvs)
    print(f"  NPV: mean=${s2_npv.mean:.2f}, std=${s2_npv.std:.2f}, "
          f"P5=${s2_npv.p5:.2f}, P95=${s2_npv.p95:.2f}, CVaR5=${s2_npv.cvar_5:.2f}")

    dist_counts = s2_batch.disturbance_counts
    print(f"  Disturbance counts: mean={dist_counts.mean():.2f}, "
          f"max={dist_counts.max()}, zero={int(np.sum(dist_counts == 0))}/{N_MC}")
    print(f"  Expected disturbances per rotation: {ROTATION * (1/20):.2f}")

    # Severity across all events
    all_sevs = []
    for path in s2_batch.disturbance_severity_paths:
        all_sevs.extend(path)
    if all_sevs:
        all_sevs_arr = np.array(all_sevs)
        print(f"  All severity draws ({len(all_sevs)} events): "
              f"mean={all_sevs_arr.mean():.4f}, std={all_sevs_arr.std():.4f}")

    # Check that stand continues from damaged state (salvage is OFF)
    # Look for a trajectory with a disturbance and check age continuity
    print("\n  Verifying post-disturbance behavior (salvage OFF):")
    found_example = False
    for traj_result in (s2_batch.trajectories or []):
        for i, rec in enumerate(traj_result.trajectory):
            if rec.disturbance and i + 1 < len(traj_result.trajectory):
                next_rec = traj_result.trajectory[i + 1]
                # Age should increment (no reset)
                if next_rec.age > rec.age:
                    print(f"    Year {rec.year}: disturbance (severity={rec.disturbance_severity:.3f}), "
                          f"age {rec.age:.0f} → {next_rec.age:.0f} (continues, no reset)")
                    found_example = True
                    break
        if found_example:
            break
    if not found_example:
        print("    No disturbance example found in stored trajectories.")

    # ------------------------------------------------------------------
    # Block 15: S3 — Full + salvage batch
    # ------------------------------------------------------------------
    sub_banner("15. S3: Full + Salvage Batch (λ=1.0, p_dist=1/20, salvage ON)")
    print(f"  Running {N_MC} trajectories...")
    s3_batch = run_batch(s3_config, n_trajectories=N_MC, seed=SEED + 2,
                         store_trajectories=True)
    s3_npv = summarize_distribution(s3_batch.npvs)
    print(f"  NPV: mean=${s3_npv.mean:.2f}, std=${s3_npv.std:.2f}, "
          f"P5=${s3_npv.p5:.2f}, P95=${s3_npv.p95:.2f}, CVaR5=${s3_npv.cvar_5:.2f}")

    # Salvage stats
    salvage_count = int(np.sum(s3_batch.salvage_counts > 0))
    print(f"\n  Salvage events:")
    print(f"    Runs with ≥1 salvage: {salvage_count}/{N_MC} ({100 * salvage_count / N_MC:.1f}%)")
    print(f"    Total salvage events:  {int(s3_batch.salvage_counts.sum())}")
    if salvage_count > 0:
        salvage_revs = s3_batch.salvage_revenues[s3_batch.salvage_revenues > 0]
        if len(salvage_revs) > 0:
            print(f"    Salvage revenue (when occurred): mean=${salvage_revs.mean():.2f}, "
                  f"std=${salvage_revs.std():.2f}")

    # Verify replanting behavior (age reset after salvage)
    print("\n  Verifying replanting/age-reset after salvage:")
    found_salvage = False
    for traj_result in (s3_batch.trajectories or []):
        for i, rec in enumerate(traj_result.trajectory):
            if rec.salvage and i + 1 < len(traj_result.trajectory):
                next_rec = traj_result.trajectory[i + 1]
                # After salvage, state should reset to initial (age=age0+1 next step)
                print(f"    Year {rec.year}: salvage (severity={rec.disturbance_severity:.3f}, "
                      f"rev=${rec.salvage_revenue:.2f})")
                print(f"      Post-salvage state: age={rec.age:.0f}, TPA={rec.tpa:.0f}, BA={rec.ba:.1f}")
                print(f"      Next year state:    age={next_rec.age:.0f}, TPA={next_rec.tpa:.0f}, "
                      f"BA={next_rec.ba:.1f}")
                found_salvage = True
                break
        if found_salvage:
            break
    if not found_salvage:
        print("    No salvage events found in stored trajectories.")

    # ------------------------------------------------------------------
    # Block 16: Cross-scenario comparison table
    # ------------------------------------------------------------------
    sub_banner("16. Cross-Scenario Comparison (NPV)")
    results_map = {
        "DET": det_result,
        "S1_noise": s1_batch,
        "S2_dist": s2_batch,
        "S3_full": s3_batch,
    }
    comparison = compare_scenarios(results_map, metric="npv")

    print(f"  {'Scenario':<12} {'Mean':>10} {'Median':>10} {'P5':>10} "
          f"{'P95':>10} {'CVaR5':>10} {'P<Det':>8}")
    print(f"  {'-' * 70}")
    for name, stats in comparison.items():
        print(
            f"  {name:<12} ${stats['mean']:>9.0f} ${stats['median']:>9.0f} "
            f"${stats['p5']:>9.0f} ${stats['p95']:>9.0f} ${stats['cvar_5']:>9.0f} "
            f"{stats['downside_prob_vs_deterministic']:>7.1%}"
        )

    # ------------------------------------------------------------------
    # Block 17: Product distribution sanity
    # ------------------------------------------------------------------
    sub_banner("17. Product Distribution Sanity Check (terminal state)")
    pmrc = PMRCModel(region=REGION)

    # Check across S3 trajectories: at terminal age, saw > CNS > pulp?
    inversions = 0
    total_checked = 0
    for traj_result in (s3_batch.trajectories or []):
        final = traj_result.trajectory[-1]
        if final.ba > 0 and final.tpa > 0 and final.hd > 0:
            py = pmrc.product_yields(
                age=final.age, tpa=final.tpa, hd=final.hd, ba=final.ba,
                unit="TVOB", region=REGION,
            )
            total_checked += 1
            if not (py.sawtimber >= py.chip_n_saw >= py.pulpwood):
                inversions += 1

    print(f"  Checked {total_checked} terminal states from S3")
    print(f"  Expected ordering: sawtimber ≥ CNS ≥ pulpwood")
    print(f"  Inversions: {inversions}/{total_checked} "
          f"({100 * inversions / max(1, total_checked):.1f}%)")
    if inversions > 0:
        print("  (Some inversions expected when disturbances reset stand to young age)")

    # ------------------------------------------------------------------
    # Block 18: Single illustrative trajectory with salvage
    # ------------------------------------------------------------------
    sub_banner("18. Illustrative Trajectory (S3, with salvage event)")
    example_traj = None
    for traj_result in (s3_batch.trajectories or []):
        if traj_result.salvage_count > 0:
            example_traj = traj_result
            break

    if example_traj is not None:
        print(f"  {'Year':>5} {'Age':>5} {'HD':>7} {'TPA':>7} {'BA':>7} "
              f"{'Dist':>6} {'Sev':>6} {'Thin':>5} {'Salv':>5}")
        for rec in example_traj.trajectory:
            dist_flag = "Y" if rec.disturbance else ""
            thin_flag = "Y" if rec.thinned else ""
            salv_flag = "Y" if rec.salvage else ""
            sev_str = f"{rec.disturbance_severity:.3f}" if rec.disturbance else ""
            print(
                f"  {rec.year:5d} {rec.age:5.0f} {rec.hd:7.1f} {rec.tpa:7.0f} "
                f"{rec.ba:7.1f} {dist_flag:>6} {sev_str:>6} {thin_flag:>5} {salv_flag:>5}"
            )
        print(f"\n  NPV: ${example_traj.npv:.2f}/ac, "
              f"Salvage events: {example_traj.salvage_count}, "
              f"Salvage revenue: ${example_traj.total_salvage_revenue:.2f}/ac")
    else:
        print("  No trajectory with salvage found. Try increasing N_MC or p_dist.")

    return det_result, s1_batch, s2_batch, s3_batch


# ###########################################################################
#  PART D — Diagnostic Plots
# ###########################################################################

def part_d(det_ages, det_hd, det_tpa, det_ba, det_thin_ba,
           det_result, s1_batch, s2_batch, s3_batch, severities):
    banner("PART D — Diagnostic Plots")
    print(f"  Saving to {OUT_DIR}/")

    # ------------------------------------------------------------------
    # Block 19: Deterministic trajectory
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(det_ages, det_hd, "k-", linewidth=1.5)
    axes[0].set_xlabel("Age (yr)")
    axes[0].set_ylabel("HD (ft)")
    axes[0].set_title("Dominant Height")

    axes[1].plot(det_ages, det_tpa, "k-", linewidth=1.5)
    axes[1].set_xlabel("Age (yr)")
    axes[1].set_ylabel("TPA")
    axes[1].set_title("Trees per Acre")

    axes[2].plot(det_ages, det_ba, "k-", linewidth=1.5)
    axes[2].set_xlabel("Age (yr)")
    axes[2].set_ylabel("BA (ft²/ac)")
    axes[2].set_title("Basal Area")

    fig.suptitle("Deterministic PMRC Rotation (no thinning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "det_trajectory.png"), dpi=150)
    plt.close(fig)
    print("  Saved det_trajectory.png")

    # ------------------------------------------------------------------
    # Block 20: NPV distributions
    # ------------------------------------------------------------------
    det_npv = det_result.npv
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(s1_batch.npvs.min(), s2_batch.npvs.min(), s3_batch.npvs.min()) - 50,
        max(s1_batch.npvs.max(), s2_batch.npvs.max(), s3_batch.npvs.max()) + 50,
        60,
    )
    ax.hist(s1_batch.npvs, bins=bins, alpha=0.5, label="S1: noise-only", density=True)
    ax.hist(s2_batch.npvs, bins=bins, alpha=0.5, label="S2: disturbance", density=True)
    ax.hist(s3_batch.npvs, bins=bins, alpha=0.5, label="S3: full+salvage", density=True)
    ax.axvline(det_npv, color="k", linestyle="--", linewidth=1.5, label=f"DET (${det_npv:.0f})")
    ax.set_xlabel("NPV ($/ac)")
    ax.set_ylabel("Density")
    ax.set_title("NPV Distribution Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npv_distributions.png"), dpi=150)
    plt.close(fig)
    print("  Saved npv_distributions.png")

    # ------------------------------------------------------------------
    # Block 21: BA fan chart for S3
    # ------------------------------------------------------------------
    if s3_batch.trajectories:
        n_years = ROTATION + 1
        ba_matrix = np.full((len(s3_batch.trajectories), n_years), np.nan)
        for i, traj in enumerate(s3_batch.trajectories):
            for rec in traj.trajectory:
                if rec.year < n_years:
                    ba_matrix[i, rec.year] = rec.ba

        years = np.arange(n_years)
        p5 = np.nanpercentile(ba_matrix, 5, axis=0)
        p25 = np.nanpercentile(ba_matrix, 25, axis=0)
        p50 = np.nanpercentile(ba_matrix, 50, axis=0)
        p75 = np.nanpercentile(ba_matrix, 75, axis=0)
        p95 = np.nanpercentile(ba_matrix, 95, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(years, p5, p95, alpha=0.15, color="steelblue", label="P5–P95")
        ax.fill_between(years, p25, p75, alpha=0.3, color="steelblue", label="P25–P75")
        ax.plot(years, p50, color="steelblue", linewidth=1.5, label="Median")
        ax.plot(range(len(det_ba)), det_ba, "k--", linewidth=1, label="Det (no thin)")
        ax.plot(range(len(det_thin_ba)), det_thin_ba, "k-.", linewidth=1, label="Det (BAT thin)")
        ax.set_xlabel("Year")
        ax.set_ylabel("BA (ft²/ac)")
        ax.set_title("S3 Basal Area Fan Chart (1000 trajectories)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "ba_fan_chart.png"), dpi=150)
        plt.close(fig)
        print("  Saved ba_fan_chart.png")

    # ------------------------------------------------------------------
    # Block 22: NPV distribution with VaR and CVaR markers (S3)
    # ------------------------------------------------------------------
    s3_npv = summarize_distribution(s3_batch.npvs)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram
    counts, bin_edges, patches = ax.hist(
        s3_batch.npvs, bins=50, color="steelblue", alpha=0.65,
        edgecolor="white", linewidth=0.4, label="S3 NPV (n=1000)",
    )

    # Shade the tail below VaR
    for patch, left_edge in zip(patches, bin_edges[:-1], strict=False):
        if left_edge + (bin_edges[1] - bin_edges[0]) <= s3_npv.var_5:
            patch.set_facecolor("firebrick")
            patch.set_alpha(0.75)

    # VaR line
    ax.axvline(s3_npv.var_5, color="firebrick", linestyle="--", linewidth=1.5,
               label=f"VaR₅ = ${s3_npv.var_5:,.0f}")

    # CVaR line
    ax.axvline(s3_npv.cvar_5, color="darkred", linestyle=":", linewidth=1.5,
               label=f"CVaR₅ = ${s3_npv.cvar_5:,.0f}")

    # Mean
    ax.axvline(s3_npv.mean, color="black", linestyle="-", linewidth=1.2,
               label=f"Mean = ${s3_npv.mean:,.0f}")

    # Median
    ax.axvline(s3_npv.median, color="black", linestyle="--", linewidth=0.9,
               alpha=0.6, label=f"Median = ${s3_npv.median:,.0f}")

    # Deterministic baseline
    ax.axvline(det_result.npv, color="forestgreen", linestyle="-", linewidth=1.5,
               label=f"Deterministic = ${det_result.npv:,.0f}")

    # Percentile annotations
    ax.axvline(s3_npv.p95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.annotate(f"P95 = ${s3_npv.p95:,.0f}", xy=(s3_npv.p95, ax.get_ylim()[1] * 0.92),
                fontsize=8, color="gray", ha="left", va="top",
                xytext=(5, 0), textcoords="offset points")

    ax.set_xlabel("NPV ($/ac)")
    ax.set_ylabel("Count")
    ax.set_title("S3 NPV Distribution with Risk Measures (λ=1.0, p=1/20, salvage)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npv_risk_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved npv_risk_distribution.png")

    # ------------------------------------------------------------------
    # Block 23: Disturbance severity histogram vs theoretical Beta
    # ------------------------------------------------------------------
    if len(severities) > 0:
        from scipy.stats import beta as beta_dist
        alpha_b = 0.30 * 12.0
        beta_b = 0.70 * 12.0

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(severities, bins=40, density=True, alpha=0.6, color="coral",
                label="Empirical (1k draws)")
        x = np.linspace(0, 1, 200)
        ax.plot(x, beta_dist.pdf(x, alpha_b, beta_b), "k-", linewidth=1.5,
                label=f"Beta({alpha_b:.1f}, {beta_b:.1f})")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Density")
        ax.set_title("Disturbance Severity: Empirical vs Theoretical")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "severity_distribution.png"), dpi=150)
        plt.close(fig)
        print("  Saved severity_distribution.png")


# ###########################################################################
#  PART E — Rotation Length Sensitivity
# ###########################################################################

def part_e():
    banner("PART E — Rotation Length Sensitivity")

    rotations = list(range(10, 46))

    # Deterministic baseline config (with thinning)
    det_base = ScenarioConfig(
        name="det_rot",
        scenario_type="deterministic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
    )

    # Scenario base configs
    noise_only_base = ScenarioConfig(
        name="noise_only",
        scenario_type="stochastic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        noise_params=NoiseParams(lambda_proc=1.0),
        disturbance_params=DisturbanceParams(p_dist=0.0),
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
        n_trajectories=N_MC, seed=SEED,
    )

    dist_only_base = ScenarioConfig(
        name="dist_only_050",
        scenario_type="stochastic",
        age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
        rotation_length=ROTATION,
        thin_params=THIN_PARAMS,
        noise_params=NoiseParams(lambda_proc=0.0),
        disturbance_params=DisturbanceParams(p_dist=1/20, severity_mean=0.30, severity_kappa=12.0),
        discount_rate=DISCOUNT_RATE,
        prices=PRICES, costs=COSTS,
        n_trajectories=N_MC, seed=SEED,
    )

    # Disturbance-level configs for Figure 2 (noise ON + varying p_dist)
    dist_levels = [
        (0.033, "tab:blue"),
        (0.050, "tab:orange"),
        (0.100, "tab:red"),
    ]
    dist_level_bases = {}
    for p, _ in dist_levels:
        dist_level_bases[p] = ScenarioConfig(
            name=f"noise_dist_{p:.3f}",
            scenario_type="stochastic",
            age0=AGE0, tpa0=TPA0, si25=SI25, region=REGION,
            rotation_length=ROTATION,
            thin_params=THIN_PARAMS,
            noise_params=NoiseParams(lambda_proc=1.0),
            disturbance_params=DisturbanceParams(p_dist=p, severity_mean=0.30, severity_kappa=12.0),
            discount_rate=DISCOUNT_RATE,
            prices=PRICES, costs=COSTS,
            n_trajectories=N_MC, seed=SEED,
        )

    from dataclasses import replace as dc_replace

    # Deterministic sweep
    det_npvs = []
    det_levs = []
    for rot in rotations:
        cfg = dc_replace(det_base, name=f"det_rot_{rot}", rotation_length=rot)
        result = run_scenario(cfg)
        det_npvs.append(result.npv)
        det_levs.append(result.lev)

    # Helper to sweep one stochastic scenario
    def _sweep_scenario(base_cfg, label):
        mean_npv, p5_npv, p95_npv = [], [], []
        mean_lev, p5_lev, p95_lev = [], [], []
        var5_npv, cvar5_npv = [], []
        print(f"  Sweeping {label} rotation {rotations[0]}–{rotations[-1]} yr "
              f"({N_MC} MC runs each)...")
        for rot in rotations:
            cfg = dc_replace(base_cfg, name=f"{label}_rot_{rot}", rotation_length=rot)
            batch = run_batch(cfg, n_trajectories=N_MC, seed=SEED)
            sn = summarize_distribution(batch.npvs)
            sl = summarize_distribution(batch.levs)
            mean_npv.append(sn.mean)
            p5_npv.append(sn.p5)
            p95_npv.append(sn.p95)
            var5_npv.append(sn.var_5)
            cvar5_npv.append(sn.cvar_5)
            mean_lev.append(sl.mean)
            p5_lev.append(sl.p5)
            p95_lev.append(sl.p95)
        return dict(mean_npv=mean_npv, p5_npv=p5_npv, p95_npv=p95_npv,
                    var5_npv=var5_npv, cvar5_npv=cvar5_npv,
                    mean_lev=mean_lev, p5_lev=p5_lev, p95_lev=p95_lev)

    # Sweep for Figure 1
    noise_data = _sweep_scenario(noise_only_base, "Noise")
    dist_data = _sweep_scenario(dist_only_base, "Dist")

    # Sweep for Figure 2
    dist_level_data = {}
    for p, _ in dist_levels:
        dist_level_data[p] = _sweep_scenario(dist_level_bases[p], f"p={p:.3f}")

    rots = np.array(rotations)

    # Print summary table
    sub_banner("Rotation Length Sensitivity — NPV")
    print(f"  {'Rot':>4} {'Det':>10} {'Noise':>10} {'Dist':>10} "
          f"{'p=.033':>10} {'p=.050':>10} {'p=.100':>10}")
    for i, rot in enumerate(rotations):
        if rot % 5 == 0:
            print(f"  {rot:4d} ${det_npvs[i]:>9.0f} ${noise_data['mean_npv'][i]:>9.0f} "
                  f"${dist_data['mean_npv'][i]:>9.0f} "
                  f"${dist_level_data[0.033]['mean_npv'][i]:>9.0f} "
                  f"${dist_level_data[0.050]['mean_npv'][i]:>9.0f} "
                  f"${dist_level_data[0.100]['mean_npv'][i]:>9.0f}")

    # ------------------------------------------------------------------
    # Figure 1: Deterministic + Noise only + Disturbance only (p=1/20)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios_f1 = [
        ("Noise only (λ=1.0)", noise_data, "tab:blue"),
        ("Disturbance only (p=1/20)", dist_data, "tab:orange"),
    ]

    det_npv_arr = np.array(det_npvs)
    det_lev_arr = np.array(det_levs)

    ax = axes[0]
    for label, data, color in scenarios_f1:
        p95_clipped = np.minimum(data["p95_npv"], det_npv_arr)
        ax.fill_between(rots, data["p5_npv"], p95_clipped,
                         alpha=0.12, color=color)
        ax.plot(rots, data["mean_npv"], color=color, linewidth=1.5, label=label)
    ax.plot(rots, det_npvs, "k-", linewidth=2, label="Deterministic", zorder=5)
    ax.set_xlabel("Rotation Length (yr)")
    ax.set_ylabel("NPV ($/ac)")
    ax.set_title("NPV vs Rotation Length")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, data, color in scenarios_f1:
        p95_clipped = np.minimum(data["p95_lev"], det_lev_arr)
        ax.fill_between(rots, data["p5_lev"], p95_clipped,
                         alpha=0.12, color=color)
        ax.plot(rots, data["mean_lev"], color=color, linewidth=1.5, label=label)
    ax.plot(rots, det_levs, "k-", linewidth=2, label="Deterministic", zorder=5)
    ax.set_xlabel("Rotation Length (yr)")
    ax.set_ylabel("LEV ($/ac)")
    ax.set_title("LEV vs Rotation Length")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Component Isolation — {N_MC} MC runs per rotation",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rotation_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("  Saved rotation_sensitivity.png")

    # ------------------------------------------------------------------
    # Figure 2: Deterministic + Noise only + 3 disturbance levels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios_f2 = [
        ("Noise only (λ=1.0)", noise_data, "tab:green"),
    ]
    for p, color in dist_levels:
        scenarios_f2.append((f"λ=1.0, p={p:.3f}", dist_level_data[p], color))

    ax = axes[0]
    for label, data, color in scenarios_f2:
        p95_clipped = np.minimum(data["p95_npv"], det_npv_arr)
        ax.fill_between(rots, data["p5_npv"], p95_clipped,
                         alpha=0.10, color=color)
        ax.plot(rots, data["mean_npv"], color=color, linewidth=1.5, label=label)
    ax.plot(rots, det_npvs, "k-", linewidth=2, label="Deterministic", zorder=5)
    ax.set_xlabel("Rotation Length (yr)")
    ax.set_ylabel("NPV ($/ac)")
    ax.set_title("NPV vs Rotation Length")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, data, color in scenarios_f2:
        p95_clipped = np.minimum(data["p95_lev"], det_lev_arr)
        ax.fill_between(rots, data["p5_lev"], p95_clipped,
                         alpha=0.10, color=color)
        ax.plot(rots, data["mean_lev"], color=color, linewidth=1.5, label=label)
    ax.plot(rots, det_levs, "k-", linewidth=2, label="Deterministic", zorder=5)
    ax.set_xlabel("Rotation Length (yr)")
    ax.set_ylabel("LEV ($/ac)")
    ax.set_title("LEV vs Rotation Length")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Disturbance Level Sensitivity (λ=1.0) — {N_MC} MC runs per rotation",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rotation_sensitivity_disturbance.png"), dpi=150)
    plt.close(fig)
    print("  Saved rotation_sensitivity_disturbance.png")


# ###########################################################################
#  Main
# ###########################################################################

def main():
    banner("MODEL WALKTHROUGH — Comprehensive Validation Script")
    print(f"  Seed:     {SEED}")
    print(f"  Stand:    age0={AGE0}, TPA0={TPA0}, SI25={SI25}, region={REGION}")
    print(f"  Rotation: {ROTATION} yr")
    print(f"  Thinning: BAT (age {THIN_PARAMS.trigger_age}, BA>{THIN_PARAMS.ba_threshold} "
          f"→ residual {THIN_PARAMS.residual_ba})")
    print(f"  Prices:   pulp=${PRICES.pulpwood}, CNS=${PRICES.chip_n_saw}, "
          f"saw=${PRICES.sawtimber} $/ton")
    print(f"  Costs:    logging=${COSTS.logging}, replanting=${COSTS.replanting} $/ac")
    print(f"  Discount: {DISCOUNT_RATE:.0%}")
    print(f"  MC runs:  {N_MC}")
    print(f"  Output:   {OUT_DIR}/")

    # Part A
    det_ages, det_hd, det_tpa, det_ba, det_vol, npv_nt, det_final, det_thin_ba = part_a()

    # Part B
    severities = part_b(det_ages, det_hd, det_tpa, det_ba, det_final)

    # Part C
    det_result, s1_batch, s2_batch, s3_batch = part_c()

    # Part D
    part_d(det_ages, det_hd, det_tpa, det_ba, det_thin_ba,
           det_result, s1_batch, s2_batch, s3_batch, severities)

    # Part E
    part_e()

    banner("DONE")
    print(f"  All validation blocks completed.")
    print(f"  Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
