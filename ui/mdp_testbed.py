"""Interactive Streamlit testbed for exploring forest stand MDP trajectories.

Run with: streamlit run ui/mdp_testbed.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ActionSpec,
    DisturbanceParams,
    MDPDiscretization,
    RiskLevel,
    get_risk_profile,
    RISK_PROFILES,
)
from core.mdp import BuongiornoConfig
from core.pmrc_model import PMRCModel

from ui.adapter import (
    SimulatorAdapter,
    SimulatorConfig,
    PolicyAdapter,
    TransitionVisualizer,
    RolloutResult,
    run_batch_rollouts,
    run_parameter_sweep,
)
from ui.plotting import (
    create_trajectory_plot,
    create_reward_plot,
    create_transition_heatmap,
    create_state_space_2d,
    create_batch_histogram,
    create_batch_boxplot,
    create_ecdf_plot,
    create_policy_comparison_table,
)


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        # Core simulation state
        "simulator": None,
        "seed": 42,
        "risk_level": "medium",
        "initialized": False,
        
        # Disturbance parameters (live adjustable)
        "p_mild": 0.02,
        "severe_mean_interval": 30.0,
        "mild_tpa_mult": 0.85,
        "severe_tpa_mult": 0.40,
        
        # Control mode
        "control_mode": "manual",  # "manual" or "policy"
        "selected_policy": "no_op",
        
        # Run history (persists across resets within session)
        "run_history": [],  # List of RolloutResult
        "current_run_id": 0,
        
        # Transition visualization state
        "transition_viz": None,
        "trajectory_positions": [],  # List of (age, tpa, ba) tuples for current run
        "all_trajectories": [],  # All trajectories for state space viz
        
        # Batch mode results
        "batch_results": None,
        "sweep_results": None,
        
        # MDP solution cache
        "mdp_solution": None,
        "mdp_risk_level": None,
        
        # Discretization settings
        "age_bins": "0,5,10,15,20,25,30,35,40",
        "tpa_bins": "100,200,300,400,500,600",
        "ba_bins": "0,40,80,120,160,200",
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_disturbance_params() -> DisturbanceParams:
    """Build DisturbanceParams from current session state."""
    return DisturbanceParams(
        p_mild=st.session_state.p_mild,
        severe_mean_interval=st.session_state.severe_mean_interval,
        mild_tpa_multiplier=st.session_state.mild_tpa_mult,
        severe_tpa_multiplier=st.session_state.severe_tpa_mult,
        mild_hd_multiplier=0.95,
        severe_hd_multiplier=0.80,
    )


def get_discretization() -> MDPDiscretization:
    """Parse discretization bins from session state."""
    try:
        age_bins = tuple(float(x) for x in st.session_state.age_bins.split(","))
        tpa_bins = tuple(float(x) for x in st.session_state.tpa_bins.split(","))
        ba_bins = tuple(float(x) for x in st.session_state.ba_bins.split(","))
        return MDPDiscretization(age_bins=age_bins, tpa_bins=tpa_bins, ba_bins=ba_bins)
    except Exception:
        return MDPDiscretization()


def reset_environment() -> None:
    """Reset the simulator to initial conditions."""
    config = SimulatorConfig(
        si25=60.0,
        region="ucp",
        initial_age=1.0,
        initial_tpa=600.0,
        dt=1.0,
        max_age=40.0,
    )
    
    risk_profile = get_risk_profile(st.session_state.risk_level)
    
    # Create simulator
    st.session_state.simulator = SimulatorAdapter(
        config=config,
        risk_profile=risk_profile,
        seed=st.session_state.seed,
    )
    st.session_state.simulator.reset()
    
    # Initialize transition visualizer
    discretization = get_discretization()
    st.session_state.transition_viz = TransitionVisualizer(discretization)
    st.session_state.transition_viz.start_new_trajectory()
    
    # Record initial state
    initial_state = st.session_state.simulator.state
    st.session_state.transition_viz.record_state(initial_state)
    st.session_state.trajectory_positions = [
        (initial_state.age, initial_state.tpa, initial_state.ba)
    ]
    
    st.session_state.initialized = True
    st.session_state.current_run_id += 1


def save_current_run() -> None:
    """Save current run to history."""
    if st.session_state.simulator is None:
        return
    
    result = st.session_state.simulator.get_rollout_result()
    result.metadata["run_id"] = st.session_state.current_run_id
    result.metadata["timestamp"] = datetime.datetime.now().isoformat()
    st.session_state.run_history.append(result)
    
    # Save trajectory for state space visualization
    if st.session_state.trajectory_positions:
        st.session_state.all_trajectories.append(
            list(st.session_state.trajectory_positions)
        )


# =============================================================================
# Sidebar Controls
# =============================================================================

def render_sidebar() -> None:
    """Render all sidebar controls."""
    st.sidebar.title("🌲 MDP Testbed")
    
    # Session & Reproducibility
    st.sidebar.header("Session & Seed")
    new_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=st.session_state.seed,
        key="seed_input",
    )
    if new_seed != st.session_state.seed:
        st.session_state.seed = new_seed
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Reset Environment", type="primary"):
        save_current_run()
        reset_environment()
        st.rerun()
    
    if col2.button("New Seed"):
        st.session_state.seed = np.random.randint(0, 999999)
        save_current_run()
        reset_environment()
        st.rerun()
    
    # Risk Settings
    st.sidebar.header("Risk Configuration")
    risk_options = list(RISK_PROFILES.keys())
    risk_idx = risk_options.index(st.session_state.risk_level)
    new_risk = st.sidebar.selectbox(
        "Risk Level",
        options=risk_options,
        index=risk_idx,
        format_func=lambda x: f"{x.title()} - {RISK_PROFILES[x].description}",
    )
    if new_risk != st.session_state.risk_level:
        st.session_state.risk_level = new_risk
        st.session_state.mdp_solution = None  # Invalidate cached solution
    
    # Live Disturbance Controls
    st.sidebar.header("Disturbance Parameters")
    st.sidebar.caption("Adjust live during simulation")
    
    st.session_state.p_mild = st.sidebar.slider(
        "Mild Disturbance Probability",
        min_value=0.0,
        max_value=0.20,
        value=st.session_state.p_mild,
        step=0.01,
        help="Annual probability of mild disturbance",
    )
    
    st.session_state.severe_mean_interval = st.sidebar.slider(
        "Severe Disturbance Interval (years)",
        min_value=5.0,
        max_value=100.0,
        value=st.session_state.severe_mean_interval,
        step=5.0,
        help="Mean return interval for severe disturbances",
    )
    
    with st.sidebar.expander("Advanced Disturbance Settings"):
        st.session_state.mild_tpa_mult = st.slider(
            "Mild TPA Multiplier",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.mild_tpa_mult,
            step=0.05,
        )
        st.session_state.severe_tpa_mult = st.slider(
            "Severe TPA Multiplier",
            min_value=0.1,
            max_value=0.8,
            value=st.session_state.severe_tpa_mult,
            step=0.05,
        )
    
    # Control Mode
    st.sidebar.header("Control Mode")
    st.session_state.control_mode = st.sidebar.radio(
        "Action Selection",
        options=["manual", "policy"],
        format_func=lambda x: "Manual Control" if x == "manual" else "Policy-Driven",
    )
    
    if st.session_state.control_mode == "policy":
        policy_options = [
            "no_op",
            "fixed_rotation_25",
            "fixed_rotation_30",
            "ba_threshold_120",
            "volume_2000",
            "mdp_optimal",
        ]
        st.session_state.selected_policy = st.sidebar.selectbox(
            "Select Policy",
            options=policy_options,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        
        if st.session_state.selected_policy == "mdp_optimal":
            if st.sidebar.button("Solve MDP"):
                with st.spinner("Solving MDP..."):
                    solve_mdp_for_current_risk()
    
    # Discretization Settings
    with st.sidebar.expander("Discretization (Transition Matrix)"):
        st.session_state.age_bins = st.text_input(
            "Age Bins",
            value=st.session_state.age_bins,
            help="Comma-separated bin edges",
        )
        st.session_state.tpa_bins = st.text_input(
            "TPA Bins",
            value=st.session_state.tpa_bins,
        )
        st.session_state.ba_bins = st.text_input(
            "BA Bins",
            value=st.session_state.ba_bins,
        )
        
        if st.button("Apply Discretization"):
            discretization = get_discretization()
            st.session_state.transition_viz = TransitionVisualizer(discretization)
            st.success(f"Applied: {discretization.n_states} states")


def solve_mdp_for_current_risk() -> None:
    """Solve MDP for current risk level and cache result."""
    pmrc = PMRCModel(region="ucp")
    config = BuongiornoConfig()
    policy_adapter = PolicyAdapter(pmrc, config)
    
    solution = policy_adapter.solve_mdp(
        st.session_state.risk_level,
        n_samples=500,
        seed=st.session_state.seed,
    )
    
    st.session_state.mdp_solution = solution
    st.session_state.mdp_risk_level = st.session_state.risk_level
    st.session_state.policy_adapter = policy_adapter


# =============================================================================
# Main Panel - Trajectory Visualization
# =============================================================================

def render_main_panel() -> None:
    """Render the main trajectory visualization panel."""
    st.header("Trajectory Visualization")
    
    if not st.session_state.initialized or st.session_state.simulator is None:
        st.info("Click 'Reset Environment' in the sidebar to initialize the simulation.")
        return
    
    sim = st.session_state.simulator
    
    # Current state display
    col1, col2, col3, col4 = st.columns(4)
    state = sim.state
    col1.metric("Age", f"{state.age:.1f} yrs")
    col2.metric("TPA", f"{state.tpa:.0f}")
    col3.metric("BA", f"{state.ba:.1f} ft²/ac")
    col4.metric("Cumulative Reward", f"${sim.cumulative_reward:.2f}")
    
    # Action controls
    st.subheader("Actions")
    
    if st.session_state.control_mode == "manual":
        render_manual_controls()
    else:
        render_policy_controls()
    
    # Disturbance trigger
    st.subheader("Manual Disturbance Trigger")
    dist_col1, dist_col2, dist_col3 = st.columns(3)
    
    if dist_col1.button("Trigger Mild Disturbance"):
        sim.trigger_disturbance("mild")
        st.warning("Mild disturbance queued for next step")
    
    if dist_col2.button("Trigger Severe Disturbance"):
        sim.trigger_disturbance("severe")
        st.error("Severe disturbance queued for next step")
    
    # Trajectory plots
    st.subheader("Stand Variables Over Time")
    history = sim.history
    
    var_options = ["age", "tpa", "ba", "volume", "hd"]
    selected_vars = st.multiselect(
        "Variables to Display",
        options=var_options,
        default=["tpa", "ba", "volume"],
    )
    
    if selected_vars:
        fig = create_trajectory_plot(history, selected_vars)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reward plot
    st.subheader("Rewards")
    fig_reward = create_reward_plot(history)
    st.plotly_chart(fig_reward, use_container_width=True)
    
    # Step history table
    with st.expander("Step History Table"):
        if history:
            df = pd.DataFrame([
                {
                    "Step": r.step,
                    "Age": f"{r.age:.1f}",
                    "TPA": f"{r.tpa:.0f}",
                    "BA": f"{r.ba:.1f}",
                    "Volume": f"{r.volume:.0f}",
                    "Action": r.action_name,
                    "Reward": f"${r.reward:.2f}",
                    "Disturbance": r.disturbance or "-",
                }
                for r in history
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No steps recorded yet.")


def render_manual_controls() -> None:
    """Render manual action selection controls."""
    sim = st.session_state.simulator
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    actions = [
        (0, "No-op", "Let stand grow"),
        (1, "Light Thin (20%)", "Remove 20% of BA"),
        (2, "Heavy Thin (40%)", "Remove 40% of BA"),
        (3, "Harvest & Replant", "Clear-cut and replant"),
    ]
    
    for col, (action_id, name, desc) in zip(
        [action_col1, action_col2, action_col3, action_col4], actions
    ):
        if col.button(name, help=desc, disabled=sim.is_done):
            execute_step(action_id)
            st.rerun()
    
    # Multi-step controls
    st.markdown("---")
    step_col1, step_col2 = st.columns(2)
    
    n_steps = step_col1.number_input("Steps", min_value=1, max_value=40, value=5)
    if step_col2.button(f"Run {n_steps} Steps (No-op)", disabled=sim.is_done):
        for _ in range(int(n_steps)):
            if sim.is_done:
                break
            execute_step(0)
        st.rerun()


def render_policy_controls() -> None:
    """Render policy-driven action controls."""
    sim = st.session_state.simulator
    policy_name = st.session_state.selected_policy
    
    # Show policy recommendation
    if policy_name == "mdp_optimal" and st.session_state.mdp_solution is not None:
        render_mdp_recommendation()
    
    col1, col2 = st.columns(2)
    
    if col1.button("Apply Policy Action", type="primary", disabled=sim.is_done):
        action = get_policy_action()
        execute_step(action)
        st.rerun()
    
    n_steps = col2.number_input("Auto-run steps", min_value=1, max_value=40, value=10)
    if st.button(f"Auto-run {n_steps} Steps with Policy", disabled=sim.is_done):
        for _ in range(int(n_steps)):
            if sim.is_done:
                break
            action = get_policy_action()
            execute_step(action)
        st.rerun()


def render_mdp_recommendation() -> None:
    """Display MDP policy recommendation for current state."""
    if st.session_state.mdp_solution is None:
        st.warning("MDP not solved. Click 'Solve MDP' in sidebar.")
        return
    
    sim = st.session_state.simulator
    policy_adapter = st.session_state.get("policy_adapter")
    
    if policy_adapter is None:
        return
    
    recommendation = policy_adapter.get_mdp_recommendation(
        sim.state,
        st.session_state.mdp_solution,
    )
    
    st.info(
        f"**MDP Recommendation:** {recommendation['action_name']} | "
        f"V(s) = ${recommendation['value']:.2f} | "
        f"State: {recommendation['discrete_state']}"
    )
    
    # Q-value table
    fig = create_policy_comparison_table(
        recommendation["q_values"],
        recommendation["action"],
        recommendation["value"],
    )
    st.plotly_chart(fig, use_container_width=True)


def get_policy_action() -> int:
    """Get action from selected policy."""
    sim = st.session_state.simulator
    policy_name = st.session_state.selected_policy
    
    if policy_name == "mdp_optimal":
        if st.session_state.mdp_solution is not None:
            policy_adapter = st.session_state.get("policy_adapter")
            if policy_adapter:
                rec = policy_adapter.get_mdp_recommendation(
                    sim.state, st.session_state.mdp_solution
                )
                return rec["action"]
        return 0
    
    # Heuristic policies
    pmrc = PMRCModel(region="ucp")
    config = BuongiornoConfig()
    policy_adapter = PolicyAdapter(pmrc, config)
    return policy_adapter.get_heuristic_action(sim.state, policy_name)


def execute_step(action: int) -> None:
    """Execute a simulation step and update visualizations."""
    sim = st.session_state.simulator
    viz = st.session_state.transition_viz
    
    prev_state = sim.state
    
    # Use current disturbance params
    dist_params = get_current_disturbance_params()
    next_state, reward, done, info = sim.step(action, dist_params)
    
    # Update transition visualization
    if viz is not None:
        viz.record_transition(prev_state, next_state)
        viz.record_state(next_state)
    
    # Update trajectory positions
    st.session_state.trajectory_positions.append(
        (next_state.age, next_state.tpa, next_state.ba)
    )
    
    if done:
        save_current_run()
        st.success("Episode complete!")


# =============================================================================
# Transition Visualization Panel
# =============================================================================

def render_transition_panel() -> None:
    """Render the transition matrix / state space visualization panel."""
    st.header("Transition Visualization")
    
    viz = st.session_state.transition_viz
    if viz is None:
        st.info("Initialize the environment to see transition visualization.")
        return
    
    # View mode selector
    view_mode = st.radio(
        "View Mode",
        options=["Transition Matrix", "State Space 2D"],
        horizontal=True,
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Reset Transition View"):
            viz.reset()
            viz.start_new_trajectory()
            st.session_state.all_trajectories = []
            st.success("Transition view cleared")
            st.rerun()
        
        st.metric("Recorded Transitions", len(viz._transition_counts))
        st.metric("Trajectories", len(viz.get_trajectory_paths()))
    
    with col1:
        if view_mode == "Transition Matrix":
            render_transition_matrix(viz)
        else:
            render_state_space_2d()


def render_transition_matrix(viz: TransitionVisualizer) -> None:
    """Render transition matrix heatmap."""
    matrix = viz.get_transition_matrix()
    labels = viz.get_state_labels()
    
    # Get current state index
    current_idx = None
    if st.session_state.simulator is not None:
        current_idx = viz.encode_state(st.session_state.simulator.state)
    
    # Get current trajectory path
    paths = viz.get_trajectory_paths()
    current_path = paths[-1] if paths else None
    
    fig = create_transition_heatmap(
        matrix,
        labels=labels,
        current_state=current_idx,
        trajectory_path=current_path,
        title=f"Empirical Transition Matrix ({viz.n_states} states)",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # State info
    if current_idx is not None:
        age, tpa, ba = viz.get_bin_midpoints(current_idx)
        st.caption(
            f"Current discrete state: {current_idx} "
            f"(Age: {age:.0f}, TPA: {tpa:.0f}, BA: {ba:.0f})"
        )


def render_state_space_2d() -> None:
    """Render 2D state space visualization."""
    trajectories = st.session_state.all_trajectories.copy()
    if st.session_state.trajectory_positions:
        trajectories.append(st.session_state.trajectory_positions)
    
    current_pos = None
    if st.session_state.simulator is not None:
        state = st.session_state.simulator.state
        current_pos = (state.age, state.tpa, state.ba)
    
    col1, col2 = st.columns(2)
    x_var = col1.selectbox("X-axis", options=["age", "tpa", "ba"], index=0)
    y_var = col2.selectbox("Y-axis", options=["age", "tpa", "ba"], index=2)
    
    fig = create_state_space_2d(
        trajectories,
        current_position=current_pos,
        x_var=x_var,
        y_var=y_var,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Batch Mode Panel
# =============================================================================

def render_batch_panel() -> None:
    """Render batch mode controls and results."""
    st.header("Batch Mode")
    
    tab1, tab2 = st.tabs(["Multiple Rollouts", "Parameter Sweep"])
    
    with tab1:
        render_batch_rollouts()
    
    with tab2:
        render_parameter_sweep()


def render_batch_rollouts() -> None:
    """Render multiple rollouts batch mode."""
    st.subheader("Run Multiple Rollouts")
    
    col1, col2, col3 = st.columns(3)
    n_rollouts = col1.number_input("Number of Rollouts", min_value=5, max_value=500, value=50)
    max_steps = col2.number_input("Max Steps per Rollout", min_value=10, max_value=100, value=40)
    base_seed = col3.number_input("Base Seed", min_value=0, value=st.session_state.seed)
    
    if st.button("Run Batch", type="primary"):
        config = SimulatorConfig()
        
        with st.spinner(f"Running {n_rollouts} rollouts..."):
            results = run_batch_rollouts(
                config=config,
                risk_level=st.session_state.risk_level,
                n_rollouts=int(n_rollouts),
                base_seed=int(base_seed),
                max_steps=int(max_steps),
            )
        
        st.session_state.batch_results = results
        st.success(f"Completed {len(results)} rollouts")
    
    # Display results
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        st.subheader("Results Summary")
        
        rewards = [r.final_reward for r in results]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Reward", f"${np.mean(rewards):.2f}")
        col2.metric("Std Dev", f"${np.std(rewards):.2f}")
        col3.metric("Min", f"${np.min(rewards):.2f}")
        col4.metric("Max", f"${np.max(rewards):.2f}")
        
        # Plots
        metric = st.selectbox(
            "Metric",
            options=["final_reward", "final_volume", "disturbance_count"],
        )
        
        fig = create_batch_histogram(results, metric=metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        render_export_controls(results, "batch")


def render_parameter_sweep() -> None:
    """Render parameter sweep mode."""
    st.subheader("Parameter Sweep")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_levels = st.multiselect(
            "Risk Levels",
            options=["low", "medium", "high"],
            default=["low", "medium", "high"],
        )
    
    with col2:
        p_mild_values = st.text_input(
            "Mild Disturbance Probabilities (comma-separated)",
            value="0.01,0.02,0.05",
        )
    
    n_rollouts = st.number_input("Rollouts per Config", min_value=5, max_value=100, value=20)
    
    if st.button("Run Sweep", type="primary"):
        try:
            p_mild_list = [float(x.strip()) for x in p_mild_values.split(",")]
        except ValueError:
            st.error("Invalid probability values")
            return
        
        config = SimulatorConfig()
        
        with st.spinner("Running parameter sweep..."):
            results = run_parameter_sweep(
                config=config,
                risk_levels=risk_levels,
                disturbance_probs=p_mild_list,
                n_rollouts_per_config=int(n_rollouts),
            )
        
        st.session_state.sweep_results = results
        st.success(f"Completed sweep: {len(results)} configurations")
    
    # Display sweep results
    if st.session_state.sweep_results:
        results = st.session_state.sweep_results
        
        st.subheader("Sweep Results")
        
        # Summary table
        summary_data = []
        for config_name, rollouts in results.items():
            rewards = [r.final_reward for r in rollouts]
            summary_data.append({
                "Configuration": config_name,
                "Mean Reward": f"${np.mean(rewards):.2f}",
                "Std Dev": f"${np.std(rewards):.2f}",
                "N Rollouts": len(rollouts),
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Comparison plots
        fig_box = create_batch_boxplot(results)
        st.plotly_chart(fig_box, use_container_width=True)
        
        fig_ecdf = create_ecdf_plot(results)
        st.plotly_chart(fig_ecdf, use_container_width=True)
        
        # Export
        render_export_controls(results, "sweep")


def render_export_controls(
    results: list[RolloutResult] | dict[str, list[RolloutResult]],
    mode: str,
) -> None:
    """Render export controls for batch results."""
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    # Prepare export data
    if mode == "batch":
        export_data = [
            {
                "seed": r.seed,
                "risk_level": r.risk_level,
                "final_reward": r.final_reward,
                "n_steps": len(r.steps),
                "n_disturbances": sum(1 for s in r.steps if s.disturbance),
                "final_volume": r.steps[-1].volume if r.steps else 0,
            }
            for r in results
        ]
    else:  # sweep
        export_data = []
        for config_name, rollouts in results.items():
            for r in rollouts:
                export_data.append({
                    "config": config_name,
                    "seed": r.seed,
                    "final_reward": r.final_reward,
                    "n_steps": len(r.steps),
                })
    
    df = pd.DataFrame(export_data)
    
    # CSV export
    csv = df.to_csv(index=False)
    col1.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{mode}_results.csv",
        mime="text/csv",
    )
    
    # JSON export
    json_str = json.dumps(export_data, indent=2)
    col2.download_button(
        "Download JSON",
        data=json_str,
        file_name=f"{mode}_results.json",
        mime="application/json",
    )


# =============================================================================
# Run History Panel
# =============================================================================

def render_history_panel() -> None:
    """Render run history panel."""
    st.header("Run History")
    
    history = st.session_state.run_history
    
    if not history:
        st.info("No completed runs yet. Complete a rollout to see history.")
        return
    
    st.metric("Total Runs", len(history))
    
    # Summary table
    summary = []
    for r in history:
        summary.append({
            "Run ID": r.metadata.get("run_id", "?"),
            "Risk Level": r.risk_level,
            "Seed": r.seed,
            "Steps": len(r.steps),
            "Final Reward": f"${r.final_reward:.2f}",
            "Disturbances": sum(1 for s in r.steps if s.disturbance),
            "Timestamp": r.metadata.get("timestamp", "")[:19],
        })
    
    st.dataframe(pd.DataFrame(summary), use_container_width=True)
    
    # Export all history
    if st.button("Export All Run History"):
        all_data = []
        for r in history:
            for s in r.steps:
                all_data.append({
                    "run_id": r.metadata.get("run_id"),
                    "risk_level": r.risk_level,
                    "seed": r.seed,
                    "step": s.step,
                    "age": s.age,
                    "tpa": s.tpa,
                    "ba": s.ba,
                    "volume": s.volume,
                    "action": s.action_name,
                    "reward": s.reward,
                    "disturbance": s.disturbance,
                })
        
        df = pd.DataFrame(all_data)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Full History CSV",
            data=csv,
            file_name="full_run_history.csv",
            mime="text/csv",
        )


# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Forest Stand MDP Testbed",
        page_icon="🌲",
        layout="wide",
    )
    
    init_session_state()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trajectory",
        "🔄 Transitions",
        "📊 Batch Mode",
        "📜 History",
    ])
    
    with tab1:
        render_main_panel()
    
    with tab2:
        render_transition_panel()
    
    with tab3:
        render_batch_panel()
    
    with tab4:
        render_history_panel()
    
    # Footer
    st.markdown("---")
    st.caption(
        "Forest Stand MDP Interactive Testbed | "
        f"Session seed: {st.session_state.seed} | "
        f"Risk level: {st.session_state.risk_level}"
    )


if __name__ == "__main__":
    main()
