"""Plotly visualization utilities for the interactive testbed."""

from __future__ import annotations

from typing import Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.adapter import StepRecord, RolloutResult


def create_trajectory_plot(
    history: list[StepRecord],
    variables: list[str] | None = None,
) -> go.Figure:
    """Create main trajectory visualization with multiple variables.
    
    Args:
        history: List of step records
        variables: Variables to plot. Defaults to ['age', 'tpa', 'ba', 'volume']
        
    Returns:
        Plotly figure with line charts and tooltips
    """
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet. Run simulation steps to see trajectory.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(height=400)
        return fig
    
    if variables is None:
        variables = ["age", "tpa", "ba", "volume"]
    
    # Create subplots
    n_vars = len(variables)
    fig = make_subplots(
        rows=n_vars, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[v.upper() for v in variables],
    )
    
    steps = [r.step for r in history]
    
    var_labels = {
        "age": "Age (years)",
        "tpa": "Trees/Acre",
        "ba": "Basal Area (ft²/ac)",
        "hd": "Dom. Height (ft)",
        "volume": "Volume (ft³/ac)",
        "reward": "Step Reward ($)",
        "cumulative_reward": "Cumulative Reward ($)",
    }
    
    colors = {
        "age": "#1f77b4",
        "tpa": "#2ca02c",
        "ba": "#d62728",
        "hd": "#9467bd",
        "volume": "#8c564b",
        "reward": "#e377c2",
        "cumulative_reward": "#17becf",
    }
    
    for i, var in enumerate(variables, 1):
        values = [getattr(r, var, 0) for r in history]
        
        # Build hover text with full state info
        hover_texts = []
        for r in history:
            dist_info = f"Disturbance: {r.disturbance}" if r.disturbance else "No disturbance"
            hover_texts.append(
                f"Step {r.step}<br>"
                f"Age: {r.age:.1f} yrs<br>"
                f"TPA: {r.tpa:.0f}<br>"
                f"BA: {r.ba:.1f} ft²/ac<br>"
                f"Volume: {r.volume:.0f} ft³/ac<br>"
                f"Action: {r.action_name}<br>"
                f"Reward: ${r.reward:.2f}<br>"
                f"{dist_info}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers",
                name=var_labels.get(var, var),
                line=dict(color=colors.get(var, "#333")),
                marker=dict(size=6),
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=i, col=1,
        )
        
        # Mark disturbance events
        dist_steps = [r.step for r in history if r.disturbance]
        dist_values = [getattr(r, var, 0) for r in history if r.disturbance]
        dist_labels = [r.disturbance for r in history if r.disturbance]
        
        if dist_steps:
            fig.add_trace(
                go.Scatter(
                    x=dist_steps,
                    y=dist_values,
                    mode="markers",
                    marker=dict(
                        size=12,
                        symbol="x",
                        color="red" if any(d == "severe" for d in dist_labels) else "orange",
                        line=dict(width=2),
                    ),
                    name="Disturbance",
                    showlegend=(i == 1),
                    hovertext=[f"{d} disturbance" for d in dist_labels],
                    hoverinfo="text",
                ),
                row=i, col=1,
            )
        
        fig.update_yaxes(title_text=var_labels.get(var, var), row=i, col=1)
    
    fig.update_xaxes(title_text="Step", row=n_vars, col=1)
    fig.update_layout(
        height=150 * n_vars + 100,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40),
        template="plotly_white",
    )
    
    return fig


def create_reward_plot(history: list[StepRecord]) -> go.Figure:
    """Create reward and cumulative reward plot."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    steps = [r.step for r in history]
    rewards = [r.reward for r in history]
    cumulative = [r.cumulative_reward for r in history]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=steps,
            y=rewards,
            name="Step Reward",
            marker_color="rgba(55, 128, 191, 0.7)",
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative Reward",
            line=dict(color="rgb(219, 64, 82)", width=2),
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Step Reward ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Reward ($)", secondary_y=True)
    fig.update_layout(
        height=300,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def create_transition_heatmap(
    matrix: np.ndarray,
    labels: list[str] | None = None,
    current_state: int | None = None,
    trajectory_path: list[int] | None = None,
    title: str = "Transition Matrix",
) -> go.Figure:
    """Create transition matrix heatmap visualization.
    
    Args:
        matrix: Transition probability matrix (n_states x n_states)
        labels: State labels
        current_state: Index of current state to highlight
        trajectory_path: List of state indices for trajectory overlay
        title: Plot title
    """
    n_states = matrix.shape[0]
    
    if labels is None:
        labels = [f"S{i}" for i in range(n_states)]
    
    # Truncate labels if too many states
    if n_states > 30:
        tick_indices = list(range(0, n_states, n_states // 10))
        tick_labels = [labels[i] for i in tick_indices]
    else:
        tick_indices = list(range(n_states))
        tick_labels = labels
    
    fig = go.Figure()
    
    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=list(range(n_states)),
            y=list(range(n_states)),
            colorscale="Blues",
            colorbar=dict(title="P(s'|s)"),
            hovertemplate="From: %{y}<br>To: %{x}<br>P: %{z:.3f}<extra></extra>",
        )
    )
    
    # Current state marker
    if current_state is not None and 0 <= current_state < n_states:
        fig.add_trace(
            go.Scatter(
                x=[current_state],
                y=[current_state],
                mode="markers",
                marker=dict(size=20, color="red", symbol="circle-open", line=dict(width=3)),
                name="Current State",
                hoverinfo="name",
            )
        )
    
    # Trajectory path overlay
    if trajectory_path and len(trajectory_path) > 1:
        for i in range(len(trajectory_path) - 1):
            from_s = trajectory_path[i]
            to_s = trajectory_path[i + 1]
            fig.add_trace(
                go.Scatter(
                    x=[from_s, to_s],
                    y=[from_s, to_s],
                    mode="lines+markers",
                    line=dict(color="rgba(255, 0, 0, 0.5)", width=2),
                    marker=dict(size=8, color="red"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Next State",
            tickmode="array",
            tickvals=tick_indices,
            ticktext=tick_labels,
            tickangle=45,
        ),
        yaxis=dict(
            title="Current State",
            tickmode="array",
            tickvals=tick_indices,
            ticktext=tick_labels,
            autorange="reversed",
        ),
        height=500,
        template="plotly_white",
    )
    
    return fig


def create_state_space_2d(
    trajectories: list[list[tuple[float, float, float]]],
    current_position: tuple[float, float, float] | None = None,
    x_var: str = "age",
    y_var: str = "ba",
) -> go.Figure:
    """Create 2D state space visualization with trajectory paths.
    
    Args:
        trajectories: List of trajectories, each a list of (age, tpa, ba) tuples
        current_position: Current state position (age, tpa, ba)
        x_var: Variable for x-axis ('age', 'tpa', or 'ba')
        y_var: Variable for y-axis ('age', 'tpa', or 'ba')
    """
    var_idx = {"age": 0, "tpa": 1, "ba": 2}
    var_labels = {"age": "Age (years)", "tpa": "Trees/Acre", "ba": "Basal Area (ft²/ac)"}
    
    xi = var_idx.get(x_var, 0)
    yi = var_idx.get(y_var, 2)
    
    fig = go.Figure()
    
    # Plot past trajectories with reduced opacity
    for i, traj in enumerate(trajectories[:-1] if len(trajectories) > 1 else []):
        if not traj:
            continue
        xs = [p[xi] for p in traj]
        ys = [p[yi] for p in traj]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                line=dict(color="gray", width=1),
                marker=dict(size=4, opacity=0.3),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    
    # Plot current trajectory
    if trajectories:
        current_traj = trajectories[-1]
        if current_traj:
            xs = [p[xi] for p in current_traj]
            ys = [p[yi] for p in current_traj]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                    name="Current Trajectory",
                    hovertemplate=f"{x_var}: %{{x:.1f}}<br>{y_var}: %{{y:.1f}}<extra></extra>",
                )
            )
    
    # Current position marker
    if current_position is not None:
        fig.add_trace(
            go.Scatter(
                x=[current_position[xi]],
                y=[current_position[yi]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="Current State",
            )
        )
    
    fig.update_layout(
        xaxis_title=var_labels.get(x_var, x_var),
        yaxis_title=var_labels.get(y_var, y_var),
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def create_batch_histogram(
    results: list[RolloutResult],
    metric: str = "final_reward",
    title: str = "Distribution of Final Rewards",
) -> go.Figure:
    """Create histogram of batch rollout results."""
    if metric == "final_reward":
        values = [r.final_reward for r in results]
        x_label = "Final Reward ($)"
    elif metric == "final_volume":
        values = [r.steps[-1].volume if r.steps else 0 for r in results]
        x_label = "Final Volume (ft³/ac)"
    elif metric == "disturbance_count":
        values = [sum(1 for s in r.steps if s.disturbance) for r in results]
        x_label = "Number of Disturbances"
    else:
        values = [r.final_reward for r in results]
        x_label = metric
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=20,
            marker_color="rgba(55, 128, 191, 0.7)",
            marker_line=dict(color="rgb(55, 128, 191)", width=1),
        )
    )
    
    # Add mean line
    mean_val = np.mean(values)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Count",
        height=350,
        template="plotly_white",
    )
    
    return fig


def create_batch_boxplot(
    results_by_config: dict[str, list[RolloutResult]],
    metric: str = "final_reward",
    title: str = "Comparison Across Configurations",
) -> go.Figure:
    """Create box plot comparing batch results across configurations."""
    fig = go.Figure()
    
    for config_name, results in results_by_config.items():
        if metric == "final_reward":
            values = [r.final_reward for r in results]
        elif metric == "final_volume":
            values = [r.steps[-1].volume if r.steps else 0 for r in results]
        else:
            values = [r.final_reward for r in results]
        
        fig.add_trace(
            go.Box(
                y=values,
                name=config_name,
                boxmean=True,
            )
        )
    
    fig.update_layout(
        title=title,
        yaxis_title=metric.replace("_", " ").title(),
        height=400,
        template="plotly_white",
    )
    
    return fig


def create_ecdf_plot(
    results_by_config: dict[str, list[RolloutResult]],
    metric: str = "final_reward",
    title: str = "Empirical CDF of Final Rewards",
) -> go.Figure:
    """Create ECDF plot for comparing distributions."""
    fig = go.Figure()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for i, (config_name, results) in enumerate(results_by_config.items()):
        if metric == "final_reward":
            values = sorted([r.final_reward for r in results])
        else:
            values = sorted([r.final_reward for r in results])
        
        n = len(values)
        ecdf_y = np.arange(1, n + 1) / n
        
        fig.add_trace(
            go.Scatter(
                x=values,
                y=ecdf_y,
                mode="lines",
                name=config_name,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="Cumulative Probability",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def create_policy_comparison_table(
    q_values: dict[Any, float],
    recommended_action: int,
    value: float,
) -> go.Figure:
    """Create a table showing Q-values and policy recommendation."""
    actions = list(q_values.keys())
    values = list(q_values.values())
    
    # Format action names
    action_names = []
    for a in actions:
        if hasattr(a, "name"):
            action_names.append(a.name)
        else:
            action_names.append(str(a))
    
    # Highlight recommended action
    colors = ["lightgreen" if a == recommended_action else "white" for a in actions]
    
    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=["Action", "Q-Value", "Recommended"],
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    action_names,
                    [f"${v:.2f}" for v in values],
                    ["✓" if a == recommended_action else "" for a in actions],
                ],
                fill_color=[colors, colors, colors],
                align="left",
            ),
        )
    ])
    
    fig.update_layout(
        title=f"Policy Recommendation (V(s) = ${value:.2f})",
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
