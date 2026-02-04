# Forest Stand MDP Interactive Testbed

An interactive Streamlit application for exploring trajectories of the forest stand MDP under different risk settings.

## Quick Start

```bash
# From the project root
streamlit run ui/mdp_testbed.py
```

The app will open in your browser at `http://localhost:8501`.

## Features Overview

### 1. Trajectory Panel (📈)

The main panel for running and visualizing individual rollouts.

**Controls:**
- **Manual Mode**: Select actions directly (No-op, Light Thin, Heavy Thin, Harvest & Replant)
- **Policy Mode**: Apply heuristic policies or solved MDP optimal policy
- **Multi-step**: Run multiple steps at once with a selected action or policy

**Visualizations:**
- Line charts for stand variables (Age, TPA, BA, Volume, Height)
- Disturbance events marked with red X markers
- Reward and cumulative reward bar/line chart
- Hover tooltips with full state information per timestep

**State Display:**
- Current Age, TPA, BA, and Cumulative Reward metrics
- Step history table with all recorded data

### 2. Transitions Panel (🔄)

Visualizes state transitions across the session.

**Transition Matrix View:**
- Heatmap of empirical transition probabilities P(s'|s)
- Current state highlighted with red circle
- Trajectory path overlaid on matrix
- Customizable discretization bins (Age, TPA, BA)

**State Space 2D View:**
- Scatter plot of trajectories in 2D state space
- Selectable axes (Age vs BA, TPA vs BA, etc.)
- Current trajectory in blue, past trajectories in gray
- Current state marked with red star

**Controls:**
- "Reset Transition View" clears accumulated data without resetting the environment
- Discretization settings in sidebar expander

### 3. Batch Mode Panel (📊)

Run multiple rollouts for statistical analysis.

**Multiple Rollouts:**
- Configure number of rollouts, max steps, and base seed
- View distribution histograms with mean line
- Metrics: final reward, final volume, disturbance count

**Parameter Sweep:**
- Compare across risk levels (low, medium, high)
- Sweep over mild disturbance probabilities
- Box plots and ECDF plots for comparison
- Summary table with mean/std statistics

**Export:**
- Download results as CSV or JSON
- Full step-by-step data available

### 4. History Panel (📜)

View and export all completed runs in the session.

- Summary table of all runs with metadata
- Export full history with step-level detail

## Sidebar Controls

### Session & Seed
- **Random Seed**: Set for reproducibility (editable)
- **Reset Environment**: Reset to initial conditions with current seed
- **New Seed**: Generate random seed and reset

### Risk Configuration
- **Risk Level**: Low, Medium, or High preset profiles
  - Low: σ_BA=0.05, 60-year severe interval
  - Medium: σ_BA=0.10, 30-year severe interval  
  - High: σ_BA=0.15, 10-year severe interval

### Disturbance Parameters (Live Adjustable)
- **Mild Disturbance Probability**: Annual probability (0-20%)
- **Severe Disturbance Interval**: Mean return interval (5-100 years)
- **Advanced**: TPA multipliers for mild/severe events

### Control Mode
- **Manual**: User selects each action
- **Policy-Driven**: Apply selected policy automatically
  - Heuristic policies: no_op, fixed_rotation_25/30, ba_threshold_120, volume_2000
  - MDP optimal: Requires solving MDP first (click "Solve MDP")

### Discretization Settings
- Customize bin edges for transition matrix visualization
- Format: comma-separated values (e.g., "0,5,10,15,20,25,30,35,40")

## Interpreting the Panels

### Trajectory Plot
- **TPA (Trees/Acre)**: Decreases with mortality and thinning, resets on harvest
- **BA (Basal Area)**: Grows over time, reduced by thinning/disturbances
- **Volume**: Derived from BA, TPA, and height; main economic metric
- **Disturbance markers**: Red X indicates when disturbance occurred

### Transition Matrix
- **Rows**: Current state (from)
- **Columns**: Next state (to)
- **Color intensity**: Transition probability
- **Red circle**: Current state position
- **Trajectory line**: Path taken through state space

### Batch Results
- **Histogram**: Distribution of outcomes across rollouts
- **Box plot**: Compare distributions across configurations
- **ECDF**: Cumulative probability for risk analysis

## Architecture

```
ui/
├── mdp_testbed.py    # Main Streamlit application
├── adapter.py        # Adapter layer (SimulatorAdapter, PolicyAdapter, TransitionVisualizer)
├── plotting.py       # Plotly visualization functions
└── README.md         # This file
```

### Key Classes

**SimulatorAdapter** (`adapter.py`)
- Wraps `StochasticPMRC` for UI interaction
- Manages state, history, and step execution
- Supports manual disturbance triggering

**PolicyAdapter** (`adapter.py`)
- Provides MDP solving and policy recommendation
- Wraps heuristic policies from `core/baselines.py`

**TransitionVisualizer** (`adapter.py`)
- Tracks state transitions for visualization
- Builds empirical transition matrices
- Manages trajectory paths

## Acceptance Criteria Checklist

- [x] Step through rollout manually with updated plots
- [x] See disturbances applied and visualized
- [x] Switch to policy mode and apply recommended actions
- [x] Adjust disturbance probabilities mid-run
- [x] Transition view persists across runs in-session
- [x] Transition view can be reset independently
- [x] Batch runs produce summary plots
- [x] Export batch results to CSV/JSON
- [x] Control discretization/bucketing for transition matrix

## Dependencies

- streamlit
- plotly
- pandas
- numpy

All dependencies should already be in the project's `pyproject.toml`.
