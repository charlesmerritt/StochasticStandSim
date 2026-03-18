# Thesis Methods Notes

Reusable prose snippets for the methods chapter.

---

## Stochastic Transition Model

The stochastic stand simulator extends the deterministic PMRC growth and yield model (Harrison and Borders, 1996) to incorporate aleatoric uncertainty from two sources: process noise on growth increments and catastrophic disturbance events.

The transition from state $s_t$ to $s_{t+1}$ proceeds in three stages: (1) deterministic PMRC projection, (2) stochastic perturbation of growth increments, and (3) disturbance shock application.

---

## Process Noise

Process noise captures year-to-year growth variability not explained by the deterministic model. Following standard practice in forest growth modeling, we apply multiplicative lognormal noise to growth increments rather than state levels:

$$\Delta x_t^{\text{stoch}} = \Delta x_t^{\text{PMRC}} \cdot \exp\left(\lambda \sigma_x \epsilon_t - \frac{(\lambda \sigma_x)^2}{2}\right)$$

where $\epsilon_t \sim \mathcal{N}(0,1)$ and $\lambda \in [0,1]$ is a global scaling parameter. The mean correction term $-(\lambda \sigma_x)^2/2$ ensures that $\mathbb{E}[\text{multiplier}] = 1$, preserving the expected growth trajectory of the deterministic model.

---

## Disturbance Model

Catastrophic disturbance events (e.g., wind, ice, fire, pest outbreak) are modeled as a compound process with Bernoulli occurrence and Beta-distributed severity.

**Occurrence**: Each year, a disturbance occurs with probability $p_{\text{dist}} = 1/n$, where $n$ represents the mean return interval in years:

$$z_t \sim \text{Bernoulli}(p_{\text{dist}})$$

**Severity**: Conditional on occurrence, severity is drawn from a Beta distribution parameterized by mean $m_q$ and concentration $\kappa$:

$$q_t \mid z_t = 1 \sim \text{Beta}(m_q \kappa, (1-m_q)\kappa)$$

**Shock application**: Severity is applied as a proportional reduction to atomic state variables:

$$x_{t+1}^{\text{post}} = x_{t+1}^{\text{pre}} \cdot (1 - c_x \cdot q_t)$$

where $c_x$ is a sensitivity coefficient. We set $c_{\text{TPA}} = c_{\text{BA}} = 1$ and $c_{\text{HD}} = 0$, reflecting the assumption that disturbances reduce stocking but do not affect dominant height.

---

## Feasibility Projection

After each stochastic update, the state is projected into the feasible region to ensure biological and model consistency. Specifically, we enforce:

- Trees per acre: $n_t \geq 100$ (PMRC lower bound)
- Basal area: $b_t \geq 0$
- Dominant height: $h_t \geq h_{t-1}$ (non-decreasing)
- Age: $a_t > 0$

---

## Scenario-Based Uncertainty Treatment

Rather than estimating stochastic kernel parameters from data, we adopt a scenario-based approach to epistemic uncertainty. We define a structured grid of scenarios varying two key parameters:

- Process noise scaling: $\lambda \in \{0, 0.25, 0.5, 1.0\}$
- Disturbance probability: $p_{\text{dist}} \in \{0, 1/30, 1/20, 1/10\}$

This yields a 4×4 matrix of 16 scenarios, enabling systematic sensitivity analysis of stand value to different uncertainty regimes.

---

## Terminal Valuation

Terminal stand value is computed as the sum of product-specific revenues:

$$V_T = \sum_{k \in \{\text{pulp}, \text{cns}, \text{saw}\}} p_k \cdot y_{k,T}$$

where $p_k$ is the stumpage price and $y_{k,T}$ is the terminal yield in product class $k$. Product yields are estimated using a Weibull diameter distribution fitted to PMRC percentile predictions.

---

## Monte Carlo Simulation

For each stochastic scenario, we generate $N = 1000$ independent trajectories using forward simulation. The distribution of terminal values $\{V_T^{(i)}\}_{i=1}^N$ is summarized using standard statistics (mean, standard deviation, percentiles) and risk metrics (Value at Risk, Conditional Value at Risk).

---

## Limitations

Several simplifying assumptions constrain the scope of this analysis:

1. **Fixed management policy**: Thinning follows a deterministic basal area threshold rule; policy optimization is not considered.

2. **Single disturbance type**: All disturbances are modeled as a generic event; type-specific dynamics (e.g., fire spread, pest population) are not represented.

3. **Independence**: Disturbance occurrence is independent across years; temporal clustering is not modeled.

4. **Instantaneous effects**: Disturbance shocks are applied immediately; delayed recovery dynamics are not represented.

5. **Fixed prices**: Stumpage prices are treated as constant; price uncertainty is not modeled.

These limitations define clear directions for future model extensions.
