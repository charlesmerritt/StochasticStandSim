# Model Specification

Formal mathematical specification for the stochastic stand simulator.

## 1. State Space

### 1.1 Atomic State Variables

| Variable | Symbol | Type | Domain | Description |
|----------|--------|------|--------|-------------|
| Age | $a_t$ | Dynamic | $\mathbb{R}^+$ | Stand age (years) |
| Dominant Height | $h_t$ | Dynamic | $\mathbb{R}^+$ | Dominant height (ft) |
| Trees per Acre | $n_t$ | Dynamic | $[100, \infty)$ | Stems per acre |
| Basal Area | $b_t$ | Dynamic | $\mathbb{R}^+$ | Basal area (ft²/ac) |
| Site Index | $S$ | Constant | $\mathbb{R}^+$ | Site index at base age 25 |
| Region | $r$ | Constant | {ucp, pucp, lcp} | PMRC coefficient region |
| Hardwood | $\phi$ | Constant | $[0, 100]$ | Percent hardwood |

State vector: $s_t = (a_t, h_t, n_t, b_t, S, r, \phi)$

### 1.2 Derived Quantities

Computed on-demand from atomics:
- QMD: $\bar{d} = \sqrt{b / (n \cdot 0.005454)}$
- Volume: $V = f_{\text{PMRC}}(a, n, h, b, r)$
- Products: $(y_{\text{pulp}}, y_{\text{cns}}, y_{\text{saw}}) = g_{\text{Weibull}}(b, n, h, r, \phi)$

## 2. Transition Model

### 2.1 Deterministic PMRC Backbone

For time step $\Delta t = 1$ year:

$$\mu_{t+1} = f_{\text{PMRC}}(s_t, a_t)$$

Projection order:
1. $a_{t+1} = a_t + \Delta t$
2. $h_{t+1}^{\text{det}} = f_h(a_t, h_t, a_{t+1})$ — Chapman-Richards
3. $n_{t+1}^{\text{det}} = f_n(n_t, S, a_t, a_{t+1})$ — Survival equation
4. $b_{t+1}^{\text{det}} = f_b(a_t, n_t, n_{t+1}, b_t, h_t, h_{t+1}, a_{t+1}, r)$

### 2.2 Process Noise

Multiplicative lognormal noise on increments:

$$\Delta x_t^{\text{stoch}} = \Delta x_t^{\text{det}} \cdot \exp\left(\lambda \sigma_x \epsilon_t - \frac{1}{2}(\lambda \sigma_x)^2\right)$$

where $\epsilon_t \sim \mathcal{N}(0, 1)$ and $\lambda = \lambda_{\text{proc}}$ is the global scaling factor.

Applied to:
- BA increment: $\sigma_b = 0.14$
- HD increment: $\sigma_h = 0$ (optional)

TPA mortality (binomial):
$$D_t \sim \text{Binomial}(n_t, p_{\text{die}})$$
where $p_{\text{die}} = \mathbb{E}[\text{mortality}] / n_t$

Recruitment:
$$R_t \sim \text{Poisson}(\lambda_R)$$
where $\lambda_R = \max(0, \alpha_0 + \alpha_1 b_t + \alpha_2 S)$

### 2.3 Disturbance Process

Occurrence:
$$z_t \sim \text{Bernoulli}(p_{\text{dist}})$$

where $p_{\text{dist}} = 1/n$ for $n$-year mean return interval.

Severity (conditional on $z_t = 1$):
$$q_t \mid z_t = 1 \sim \text{Beta}(\alpha, \beta)$$

with $\alpha = m_q \kappa$ and $\beta = (1 - m_q) \kappa$.

Default parameters: $m_q = 0.30$, $\kappa = 12$.

### 2.4 Shock Application

Proportional reduction to atomics:
$$x_{t+1}^{\text{post}} = x_{t+1}^{\text{pre}} \cdot (1 - c_x \cdot q_t)$$

Sensitivity coefficients:
| Variable | $c_x$ |
|----------|-------|
| TPA | 1.0 |
| BA | 1.0 |
| HD | 0.0 |

### 2.5 Feasibility Projection

$$s_{t+1} = g(\tilde{s}_{t+1})$$

where $g$ enforces:
- $n \geq 100$
- $b \geq 0$
- $h \geq h_t$ (non-decreasing)
- $a > 0$

## 3. Valuation

### 3.1 Terminal Value

$$V_T = \sum_{k \in \{\text{pulp}, \text{cns}, \text{saw}\}} p_k \cdot y_{k,T}$$

where $p_k$ is stumpage price and $y_{k,T}$ is terminal yield in product class $k$.

### 3.2 NPV

$$\text{NPV} = -C_0 + \frac{R_{\text{thin}}}{(1+r)^{t_{\text{thin}}}} + \frac{R_{\text{harvest}}}{(1+r)^T}$$

### 3.3 LEV

$$\text{LEV} = \text{NPV} \cdot \frac{(1+r)^T}{(1+r)^T - 1}$$

## 4. Scenario Parameters

### 4.1 Process Noise

| Parameter | Symbol | Values |
|-----------|--------|--------|
| Noise scale | $\lambda_{\text{proc}}$ | {0, 0.25, 0.5, 1.0} |

### 4.2 Disturbance

| Parameter | Symbol | Values |
|-----------|--------|--------|
| Probability | $p_{\text{dist}}$ | {0, 1/30, 1/20, 1/10} |
| Severity mean | $m_q$ | 0.30 (fixed) |
| Concentration | $\kappa$ | 12 (fixed) |

### 4.3 Full Matrix

16 scenarios: $\lambda_{\text{proc}} \times p_{\text{dist}}$ = 4 × 4
