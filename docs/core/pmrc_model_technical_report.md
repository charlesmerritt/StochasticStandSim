# PMRC Loblolly Pine Growth and Yield Model
## Technical Documentation of the 1996 PMRC System and the Supplied Baseline Implementation

### Scope

This report documents the PMRC growth and yield system for **site-prepared loblolly pine plantations** in the Carolinas, Georgia, Alabama, and Florida, based primarily on the 1996 PMRC technical report, with cross-checks against the supplied R script and workbook. The goal is to explain:

- what type of model this is
- the state variables and inputs
- the governing equations
- how the submodels depend on one another
- how the system is used in forecasting, thinning, and fertilization scenarios
- where the supplied implementation aligns with, simplifies, or potentially departs from the published model

This is not just a transcription of formulas. It is an architectural description of the full modeling system.

---

## 1. Model family and modeling philosophy

The PMRC system is a **stand-level empirical growth and yield model** with optional **implicit stand structure recovery** and **stand-table projection** components.

At a high level, the system consists of three alternative simulation pathways that share the same whole-stand drivers:

1. **Whole-stand prediction/projection system**
2. **Weibull diameter distribution recovery system**
3. **Stand-table projection system**

All three are driven by whole-stand estimates of:

- dominant height (`HD`)
- trees per acre (`TPA`)
- basal area per acre (`BA`)

Those whole-stand quantities then feed downstream yield and merchandizing calculations.

The 1996 report explicitly frames the system as having six major components:

1. individual-tree volume, weight, and taper equations
2. whole-stand growth and yield equations
3. Weibull-based diameter distribution prediction
4. stand-table projection
5. thinning response
6. midrotation fertilization response

So the PMRC model is best understood as a **modular stand simulator**, not a single equation.

---

## 2. Geographic stratification and regionalization

The model uses three physiographic regions in the underlying report:

- **Lower Coastal Plain (LCP)**
- **Upper Coastal Plain (UCP)**
- **Piedmont**

However, many fitted components are parameterized as only **two regional systems**:

- **LCP**
- **Piedmont + Upper Coastal Plain**

The supplied R script encodes these as:

- `LCP`
- `PUCP`

where `PUCP` means the combined **Piedmont and Upper Coastal Plain** parameterization.

This is important because many implementation decisions must branch on region.

---

## 3. Core state variables and notation

The PMRC notation used in the report is:

- `A` = stand age
- `HD` = dominant height
- `TPA` = trees per acre
- `SI25` = site index, base age 25
- `BA` = basal area per acre
- `TVOB` = total volume per acre, outside bark
- `TVIB` = total volume per acre, inside bark
- `GWOB` = total green weight per acre, outside bark
- `DWIB` = total dry weight per acre, inside bark
- `Dq` = quadratic mean diameter
- `Dbh` = diameter at breast height
- `H` = total tree height

The supplied R code also uses:

- `qmd` as quadratic mean diameter
- `PHWD` as percent hardwood basal area relative to pine basal area
- `CI` as thinning competition index
- `RHD`, `RBA` as fertilizer response adjustments

---

## 4. Data basis and domain of intended use

The report states that the model was fit from PMRC permanent plot data collected over roughly 19 years, with measurements concentrated in ages 10 to 25 and more limited support at older ages and very high densities. The report also notes that excluded plots included those affected by fire, beetles, wind, harvest damage, thinning, plot-layout changes, and heavy wildling contamination.

Implication:

- This is an **empirical regional plantation model**
- It is intended for **site-prepared loblolly pine plantations**
- It is strongest inside the age-density-site-index ranges represented by the PMRC plot network
- Long extrapolations are possible, but they remain extrapolations

---

## 5. System architecture

### 5.1 Dependency graph

The practical dependency graph is:

1. **Input stand descriptors**
   - age
   - site index or dominant height
   - trees per acre
   - basal area if known
   - region
   - optional hardwood competition
   - optional thinning regime
   - optional fertilization regime

2. **Whole-stand drivers**
   - dominant height projection/prediction
   - survival / trees per acre projection
   - basal area prediction/projection

3. **Yield layer**
   - whole-stand total yield by unit
   - product allocation from total yield

4. **Implicit structure layer**
   - percentile prediction
   - Weibull parameter recovery
   - stand table
   - height-by-diameter assignment

5. **Treatment extensions**
   - thinning modifies `TPA`, `BA`, and subsequent `BA` trajectory through the competition index
   - fertilization adds response terms to `HD` and `BA`, which then indirectly propagate to yield

### 5.2 Operational consequence

Because the modules are interdependent, errors in early-state variables propagate:

- if `HD` is wrong, `BA` will be wrong
- if `TPA` is wrong, both `BA` and yield will be wrong
- if `BA` is wrong, yield and product allocation will be wrong
- if `qmd` is wrong, product breakdown will be distorted
- if thinning is handled inconsistently with the basal-area system, post-thin yield paths become internally inconsistent

---

## 6. Whole-stand model

This is the core of the supplied implementation.

### 6.1 Dominant height projection and site index system

The 1996 report uses a Chapman-Richards dominant-height model.

#### Projection form

\[
HD_2 = HD_1 \left(\frac{1 - e^{-0.014452 A_2}}{1 - e^{-0.014452 A_1}}\right)^{0.8216}
\]

This is implemented in the R script as:

```r
hdom_proj_fun <- function(hdom1, age1, age2){
  hdom2 <- hdom1 * ((1-exp(-0.014452*age2)) / (1-exp(-0.014452*age1)))**0.8216
  return(hdom2)
}
```

#### Height prediction from site index

\[
HD = SI_{25}\left(\frac{0.30323}{1 - e^{-0.014452 A}}\right)^{-0.8216}
\]

Implemented as:

```r
hdom_pred_fun <- function(si.25, age1){
  si.25 * (0.30323/(1-exp(-0.014452*age1)))**-0.8216
}
```

#### Interpretation

This does two jobs:

- **project** height forward from a known height
- **impute** height from site index when dominant height is not observed

In the supplied implementation, if `hd == 0`, the model fills height from `SI25` and age before continuing simulation.

### 6.2 Survival / trees per acre projection

The report uses a survival equation with a **hard lower asymptote of 100 TPA**:

\[
TPA_2 = 100 + \left[(TPA_1 - 100)^{-0.745339} + 0.0003425^2 \cdot SI_{25}(A_2^{1.97472} - A_1^{1.97472})\right]^{-1/0.745339}
\]

Implemented in the R script as:

```r
tpa_fun <- function(init_tpa, si, age1, age2){
  tpa2 <- 100 + ((init_tpa-100)**-0.745339 +
                 0.0003425**2 * si * (age2**1.97472-age1**1.97472))**(-1/0.745339)
  return(tpa2)
}
```

#### Interpretation

- Mortality increases with age
- Mortality also increases with site quality
- The asymptote prevents the model from driving TPA toward zero under long projections

#### Important caveat

The report explicitly warns that equation (15) should **not** be used when the initial density is already `<= 100 TPA`.

The supplied R script addresses this only in the **post-thinning low-density case**, where it replaces the PMRC survival model with an ad hoc decline:

```r
age_tpa <- ifelse(postthin_tpa>100,
                  tpa_fun(...),
                  (postthin_tpa*(0.99)^(age-thin_age)))
```

This is not in the 1996 report. It is an implementation safeguard.

### 6.3 Basal area prediction and projection

The PMRC system uses both a **state prediction equation** and a **state transition equation**.

#### Basal area prediction form

\[
\ln(BA) = b_0 + \frac{b_1}{A} + b_2 \ln(TPA) + b_3 \ln(HD) + b_4 \frac{\ln(TPA)}{A} + b_5 \frac{\ln(HD)}{A}
\]

#### Basal area projection form

\[
\ln(BA_2) = \ln(BA_1) + b_1\left(\frac{1}{A_2} - \frac{1}{A_1}\right)
+ b_2[\ln(TPA_2)-\ln(TPA_1)]
+ b_3[\ln(HD_2)-\ln(HD_1)]
+ b_4\left[\frac{\ln(TPA_2)}{A_2}-\frac{\ln(TPA_1)}{A_1}\right]
+ b_5\left[\frac{\ln(HD_2)}{A_2}-\frac{\ln(HD_1)}{A_1}\right]
\]

#### Regional coefficients

**Lower Coastal Plain**
- `b0 = 0.0`
- `b1 = -42.689283`
- `b2 = 0.367244`
- `b3 = 0.659985`
- `b4 = 2.012724`
- `b5 = 7.703502`

**Piedmont + Upper Coastal Plain**
- `b0 = -0.855557`
- `b1 = -36.050347`
- `b2 = 0.299071`
- `b3 = 0.980246`
- `b4 = 3.309212`
- `b5 = 3.787258`

Implemented directly in the script via `ba_pred_fun()` and `ba_proj_fun()`.

#### Interpretation

This is the most important density-growth bridge in the system:

- `TPA` captures stocking
- `HD` captures site and developmental stage
- `BA` becomes the key carry-forward state for yield prediction

#### Why prediction and projection both exist

- use **prediction** when only current age, TPA, and dominant height are known
- use **projection** when a current `BA` estimate already exists and should be propagated consistently

The supplied implementation does exactly this:
- if `ba == 0`, it imputes basal area with `ba_pred_fun`
- then it projects forward annually with `ba_proj_fun`

### 6.4 Piedmont hardwood competition extension

For Piedmont stands, the report provides an alternate basal-area prediction equation including hardwood pressure:

\[
\ln(BA) =
-0.904066
-\frac{33.811815}{A}
+0.321301\ln(TPA)
+0.985342\ln(HD)
+3.381071\frac{\ln(TPA)}{A}
+2.548207\frac{\ln(HD)}{A}
-0.003689 \cdot PHWD
\]

The supplied script includes:

```r
ba_hdwd_piedmont <- function(hdom, tpa, phwd, age){
  ba <- exp(-0.904066 + (-33.811815/age) + 0.321301*log(tpa) +
            0.985342*log(hdom) + 3.381071*(log(tpa)/age) +
            2.548207*(log(hdom)/age) - 0.003689*phwd)
}
```

This component is **present but not central** in the main baseline projection loop.

---

## 7. Whole-stand yield models

The PMRC system predicts whole-stand totals in four units:

- `TVOB`
- `TVIB`
- `GWOB`
- `DWIB`

### 7.1 Piedmont + Upper Coastal Plain yield form

\[
\ln(Y) = b_0 + b_1\ln(HD) + b_2\ln(BA) + b_3\frac{\ln(TPA)}{A}
+ b_4\frac{\ln(HD)}{A} + b_5\frac{\ln(BA)}{A}
\]

### 7.2 Lower Coastal Plain yield form

\[
\ln(Y) = b_0 + b_1\ln(TPA) + b_2\ln(HD) + b_3\ln(BA)
+ b_4\frac{\ln(TPA)}{A} + b_5\frac{\ln(BA)}{A}
\]

### 7.3 Coefficients used in the baseline script

#### PUCP

- `TVOB`: `(0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411)`
- `TVIB`: `(0.0, 0.350394, 1.263708, -8.608165, 7.193937, 6.309586)`
- `GWOB`: `(-3.818016, 0.430179, 1.276768, -8.088792, 7.428472, 5.554509)`
- `DWIB`: `(-4.987560, 0.446433, 1.348843, -7.757842, 7.857337, 4.222016)`

#### LCP

- `TVOB`: `(-1.520877, 0.200680, 1.207586, 0.703405, -5.139064, 6.744164)`
- `TVIB`: `(-2.088857, 0.177587, 1.303770, 0.726950, -5.091474, 6.676532)`
- `GWOB`: `(-5.175922, 0.198424, 1.232028, 0.705769, -5.129853, 6.731477)`
- `DWIB`: `(-6.332502, 0.145815, 1.296629, 0.814967, -4.660198, 5.383589)`

These are encoded in `yield_pred_fun()`.

### 7.4 Interpretation

Yield is not projected with a separate transition equation. Instead it is **predicted as a function of the current whole-stand state**:

- age
- TPA
- HD
- BA
- region
- requested yield unit

So the PMRC system is state-based:

1. project the stand state
2. predict yield from the projected state

This means yield inherits all assumptions and errors from the upstream whole-stand models.

---

## 8. Product allocation / merchantable yield breakdown

The model then partitions total whole-stand yield into merchantable components using top-diameter and DBH threshold rules.

### 8.1 Functional form

The report gives:

\[
Y_m = Y \exp\left( b_1 (t/D_q)^{b_2} + b_3 TPA^{b_4}(d/D_q)^{b_5} \right)
\]

where:

- `Ym` = merchantable per-acre yield for trees at or above DBH threshold `d`
- `Y` = total stand yield
- `t` = merchantable top diameter
- `d` = DBH threshold
- `Dq` = quadratic mean diameter

### 8.2 Product definitions

The report figures use product rules approximately like:

- pulpwood: trees `> 4.5"` DBH to a `2"` top
- chip-n-saw: trees `8.5"` to `11.5"` DBH to a `4"` top
- sawtimber: trees `> 11.5"` DBH to an `8"` top

The workbook `specs` sheet contains a related but not identical product table:

- pulp: `top = 3`, `dbh = 4.6`
- cns: `top = 6`, `dbh = 8.6`
- saw: `top = 8`, `dbh = 12.6`

So the workbook appears to reflect a **customized merchandizing setup**, not a verbatim copy of the figure captions in the 1996 report.

### 8.3 Implementation formula in script

The R script implements the product allocation using `product_function()` with separate coefficients by unit and region. It also computes `qmd` as:

\[
D_q = \sqrt{\frac{BA/TPA}{0.005454154}}
\]

which is the standard English-unit conversion from basal area and stems per acre.

### 8.4 Important implementation concern

The supplied text file contains what appears to be a typo in the LCP `DWIB` product breakdown exponent:

```r
(top.dia/qmd)**4054202
```

This should almost certainly be:

```r
(top.dia/qmd)**4.054202
```

matching the reported Table 20 coefficient.

This should be treated as a transcription or OCR corruption, not as an intended model specification.

### 8.5 Structural limitation

Because this function uses whole-stand `Dq` as the reference diameter, it is a **ratio-based stand-level merchandizing approximation**, not a full tree-level merchandising routine. It is fast and convenient, but it does not explicitly integrate over the entire stand table unless the user separately invokes the Weibull or stand-table layer.

---

## 9. Weibull diameter distribution recovery layer

This is the first of the system's implicit stand-structure modules.

### 9.1 Motivation

The whole-stand layer only gives aggregates. To obtain a stand table or a stock table, the model recovers a diameter distribution using predicted percentiles and a Weibull distribution.

### 9.2 Percentile model

The report predicts diameter percentiles using:

\[
\ln(P_x) = a_0 + a_1 \ln\left(\frac{BA}{TPA}\right)
\]

for percentiles:

- `P0`
- `P25`
- `P50`
- `P95`

Regional coefficients are provided in the report and determine the recovered Weibull parameters.

### 9.3 Interpretation

This is a **parameter-recovery approach** rather than direct parameter regression.

That means:

1. predict selected diameter percentiles from stand state
2. infer Weibull parameters from those percentiles
3. reconstruct a diameter distribution whose implied `Dq` is compatible with whole-stand `BA` and `TPA`

This compatibility is one of the most important design features of the PMRC system. The structural recovery is intended to remain consistent with whole-stand density.

### 9.4 Hardwood extension in Piedmont

The report also gives Piedmont-specific hardwood-adjusted percentile equations:

- `P0`: no hardwood term
- `P25`: includes negative hardwood effect
- `P50`: no hardwood term
- `P95`: includes positive hardwood effect

The practical effect reported is increasing positive skew in the pine diameter distribution with increasing hardwood competition.

### 9.5 What the supplied baseline does

The supplied R script **does not explicitly implement the Weibull recovery layer** in the visible sections provided. Its main operational path is the whole-stand simulator plus thinning/fertilization logic.

So this layer is part of the **underlying PMRC system**, but not the primary pathway in the supplied baseline code.

---

## 10. Stand-table projection model

This is the second implicit stand-structure pathway.

### 10.1 Relative size projection equation

The report uses the Pienaar and Harrison relative-size model:

\[
b_{2i} = b_2 \left(\frac{b_{1i}}{b_1}\right)^{\left(\frac{A_2}{A_1}\right)^{\beta}}
\]

where:

- `b1` = average tree basal area at time 1
- `b2` = average tree basal area at time 2
- `b1i` = basal area of tree or class `i` at time 1
- `b2i` = basal area of tree or class `i` at time 2
- `β` = fitted parameter

Regional `β` values in the report:

- `PUCP`: `-0.2277`
- `LCP`: `-0.0525`

### 10.2 Interpretation

This model projects individual-tree or diameter-class basal areas as a function of their size relative to the stand mean.

The report notes an interesting property:

- estimated `β` values are negative
- therefore, relative sizes tend to move somewhat toward the mean over time

That is not perfectly aligned with one of the original conceptual assumptions, but the report says the projected stand tables still behave well over short to medium projection intervals.

### 10.3 Practical use

Use this pathway when an actual stand table already exists, either from:

- inventory
- a recovered Weibull diameter distribution

Then:

1. project the whole-stand driver state
2. project relative class sizes
3. rebuild the stand table consistently

Again, this layer is described in the report but is not the main path in the provided R baseline.

---

## 11. Height-by-diameter function

To convert a stand table into a stock table, the report uses a class-level height equation:

\[
H_i = HD \cdot a_1 \left(1 - a_2 e^{-a_3 \cdot DBH_i/D_q}\right)
\]

The OCR of the exact typography is imperfect, but the main structure is clear:

- class height is anchored to dominant height
- class height depends on class DBH relative to `Dq`
- coefficients vary by region

The report gives regional parameter estimates for the combined PUCP region and for LCP.

This is needed if the user wants:

- merchantable volume by class
- stock tables
- height-sensitive merchandizing

The provided baseline script does not visibly use this function in the main forecasting loop.

---

## 12. Thinning extension

The supplied baseline uses thinning heavily, and this part is very important.

### 12.1 Thinned basal area removed

The report gives a thinning-removal equation:

\[
\frac{BA_t}{BA} =
\frac{TPA_r}{TPA}
+
\left(1-\frac{TPA_r}{TPA}\right)
\left(\frac{TPA_s}{TPA - TPA_r}\right)^{1.2345}
\]

where:

- `BAt` = basal area removed
- `BA` = pre-thin basal area
- `TPAr` = TPA removed by row thinning
- `TPAs` = TPA removed by selective thinning

The script includes a corresponding function:

```r
thin_removed_fun <- function(ba_init, tpa_before, tpa_row_remove, tpa_select_remove){
  ba_init*(tpa.row.remove/tpa_before) +
    (1-(tpa_row_remove/tpa_before)) *
    (tpa_select_remove/(tpa_before-tpa_row_remove))**1.2345
}
```

However, as written, this function appears to contain a naming typo:
- `tpa.row.remove` should probably be `tpa_row_remove`

The main thinning logic in the script does not appear to rely directly on this function. Instead it manually computes row-thin and selective-thin removals.

### 12.2 Competition index

The report defines:

\[
CI = 1 - \frac{BA_{thinned}}{BA_{unthinned}}
\]

Implemented as:

```r
competition_index <- function(ba_thinned, ba_unthinned){
  1 - (ba_thinned/ba_unthinned)
}
```

### 12.3 Competition index decay

The report then projects thinning response via:

\[
CI_2 = CI_1 e^{-\beta(A_2 - A_1)}
\]

Regional decay parameters:

- `PUCP`: `β = 0.076472`
- `LCP`: `β = 0.110521`

Implemented as:

```r
projected_ci <- function(ci_init, age1, age2, region){
  ifelse(region=="PUCP", ci_init*exp(-0.076472*(age2-age1)),
         ifelse(region=="LCP", ci_init*exp(-0.110521*(age2-age1)), 0))
}
```

### 12.4 Recovering thinned basal area trajectory

Then:

\[
BA_{thinned,2} = BA_{unthinned,2}(1-CI_2)
\]

Implemented as:

```r
ba_thinned_projected <- function(ba_unthinned, ci_projected){
  ba_unthinned*(1-ci_projected)
}
```

### 12.5 How the supplied baseline applies thinning

The R script uses the following operational workflow:

1. project the unthinned stand to the thinning age
2. compute pre-thin `TPA`, `BA`, and `HD`
3. set target residual basal area from the regime table
4. assume a fixed **fourth-row thin** removing 25% of stems
5. assign the remainder of the basal area removal to selective thinning
6. back-solve selective TPA removal with `tpa_select_remove()`
7. compute post-thin `TPA` and `BA`
8. compute an unthinned counterpart basal area using `ba_pred_fun(postthin_tpa, prethin_hd, thin_age, region)`
9. compute initial competition index
10. project annual post-thin basal area using the CI decay model

### 12.6 Important implementation note

The report describes a general thinning-response framework. The supplied R implementation adds a **specific operational assumption**:

- every thinning includes a **25% row thin** before selective removal

That is not a universal PMRC assumption. It is a modeling choice in the baseline script.

### 12.7 Another important note

The script uses residual basal area targets from the workbook. So the workbook is not just input data, it encodes a management-policy layer on top of the PMRC transition system.

---

## 13. Midrotation fertilization extension

This is the other major treatment extension.

### 13.1 Dominant height response

The report gives the fertilizer adjustment:

\[
RHD = (0.00106N + 0.2506P_Z)Y_t e^{-0.1096Y_t}
\]

where:

- `N` = pounds of elemental nitrogen per acre
- `PZ` = indicator for phosphorus treatment
- `Yt` = years since treatment

Implemented as:

```r
hd_response_fert <- function(N, P, Yst){
  r_hd <- (0.00106*N + 0.2506*P) * Yst * exp(-0.1096*Yst)
  return(r_hd)
}
```

Note that in the script, `P` functions like the indicator `PZ`, not a pounds-of-P quantity.

### 13.2 Basal area response

\[
RBA = (0.0121N + 1.3639P_Z)Y_t e^{-0.2635Y_t}
\]

Implemented as:

```r
ba_response_fert <- function(N, P, Yst){
  r_ba <- (0.0121*N + 1.3639*P) * Yst * exp(-0.2635*Yst)
  return(r_ba)
}
```

### 13.3 How fertilization enters yield

The report explicitly says **no separate yield adjustment is needed**. Fertilization changes yield by changing the upstream state variables:

- dominant height
- basal area

Then the usual yield equations use the adjusted state.

### 13.4 Workbook treatment encoding

The workbook regime sheets include:

- `fert1_age`, `fert2_age`, `fert3_age`
- `fert1_N`, `fert2_N`, `fert3_N`
- `fert1_P`, `fert2_P`, `fert3_P`

So the workbook is designed for multiple fertilization events, even though the report mainly documents a midrotation response form.

### 13.5 Modeling warning

The report explicitly warns that thinning and fertilization models were developed from **independent datasets**, and interactions are not empirically validated. The workbook and script nonetheless allow mixed regimes. That is a legitimate scenario engine, but it should be documented as **beyond the strict empirical support of the original PMRC report**.

---

## 14. Workbook structure and role in the implementation

The supplied workbook is not just an auxiliary file. It encodes a scenario-generation interface.

### 14.1 Key sheets

#### `stand`
Contains initial stand conditions:

- stand id
- age
- basal area
- trees per acre
- dominant height
- site index
- region

If `ba` or `hd` are zero, the script imputes them from the PMRC equations.

#### `regimes`
Contains management and fertilization schedules:

- thinning ages
- residual basal area targets
- fertilization ages
- N and P treatments

#### `specs`
Contains product rules and stumpage-like fields:

- species
- product
- top diameter
- DBH threshold
- piece length
- stumpage

### 14.2 Implementation meaning

The PMRC equations define the biological and empirical stand-dynamics kernel.  
The workbook defines the **scenario policy layer**.

This separation is valuable and should be preserved in future refactors:
- stand state inputs should remain separate from
- regime or management policy inputs

---

## 15. Annual simulation workflow in the supplied baseline

The main baseline simulator operates roughly as follows.

### 15.1 Unthinned stands

For each stand-regime combination:

1. read initial stand state
2. if `HD` is missing, predict it from `SI25`
3. if `BA` is missing, predict it from age, `TPA`, `HD`, region
4. for each future year:
   - project dominant height
   - project trees per acre
   - project basal area
5. predict yields from state
6. optionally allocate total yield into products

### 15.2 Thinned stands

For stands with scheduled thinning:

1. project forward to first thinning age
2. compute pre-thin state
3. impose target post-thin residual basal area
4. split removal into row-thin plus selective-thin components
5. derive post-thin `TPA`
6. compute post-thin competition index relative to unthinned counterpart
7. for each future year:
   - project `HD`
   - project post-thin `TPA`
   - project unthinned counterpart `BA`
   - decay competition index
   - compute thinned `BA` as adjusted counterpart `BA`
8. continue yield prediction on the adjusted state trajectory

### 15.3 Fertilized stands

The response terms are additive. Operationally the logic is:

1. compute base PMRC state trajectory
2. for years after treatment:
   - add `RHD`
   - add `RBA`
3. feed adjusted state into the yield equation

The visible code excerpt defines the response functions, but the full downstream use is not shown in the snippet provided. Still, the intended use is clear from the report and the workbook.

---

## 16. Interdependencies and model consistency

### 16.1 Strong dependencies

The PMRC system is tightly coupled:

- `SI25` or initial `HD` determines height trajectory
- height trajectory influences basal area trajectory
- survival trajectory influences both `TPA` and `Dq`
- `BA` and `TPA` together determine `Dq`
- `HD`, `BA`, `TPA`, and age determine total yield
- `Dq`, `TPA`, and merchandizing thresholds determine product yield

### 16.2 Consistency principle

The whole reason the report includes both whole-stand and implicit stand-structure modules is to preserve **consistency** between:

- stand totals
- diameter distributions
- stand tables
- merchandized output

That is why percentile recovery is driven by `BA/TPA`, and why stand-table projection is anchored to average basal area.

### 16.3 Practical implication for reimplementation

A faithful reimplementation should treat the model as a **state machine with compatible submodels**, not as a bag of independent equations.

---

## 17. Known implementation issues and ambiguities in the supplied baseline

This section is especially important for future engineering work.

### 17.1 OCR and transcription issues

The plain-text script appears to contain at least one obvious corruption:

- LCP `DWIB` product exponent `4.054202` appears as `4054202`

This should be fixed before relying on product allocation output.

### 17.2 Possible sign loss in table OCR

Some coefficient tables in OCR text lose minus signs, especially in intercept terms. The R script appears to preserve the intended signs more faithfully for several equations. When the PDF table OCR and script disagree, the script often matches the logical pattern better.

### 17.3 Typo in `thin_removed_fun`

The function body uses `tpa.row.remove` rather than `tpa_row_remove`. As written, that function would fail if called directly.

### 17.4 Low-density workaround is ad hoc

The 1996 report only says equation (15) should not be used at `TPA <= 100`.  
The script substitutes a 1% annual decay after post-thin densities below 100 TPA. That is a pragmatic hack, not a PMRC-published equation.

### 17.5 Thinning policy is coupled to implementation assumptions

The script hardcodes:
- 25% row thinning
- then selective removal to hit target residual BA

That should be documented as a regime assumption, not a universal property of the PMRC thinning equations.

### 17.6 Mixed thinning plus fertilization scenarios exceed strict empirical support

The report explicitly warns against assuming validated treatment interactions. The workbook and script support such combinations anyway. This is acceptable as a scenario device, but it should be labeled exploratory.

---

## 18. Recommended mental model for users and developers

The best way to think about the PMRC system is:

### Level 1: whole-stand state engine
This is the core:
- `HD`
- `TPA`
- `BA`

### Level 2: yield translator
Maps state to:
- volume
- weight
- product classes

### Level 3: structure recovery
Maps state to:
- percentiles
- Weibull distribution
- stand table
- stock table

### Level 4: treatment modifiers
Adjusts the state engine for:
- thinning
- fertilization
- optionally hardwood competition in Piedmont

This layered view makes the model easier to refactor, validate, and extend.

---

## 19. Suggested refactor boundaries for a modern implementation

A clean reimplementation would separate the system into modules like:

### `site_height.py`
- `hdom_pred`
- `hdom_proj`
- `site_index_from_height` if needed

### `survival.py`
- `tpa_proj`
- guardrails for `TPA <= 100`

### `basal_area.py`
- `ba_pred`
- `ba_proj`
- `ba_pred_piedmont_hardwood`

### `yield.py`
- `yield_pred`
- units enum
- region enum

### `products.py`
- `qmd_from_ba_tpa`
- `product_breakdown`
- product-rule definitions from config

### `weibull.py`
- percentile prediction
- parameter recovery
- class allocation

### `stand_table.py`
- relative-size projection
- height-by-diameter

### `thinning.py`
- basal-area removal
- post-thin state reconstruction
- competition index transition

### `fertilization.py`
- `hd_response`
- `ba_response`
- schedule application

### `scenario.py`
- stand inputs
- treatment schedule inputs
- simulation loop orchestration

This decomposition matches the mathematics much better than one monolithic script.

---

## 20. Summary

The PMRC system is a mature empirical plantation model with a modular architecture:

- **height** is projected from a Chapman-Richards dominant height function
- **survival** is projected with an age-site-density equation with a 100 TPA asymptote
- **basal area** is predicted or projected from age, height, and stocking
- **yield** is predicted from the whole-stand state
- **merchantable products** are allocated from total yield using top-diameter and DBH-threshold ratio functions
- **diameter distributions** can be recovered through Weibull percentile prediction
- **stand tables** can be projected through a relative-size model
- **thinning** is modeled through residual basal area and a competition-index recovery framework
- **fertilization** is modeled as additive response terms on dominant height and basal area

The supplied R script implements the **whole-stand core**, **yield layer**, **product allocation**, **thinning extension**, and **fertilization response functions**, while the workbook acts as a scenario and treatment-policy interface. The PDF remains the authoritative source for the mathematical structure, while the script is the best guide to practical implementation details and modern usage.

---

## 21. Source mapping used for this report

Primary sources consulted:

- `TR1996-1 Embedded Fonts.pdf`
- `GY_System_20241120.txt`
- `model_input_fert.xlsx`

### What each source contributed

#### PDF technical report
Authoritative source for:
- model architecture
- published equations
- regional parameterization
- thinning and fertilization theory
- intended scope and caveats

#### R script
Best source for:
- implementation details
- exact branching logic
- annual simulation workflow
- workbook integration
- practical defaults and safeguards

#### Workbook
Best source for:
- expected input schema
- regime encoding
- product specification structure
- how the model is being operationalized in scenario analysis
