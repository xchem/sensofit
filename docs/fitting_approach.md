# Fitting Approach — Detailed Documentation

## Overview

This library fits 1:1 Langmuir binding kinetics to GCI sensorgrams from
Creoptix WAVE instruments. The pipeline has two stages:

1. **Direct Kinetics (DK)** — closed-form linear solver for initial estimates
2. **ODE Refinement** — nonlinear least-squares over the full sensorgram

The two-stage approach combines the speed and determinism of DK with the
accuracy of a full ODE model that properly handles pulsed injection.

---

## 1. The 1:1 Langmuir Binding Model

The binding interaction is described by a single ODE:

$$
\frac{dR}{dt} = k_a \cdot c(t) \cdot (R_{max} - R) - k_d \cdot R
$$

| Symbol     | Meaning                              | Units       |
|------------|--------------------------------------|-------------|
| $R(t)$     | Binding response                     | pg/mm²      |
| $c(t)$     | Analyte concentration at surface     | M           |
| $k_a$      | Association rate constant            | M⁻¹ s⁻¹    |
| $k_d$      | Dissociation rate constant           | s⁻¹         |
| $R_{max}$  | Maximum binding capacity             | pg/mm²      |
| $K_D$      | Equilibrium dissociation constant    | M           |

At equilibrium: $K_D = k_d / k_a$

---

## 2. Signal Preprocessing

### 2a. Reference Subtraction

The raw .cxw data contains signals from multiple flow cells. The active
channel (with immobilised ligand, e.g. FC2) is subtracted by the reference
channel (unmodified surface, e.g. FC1) during loading:

$$
\text{signal} = \text{raw\_active} - \text{raw\_reference}
$$

This removes bulk refractive index (RI) changes common to both surfaces.

### 2b. Double Referencing

A blank cycle (buffer-only injection, same flow path) is subtracted from
the sample signal to remove systematic drift:

$$
\text{signal\_dr} = (\text{signal\_sample} - \text{baseline\_sample}) - (\text{signal\_blank} - \text{baseline\_blank})
$$

The nearest *preceding* blank is preferred. If subtraction yields negative
peak response (blank overcorrection), subsequent preceding blanks are tried.

**Implementation:** `models.double_reference(sample, blanks)`

### 2c. Non-Specific Binder Detection

Before fitting, the reference channel dissociation signal is checked.
Compounds retained on the unmodified reference surface (> 2 pg/mm² at
2–5 s after rinse onset) are flagged as non-specific binders and excluded
from kinetic fitting.

**Implementation:** `models.is_nonspecific_binder(sample)`

---

## 3. Concentration Profile c(t)

Creoptix GCI uses **pulsed injection**: the analyte alternates with buffer
in rapid sub-pulses (~3.5 s period). This creates a modulated c(t) at the
sensor surface.

Two concentration profiles are constructed from the DMSO calibration cycle:

### 3a. Envelope c(t) — for Direct Kinetics

A rolling maximum extracts the upper envelope of the DMSO reference signal
(which directly measures RI ∝ concentration). This smooth profile
represents the "effective" analyte concentration ignoring pulse structure.

```
┌─────────────────────────────────────────────────┐
│  Envelope c(t)                                  │
│      ╭──────────────────╮                       │
│     ╱                    ╲                      │
│────╯                      ╲────────────         │
│  baseline    injection      rinse/dissociation  │
└─────────────────────────────────────────────────┘
```

**Implementation:** `models.build_concentration_profile(dmso, C_analyte)`

### 3b. Pulsed c(t) — for ODE Fitting

The raw DMSO reference signal (baseline-subtracted, normalized) preserves
the pulse structure. During analyte pulses, c(t) is high; during buffer
pulses, c(t) drops toward zero.

```
┌─────────────────────────────────────────────────┐
│  Pulsed c(t)                                    │
│      ┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐                       │
│      │└┘└┘└┘└┘└┘└┘└┘└┘│                        │
│──────┘                  └───────────            │
│  baseline    injection      rinse/dissociation  │
└─────────────────────────────────────────────────┘
```

**Implementation:** `models.build_pulsed_concentration_profile(dmso, C_analyte)`

---

## 4. Weight Masks

Not all time points contribute equally to accurate fitting.

### 4a. Dissociation-Only Mask (for DK)

$w = 1$ from Rinse to RinseEnd; $w = 0$ elsewhere.

Used by DK to extract $k_d$ from the clean dissociation phase where $c(t) = 0$
and the ODE reduces to $R(t) = R_0 \cdot e^{-k_d t}$.

**Implementation:** `models.build_weight_mask(t, markers)`

### 4b. Full Weight Mask (for ODE)

$w = 1$ during buffer pulses in the association phase AND during dissociation.
$w = 0$ during analyte pulses (RI bulk artefacts) and baseline.

```
┌─────────────────────────────────────────────────────┐
│  Weight mask                                        │
│         ╷ ╷ ╷ ╷ ╷ ╷ ╷ ╷ ╷ ╷┌──────────────┐         │
│  w=1    │ │ │ │ │ │ │ │ │ ││              │         │
│                            │   dissoc     │         │
│  w=0 ───┘ └ └ └ └ └ └ └ └ └┘              └───      │
│  baseline  buffer pulses                            │
└─────────────────────────────────────────────────────┘
```

The pulse timing is detected from the DMSO calibration reference signal
using a threshold-based classifier (`models.build_pulse_mask`).

**Implementation:** `models.build_full_weight_mask(t, markers, dmso)`

---

## 5. Stage 1: Direct Kinetics (DK)

### Principle

Reparameterize the Langmuir ODE so it becomes *linear* in intermediate
parameters $\mathbf{k} = (k_1, k_2, k_3) = (k_a R_{max},\; k_a,\; k_d)$:

$$
\frac{dR}{dt} = k_1 \cdot c(t) - k_2 \cdot c(t) \cdot R(t) - k_3 \cdot R(t)
$$

Rearranging:

$$
\frac{dR}{dt} = \begin{bmatrix} -c & c R & R \end{bmatrix} \cdot \begin{bmatrix} k_1 \\ k_2 \\ k_3 \end{bmatrix}
$$

$$
\Rightarrow \mathbf{X} \mathbf{k} + \mathbf{b} = 0
$$

where $\mathbf{b} = dR/dt$ and each row of $\mathbf{X}$ is $[-c_i,\; c_i R_i,\; R_i]$.

### Solution

With optional Tikhonov regularisation $\Lambda$:

$$
\mathbf{k} = -(\mathbf{X}^T \mathbf{X} + \Lambda)^{-1} \mathbf{X}^T \mathbf{b}
$$

This is solved in a single matrix operation — no iteration needed.

### What DK Provides

| Parameter | Source | Reliability |
|-----------|--------|-------------|
| $k_d$     | $k_3$ from linear solver (dissociation-only fit) | **High** — dissociation is clean, c=0 |
| $k_a$     | Estimated from $k_{obs}$ or steady-state heuristic | Moderate — initial estimate only |
| $R_{max}$ | Derived from $k_a$, $k_d$, $R_0$, $c$ | Moderate — coupled to $k_a$ estimate |

### ka Estimation Strategy

Two approaches, tried in order:

1. **kobs method**: From observed association kinetics,
   $R(t) \approx R_{eq}(1 - e^{-k_{obs} t})$ where $k_{obs} = k_a c + k_d$.
   Find $t_{1/2}$ (time to reach 50% of $R_0$), compute
   $k_{obs} = \ln(2) / t_{1/2}$, then $k_a = (k_{obs} - k_d) / c$.

2. **Steady-state fallback**: At dissociation onset, estimate saturation
   fraction $f_{sat}$ from the derivative $dR/dt$ near rinse:
   $\eta = \frac{dR/dt}{k_d \cdot R_0}$, then $f_{sat} = 1/(1+\eta)$.
   Use $R_{max} = R_0 / f_{sat}$ and $k_a = k_d R_0 / (c \cdot (R_{max} - R_0))$.

**Implementation:** `direct_kinetics.fit_sample(sample, dmso_cals, blanks)`

---

## 6. Stage 2: ODE Refinement

### Three-Phase Architecture

#### Phase 1: Closed-Form Linear Regression for (R₀, Rss)

With $k_d$ fixed from DK, the dissociation-only model:

$$
R(t) = R_0 \cdot e^{-k_d (t - t_0)} + R_{ss}
$$

is *linear* in $(R_0, R_{ss})$. Solve via ordinary least squares on the
dissociation window (Rinse + 1s → RinseEnd), skipping the first second
to avoid transport lag artefacts.

#### Phase 2: Derive ka from Steady-State

At dissociation onset ($t = t_{rinse}$), the system is near steady state:

$$
k_a = \frac{k_d \cdot R_0}{c(t_{rinse}) \cdot (R_{max} - R_0)}
$$

This provides a physics-informed starting point for optimisation.

#### Phase 3: Multi-Start TRF Optimisation

Optimise $(k_a, R_{max})$ while holding $k_d$ fixed. The cost function is:

$$
\min_{k_a, R_{max}} \sum_i w_i \cdot \left( R_i^{obs} - R_i^{sim}(k_a, k_d, R_{max}) \right)^2
$$

where $R^{sim}$ is obtained by integrating the Langmuir ODE with pulsed c(t)
using `scipy.integrate.solve_ivp` (RK45, rtol=1e-8).

**Multi-start protocol:**
- Start 1: Phase 2 estimates
- Start 2: DK estimates
- Starts 3+: Log-normal perturbations of Phase 2 estimates (σ=0.5 in log-space)

Each start is optimised with `scipy.optimize.least_squares` (TRF method,
bounded: $k_a \in [0.1, 10^8]$, $R_{max} \in [1, 10^4]$).

**Aggregation:** Take the *median* of all converged parameter sets. This is
more robust than taking the best cost, since the cost surface can have
shallow local minima that give similar costs but different parameters.

### Why Fix kd?

$k_d$ is well-determined from the dissociation phase alone (where $c = 0$,
the ODE has an exact exponential solution). Allowing $k_d$ to vary in the
full ODE fit introduces unnecessary degrees of freedom and can lead to
parameter coupling artefacts. The DK estimate of $k_d$ is typically very
accurate.

### Standard Errors

Asymptotic standard errors are computed from the best Jacobian:

$$
\text{Cov}(k_a, R_{max}) = \hat{\sigma}^2 \cdot (\mathbf{J}^T \mathbf{J})^{-1}
$$

where $\hat{\sigma}^2 = \sum r_i^2 / (n - 2)$ is the residual variance.

**Implementation:** `ode_fitting.ode_fit(t, signal, c_func, w, markers, ka0, kd0, Rmax0)`

---

## 7. Batch Processing

`batch.batch_fit(filepath, mode='ode')` processes all samples:

1. Load .cxw file
2. For each sample:
   - Check for non-specific binding → skip if NSB
   - Fit with DK (mode='dk') or DK→ODE (mode='ode')
   - Record parameters, uncertainties, and QC metrics
3. Flag poor fits based on:
   - kd hitting upper bound (≥ 10 s⁻¹)
   - ka hitting lower bound (≤ 1 M⁻¹s⁻¹)
   - Low Rmax (≤ 0.5 pg/mm²)
   - High residual (σ > 10)
   - Fit failure
4. Return sorted DataFrame (by compound, then concentration)

---

## 8. Concentration Error Sensitivity

Notebook `03_concentration_error.ipynb` demonstrates that:

- $k_d$ is **invariant** to concentration errors (fitted from dissociation only)
- $k_a \propto 1/c$ — scales inversely with concentration
- $K_D = k_d/k_a$ scales roughly linearly with concentration error
- ±30% concentration error → ~30% KD error
- ±90% → KD can shift by an order of magnitude

This motivates accurate LCMS concentration measurements for reliable $K_D$
determination, while $k_d$ rankings are robust to concentration uncertainty.
