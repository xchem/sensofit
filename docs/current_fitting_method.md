# Current per-channel fitting method

Select `joint_reference_offset_prefit_basin` in `batch_fit` or the CLI to fit
each active channel independently. The historical fitter remains the default.

## Fitting procedure

1. Baseline- and blank-correct active and reference channels separately. Fit clean
   buffer and final dissociation, excluding ±0.5 s around Injection, Rinse, and
   RinseEnd. Use exact Langmuir propagation with left-endpoint concentration.
2. Derive theoretical Rmax from captured ligand response and the analyte/ligand
   molecular-weight ratio; constrain Rmax to 0.2–5 times this capacity. Seed kinetics from
   early dissociation and pulse transitions.
3. Fit log kinetics with soft-L1 loss, selecting the best converged start. Project
   reference scale (0–2) and offset by bounded linear least squares; the scale
   prior is centered at one and weighted by reference information.
4. Compare unrestricted and pre-fit basins: weak pKD 2–5.5, medium 4–6.5, tight
   6–14. Nine starts cross three capacities with three feasible association rates.
   Both fits must succeed; select the constrained fit if its squared data-residual
   cost is within `max_cost_ratio` (default 1.1) times the unrestricted cost.
   This comparison uses ordinary squared residuals, not the soft-L1 start cost.
5. Skip pre-fit no-binding traces unless `fit_no_binding=True`, which fits only
   the unrestricted basin and retains the classification. Raw unidentifiability
   excludes traces only with an injection issue. Failed/skipped keys retain reasons.

The score is heuristic; physical and pre-fit bounds can influence affinity.
Scoring stops 1 s before RinseEnd; the drift endpoint uses the preceding 2 s.
QC reports boundary hits, local errors, start spread, and dissociation mismatch.
Local errors do not establish affinity identifiability.

## Inputs and use

Capture response comes from immobilization traces, then Wizard metadata, then
`<cxw stem>.capture_levels.json` with `{"capture_levels_pg_per_mm2": {"FC2": 2400.0}}`.
Record any imputation basis. Missing physical metadata raises a fitting error;
nonpositive concentration uses fixed seeds (ka=1e3, kd=1e-3, Rmax=10).
Da/kDa molecular weights are converted to daltons.

```python
from sensofit import batch_fit, flag_poor_fits

frame, data, fits = batch_fit(
    "experiment.cxw", mode="ode",
    ode_fit_variant="joint_reference_offset_prefit_basin",
    rng_seed=0, n_parallel_jobs=6,
)
frame = flag_poor_fits(frame)
```

The method uses at least nine starts. Reproduce results with identical inputs,
sample order, seed, and numerical-library versions. `prefit_thresholds` defaults
to `(0.80, 2.05, 2.97)`: finite, increasing no-binding/weak/medium/tight boundaries;
equality enters the higher regime. `max_cost_ratio` must be finite and ≥1.
These current-method options leave other exclusion and QC checks unchanged.

```bash
python -m sensofit experiment.cxw --mode ode \
  --ode-fit-variant joint_reference_offset_prefit_basin --rng-seed 0 -o results/

# Optional settings:
# --prefit-thresholds 0.80 2.05 2.97 --fit-no-binding --max-cost-ratio 1.1
```
