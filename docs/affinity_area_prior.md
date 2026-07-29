# Pre-fit affinity area prior

SensoFit can characterise a trace before kinetic fitting using a cheap summary
of the raw active and reference channels. The result contains an ordinal score,
and the component diagnostics needed to audit it. It deliberately does not
assign a binding class or modify the kinetic fit.

The screen baseline-corrects both channels using the 20–2 second
pre-injection window, then records time-averaged areas (AUC divided by window
duration) during association (injection +0.5 seconds to rinse −0.5 seconds),
late association (approximately the final 10 seconds), early dissociation
(1–10 seconds after rinse), and the 30–60 second dissociation tail. The final
five seconds before `RinseEnd` are excluded to avoid marker-boundary artefacts.

The current score uses six interpretable, monotonic terms:

```text
association_selectivity =
    clipped positive association active-minus-reference mean
    / association response scale

retention_ratio =
    clipped positive early-dissociation active-minus-reference mean
    / late-association active-minus-reference mean

score = 2.00
        + 0.2 * -log10(concentration_M)
        + 1.0 * log10(association_selectivity)
        + (1/3) * log10(retention_ratio)
        + 0.15 * log10(tail_survival_ratio)
        + positive_early_decay_delta_BIC
          / (positive_early_decay_delta_BIC + 100)
        + 0.5 * sqrt(retained_tail_fraction)

retained_tail_fraction = retention_ratio * tail_survival_ratio
```

The association response scale is the sum of the absolute active and reference
association means over the injection +0.5 second to rinse −0.5 second window,
with floors based on baseline noise and one response unit. Selectivity
approximates specific occupancy relative to the common-mode response.
Retention divides the 1–10 second post-rinse mean by the mean over the final
approximately 10 seconds of association (rinse −10 to −0.5 seconds), adding
early off-rate information. The concentration term uses the injected analyte
concentration recorded for the whole trace rather than a timed response.

`tail_survival_ratio` is the response retained at 30–60 seconds after rinse
relative to the response retained at 1–10 seconds. This ratio cancels the
association response scale and therefore targets decay shape rather than
binding amplitude. When both dissociation windows are at the noise floor, the
term is neutral.

The early-decay term compares two small local models from 0.15–10 seconds
after rinse. Both contain a scaled copy of the reference trace, an intercept,
and linear drift. The alternative adds a positive exponential decay. `koff`
is selected from a fixed one-dimensional grid and the other parameters use
linear least squares; this is a local shape comparison rather than a full
kinetic fit. The BIC difference includes a two-parameter penalty for amplitude
and `koff`. Only positive evidence contributes to the score. Because BIC
establishes that an exponential component is convincing rather than that it
is slow, its contribution is bounded as `B / (B + 100)`, where
`B = max(early_decay_delta_BIC, 0)`. This gives half credit at a positive
delta-BIC of 100 and approaches, but never reaches, a maximum contribution of
one.

The retained-tail persistence term uses
`retention_ratio * tail_survival_ratio`, which simplifies to the
drift-corrected 30–60 second response divided by the late-association response.
Its contribution is `0.5 * sqrt(retained_tail_fraction)`. This provides direct
positive evidence when a substantial absolute fraction of the association
response remains well into dissociation, while the square root prevents the
strongest tails from dominating the score. Using the absolute retained
fraction also avoids rewarding response-floor traces whose tail-survival ratio
is defined as neutral.

The 30–60 second mean receives a narrowly gated differential-drift correction
when the active channel finishes below the reference channel. A straight drift
line is anchored at injection and estimated from the final three seconds, but
it is applied only to the tail mean—not to association or early retention.
Correction requires association selectivity of at least `0.10` and early
retention of at least `0.25`.

The endpoint condition is smooth rather than binary. If `d` is the amount by
which the endpoint lies below its baseline and `sigma` is baseline noise, the
applied fraction of the inferred drift is

```text
drift_weight = d^2 / (d^2 + (3 * sigma)^2)
```

Positive endpoints receive zero correction; increasingly convincing negative
drift approaches full correction continuously. The raw and corrected tail
means, endpoint, applied slope, drift weight, and early-decay diagnostics are
all exported.

The original coefficients were a non-negative least-squares fit to ordered targets
(`no_binding=0`, `weak=1`, `medium=2`, `tight=3`). Requiring non-negative
slopes fixes each feature's physical direction. The first three slopes are
rounded to exactly `0.2`, `1`, and `1/3`; the tail-survival coefficient is
`0.15`. The bounded BIC term has unit maximum contribution, and the retained-
tail persistence coefficient is `0.5`. No compound or assay identifiers are
included.

The intercept is fixed at the round value `2.00`. Classification thresholds
are not part of the estimator because the score should be validated on an
independent assay panel before cutoffs are treated as transferable.

Call the estimator directly with a sample returned by the SensoFit data
loader:

```python
from sensofit.affinity_prior import estimate_affinity_area_prior

prior = estimate_affinity_area_prior(sample)
print(prior.score)
print(prior.association_selectivity, prior.retention_ratio)
```

If the raw channels, markers, baseline, or positive concentration are
unavailable, the estimator raises `ValueError` with the failed requirement.

To reproduce the experimentalist-label diagnostic:

```bash
python -m scripts.benchmark_affinity_area_prior \
  --data-dir data \
  --output runs/affinity_area_prior.csv
```

The optional `affinity_concentration_overrides.csv` file makes
experimentalist-supplied corrections to missing or incorrect source metadata
explicit without changing the shared benchmark annotation schema. The current
override uses `2.5e-6` M for cycle 205 of
`20260323_EV712A_Binding_assay.cxw`.

On the 48 annotated affinity-regime traces, all traces were scored and the
in-sample Spearman correlation with ordered labels (`no_binding`, `weak`,
`medium`, `tight`) was 0.924. For comparison with those labels, the benchmark
script applies reporting cutoffs of 0.80, 2.00, and 2.97. These classify 45/48
traces correctly (93.8%): 4/4 non-binders, 13/14 weak, 20/21 medium, and 8/9
tight.
Recalibrating the thresholds under leave-one-compound-out validation classified
44/48 traces correctly (91.7%), showing that one of the two apparent gains is
threshold-sensitive. Refitting the monotonic coefficient model under the same
compound-held-out scheme gave Spearman 0.895 and ordinal mean absolute error
0.288. Medium and tight traces still overlap, so the cutoffs remain a benchmark
diagnostic rather than part of the estimator API.
