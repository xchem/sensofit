"""Interpretable pre-fit affinity characterisation for GCI sensorgrams.

The score combines analyte concentration with baseline-corrected association,
early-dissociation and retained-tail summaries.  A bounded local BIC comparison
adds evidence that a reference-adjusted exponential decay is present.  Every
component is returned as a diagnostic rather than hidden inside a classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import trapezoid


REGIME_PKD_BOUNDS = {
    # Overlap conventional cut-offs: this coarse score proposes a basin, not a class.
    "no_binding": (2.0, 6.0),
    "weak": (2.0, 5.5),
    "medium": (4.0, 6.5),
    # pKD 14 is the optimizer's physical maximum (ka <= 1e8, kd >= 1e-6).
    "tight": (6.0, 14.0),
}

# Only binding regimes propose a constrained basin.
PREFIT_BASIN_PKD_BOUNDS = {
    key: value for key, value in REGIME_PKD_BOUNDS.items()
    if key != "no_binding"
}

# The original concentration, selectivity and retention coefficients were a
# non-negative least-squares fit to ordered experimentalist labels, then
# rounded to simple values.  Tail survival, bounded BIC and persistence were
# added as physically directed refinements.  See the benchmark script.
ORDINAL_INTERCEPT = 2.0
CONCENTRATION_COEFFICIENT = 0.2
SELECTIVITY_COEFFICIENT = 1.0
RETENTION_COEFFICIENT = 1.0 / 3.0
TAIL_SURVIVAL_COEFFICIENT = 0.15
EARLY_DECAY_BIC_HALF_SATURATION = 100.0
PERSISTENCE_COEFFICIENT = 0.5
NO_BINDING_THRESHOLD = 0.8
WEAK_THRESHOLD = 2.05
MEDIUM_THRESHOLD = 2.97

# A negative active-minus-reference endpoint is treated as differential
# channel drift only when the earlier trace already contains independent
# binder-like evidence.  This prevents an endpoint correction from turning
# noise in weak/non-binding traces into apparent late retention.
TAIL_DRIFT_MIN_SELECTIVITY = 0.10
TAIL_DRIFT_MIN_RETENTION = 0.25
TAIL_DRIFT_SHRINKAGE_SNR = 3.0
END_GUARD_SECONDS = 5.0


REGIME_SCORE_THRESHOLDS = {
    "no_binding_to_weak": 0.80,
    "weak_to_medium": 2.05,
    "medium_to_tight": 2.97,
}


def _validate_prefit_thresholds(thresholds):
    thresholds = np.asarray(
        tuple(REGIME_SCORE_THRESHOLDS.values()) if thresholds is None else thresholds,
        dtype=float)
    if (thresholds.shape != (3,) or not np.isfinite(thresholds).all()
            or np.any(np.diff(thresholds) <= 0)):
        raise ValueError("prefit_thresholds must contain three finite, increasing scores")
    return thresholds


@dataclass(frozen=True)
class AffinityAreaPrior:
    """Pre-fit affinity regime and its raw-channel evidence."""

    usable: bool = False
    reason: str = ""
    regime: str = "unknown"
    score: float = np.nan
    pKD_lower: float = np.nan
    pKD_upper: float = np.nan
    concentration_M: float = np.nan
    baseline_noise: float = np.nan
    association_active_mean: float = np.nan
    association_reference_mean: float = np.nan
    association_between_mean: float = np.nan
    late_association_between_mean: float = np.nan
    early_dissociation_active_mean: float = np.nan
    early_dissociation_reference_mean: float = np.nan
    early_dissociation_between_mean: float = np.nan
    raw_tail_dissociation_between_mean: float = np.nan
    tail_dissociation_between_mean: float = np.nan
    late_dissociation_between_mean: float = np.nan
    tail_drift_corrected: bool = False
    tail_drift_weight: float = 0.0
    tail_drift_slope: float = np.nan
    tail_endpoint_between: float = np.nan
    early_decay_delta_bic: float = np.nan
    early_decay_amplitude: float = np.nan
    early_decay_koff: float = np.nan
    early_decay_reference_scale: float = np.nan
    response_scale: float = np.nan
    early_dissociation_fraction: float = np.nan
    association_selectivity: float = np.nan
    retention_ratio: float = np.nan
    tail_survival_ratio: float = np.nan
    concentration_evidence: float = np.nan
    selectivity_evidence: float = np.nan
    retention_evidence: float = np.nan
    tail_survival_evidence: float = np.nan
    early_decay_evidence: float = np.nan
    retained_tail_fraction: float = np.nan
    persistence_evidence: float = np.nan

    def as_dict(self) -> dict:
        return asdict(self)


def _score_to_affinity_label(score):
    """Map a pre-fit affinity score to the benchmark label vocabulary."""
    if score is None or not np.isfinite(score):
        return 'unknown'
    if score < NO_BINDING_THRESHOLD:
        return 'no binding'
    if score < WEAK_THRESHOLD:
        return 'weak'
    if score < MEDIUM_THRESHOLD:
        return 'medium'
    return 'tight'


def _window_mean(t, values, start, stop):
    mask = (
        np.isfinite(t) & np.isfinite(values)
        & (t >= float(start)) & (t <= float(stop))
    )
    if mask.sum() < 3:
        return np.nan
    selected_t = t[mask]
    duration = float(selected_t[-1] - selected_t[0])
    if duration <= 0:
        return np.nan
    return float(trapezoid(values[mask], selected_t) / duration)


def _unusable(reason, concentration=np.nan):
    return AffinityAreaPrior(reason=reason, concentration_M=concentration)


def _early_decay_reference_evidence(
    t, active, reference, rinse,
):
    """Compare reference-template models over the first 10 s after rinse.

    The alternative adds a positive exponential to a scaled reference trace,
    intercept, and linear drift.  This is a local shape comparison rather than
    a full kinetic fit: koff is selected from a fixed one-dimensional grid and
    all remaining parameters are ordinary linear least squares.
    """
    mask = (
        np.isfinite(t) & np.isfinite(active) & np.isfinite(reference)
        & (t >= rinse + 0.15) & (t <= rinse + 10.0)
    )
    if mask.sum() < 12:
        return np.nan, np.nan, np.nan, np.nan
    dt = t[mask] - rinse
    yy = active[mask]
    rr = reference[mask]
    null_design = np.column_stack([rr, np.ones(len(dt)), dt])
    null_parameters, *_ = np.linalg.lstsq(
        null_design, yy, rcond=None)
    null_residual = yy - null_design @ null_parameters
    null_rss = max(float(np.sum(null_residual ** 2)), 1e-12)

    best = None
    for koff in np.geomspace(0.01, 5.0, 160):
        design = np.column_stack([
            rr, np.exp(-koff * dt), np.ones(len(dt)), dt,
        ])
        parameters, *_ = np.linalg.lstsq(design, yy, rcond=None)
        amplitude = float(parameters[1])
        if amplitude < 0:
            continue
        residual = yy - design @ parameters
        rss = max(float(np.sum(residual ** 2)), 1e-12)
        if best is None or rss < best[0]:
            best = (
                rss, float(amplitude), float(koff),
                float(parameters[0]),
            )
    if best is None:
        return 0.0, 0.0, np.nan, float(null_parameters[0])

    # Relative to the null, the alternative adds amplitude and koff.  The
    # two-parameter BIC penalty prevents small improvements from becoming
    # positive evidence merely because the koff grid was searched.
    delta_bic = float(
        len(dt) * np.log(null_rss / best[0])
        - 2.0 * np.log(len(dt))
    )
    return delta_bic, best[1], best[2], best[3]


def estimate_affinity_area_prior(sample, *, prefit_thresholds=None,
                                 end_guard_s=END_GUARD_SECONDS,
                                 tail_endpoint_window_s=3.0) -> AffinityAreaPrior:
    """Characterise a trace before kinetic fitting.

    The monotonic score combines concentration, association selectivity,
    early retention, tail survival, bounded decay-model evidence and absolute
    retained-tail persistence.
    """
    thresholds = _validate_prefit_thresholds(prefit_thresholds)
    try:
        concentration = float(sample["concentration_M"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "sample must contain a numeric concentration_M"
        ) from error
    if not np.isfinite(concentration) or concentration <= 0:
        return _unusable("non_positive_concentration", concentration)

    try:
        t = np.asarray(sample["time"], dtype=float)
        active = np.asarray(sample["raw_active"], dtype=float)
        reference = np.asarray(sample["raw_reference"], dtype=float)
        markers = sample["markers"]
        injection = float(markers["Injection"])
        rinse = float(markers["Rinse"])
        rinse_end = float(markers["RinseEnd"])
    except (KeyError, TypeError, ValueError):
        return _unusable("missing_raw_channels_or_markers", concentration)
    if (
        t.ndim != 1 or active.shape != t.shape or reference.shape != t.shape
        or not injection < rinse < rinse_end
    ):
        return _unusable("invalid_raw_channels_or_markers", concentration)

    baseline = (
        np.isfinite(t) & np.isfinite(active) & np.isfinite(reference)
        & (t >= max(float(t[0]), injection - 20.0))
        & (t <= injection - 2.0)
    )
    if baseline.sum() < 5:
        return _unusable("insufficient_baseline", concentration)
    active = active - float(np.median(active[baseline]))
    reference = reference - float(np.median(reference[baseline]))
    between = active - reference
    baseline_center = float(np.median(between[baseline]))
    noise = float(
        1.4826 * np.median(np.abs(between[baseline] - baseline_center))
    )

    association_start = injection + 0.5
    association_stop = rinse - 0.5
    analysis_end = rinse_end - end_guard_s
    early_stop = min(rinse + 10.0, analysis_end)
    late_start = max(rinse + 1.0, rinse_end - 20.0)
    windows = {
        "association_active_mean": _window_mean(
            t, active, association_start, association_stop),
        "association_reference_mean": _window_mean(
            t, reference, association_start, association_stop),
        "association_between_mean": _window_mean(
            t, between, association_start, association_stop),
        "late_association_between_mean": _window_mean(
            t, between, max(association_start, rinse - 10.0),
            association_stop),
        "early_dissociation_active_mean": _window_mean(
            t, active, rinse + 1.0, early_stop),
        "early_dissociation_reference_mean": _window_mean(
            t, reference, rinse + 1.0, early_stop),
        "early_dissociation_between_mean": _window_mean(
            t, between, rinse + 1.0, early_stop),
        "raw_tail_dissociation_between_mean": _window_mean(
            t, between, rinse + 30.0, min(rinse + 60.0, analysis_end)),
        "late_dissociation_between_mean": _window_mean(
            t, between, late_start, analysis_end),
    }
    if not np.isfinite(list(windows.values())).all():
        return _unusable("insufficient_window_data", concentration)

    response_scale = max(
        abs(windows["association_active_mean"])
        + abs(windows["association_reference_mean"]),
        3.0 * noise,
        1.0,
    )
    selectivity = float(np.clip(
        max(windows["association_between_mean"], 0.0) / response_scale,
        1e-3,
        0.999,
    ))
    retention = float(np.clip(
        max(windows["early_dissociation_between_mean"], 0.0)
        / max(
            windows["late_association_between_mean"],
            noise,
            1e-6,
        ),
        1e-3,
        1.5,
    ))

    # Correct only the 30--60 s tail for a negative differential-channel
    # drift.  The linear drift is anchored at injection and estimated from
    # the final 3 s; association and early-retention evidence are left
    # untouched.  Requiring both of those independent pieces of evidence
    # makes this a conditional measurement correction rather than a binder
    # classifier based on the endpoint.
    # Use the final 3 s before the end guard as the endpoint.  A longer window
    # can be centred inside the 30--60 s tail itself and consequently subtract
    # away genuine curvature/retention that the correction should preserve.
    endpoint_start = max(rinse + 1.0, analysis_end - tail_endpoint_window_s)
    endpoint = _window_mean(t, between, endpoint_start, analysis_end)
    endpoint_time = 0.5 * (endpoint_start + analysis_end)
    eligible_tail_drift = bool(
        selectivity >= TAIL_DRIFT_MIN_SELECTIVITY
        and retention >= TAIL_DRIFT_MIN_RETENTION
        and np.isfinite(endpoint)
        and endpoint < baseline_center
        and endpoint_time > injection
    )
    drift_depth = max(baseline_center - endpoint, 0.0)
    shrinkage_scale = TAIL_DRIFT_SHRINKAGE_SNR * max(noise, 1e-6)
    tail_drift_weight = (
        float(
            drift_depth ** 2
            / (drift_depth ** 2 + shrinkage_scale ** 2)
        )
        if eligible_tail_drift else 0.0
    )
    correct_tail_drift = tail_drift_weight > 0.0
    tail_drift_slope = 0.0
    if correct_tail_drift:
        tail_drift_slope = float(
            tail_drift_weight * (endpoint - baseline_center)
            / (endpoint_time - injection)
        )
        drift = tail_drift_slope * np.clip(t - injection, 0.0, None)
        tail_between = between - drift
        tail_mean = _window_mean(
            t, tail_between, rinse + 30.0,
            min(rinse + 60.0, analysis_end),
        )
    else:
        tail_mean = windows["raw_tail_dissociation_between_mean"]
    windows["tail_dissociation_between_mean"] = tail_mean

    tail_retention = float(np.clip(
        max(windows["tail_dissociation_between_mean"], 0.0)
        / max(
            windows["late_association_between_mean"],
            noise,
            1e-6,
        ),
        1e-3,
        1.5,
    ))
    # This ratio isolates decay shape: association amplitude and selectivity
    # cancel because both numerator and denominator share the same scale.
    # If both windows are at the response floor, survival is neutral rather
    # than spuriously perfect evidence for tight binding.
    tail_survival = (
        1.0
        if retention <= 1.001e-3 and tail_retention <= 1.001e-3
        else float(np.clip(tail_retention / retention, 1e-3, 1.0))
    )
    fraction = max(
        windows["early_dissociation_between_mean"] / response_scale, 1e-3,
    )
    concentration_evidence = float(
        CONCENTRATION_COEFFICIENT * -np.log10(concentration))
    selectivity_evidence = float(
        SELECTIVITY_COEFFICIENT * np.log10(selectivity))
    retention_evidence = float(
        RETENTION_COEFFICIENT * np.log10(retention))
    tail_survival_evidence = float(
        TAIL_SURVIVAL_COEFFICIENT * np.log10(tail_survival))
    (
        early_decay_delta_bic,
        early_decay_amplitude,
        early_decay_koff,
        early_decay_reference_scale,
    ) = _early_decay_reference_evidence(
        t, active, reference, rinse,
    )
    # BIC establishes whether an exponential component is credible, not how
    # tight the binder is.  Convert it to bounded evidence: half credit at a
    # positive delta-BIC of 100 and an asymptotic maximum of one.
    positive_delta_bic = max(early_decay_delta_bic, 0.0)
    early_decay_evidence = float(
        positive_delta_bic
        / (positive_delta_bic + EARLY_DECAY_BIC_HALF_SATURATION)
        if np.isfinite(early_decay_delta_bic) else 0.0
    )
    # Retention times tail survival is the absolute fraction of the late
    # association response remaining at 30--60 s.  Its square root supplies a
    # smooth positive persistence contribution without allowing the strongest
    # tails to dominate the score.
    retained_tail_fraction = float(np.clip(
        retention * tail_survival, 1e-6, 1.5,
    ))
    persistence_evidence = float(
        PERSISTENCE_COEFFICIENT * np.sqrt(retained_tail_fraction)
    )
    score = float(
        ORDINAL_INTERCEPT
        + concentration_evidence
        + selectivity_evidence
        + retention_evidence
        + tail_survival_evidence
        + early_decay_evidence
        + persistence_evidence
    )
    if score < thresholds[0]:
        regime = "no_binding"
    elif score < thresholds[1]:
        regime = "weak"
    elif score < thresholds[2]:
        regime = "medium"
    else:
        regime = "tight"
    lower, upper = REGIME_PKD_BOUNDS[regime]
    return AffinityAreaPrior(
        usable=True, regime=regime, pKD_lower=lower, pKD_upper=upper,
        score=score,
        concentration_M=concentration,
        baseline_noise=noise,
        tail_drift_corrected=correct_tail_drift,
        tail_drift_weight=tail_drift_weight,
        tail_drift_slope=tail_drift_slope,
        tail_endpoint_between=endpoint,
        early_decay_delta_bic=early_decay_delta_bic,
        early_decay_amplitude=early_decay_amplitude,
        early_decay_koff=early_decay_koff,
        early_decay_reference_scale=early_decay_reference_scale,
        response_scale=response_scale,
        early_dissociation_fraction=fraction,
        association_selectivity=selectivity,
        retention_ratio=retention,
        tail_survival_ratio=tail_survival,
        concentration_evidence=concentration_evidence,
        selectivity_evidence=selectivity_evidence,
        retention_evidence=retention_evidence,
        tail_survival_evidence=tail_survival_evidence,
        early_decay_evidence=early_decay_evidence,
        retained_tail_fraction=retained_tail_fraction,
        persistence_evidence=persistence_evidence,
        **windows,
    )
