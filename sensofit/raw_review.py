"""Deterministic raw-signal diagnostics for kinetic identifiability."""

from __future__ import annotations

import numpy as np

from .models import build_pulsed_concentration_profile, double_reference


def _median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _safe_ratio(numerator, denominator):
    if np.isfinite(numerator) and np.isfinite(denominator) and abs(denominator) > 1e-12:
        return float(numerator / denominator)
    return np.nan


def _correlation(left, right, mask):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 5 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def calculate_raw_identifiability(sample, blank=None, dmso=None):
    """Assess kinetic identifiability from raw-signal retention, coupling, and SNR."""
    t = np.asarray(sample["time"], dtype=float)
    markers = sample.get("markers", {})
    injection = float(markers.get("Injection", t[0]))
    rinse = float(markers.get("Rinse", t[-1]))
    rinse_end = float(markers.get("RinseEnd", t[-1]))
    association = (t >= injection) & (t < rinse)
    pre_rinse = (t >= max(injection, rinse - 5.0)) & (t < rinse)
    early_dissociation = (t >= rinse + 2.0) & (t <= min(rinse + 12.0, rinse_end))
    late_dissociation = (t >= max(rinse, rinse_end - 10.0)) & (t <= rinse_end)
    baseline = t < injection

    signal, blank_index = double_reference(sample, blank)
    signal = np.asarray(signal, dtype=float)
    association_level = _median(signal[pre_rinse])
    early_level = _median(signal[early_dissociation])
    late_level = _median(signal[late_dissociation])
    early_retention = _safe_ratio(early_level, association_level)
    late_retention = _safe_ratio(late_level, association_level)

    active = np.asarray(sample.get("raw_active", signal), dtype=float)
    reference = np.asarray(sample.get("raw_reference", np.zeros_like(t)), dtype=float)
    active_reference_correlation = _correlation(active, reference, association)
    differential_delivery_correlation = np.nan
    if dmso is not None:
        try:
            c_func, _ = build_pulsed_concentration_profile(
                dmso, float(sample.get("concentration_M", 0.0))
            )
            delivery = np.asarray(c_func(t), dtype=float)
            differential_delivery_correlation = _correlation(
                signal, delivery, association
            )
        except (TypeError, ValueError):
            pass

    baseline_noise = float(np.nanstd(signal[baseline])) if baseline.any() else np.nan
    association_peak = float(np.nanmax(signal[association])) if association.any() else np.nan
    delivery_coupled = bool(
        (np.isfinite(differential_delivery_correlation) and differential_delivery_correlation >= 0.80)
        or (np.isfinite(active_reference_correlation) and active_reference_correlation >= 0.98))
    low_retention = bool(
        np.isfinite(early_retention) and np.isfinite(late_retention)
        and early_retention < 0.35 and late_retention < 0.10)
    reasons = []
    if delivery_coupled:
        reasons.append("association_delivery_coupled")
    if low_retention:
        reasons.append("low_post_rinse_retention")
    identifiable = not (delivery_coupled and low_retention)
    if (np.isfinite(association_peak) and np.isfinite(baseline_noise)
            and association_peak <= 5.0 * baseline_noise):
        identifiable = False
        reasons.append("low_raw_signal_to_noise")

    return {
        "raw_blank_index": blank_index,
        "raw_baseline_noise": baseline_noise,
        "raw_association_peak": association_peak,
        "raw_pre_rinse_level": association_level,
        "raw_early_dissociation_level": early_level,
        "raw_late_dissociation_level": late_level,
        "raw_early_retention_fraction": early_retention,
        "raw_late_retention_fraction": late_retention,
        "raw_active_reference_correlation": active_reference_correlation,
        "raw_delivery_correlation": differential_delivery_correlation,
        "raw_delivery_coupled": delivery_coupled,
        "raw_low_retention": low_retention,
        "raw_affinity_identifiable": bool(identifiable),
        "raw_affinity_unidentifiable_reason": "" if identifiable else "; ".join(reasons),
    }
