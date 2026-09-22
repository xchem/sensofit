"""Synthetic recovery and exclusion behavior for the current fitting method."""

from types import SimpleNamespace
import numpy as np
import pytest
from sensofit import batch, current_fitting
from sensofit.current_fitting import ode_fit
from sensofit.data_loader import _parse_mw
from sensofit.models import (
    build_full_weight_mask,
    simulate_sensorgram_zoh as simulate_sensorgram,
)


@pytest.mark.parametrize("text, expected", [
    ("1 kDa", 1000.0), ("0.455 kDa", 455.0), ("455 Da", 455.0), ("", 0.0),
])
def test_cxw_molecular_weight_units(text, expected):
    assert _parse_mw(text) == expected


def test_mask_keeps_buffer_and_dissociation_away_from_transitions():
    time = np.arange(0.0, 11.0, 0.25)
    markers = {"Injection": 2.0, "Rinse": 8.0, "RinseEnd": 10.0}
    analyte = ((time >= 3) & (time <= 4)) | ((time >= 6) & (time <= 7))
    dmso = dict(time=time, raw_reference=10.0 * analyte,
                baseline_duration_s=1.0, markers=markers)
    weights = build_full_weight_mask(time, markers, dmso,
                                    association_weight=0.0, transition_window_s=0.5)
    np.testing.assert_array_equal(weights[(time < 2) | (time > 10) | analyte], 0.0)
    for marker_time in markers.values():
        np.testing.assert_array_equal(weights[np.abs(time - marker_time) <= 0.5], 0.0)
    for clean_time in (2.75, 5.0, 7.25, 9.0):
        assert weights[time == clean_time] == 1.0


@pytest.mark.parametrize("bounds, n_points", [((7.0, 7.5), 81), (None, 161)])
def test_joint_reference_recovery_and_pkd_coordinates(bounds, n_points):
    time = np.linspace(0.0, 20.0, n_points)
    markers = {"Injection": 2.0, "Rinse": 10.0, "RinseEnd": 20.0}
    c_func = lambda t: np.where(
        (np.asarray(t) >= 2.0) & (np.asarray(t) < 10.0), 1e-6, 0.0)
    truth = np.array([2e5, 4e-3, 25.0])
    binding = simulate_sensorgram(time, *truth, c_func)
    reference = np.zeros_like(time) if bounds else 3.0 * np.sin(time / 2.7) + 0.1 * time
    signal = binding if bounds else binding + reference - 0.3
    weights = (time >= 2.0).astype(float) if bounds else np.ones_like(time)
    result = ode_fit(
        time, signal, c_func, weights, markers, ka0=truth[0], kd0=truth[1],
        Rmax0=truth[2], n_starts=1, rng_seed=0, reference_signal=reference,
        pKD_bounds=bounds)
    assert result["success"]
    if bounds:
        assert 7.0 <= -np.log10(result["KD"]) <= 7.5
        assert (result["pKD_lower_bound"], result["pKD_upper_bound"]) == bounds
    else:
        assert result["reference_scale"] == pytest.approx(1.0, abs=0.01)
        assert result["reference_scale_prior_fraction"] == 1.0
        assert result["offset"] == pytest.approx(0.3, abs=0.01)
        np.testing.assert_allclose(result["KD"], truth[1] / truth[0], rtol=0.05)


@pytest.mark.parametrize("regime, identifiable, injection, override, expected", [
    ("no_binding", True, False, False, "prefit_no_binding"),
    ("no_binding", True, False, True, ""),
    ("no_binding", False, True, True, "raw_unidentifiable_with_injection_issue"),
    ("medium", False, True, False, "raw_unidentifiable_with_injection_issue"),
    ("medium", False, False, False, ""),
    ("medium", True, True, False, ""),
])
def test_skip_policy_requires_the_agreed_evidence(regime, identifiable, injection, override, expected):
    prior = SimpleNamespace(usable=True, regime=regime)
    reason = current_fitting._prefit_basin_skip_reason(
        prior, {"raw_affinity_identifiable": identifiable},
        injection_issue=injection, fit_no_binding=override)
    assert reason == expected


@pytest.mark.parametrize("thresholds, override, ratio, regime, cost, concentration", [
    (None, False, 1.1, "no_binding", None, 1e-6),
    (None, True, 1.1, "no_binding", 100.0, 1e-6),
    ((-10, -5, 10), False, 1.1, "medium", 100.0, 1e-6),
    ((-10, -5, 10), False, 1.2, "medium", 115.0, 1e-6),
    (None, False, 1.1, "unknown", 100.0, 0.0),
    (None, False, 1.1, "unknown", 100.0, -1e-6),
])
def test_batch_prefit_controls(monkeypatch, thresholds, override, ratio,
                              regime, cost, concentration):
    t = np.arange(0.0, 81.0, 0.5)
    markers = dict(Injection=5.0, Rinse=40.0, RinseEnd=80.0)
    sample = dict(index=3, rk_serie_id="1", channel="FC2-FC1", cycle_type="Sample",
                  compound="negative-control", concentration_M=concentration, mw=400.0,
                  time=t, signal=np.zeros_like(t), raw_active=np.zeros_like(t),
                  raw_reference=np.zeros_like(t), baseline_duration_s=4.0, markers=markers)
    pulses = ((t >= 10) & (t < 15)) | ((t >= 20) & (t < 25)) | ((t >= 30) & (t < 35))
    dmso = dict(sample, raw_active=100.0 * pulses, raw_reference=100.0 * pulses)
    data = dict(samples=[sample], dmso_cals=[dmso], blanks=[],
                project={"ligand_mw_Da": 60000.0},
                capture_level_metadata={"capture_levels_pg_per_mm2": {"FC2": 3000.0}})
    monkeypatch.setattr(batch, "load_experiment", lambda *a, **k: data)
    monkeypatch.setattr(batch, "sensorgram_heuristics", lambda *a, **k: [])

    def fit(t, *args, **kwargs):
        assert cost is not None, "A skipped trace reached the optimizer"
        return dict(ka=1e3, kd=1e-3, KD=1e-6, Rmax=10.0, success=True,
                    cost=115.0 if "pKD_bounds" in kwargs else 100.0,
                    sigma_residual=0.1, R_fit=np.zeros_like(t))

    monkeypatch.setattr(current_fitting, "ode_fit", fit)
    frame, _, results = batch.batch_fit(
        "synthetic.cxw", mode="ode", progress=False,
        ode_fit_variant="joint_reference_offset_prefit_basin", rng_seed=0,
        prefit_thresholds=thresholds, fit_no_binding=override, max_cost_ratio=ratio)
    result, row = results[0], frame.iloc[0]
    assert row["affinity_area_regime"] == regime
    assert row["prefit_basin_cost_ratio_max"] == ratio
    if cost is None:
        assert result is None and np.isnan(row["KD"])
        assert row["kinetic_fit_skip_reason"] == "prefit_no_binding"
    else:
        assert result["cost"] == cost
        assert result["prefit_basin_attempted"] == (regime == "medium")
        if concentration <= 0:
            seeds = [result[k] for k in ("ka_seed", "kd_seed", "Rmax_seed", "KD_seed")]
            assert seeds == [1e3, 1e-3, 10.0, 1e-6]
            assert result["seed_method"] == "control_defaults"


@pytest.mark.parametrize("options", [
    {"prefit_thresholds": values} for values in ((1, 2), (1, 1, 2), (3, 2, 1), (1, 2, np.nan))
] + [{"max_cost_ratio": value} for value in (0.99, np.nan, np.inf)])
def test_invalid_prefit_options_fail_even_without_samples(monkeypatch, options):
    monkeypatch.setattr(batch, "load_experiment",
                        lambda *a, **k: dict(samples=[], dmso_cals=[], blanks=[]))
    with pytest.raises(ValueError, match="prefit"):
        batch.batch_fit("empty.cxw", mode="ode",
                        ode_fit_variant="joint_reference_offset_prefit_basin", **options)


def test_zoh_propagation_matches_piecewise_analytic_solution():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 9.0])
    c = lambda x: np.where(np.asarray(x) < 4.0, 2e-6, 0.0)
    ka, kd, rmax = 1e5, 0.05, 20.0
    observed = simulate_sensorgram(t, ka, kd, rmax, c)
    rate = ka * 2e-6 + kd
    req = ka * 2e-6 * rmax / rate
    at_switch = req * (1 - np.exp(-rate * 4.0))
    expected = np.where(
        t <= 4.0, req * (1 - np.exp(-rate * t)), at_switch * np.exp(-kd * (t - 4.0))
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)
