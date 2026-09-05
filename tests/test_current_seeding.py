"""Physical initialization, basin selection, and reporting regression tests."""

import numpy as np
import pandas as pd
import pytest
from sensofit.batch import flag_poor_fits
from sensofit.data_loader import estimate_capture_levels
from sensofit.models import simulate_sensorgram_zoh as simulate_sensorgram
from sensofit.current_fitting import (
    physical_pulse_seed_candidates,
    prefit_basin_seed_candidates,
    select_prefit_basin_fit,
)


@pytest.mark.parametrize("before, after, imputed, expected", [
    ({"FC1": 10.0, "FC2": 100.0}, {"FC1": 9.0, "FC2": 3100.0}, {}, {"FC2": 3000.0}),
    ({}, {}, {"FC2": 2400.0, "FC3": 2400.0, "FC4": 2400.0},
     {"FC2": 2400.0, "FC3": 2400.0, "FC4": 2400.0}),
    ({"FC2": 100.0}, {"FC2": 2600.0}, {"FC2": 2400.0}, {"FC2": 2500.0}),
])
def test_capture_levels_use_positive_measurements_before_imputation(before, after, imputed, expected):
    cycles = [{"capture_level": {fc: np.repeat(level, 2) for fc, level in edge.items()}}
              for edge in (before, after) if edge]
    data = dict(immobilizations=[dict(cycles=cycles)],
                capture_level_metadata={"capture_levels_pg_per_mm2": imputed})
    assert estimate_capture_levels(data) == expected


def test_physical_pulse_seed_recovers_synthetic_kinetics():
    t = np.arange(0.0, 61.0, 0.1)
    concentration, truth = 1e-6, np.array([1e5, 0.05, 20.0])
    markers = {"Injection": 5.0, "Rinse": 30.0, "RinseEnd": 60.0}
    pulses = [(8.0, 10.0), (14.0, 17.0), (21.0, 25.0)]

    def concentration_profile(values):
        values = np.asarray(values)
        output = np.zeros_like(values, dtype=float)
        for start, end in pulses:
            output[(values >= start) & (values < end)] = concentration
        return output

    signal = simulate_sensorgram(t, *truth, concentration_profile)
    dmso_response = concentration_profile(t) / concentration
    dmso = dict(time=t, raw_reference=dmso_response, raw_active=dmso_response,
                baseline_duration_s=4.0, markers=markers)
    sample = dict(time=t, signal=signal, concentration_M=concentration,
                  mw=400.0, markers=markers)
    result = physical_pulse_seed_candidates(
        sample, sample["signal"], dmso, capture_level=3000.0, ligand_mw_Da=60000.0)
    assert result["Rmax_theory"] == truth[2]
    assert result["Rmax_lower"] == 0.2 * truth[2]
    assert result["Rmax_upper"] == 5.0 * truth[2]
    assert result["pulse_counts"][0] >= 2
    assert result["seed_sources"][0] == "pulse_transition"
    np.testing.assert_allclose(result["starts"][0], truth, rtol=0.03)


@pytest.mark.parametrize("regime, kd, bounds, pkds", [
    ("medium", 0.02, (4.0, 6.5), [4.0, 5.25, 6.5]),
    ("tight", 0.01, (6.0, 14.0), [6.0, 8.0, 10.0]),
    ("tight", 0.1, (6.0, 14.0), [6.0, 8.0, 9.0]),
    ("no_binding", 0.02, None, []),
])
def test_prefit_seed_grid_respects_affinity_and_rate_bounds(regime, kd, bounds, pkds):
    physical = dict(kd=kd, Rmax_lower=4.0, Rmax_theory=20.0, Rmax_upper=100.0)
    result = prefit_basin_seed_candidates(physical, regime)
    assert result["pKD_bounds"] == bounds
    assert result["pKD_values"] == pkds
    if not pkds:
        assert result["starts"] == []
        return
    starts = np.asarray(result["starts"])
    assert np.unique(starts, axis=0).shape == (9, 3)
    assert starts[:, 0].max() <= 1e8 * (1 + 1e-12)
    np.testing.assert_allclose(np.unique(np.log10(starts[:, 0] / starts[:, 1])), pkds)
    np.testing.assert_allclose(np.unique(starts[:, 2]), [4.0, 20.0, 100.0])


@pytest.mark.parametrize("success, cost, selected, reason", [
    (True, 110.0, True, "prefit_basin_within_cost_tolerance"),
    (True, 110.01, False, "prefit_basin_exceeds_cost_tolerance"),
    (False, 100.0, False, "unrestricted_failed_no_cost_comparison"),
])
def test_prefit_basin_selection_requires_a_successful_fit_within_cost_tolerance(
        success, cost, selected, reason):
    unrestricted = dict(success=success, cost=100.0 if success else np.nan, KD=1e-3)
    constrained = dict(success=True, cost=cost, KD=1e-5)
    result, metadata = select_prefit_basin_fit(unrestricted, constrained, max_cost_ratio=1.10)
    assert result["KD"] == (constrained if selected else unrestricted)["KD"]
    assert metadata["prefit_basin_selected"] == selected
    assert metadata["prefit_basin_selection_reason"] == reason
    np.testing.assert_allclose(metadata["prefit_basin_cost_ratio"],
                               cost / 100.0 if success else np.nan)


@pytest.mark.parametrize("fields, reason", [
    ({"Rmax": 100.0, "Rmax_lower_bound": 4.0, "Rmax_upper_bound": 100.0},
     "Rmax_at_physical_upper_bound"),
    ({"prefit_basin_bound_hit": True}, "pKD_at_prefit_basin_bound"),
])
def test_fitting_bound_hits_are_flagged(fields, reason):
    row = dict(success=True, ka=1e5, kd=0.05, Rmax=20.0, sigma_res=0.5)
    flagged = flag_poor_fits(pd.DataFrame([{**row, **fields}])).iloc[0]
    assert flagged["flag"]
    assert reason in flagged["flag_reason"]


@pytest.mark.parametrize("fields, reason", [
    (dict(success=np.nan, concentration_M=1e-6, ka=np.nan, kd=np.nan, Rmax=np.nan,
          sigma_res=np.nan, kinetic_fit_skipped=True, kinetics_reportable=False,
          kinetic_fit_skip_reason="prefit_no_binding"), "prefit_no_binding"),
    (dict(ka=1.0, kd=1e-4, binding_amplitude=0.02), "negligible_fitted_binding_amplitude"),
    (dict(concentration_M=0.0, binding_amplitude=0.0), ""),
    (dict(ka=30000.0, kd=10.0, Rmax_lower_bound=20.0, Rmax_upper_bound=50.0,
          binding_amplitude=1.8, sigma_res=0.1), "bound_limited_low_binding_amplitude"),
])
def test_affinity_reporting_gates(fields, reason):
    row = dict(success=True, concentration_M=2.5e-5, ka=1000.0, kd=0.001,
               Rmax=20.0, sigma_res=0.2)
    flagged = flag_poor_fits(pd.DataFrame([{**row, **fields}])).iloc[0]
    assert bool(flagged["affinity_identifiable"]) == (not reason)
    assert flagged["affinity_unidentifiable_reason"] == reason
    if fields.get("kinetic_fit_skipped"):
        assert flagged["flag"]
        assert flagged["flag_reason"] == "kinetic_fit_skipped:prefit_no_binding"
