"""Tests for safe ODE seed generation."""

import numpy as np

import sensofit.ode_fitting as ode_fitting


def test_zero_lpf_ka_uses_default_seeds_without_divide_warning(monkeypatch):
    time = np.arange(0.0, 31.0)
    sample = {
        'index': 12,
        'rk_serie_id': 1,
        'channel': 'FC2-FC1',
        'time': time,
        'signal': np.ones_like(time),
        'concentration_M': 1e-6,
        'markers': {'Injection': 5.0, 'Rinse': 20.0, 'RinseEnd': 30.0},
    }

    monkeypatch.setattr(
        ode_fitting, 'double_reference',
        lambda sample, blank: (sample['signal'].copy(), None))
    monkeypatch.setattr(
        ode_fitting, 'fit_last_disso',
        lambda *args, **kwargs: (1e-3, None, None, None))
    monkeypatch.setattr(
        ode_fitting, 'fit_last_asso',
        lambda *args, **kwargs: (0.0, None, None, None, None, None))
    monkeypatch.setattr(
        ode_fitting, 'build_pulsed_concentration_profile',
        lambda dmso, concentration: (lambda t: concentration, None))
    monkeypatch.setattr(
        ode_fitting, 'build_full_weight_mask',
        lambda *args, **kwargs: np.ones_like(time))
    monkeypatch.setattr(
        ode_fitting, 'ode_fit',
        lambda *args, **kwargs: {
            'R_fit': np.zeros_like(args[0]),
            'residuals': np.zeros_like(args[0]),
            'success': True,
        })

    with np.errstate(divide='raise', invalid='raise'):
        result = ode_fitting.fit_sample(sample, dmso=None)

    assert result['ka_seed'] == 1e3
    assert result['kd_seed'] == 1e-3
    assert result['Rmax_seed'] == 10.0
