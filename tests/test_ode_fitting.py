"""Tests for creoptix_fitting.ode_fitting — ODE refinement."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from creoptix_fitting.data_loader import load_cxw
from creoptix_fitting.ode_fitting import ode_fit, fit_sample as ode_fit_sample
from creoptix_fitting.direct_kinetics import fit_sample as dk_fit_sample
from creoptix_fitting.models import (
    select_dmso_cal, build_pulsed_concentration_profile,
    build_full_weight_mask, is_nonspecific_binder,
)

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


@pytest.fixture(scope="module")
def data():
    return load_cxw(CXW)


@pytest.fixture(scope="module")
def first_sample(data):
    return data['samples'][0]


@pytest.fixture(scope="module")
def specific_binders(data):
    return [s for s in data['samples'] if not is_nonspecific_binder(s)[0]]


class TestODEFit:
    def test_returns_required_keys(self, data, first_sample):
        dk = dk_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        dmso = select_dmso_cal(first_sample['index'], data['dmso_cals'])
        c_func, _ = build_pulsed_concentration_profile(dmso, first_sample['concentration_M'])
        w = build_full_weight_mask(dk['t'], first_sample['markers'], dmso)

        result = ode_fit(dk['t'], dk['signal'], c_func, w,
                         first_sample['markers'],
                         ka0=dk['ka'], kd0=dk['kd'], Rmax0=dk['Rmax'])

        for key in ['ka', 'kd', 'Rmax', 'KD', 'R_fit', 'residuals',
                     'success', 'n_converged']:
            assert key in result, f"Missing key: {key}"

    def test_KD_equals_kd_over_ka(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        np.testing.assert_allclose(result['KD'], result['kd'] / result['ka'],
                                   rtol=1e-10)

    def test_kd_close_to_dk(self, data, first_sample):
        """ODE should use kd from DK (fixed), so they should match."""
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        np.testing.assert_allclose(result['kd'], result['dk_kd'],
                                   rtol=0.01)


class TestFitSample:
    def test_fit_succeeds(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert result['success']

    def test_parameters_positive(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert result['ka'] > 0
        assert result['kd'] > 0
        assert result['Rmax'] > 0

    def test_R_fit_shape(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert result['R_fit'].shape == result['t'].shape

    def test_dk_estimates_stored(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert 'dk_ka' in result
        assert 'dk_kd' in result
        assert 'dk_Rmax' in result
        assert result['dk_ka'] > 0

    def test_multiple_samples(self, data, specific_binders):
        """Fit first 3 specific binders — all should succeed."""
        for s in specific_binders[:3]:
            result = ode_fit_sample(s, data['dmso_cals'], blanks=data['blanks'])
            assert result['success'], f"Failed: {s['compound']}"
            assert np.isfinite(result['KD'])

    def test_sigma_residual_reasonable(self, data, first_sample):
        result = ode_fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        # Residual should be small relative to signal amplitude
        assert result['sigma_residual'] < 50.0
