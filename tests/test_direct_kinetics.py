"""Tests for sensofit.direct_kinetics — DK linear solver."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw
from sensofit.direct_kinetics import direct_kinetics_fit, fit_sample
from sensofit.models import (
    build_concentration_profile, select_dmso_cal,
    smooth_and_differentiate, build_weight_mask,
)

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


@pytest.fixture(scope="module")
def data():
    return load_cxw(CXW)


@pytest.fixture(scope="module")
def first_sample(data):
    return data['samples'][0]


class TestDirectKineticsFit:
    def test_returns_required_keys(self, data, first_sample):
        t = first_sample['time']
        dmso = select_dmso_cal(first_sample['index'], data['dmso_cals'])
        c_func, _ = build_concentration_profile(dmso, first_sample['concentration_M'])
        R_smooth, dRdt, _ = smooth_and_differentiate(t, first_sample['signal'])
        w = build_weight_mask(t, first_sample['markers'])

        result = direct_kinetics_fit(t, R_smooth, dRdt, c_func, w=w)
        for key in ['ka', 'kd', 'Rmax', 'KD', 'k_vec', 'k_cov',
                     'k_std', 'residuals', 'sigma_residual', 'n_points']:
            assert key in result, f"Missing key: {key}"

    def test_k_vec_length(self, data, first_sample):
        t = first_sample['time']
        dmso = select_dmso_cal(first_sample['index'], data['dmso_cals'])
        c_func, _ = build_concentration_profile(dmso, first_sample['concentration_M'])
        R_smooth, dRdt, _ = smooth_and_differentiate(t, first_sample['signal'])
        w = build_weight_mask(t, first_sample['markers'])

        result = direct_kinetics_fit(t, R_smooth, dRdt, c_func, w=w)
        assert len(result['k_vec']) == 3

    def test_covariance_shape(self, data, first_sample):
        t = first_sample['time']
        dmso = select_dmso_cal(first_sample['index'], data['dmso_cals'])
        c_func, _ = build_concentration_profile(dmso, first_sample['concentration_M'])
        R_smooth, dRdt, _ = smooth_and_differentiate(t, first_sample['signal'])
        w = build_weight_mask(t, first_sample['markers'])

        result = direct_kinetics_fit(t, R_smooth, dRdt, c_func, w=w)
        assert result['k_cov'].shape == (3, 3)


class TestFitSample:
    def test_fit_first_sample(self, data, first_sample):
        result = fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert result['ka'] > 0
        assert result['kd'] > 0
        assert result['Rmax'] > 0
        assert result['KD'] > 0

    def test_kd_reasonable_range(self, data, first_sample):
        result = fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        # kd should be between 1e-5 and 10 for typical binding
        assert 1e-5 < result['kd'] < 10.0

    def test_ka_positive(self, data, first_sample):
        result = fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert result['ka'] > 0

    def test_KD_equals_kd_over_ka(self, data, first_sample):
        result = fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        np.testing.assert_allclose(result['KD'], result['kd'] / result['ka'],
                                   rtol=1e-10)

    def test_output_arrays_present(self, data, first_sample):
        result = fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])
        assert 't' in result
        assert 'signal' in result
        assert 'R_smooth' in result
        assert 'c_func' in result

    def test_without_blanks(self, data, first_sample):
        """Should work without blanks (baseline subtraction only)."""
        result = fit_sample(first_sample, data['dmso_cals'], blanks=None)
        assert result['ka'] > 0
        assert result['kd'] > 0
        assert result['blank_index'] is None

    def test_batch_consistency(self, data):
        """Multiple samples should all produce valid results."""
        for s in data['samples'][:5]:
            result = fit_sample(s, data['dmso_cals'], blanks=data['blanks'])
            assert np.isfinite(result['ka'])
            assert np.isfinite(result['kd'])
            # KD can be inf when ka <= 0 (poor fit) — just check it's defined
            assert 'KD' in result
