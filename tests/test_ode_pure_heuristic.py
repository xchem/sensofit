"""Tests for creoptix_fitting.ode_pure_heuristic — pure ODE with heuristic seeds."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from creoptix_fitting.data_loader import load_cxw
from creoptix_fitting.ode_pure_heuristic import fit_sample

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


@pytest.fixture(scope="module")
def data():
    return load_cxw(CXW)


@pytest.fixture(scope="module")
def first_sample(data):
    return data['samples'][0]


@pytest.fixture(scope="module")
def first_result(data, first_sample):
    return fit_sample(first_sample, data['dmso_cals'], blanks=data['blanks'])


class TestFitSample:
    def test_returns_required_keys(self, first_result):
        for key in ['ka', 'kd', 'Rmax', 'KD', 'R_fit', 'residuals',
                     'success', 'n_converged', 't', 'signal']:
            assert key in first_result, f"Missing key: {key}"

    def test_KD_equals_kd_over_ka(self, first_result):
        np.testing.assert_allclose(first_result['KD'],
                                   first_result['kd'] / first_result['ka'],
                                   rtol=1e-10)

    def test_parameters_positive(self, first_result):
        assert first_result['ka'] > 0
        assert first_result['kd'] > 0
        assert first_result['Rmax'] > 0

    def test_R_fit_shape(self, first_result):
        assert first_result['R_fit'].shape == first_result['t'].shape

    def test_fit_succeeds(self, first_result):
        assert first_result['success']

    def test_heuristic_estimates_stored(self, first_result):
        assert 'heuristic_ka' in first_result
        assert 'heuristic_kd' in first_result
        assert first_result['heuristic_ka'] > 0

    def test_sigma_residual_reasonable(self, first_result):
        if first_result['success']:
            assert first_result['sigma_residual'] < 20.0
