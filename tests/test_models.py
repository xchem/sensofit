"""Tests for sensofit.models — preprocessing and concentration profiles."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw
from sensofit.models import (
    build_concentration_profile, build_pulsed_concentration_profile,
    select_dmso_cal, select_blank,
    double_reference, build_weight_mask, build_full_weight_mask,
    smooth_and_differentiate, simulate_sensorgram,
    is_nonspecific_binder, build_pulse_mask,
)

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


@pytest.fixture(scope="module")
def data():
    return load_cxw(CXW)


@pytest.fixture(scope="module")
def sample(data):
    return data['samples'][0]


@pytest.fixture(scope="module")
def dmso(data, sample):
    return select_dmso_cal(sample['index'], data['dmso_cals'])


class TestConcentrationProfile:
    def test_envelope_shape(self, dmso, sample):
        c_func, c_raw = build_concentration_profile(dmso, sample['concentration_M'])
        assert len(c_raw) == len(dmso['time'])

    def test_envelope_non_negative(self, dmso, sample):
        _, c_raw = build_concentration_profile(dmso, sample['concentration_M'])
        assert np.all(c_raw >= 0)

    def test_envelope_scales_with_concentration(self, dmso):
        _, c1 = build_concentration_profile(dmso, 1e-6)
        _, c2 = build_concentration_profile(dmso, 2e-6)
        np.testing.assert_allclose(c2.max(), c1.max() * 2, rtol=0.01)

    def test_envelope_callable(self, dmso, sample):
        c_func, _ = build_concentration_profile(dmso, sample['concentration_M'])
        t = dmso['time']
        vals = c_func(t)
        assert len(vals) == len(t)

    def test_pulsed_shape(self, dmso, sample):
        c_func, c_raw = build_pulsed_concentration_profile(dmso, sample['concentration_M'])
        assert len(c_raw) == len(dmso['time'])

    def test_pulsed_non_negative(self, dmso, sample):
        _, c_raw = build_pulsed_concentration_profile(dmso, sample['concentration_M'])
        assert np.all(c_raw >= 0)


class TestDMSOSelection:
    def test_select_nearest(self, data):
        dmso_cals = data['dmso_cals']
        sample_idx = data['samples'][0]['index']
        dmso = select_dmso_cal(sample_idx, dmso_cals)
        assert dmso in dmso_cals

    def test_deterministic(self, data):
        sample_idx = data['samples'][0]['index']
        d1 = select_dmso_cal(sample_idx, data['dmso_cals'])
        d2 = select_dmso_cal(sample_idx, data['dmso_cals'])
        assert d1['index'] == d2['index']


class TestBlankSelection:
    def test_select_preceding(self, data):
        """Should prefer preceding blank."""
        sample = data['samples'][-1]  # late sample
        blank = select_blank(sample['index'], data['blanks'])
        assert blank['index'] < sample['index']


class TestDoubleReferencing:
    def test_returns_array_and_index(self, data, sample):
        sig, blank_idx = double_reference(sample, data['blanks'])
        assert isinstance(sig, np.ndarray)
        assert len(sig) == len(sample['time'])

    def test_positive_peak_at_rinse(self, data, sample):
        sig, _ = double_reference(sample, data['blanks'])
        t = sample['time']
        rinse = sample['markers'].get('Rinse', t[-1])
        peak_mask = (t >= rinse - 5) & (t <= rinse)
        assert sig[peak_mask].mean() > 0


class TestWeightMask:
    def test_dissociation_mask_shape(self, sample):
        t = sample['time']
        w = build_weight_mask(t, sample['markers'])
        assert w.shape == t.shape

    def test_dissociation_mask_binary(self, sample):
        t = sample['time']
        w = build_weight_mask(t, sample['markers'])
        assert set(np.unique(w)).issubset({0.0, 1.0})

    def test_dissociation_has_nonzero(self, sample):
        t = sample['time']
        w = build_weight_mask(t, sample['markers'])
        assert w.sum() > 0

    def test_full_mask_shape(self, data, sample):
        dmso = select_dmso_cal(sample['index'], data['dmso_cals'])
        w = build_full_weight_mask(sample['time'], sample['markers'], dmso)
        assert w.shape == sample['time'].shape

    def test_full_mask_more_points_than_dissoc(self, data, sample):
        """Full mask should weight buffer pulses too, so more nonzero points."""
        dmso = select_dmso_cal(sample['index'], data['dmso_cals'])
        w_dissoc = build_weight_mask(sample['time'], sample['markers'])
        w_full = build_full_weight_mask(sample['time'], sample['markers'], dmso)
        assert w_full.sum() >= w_dissoc.sum()


class TestSmoothing:
    def test_output_shapes(self, sample):
        t = sample['time']
        R_smooth, dRdt, spline = smooth_and_differentiate(t, sample['signal'])
        assert R_smooth.shape == t.shape
        assert dRdt.shape == t.shape

    def test_smooth_reduces_noise(self, sample):
        t = sample['time']
        R_smooth, _, _ = smooth_and_differentiate(t, sample['signal'])
        # Smoothed signal should have lower variance of 2nd differences
        raw_jitter = np.std(np.diff(sample['signal'], n=2))
        smooth_jitter = np.std(np.diff(R_smooth, n=2))
        assert smooth_jitter < raw_jitter


class TestSimulation:
    def test_simulate_returns_array(self, dmso, sample):
        c_func, _ = build_concentration_profile(dmso, sample['concentration_M'])
        t = sample['time']
        R = simulate_sensorgram(t, 1e4, 0.05, 100.0, c_func)
        assert R.shape == t.shape

    def test_simulate_starts_at_R0(self, dmso, sample):
        c_func, _ = build_concentration_profile(dmso, sample['concentration_M'])
        t = sample['time']
        R = simulate_sensorgram(t, 1e4, 0.05, 100.0, c_func, R0=5.0)
        np.testing.assert_allclose(R[0], 5.0, atol=0.1)

    def test_simulate_bounded_by_Rmax(self, dmso, sample):
        c_func, _ = build_concentration_profile(dmso, sample['concentration_M'])
        t = sample['time']
        Rmax = 100.0
        R = simulate_sensorgram(t, 1e4, 0.05, Rmax, c_func)
        assert R.max() <= Rmax + 1.0  # small numerical tolerance


class TestNonspecificBinder:
    def test_returns_tuple(self, sample):
        nsb, ref_dissoc = is_nonspecific_binder(sample)
        assert isinstance(nsb, bool)
        assert isinstance(ref_dissoc, float)


class TestPulseMask:
    def test_mask_shape(self, dmso):
        mask = build_pulse_mask(dmso)
        assert mask.shape == dmso['time'].shape
        assert mask.dtype == bool

    def test_baseline_is_buffer(self, dmso):
        mask = build_pulse_mask(dmso)
        t = dmso['time']
        inj = dmso['markers'].get('Injection', t[0])
        # Before injection should be all buffer
        assert np.all(mask[t < inj])
