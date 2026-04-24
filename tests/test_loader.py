"""Tests for sensofit.data_loader — load_cxw() parser."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


@pytest.fixture(scope="module")
def data():
    """Load experiment data once for all tests in this module."""
    return load_cxw(CXW)


class TestConfig:
    def test_config_keys(self, data):
        cfg = data['config']
        assert 'active_fc' in cfg
        assert 'reference_fc' in cfg
        assert 'active_channel' in cfg
        assert 'reference_channel' in cfg

    def test_active_and_reference_differ(self, data):
        cfg = data['config']
        assert cfg['active_fc'] != cfg['reference_fc']
        assert cfg['active_channel'] != cfg['reference_channel']

    def test_channel_mapping(self, data):
        cfg = data['config']
        # FC n → channel 5 + 2n
        assert cfg['active_channel'] == 5 + 2 * cfg['active_fc']
        assert cfg['reference_channel'] == 5 + 2 * cfg['reference_fc']


class TestSamples:
    def test_samples_not_empty(self, data):
        assert len(data['samples']) > 0

    def test_sample_has_required_keys(self, data):
        s = data['samples'][0]
        for key in ['index', 'cycle_type', 'guid', 'compound',
                     'concentration_M', 'markers', 'time', 'signal',
                     'raw_active', 'raw_reference']:
            assert key in s, f"Missing key: {key}"

    def test_sample_arrays_same_length(self, data):
        s = data['samples'][0]
        n = len(s['time'])
        assert len(s['signal']) == n
        assert len(s['raw_active']) == n
        assert len(s['raw_reference']) == n

    def test_signal_is_reference_subtracted(self, data):
        s = data['samples'][0]
        expected = s['raw_active'] - s['raw_reference']
        np.testing.assert_allclose(s['signal'], expected)

    def test_sample_has_markers(self, data):
        s = data['samples'][0]
        assert 'Injection' in s['markers']
        assert 'Rinse' in s['markers']

    def test_marker_ordering(self, data):
        s = data['samples'][0]
        m = s['markers']
        assert m['Injection'] < m['Rinse']
        if 'RinseEnd' in m:
            assert m['Rinse'] < m['RinseEnd']

    def test_concentration_positive(self, data):
        for s in data['samples']:
            assert s['concentration_M'] > 0, f"Sample {s['compound']} has zero concentration"

    def test_time_monotonic(self, data):
        s = data['samples'][0]
        assert np.all(np.diff(s['time']) > 0)


class TestDMSOCals:
    def test_dmso_cals_not_empty(self, data):
        assert len(data['dmso_cals']) > 0

    def test_dmso_cycle_type(self, data):
        for d in data['dmso_cals']:
            assert d['cycle_type'] == 'DMSO Cal.'

    def test_dmso_has_signal_data(self, data):
        d = data['dmso_cals'][0]
        assert len(d['time']) > 0
        assert len(d['signal']) == len(d['time'])


class TestBlanks:
    def test_blanks_not_empty(self, data):
        assert len(data['blanks']) > 0

    def test_blank_cycle_type(self, data):
        for b in data['blanks']:
            assert b['cycle_type'] == 'Blank'


class TestAllCycles:
    def test_all_cycles_has_entries(self, data):
        assert len(data['all_cycles']) > 0

    def test_cycle_types_present(self, data):
        types = {c['cycle_type'] for c in data['all_cycles']}
        assert 'Sample' in types or 'ControlSample' in types
        assert 'Blank' in types
        assert 'DMSO Cal.' in types

    def test_indices_unique(self, data):
        indices = [c['index'] for c in data['all_cycles']]
        assert len(indices) == len(set(indices))


class TestOtherCycles:
    def test_other_cycles_key_exists(self, data):
        assert 'other_cycles' in data
        assert isinstance(data['other_cycles'], list)

    def test_other_cycles_have_signal_data(self, data):
        # If any non-fitting cycle was loaded, it must carry the same
        # signal arrays as samples / blanks / dmso_cals.
        for c in data['other_cycles']:
            assert c['cycle_type'] not in (
                'Sample', 'ControlSample', 'DMSO Cal.', 'Blank')
            for k in ('time', 'signal', 'raw_active', 'raw_reference',
                      'channel'):
                assert k in c
