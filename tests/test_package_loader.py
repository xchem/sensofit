"""Tests for sensofit.package_loader — re-ingestion of exported packages."""

import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw
from sensofit.dataexporter import export_package
from sensofit.package_loader import (
    list_experiments,
    load_package,
    load_experiment,
)

REPO = os.path.join(os.path.dirname(__file__), '..')
CXW_WITH_EVALS = os.path.join(REPO, '20260318_EV71 2A Binding assay.cxw')
CXW_NO_EVALS = os.path.join(REPO, '20260323_EV71 2A Binding assay.cxw')

# CSV truncation at :.6g loses ~6 significant digits relative to load_cxw,
# so signal arrays match within ~1e-6 relative tolerance.
SIG_RTOL = 1e-5
SIG_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------

def _pkg_zip(tmp_path, cxw, name='pkg'):
    out = tmp_path / f'{name}.zip'
    return export_package([cxw], str(out), package_name=name)


@pytest.fixture(scope='module')
def pkg_with_evals(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('pkg_evals')
    return _pkg_zip(tmp, CXW_WITH_EVALS, name='evals_pkg')


@pytest.fixture(scope='module')
def pkg_no_evals(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('pkg_noevals')
    return _pkg_zip(tmp, CXW_NO_EVALS, name='noevals_pkg')


@pytest.fixture(scope='module')
def ref_with_evals():
    return load_cxw(CXW_WITH_EVALS)


@pytest.fixture(scope='module')
def ref_no_evals():
    return load_cxw(CXW_NO_EVALS)


def _by_key(entries):
    """Index a sample / blank / dmso list by (cycle_index, channel)."""
    return {(e['index'], e.get('channel', '')): e for e in entries}


# ---------------------------------------------------------------------------
# list_experiments
# ---------------------------------------------------------------------------

class TestListExperiments:
    def test_returns_one_folder_per_cxw(self, pkg_with_evals):
        names = list_experiments(pkg_with_evals)
        assert len(names) == 1
        assert names[0]  # non-empty

    def test_works_on_unzipped_directory(self, pkg_with_evals, tmp_path):
        unz = tmp_path / 'unz'
        unz.mkdir()
        with zipfile.ZipFile(pkg_with_evals) as zf:
            zf.extractall(unz)
        names_zip = list_experiments(pkg_with_evals)
        names_dir = list_experiments(str(unz))
        assert names_zip == names_dir


# ---------------------------------------------------------------------------
# Top-level shape parity with load_cxw
# ---------------------------------------------------------------------------

class TestRoundTripShape:
    def test_top_level_keys_match(self, pkg_with_evals, ref_with_evals):
        loaded = load_package(pkg_with_evals)
        assert set(loaded.keys()) == set(ref_with_evals.keys())

    def test_counts_match(self, pkg_with_evals, ref_with_evals):
        loaded = load_package(pkg_with_evals)
        assert len(loaded['samples']) == len(ref_with_evals['samples'])
        assert len(loaded['dmso_cals']) == len(ref_with_evals['dmso_cals'])
        assert len(loaded['blanks']) == len(ref_with_evals['blanks'])
        assert len(loaded['other_cycles']) == \
            len(ref_with_evals['other_cycles'])
        # all_cycles in load_cxw includes metadata-only entries
        # (e.g. some priming cycles with no HDF5 data); the package
        # only retains cycles that actually have signal CSVs, so its
        # all_cycles equals the union of the four buckets divided by
        # the number of channels.
        n_chan = len(loaded['config']['channel_pairs'])
        loaded_cycle_units = (len(loaded['samples'])
                              + len(loaded['dmso_cals'])
                              + len(loaded['blanks'])
                              + len(loaded['other_cycles']))
        assert len(loaded['all_cycles']) * n_chan == loaded_cycle_units

    def test_config_round_trips(self, pkg_with_evals, ref_with_evals):
        loaded = load_package(pkg_with_evals)
        rcfg = ref_with_evals['config']
        lcfg = loaded['config']
        assert lcfg['active_fc'] == rcfg['active_fc']
        assert lcfg['reference_fc'] == rcfg['reference_fc']
        assert lcfg['active_channel'] == rcfg['active_channel']
        assert lcfg['reference_channel'] == rcfg['reference_channel']
        assert len(lcfg['channel_pairs']) == len(rcfg['channel_pairs'])


# ---------------------------------------------------------------------------
# Per-cycle field equivalence + signal arrays
# ---------------------------------------------------------------------------

class TestSampleEquivalence:
    def test_signals_match_within_csv_truncation(self, pkg_with_evals,
                                                  ref_with_evals):
        loaded = load_package(pkg_with_evals)
        l_idx = _by_key(loaded['samples'])
        for s in ref_with_evals['samples']:
            key = (s['index'], s.get('channel', ''))
            assert key in l_idx, f'sample missing in package: {key}'
            ls = l_idx[key]
            assert ls['time'].shape == s['time'].shape
            np.testing.assert_allclose(ls['time'], s['time'],
                                        rtol=SIG_RTOL, atol=SIG_ATOL)
            np.testing.assert_allclose(ls['signal'], s['signal'],
                                        rtol=SIG_RTOL, atol=SIG_ATOL)

    def test_metadata_fields_match(self, pkg_with_evals, ref_with_evals):
        loaded = load_package(pkg_with_evals)
        l_idx = _by_key(loaded['samples'])
        for s in ref_with_evals['samples']:
            ls = l_idx[(s['index'], s.get('channel', ''))]
            assert ls['compound'] == s['compound']
            assert ls['cycle_type'] == s['cycle_type']
            assert ls['concentration_M'] == pytest.approx(
                s['concentration_M'], rel=1e-9, abs=1e-30)
            assert ls['mw'] == pytest.approx(s['mw'])

    def test_all_cycles_ordered_by_index(self, pkg_with_evals,
                                          ref_with_evals):
        loaded = load_package(pkg_with_evals)
        loaded_order = [c['index'] for c in loaded['all_cycles']]
        assert loaded_order == sorted(loaded_order)
        # Every loaded cycle must exist in the reference at the same index.
        ref_idx = {c['index'] for c in ref_with_evals['all_cycles']}
        for i in loaded_order:
            assert i in ref_idx


# ---------------------------------------------------------------------------
# Channel filter parity
# ---------------------------------------------------------------------------

class TestChannelFilter:
    def test_single_channel_filter_matches_load_cxw(self, pkg_with_evals):
        # Pick whatever active FC is present in the file.
        full = load_package(pkg_with_evals)
        fc = full['config']['channel_pairs'][0]['active_fc']
        loaded = load_package(pkg_with_evals, channels=[fc])
        ref = load_cxw(CXW_WITH_EVALS, channels=[fc])
        assert len(loaded['samples']) == len(ref['samples'])
        for p in loaded['config']['channel_pairs']:
            assert p['active_fc'] == fc

    def test_invalid_channel_raises(self, pkg_with_evals):
        with pytest.raises(ValueError):
            load_package(pkg_with_evals, channels=[99])


# ---------------------------------------------------------------------------
# Evaluations (kinetic fits) round-trip
# ---------------------------------------------------------------------------

class TestEvaluations:
    def test_no_evals_when_cxw_has_none(self, pkg_no_evals, ref_no_evals):
        loaded = load_package(pkg_no_evals)
        assert loaded['evaluations'] == []
        assert ref_no_evals['evaluations'] == []

    def test_evals_round_trip_values(self, pkg_with_evals, ref_with_evals):
        loaded = load_package(pkg_with_evals)
        ref_evs = ref_with_evals['evaluations']
        assert ref_evs, 'sanity: reference CXW should have evaluations'
        assert len(loaded['evaluations']) == len(ref_evs)

        # Match by (cycle_index, channel label)
        def key(e):
            return (e['cycle_index'], e.get('channel', ''))
        l_idx = {key(e): e for e in loaded['evaluations']}

        for re in ref_evs:
            le = l_idx.get(key(re))
            assert le is not None, f'missing eval for {key(re)}'
            for fld in ('ka', 'kd', 'KD', 'Rmax', 'sqrt_chi2'):
                rv, lv = re.get(fld), le.get(fld)
                if rv is None:
                    assert lv is None
                else:
                    # CSV uses %.6g for these values
                    assert lv == pytest.approx(rv, rel=1e-5, abs=1e-30)
            assert le['active_fc'] == re['active_fc']
            assert le['reference_fc'] == re['reference_fc']
            assert le['compound'] == re['compound']


# ---------------------------------------------------------------------------
# Non-fitting (priming / conditioning / regeneration) round-trip
# ---------------------------------------------------------------------------

class TestOtherCycles:
    def test_other_cycles_present(self, ref_with_evals):
        # The EV71-2A binding assay contains regeneration cycles; if
        # this assertion ever fires, pick a different fixture CXW.
        assert len(ref_with_evals['other_cycles']) > 0

    def test_other_cycles_round_trip_count(self, pkg_with_evals,
                                            ref_with_evals):
        loaded = load_package(pkg_with_evals)
        assert len(loaded['other_cycles']) == \
            len(ref_with_evals['other_cycles'])

    def test_other_cycles_signals_round_trip(self, pkg_with_evals,
                                              ref_with_evals):
        loaded = load_package(pkg_with_evals)
        l_idx = _by_key(loaded['other_cycles'])
        for s in ref_with_evals['other_cycles']:
            ls = l_idx[(s['index'], s.get('channel', ''))]
            np.testing.assert_allclose(ls['signal'], s['signal'],
                                        rtol=SIG_RTOL, atol=SIG_ATOL)
            assert ls['cycle_type'] == s['cycle_type']
            # Non-fitting cycle_types should not collide with the four
            # fitting buckets.
            assert ls['cycle_type'] not in (
                'Sample', 'ControlSample', 'DMSO Cal.', 'Blank')


# ---------------------------------------------------------------------------
# Path forms (zip vs unzipped directory) and dispatcher
# ---------------------------------------------------------------------------

class TestPathForms:
    def test_directory_input_matches_zip(self, pkg_with_evals, tmp_path):
        unz = tmp_path / 'unz'
        unz.mkdir()
        with zipfile.ZipFile(pkg_with_evals) as zf:
            zf.extractall(unz)
        d = load_package(str(unz))
        z = load_package(pkg_with_evals)
        assert len(d['samples']) == len(z['samples'])
        if d['samples']:
            np.testing.assert_array_equal(d['samples'][0]['signal'],
                                           z['samples'][0]['signal'])

    def test_load_experiment_dispatches_cxw(self):
        d = load_experiment(CXW_NO_EVALS)
        assert 'samples' in d and 'config' in d

    def test_load_experiment_dispatches_zip(self, pkg_with_evals):
        d = load_experiment(pkg_with_evals)
        assert 'samples' in d and 'config' in d
