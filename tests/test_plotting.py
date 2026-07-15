"""Tests for sensofit.plotting — fit plot generation."""

import sys
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw
from sensofit.direct_kinetics import fit_sample as dk_fit_sample
from sensofit.models import select_dmso_cal
from sensofit.plotting import (
    _find_selected_blank,
    _sanitise_filename,
    plot_fit,
    save_fit_plots,
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
def dk_result(sample, data):
    return dk_fit_sample(sample, data['dmso_cals'], blanks=data['blanks'])


class TestPlotFit:
    def test_raw_channels_and_selected_blank_are_plotted(self):
        t = np.arange(4, dtype=float)
        sample = {
            'compound': 'test',
            'concentration_M': 1e-6,
            'channel': 'FC2-FC1',
            'baseline_duration_s': 1.0,
            'time': t,
            'raw_active': np.array([10.0, 12.0, 20.0, 22.0]),
            'raw_reference': np.array([5.0, 6.0, 8.0, 9.0]),
        }
        blank = {
            'index': 8,
            'baseline_duration_s': 1.0,
            'time': t,
            'signal': np.array([1.0, 2.0, 3.0, 4.0]),
        }
        result = {
            't': t,
            'signal': np.array([0.0, 1.0, 2.0, 3.0]),
            'R_fit': np.array([0.0, 1.0, 2.0, 3.0]),
            'ka': 1.0,
            'kd': 1.0,
            'KD': 1.0,
            'Rmax': 1.0,
        }

        fig = plot_fit(result, sample, blank=blank)
        assert len(fig.axes) == 2
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert labels == ['Ref. channel', 'Active channel', 'Blank']
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [-0.5, 0.5, 2.5, 3.5])
        np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), [-1.0, 1.0, 9.0, 11.0])
        np.testing.assert_allclose(fig.axes[0].lines[2].get_ydata(), [-0.5, 0.5, 1.5, 2.5])

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_selected_blank_is_matched_to_sample_context(self):
        sample = {'channel': 'FC2-FC1', 'rk_serie_id': 1}
        result = {'blank_index': 8}
        wrong_channel = {'index': 8, 'channel': 'FC3-FC1', 'rk_serie_id': 1}
        selected = {'index': 8, 'channel': 'FC2-FC1', 'rk_serie_id': 1}

        assert _find_selected_blank(result, sample, [wrong_channel, selected]) is selected

    def test_model_trace_is_labelled_fit(self):
        t = np.array([0.0, 1.0])
        result = {
            't': t, 'signal': t, 'R_fit': t,
            'ka': 1.0, 'kd': 1.0, 'KD': 1.0, 'Rmax': 1.0,
        }
        sample = {'compound': 'test', 'concentration_M': 1e-6}

        fig = plot_fit(result, sample)
        labels = fig.axes[0].get_legend_handles_labels()[1]

        assert 'Fit' in labels
        assert 'ODE fit' not in labels
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_failed_fit_is_clearly_labelled(self):
        t = np.array([0.0, 1.0])
        result = {
            't': t, 'signal': t, 'R_fit': t,
            'ka': 1.0, 'kd': 1.0, 'KD': 1.0, 'Rmax': 1.0,
            'sqrt_chi2': 0.5, 'success': False,
        }
        sample = {'compound': 'test', 'concentration_M': 1e-6}

        fig = plot_fit(result, sample)
        ax = fig.axes[0]
        labels = ax.get_legend_handles_labels()[1]
        text = '\n'.join(item.get_text() for item in ax.texts)

        assert 'Fallback estimate' in labels
        assert 'Fit' not in labels
        assert 'FIT FAILED' in text
        assert 'sqrt(chi2)' not in text
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_figure(self, dk_result, sample):
        fig = plot_fit(dk_result, sample)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_title(self, dk_result, sample):
        fig = plot_fit(dk_result, sample, title='Custom')
        ax = fig.axes[0]
        assert ax.get_title() == 'Custom'
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSaveFitPlots:
    def test_saves_pngs(self, dk_result, sample):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame([{
                'cycle_index': sample['index'],
                'channel': sample.get('channel', ''),
                'rk_serie_id': sample.get('rk_serie_id', ''),
            }])
            paths = save_fit_plots(df, [sample], [dk_result], tmpdir, mode='dk')
            assert len(paths) == 1
            assert paths[0] is not None
            assert os.path.isfile(paths[0])
            assert paths[0].endswith('.png')

    def test_skips_none_results(self, sample):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame([{
                'cycle_index': sample['index'],
                'channel': sample.get('channel', ''),
                'rk_serie_id': sample.get('rk_serie_id', ''),
            }])
            paths = save_fit_plots(df, [sample], [None], tmpdir)
            assert paths == [None]

    def test_saves_failed_fit_with_nan_residual(self):
        t = np.array([0.0, 1.0])
        sample = {
            'index': 22, 'channel': 'FC2-FC1', 'rk_serie_id': 1,
            'compound': 'test', 'concentration_M': 1e-6,
        }
        result = {
            't': t, 'signal': t, 'R_fit': t,
            'ka': 1.0, 'kd': 1.0, 'KD': 1.0, 'Rmax': 1.0,
            'sigma_residual': np.nan, 'success': False,
        }
        df = pd.DataFrame([{
            'cycle_index': 22, 'channel': 'FC2-FC1', 'rk_serie_id': 1,
            'sigma_res': np.nan, 'success': False,
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_fit_plots(df, [sample], [result], tmpdir)

            assert paths[0] is not None
            assert os.path.isfile(paths[0])


class TestSanitiseFilename:
    def test_keeps_safe_chars(self):
        assert _sanitise_filename('abc-123_v2.0') == 'abc-123_v2.0'

    def test_replaces_unsafe(self):
        assert ' ' not in _sanitise_filename('hello world')
        assert '/' not in _sanitise_filename('a/b/c')
