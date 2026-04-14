"""Tests for sensofit.plotting — fit plot generation."""

import sys
import os
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.data_loader import load_cxw
from sensofit.direct_kinetics import fit_sample as dk_fit_sample
from sensofit.models import select_dmso_cal
from sensofit.plotting import plot_fit, save_fit_plots, _sanitise_filename

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
            paths = save_fit_plots([dk_result], [sample], tmpdir, mode='dk')
            assert len(paths) == 1
            assert paths[0] is not None
            assert os.path.isfile(paths[0])
            assert paths[0].endswith('.png')

    def test_skips_none_results(self, sample):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_fit_plots([None], [sample], tmpdir)
            assert paths == [None]


class TestSanitiseFilename:
    def test_keeps_safe_chars(self):
        assert _sanitise_filename('abc-123_v2.0') == 'abc-123_v2.0'

    def test_replaces_unsafe(self):
        assert ' ' not in _sanitise_filename('hello world')
        assert '/' not in _sanitise_filename('a/b/c')
