"""Tests for creoptix_fitting.batch — batch fitting and QC flagging."""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from creoptix_fitting.batch import batch_fit, flag_poor_fits

CXW = os.path.join(os.path.dirname(__file__), '..',
                    '20250826_DENV-2 NS2B3 Binding Assay.cxw')


class TestBatchFitDK:
    @pytest.fixture(scope="class")
    def dk_results(self):
        df, data = batch_fit(CXW, mode='dk', progress=False)
        return df, data

    def test_returns_dataframe(self, dk_results):
        df, _ = dk_results
        assert isinstance(df, pd.DataFrame)

    def test_all_samples_present(self, dk_results):
        df, data = dk_results
        assert len(df) == len(data['samples'])

    def test_required_columns(self, dk_results):
        df, _ = dk_results
        for col in ['compound', 'concentration_M', 'ka', 'kd', 'KD',
                     'KD_uM', 'Rmax', 'fit_mode', 'success']:
            assert col in df.columns, f"Missing column: {col}"

    def test_fit_mode_dk(self, dk_results):
        df, _ = dk_results
        non_nsb = df[df['fit_mode'] != 'nsb']
        assert (non_nsb['fit_mode'] == 'dk').all()

    def test_some_succeed(self, dk_results):
        df, _ = dk_results
        assert df['success'].sum() > 0


class TestFlagPoorFits:
    @pytest.fixture(scope="class")
    def flagged_df(self):
        df, _ = batch_fit(CXW, mode='dk', progress=False)
        return flag_poor_fits(df)

    def test_flag_columns_added(self, flagged_df):
        assert 'flag' in flagged_df.columns
        assert 'flag_reason' in flagged_df.columns

    def test_flag_is_boolean(self, flagged_df):
        assert flagged_df['flag'].dtype == bool

    def test_nsb_flagged(self, flagged_df):
        nsb = flagged_df[flagged_df['nonspecific']]
        if len(nsb) > 0:
            assert nsb['flag'].all()

    def test_unflagged_have_empty_reason(self, flagged_df):
        unflagged = flagged_df[~flagged_df['flag']]
        assert (unflagged['flag_reason'] == '').all()
