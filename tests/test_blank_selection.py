"""Synthetic tests for blank quality control and selection."""

import numpy as np

from sensofit.models import (_is_blank_valid, _is_blank_valid_legacy,
                             select_blank)


def _blank(index=10, signal=None):
    time = np.arange(0.0, 71.0)
    if signal is None:
        signal = np.zeros_like(time)
    return {
        'index': index,
        'channel': 'FC2-FC1',
        'rk_serie_id': 1,
        'time': time,
        'signal': np.asarray(signal, dtype=float),
        'baseline_duration_s': 20.0,
        'markers': {
            'Injection': 25.0,
            'Rinse': 40.0,
            'RinseEnd': 70.0,
        },
    }


def test_stable_flat_blank_is_valid():
    assert _is_blank_valid(_blank())


def test_unsettled_baseline_is_invalid():
    signal = np.zeros(71)
    signal[:10] = 3.0
    assert not _is_blank_valid(_blank(signal=signal))


def test_three_moderately_negative_points_are_allowed():
    signal = np.zeros(71)
    signal[30:33] = -2.0
    assert _is_blank_valid(_blank(signal=signal))


def test_three_points_strictly_below_minus_five_are_invalid():
    signal = np.zeros(71)
    signal[30:33] = -5.01
    assert not _is_blank_valid(_blank(signal=signal))


def test_three_points_at_minus_five_are_allowed():
    signal = np.zeros(71)
    signal[30:33] = -5.0
    assert _is_blank_valid(_blank(signal=signal))


def test_injection_phase_mean_below_minus_two_is_invalid():
    signal = np.zeros(71)
    signal[25:40] = -2.01
    assert not _is_blank_valid(_blank(signal=signal))


def test_short_negative_period_with_phase_mean_above_minus_two_is_allowed():
    signal = np.zeros(71)
    signal[30:39] = -2.01
    assert _is_blank_valid(_blank(signal=signal))


def test_dissociation_phase_mean_below_minus_two_is_invalid():
    signal = np.zeros(71)
    signal[40:71] = -2.01
    assert not _is_blank_valid(_blank(signal=signal))


def test_single_large_negative_trough_is_not_a_sustained_response():
    signal = np.zeros(71)
    signal[65] = -10.0
    assert _is_blank_valid(_blank(signal=signal))


def test_meaningful_negative_dissociation_drift_is_invalid():
    signal = np.zeros(71)
    signal[40:] = np.linspace(2.0, -1.0, 31)
    assert not _is_blank_valid(_blank(signal=signal))


def test_selection_falls_back_from_invalid_to_previous_blank():
    valid = _blank(index=10)
    invalid_signal = np.zeros(71)
    invalid_signal[30:33] = -5.01
    invalid = _blank(index=20, signal=invalid_signal)

    selected = select_blank(25, [valid, invalid])

    assert selected['index'] == 10


def test_legacy_selection_preserves_previous_blank_choice():
    valid = _blank(index=10)
    negative_signal = np.zeros(71)
    negative_signal[25:40] = -2.01
    newly_invalid = _blank(index=20, signal=negative_signal)

    assert _is_blank_valid_legacy(newly_invalid)
    assert select_blank(25, [valid, newly_invalid],
                        selection='legacy')['index'] == 20
    assert select_blank(25, [valid, newly_invalid],
                        selection='current')['index'] == 10


def test_linear_drifted_blank_is_rejected_without_rescue():
    drifting = _blank(index=20, signal=-0.05 * np.arange(71))

    assert not _is_blank_valid(drifting)
    assert select_blank(25, [drifting], selection='current') is None


def test_dissociation_only_drift_is_not_corrected():
    signal = np.zeros(71)
    signal[40:] = np.linspace(2.0, -3.0, 31)
    drifting = _blank(index=20, signal=signal)

    assert select_blank(25, [drifting], selection='current') is None


def test_unknown_blank_selection_mode_is_rejected():
    with np.testing.assert_raises_regex(ValueError, 'selection must be'):
        select_blank(25, [_blank()], selection='unknown')
