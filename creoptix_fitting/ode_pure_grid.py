"""Pure-ODE fitter with grid-based initial guesses.

Fits all three Langmuir parameters (ka, kd, Rmax) simultaneously via
forward ODE integration — no Direct Kinetics dependency.

Initial guesses are drawn from a fixed Cartesian grid:
  ka   ∈ {1e2, 1e4, 1e6}
  kd   ∈ {0.01, 0.1, 1.0}
  Rmax ∈ {peak×1.2, peak×2.0, peak×5.0}

This gives 27 starting points, providing broad coverage of the
parameter space without any randomness.  Results are aggregated
via median over converged fits.
"""

import numpy as np
from itertools import product
from .models import (build_pulsed_concentration_profile, select_dmso_cal,
                     build_full_weight_mask, double_reference)
from .ode_core import _run_fit, aggregate_fits, BOUNDS_LO, BOUNDS_HI


KA_GRID = [1e2, 1e4, 1e6]
KD_GRID = [0.01, 0.1, 1.0]
RMAX_MULT = [1.2, 2.0, 5.0]


def _build_grid(peak_signal):
    """Build a 27-point Cartesian grid of starting points.

    Parameters
    ----------
    peak_signal : float
        Peak absolute signal value, used to scale Rmax guesses.

    Returns
    -------
    starts : list[np.ndarray]
        Each element is [ka, kd, Rmax].
    """
    peak = max(abs(peak_signal), 1.0)
    rmax_vals = [peak * m for m in RMAX_MULT]
    starts = []
    for ka, kd, rmax in product(KA_GRID, KD_GRID, rmax_vals):
        p = np.clip([ka, kd, rmax], BOUNDS_LO, BOUNDS_HI)
        starts.append(p)
    return starts


def ode_fit(t, signal, c_func, w, markers, method='trf'):
    """Fit 1:1 Langmuir via pure ODE with grid initial guesses.

    Parameters
    ----------
    t, signal : np.ndarray
        Time and double-referenced signal.
    c_func : callable
        c(t) → concentration (M), pulsed profile.
    w : np.ndarray
        Weight mask.
    markers : dict
        Cycle markers.
    method : str
        Optimizer: 'trf', 'dogbox', 'lm', 'Nelder-Mead', 'L-BFGS-B'.

    Returns
    -------
    result : dict
        Fit results with ka, kd, Rmax, KD, R_fit, etc.
    """
    peak_signal = np.max(np.abs(signal))
    starts = _build_grid(peak_signal)

    fits = []
    for p0 in starts:
        result = _run_fit(p0, t, signal, c_func, w, method=method)
        if result is not None:
            fits.append(result)

    out = aggregate_fits(fits, t, signal, c_func, w)
    out['n_starts'] = len(starts)
    return out


def fit_sample(sample, dmso_cals, blanks=None, method='trf'):
    """Fit a single sample using pure ODE with grid seeds.

    Handles all preprocessing (double referencing, c(t) construction,
    weight mask) — no Direct Kinetics step.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    dmso_cals : list[dict]
        DMSO calibration cycles.
    blanks : list[dict] or None
        Blank cycles for double referencing.
    method : str
        Optimizer method.

    Returns
    -------
    result : dict
        Full fit results plus preprocessed arrays.
    """
    t = sample['time']

    # Double referencing
    if blanks:
        signal, blank_index = double_reference(sample, blanks)
    else:
        inj = sample['markers'].get('Injection', t[0])
        bl_mask = t < inj
        baseline = sample['signal'][bl_mask].mean() if bl_mask.any() else 0.0
        signal = sample['signal'] - baseline
        blank_index = None

    # Pulsed c(t)
    dmso = select_dmso_cal(sample['index'], dmso_cals)
    c_func, c_raw = build_pulsed_concentration_profile(
        dmso, sample['concentration_M'])

    # Weight mask
    w = build_full_weight_mask(t, sample['markers'], dmso)

    # Fit
    result = ode_fit(t, signal, c_func, w, sample['markers'], method=method)

    result['t'] = t
    result['signal'] = signal
    result['c_func'] = c_func
    result['c_raw'] = c_raw
    result['dmso_index'] = dmso['index']
    result['blank_index'] = blank_index

    return result
