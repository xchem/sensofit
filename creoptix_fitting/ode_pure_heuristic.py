"""Pure-ODE fitter with heuristic initial guesses.

Fits all three Langmuir parameters (ka, kd, Rmax) simultaneously via
forward ODE integration — no Direct Kinetics dependency.

Initial guesses are derived from the signal shape:
  - kd from exponential decay during dissociation
  - ka from steady-state formula at dissociation onset
  - Rmax from peak signal × 1.5

Multi-start perturbations around the heuristic estimates improve
robustness.  Results are aggregated via median over converged fits.
"""

import numpy as np
from scipy.optimize import least_squares
from .models import (build_pulsed_concentration_profile, select_dmso_cal,
                     build_full_weight_mask, double_reference)
from .ode_core import _run_fit, aggregate_fits, BOUNDS_LO, BOUNDS_HI


def _estimate_kd(t, signal, markers, skip_s=1.0):
    """Estimate kd from exponential decay in the dissociation phase.

    Fits R(t) = R0 · exp(-kd · (t - t0)) + Rss via least_squares
    on the Rinse → RinseEnd window (skipping transport lag).

    Returns
    -------
    kd_est, R0_est, Rss_est : float
    """
    rinse = markers.get('Rinse', t[0])
    rinse_end = markers.get('RinseEnd', t[-1])

    t0 = rinse + skip_s
    mask = (t >= t0) & (t <= rinse_end)
    t_d = t[mask]
    s_d = signal[mask]

    if len(t_d) < 5:
        # Not enough dissociation data — return rough defaults
        peak = np.max(np.abs(signal))
        return 0.1, max(peak, 1.0), 0.0

    # Initial guesses for (R0, kd, Rss)
    R0_init = s_d[0]
    Rss_init = s_d[-1]
    kd_init = 0.1

    def residuals(params):
        R0, kd, Rss = params
        return s_d - (R0 * np.exp(-kd * (t_d - t0)) + Rss)

    try:
        opt = least_squares(residuals, [R0_init, kd_init, Rss_init],
                            bounds=([0, 1e-6, -1e3], [1e4, 10.0, 1e3]),
                            method='trf', max_nfev=100)
        R0_est, kd_est, Rss_est = opt.x
        kd_est = max(kd_est, 1e-5)
    except Exception:
        kd_est = 0.1
        R0_est = max(abs(s_d[0]), 1.0)
        Rss_est = 0.0

    return kd_est, R0_est, Rss_est


def _estimate_ka_Rmax(kd_est, R0_est, t, signal, c_func, markers):
    """Estimate ka and Rmax from steady-state at dissociation onset.

    At steady state: dR/dt ≈ 0, so ka·c·(Rmax − R) = kd·R
        → ka = kd·R / (c·(Rmax − R))

    Rmax is estimated as peak signal × 1.5.
    """
    rinse = markers.get('Rinse', t[-1])
    peak = max(R0_est, np.max(signal))
    Rmax_est = peak * 1.5

    # Concentration just before rinse
    c_plateau = float(c_func(rinse - 2.0))

    if c_plateau > 0 and Rmax_est > R0_est:
        ka_est = kd_est * R0_est / (c_plateau * (Rmax_est - R0_est))
    else:
        ka_est = 1e4  # fallback

    ka_est = np.clip(ka_est, BOUNDS_LO[0], BOUNDS_HI[0])
    Rmax_est = np.clip(Rmax_est, BOUNDS_LO[2], BOUNDS_HI[2])

    return ka_est, Rmax_est


def _perturb_starts(ka0, kd0, Rmax0, n_starts):
    """Generate multi-start points via log-normal perturbations.

    The first start is always the unperturbed heuristic estimate.
    """
    rng = np.random.default_rng()
    starts = [np.array([ka0, kd0, Rmax0])]
    for _ in range(n_starts - 1):
        log_perturb = rng.normal(0, 0.5, size=3)
        p = np.array([ka0, kd0, Rmax0]) * np.exp(log_perturb)
        starts.append(np.clip(p, BOUNDS_LO, BOUNDS_HI))
    return starts


def ode_fit(t, signal, c_func, w, markers, n_starts=5, method='trf'):
    """Fit 1:1 Langmuir via pure ODE with heuristic initial guesses.

    Parameters
    ----------
    t, signal : np.ndarray
        Time and double-referenced signal.
    c_func : callable
        c(t) → concentration (M), pulsed profile.
    w : np.ndarray
        Weight mask (1 during buffer pulses + dissociation, 0 elsewhere).
    markers : dict
        Cycle markers with 'Rinse', 'RinseEnd', etc.
    n_starts : int
        Number of multi-start points.
    method : str
        Optimizer: 'trf', 'dogbox', 'lm', 'Nelder-Mead', 'L-BFGS-B'.

    Returns
    -------
    result : dict
        Fit results with ka, kd, Rmax, KD, R_fit, etc.
    """
    # Heuristic initial estimates
    kd_est, R0_est, Rss_est = _estimate_kd(t, signal, markers)
    ka_est, Rmax_est = _estimate_ka_Rmax(kd_est, R0_est, t, signal,
                                          c_func, markers)

    # Generate starting points
    starts = _perturb_starts(ka_est, kd_est, Rmax_est, n_starts)

    # Run fits
    fits = []
    for p0 in starts:
        result = _run_fit(p0, t, signal, c_func, w, method=method)
        if result is not None:
            fits.append(result)

    out = aggregate_fits(fits, t, signal, c_func, w)
    out['n_starts'] = n_starts
    out['heuristic_ka'] = ka_est
    out['heuristic_kd'] = kd_est
    out['heuristic_Rmax'] = Rmax_est
    return out


def fit_sample(sample, dmso_cals, blanks=None, n_starts=5, method='trf'):
    """Fit a single sample using pure ODE with heuristic seeds.

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
    n_starts : int
        Number of multi-start points.
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
    result = ode_fit(t, signal, c_func, w, sample['markers'],
                     n_starts=n_starts, method=method)

    result['t'] = t
    result['signal'] = signal
    result['c_func'] = c_func
    result['c_raw'] = c_raw
    result['dmso_index'] = dmso['index']
    result['blank_index'] = blank_index

    return result
