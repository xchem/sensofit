"""Direct Kinetics — closed-form linear solver for 1:1 Langmuir kinetics.

Reparameterize the Langmuir ODE discrepancy so it is LINEAR in
intermediate parameters k = (k1, k2, k3) = (ka·Rmax, ka, kd):

    dR/dt = k1·c(t) - k2·c(t)·R(t) - k3·R(t)

    ⟹  dR/dt + [-c, c·R, R] · k = 0
    ⟹  X·k + b = 0

Solution (with optional Tikhonov regularisation):
    k = -(X^T X + Λ')^{-1} X^T b

No iteration, deterministic, ~ms per sensorgram.

Reference: Creoptix patent US20210241847A1, Example 1 (Eq. 41–52).
"""

import numpy as np
from .models import (build_concentration_profile, select_dmso_cal,
                     smooth_and_differentiate,
                     double_reference)


def direct_kinetics_fit(t, R_smooth, dRdt, c, w=None, lambda_reg=0.0):
    """Closed-form Direct Kinetics solver for 1:1 Langmuir binding.

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    R_smooth : np.ndarray
        Smoothed binding response (pg/mm²).
    dRdt : np.ndarray
        Time derivative of the smoothed response.
    c : np.ndarray or callable
        Analyte concentration profile — array same length as t, or c(t) callable.
    w : np.ndarray or None
        Weight mask (same length as t).  None = uniform weights.
    lambda_reg : float
        Tikhonov regularisation parameter (scalar applied to all 3 diagonal
        elements of Λ').

    Returns
    -------
    result : dict
        ka, kd, Rmax, KD             — physical kinetic parameters
        Rmax_corrected               — bias-corrected Rmax (ratio-of-normals)
        k_vec                        — intermediate parameters (k1, k2, k3)
        k_cov                        — 3×3 covariance matrix of k
        k_std                        — standard deviations of k
        residuals                    — discrepancy at each time point
        sigma_residual               — residual standard deviation
        n_points                     — number of data points used
    """
    if callable(c):
        c = c(t)

    # Observation matrix  X_i = [-c(t_i),  c(t_i)·R(t_i),  R(t_i)]
    X = np.column_stack([-c, c * R_smooth, R_smooth])
    b = dRdt

    # Apply weights via row scaling  (avoids building a huge diagonal matrix)
    if w is not None:
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        bw = b * sw
        n_eff = w.sum()
    else:
        Xw = X
        bw = b
        n_eff = len(t)

    # Regularisation matrix
    Lambda = lambda_reg * np.eye(3)

    # k = -(X^T X + Λ')^{-1} X^T b
    XtX = Xw.T @ Xw + Lambda
    Xtb = Xw.T @ bw
    k = -np.linalg.solve(XtX, Xtb)

    k1, k2, k3 = k

    # ---------- Physical parameters ----------
    ka = k2
    kd = k3
    Rmax = k1 / k2 if abs(k2) > 1e-30 else np.inf
    KD = kd / ka if abs(ka) > 1e-30 else np.inf

    # ---------- Residuals & covariance ----------
    residuals = Xw @ k + bw
    p = 3
    dof = max(n_eff - p, 1)
    sigma2 = np.sum(residuals ** 2) / dof

    XtX_inv = np.linalg.inv(XtX)
    k_cov = sigma2 * XtX_inv
    k_std = np.sqrt(np.maximum(np.diag(k_cov), 0.0))

    # ---------- Bias correction for Rmax = k1/k2 ----------
    # For k1 ~ N(μ1, σ1), k2 ~ N(μ2, σ2):
    #   E[k1/k2] ≈ (μ1/μ2) · (1 + σ2²/μ2²)
    # Corrected: Rmax_c = 2·Rmax_naive − E[k1/k2]
    #                    = Rmax · (1 − σ2²/k2²)
    if abs(k2) > 1e-30 and k_std[1] < abs(k2):
        Rmax_corrected = Rmax * (1.0 - k_std[1] ** 2 / k2 ** 2)
    else:
        Rmax_corrected = Rmax

    return {
        'ka': ka,
        'kd': kd,
        'Rmax': Rmax,
        'KD': KD,
        'Rmax_corrected': Rmax_corrected,
        'k_vec': k,
        'k_cov': k_cov,
        'k_std': k_std,
        'residuals': residuals,
        'sigma_residual': np.sqrt(sigma2),
        'n_points': int(n_eff),
    }


def fit_sample(sample, dmso_cals, blanks=None, lambda_reg=0.0,
               smoothing_factor=None):
    """Fit a single sample cycle using Direct Kinetics.

    High-level wrapper that:
    1. Double-references the sample with the nearest preceding blank
       (falls back to next preceding blank if subtraction is negative).
    2. Constructs c(t) from the nearest DMSO cal.
    3. Fits the full sensorgram (Injection→RinseEnd) using the
       Direct Kinetics linear solver to obtain ka and kd directly.

    Parameters
    ----------
    sample : dict
        A sample dict from ``load_cxw()``.
    dmso_cals : list[dict]
        DMSO calibration cycles from ``load_cxw()``.
    blanks : list[dict] or None
        Blank cycles for double referencing.  If None, only baseline-
        subtraction is applied.
    lambda_reg : float
        Tikhonov regularisation parameter.
    smoothing_factor : float or None
        Smoothing parameter for the spline (None = automatic / GCV).

    Returns
    -------
    result : dict
        ka, kd, Rmax, KD, and auxiliary arrays.
    """
    t = sample['time']
    dmso = select_dmso_cal(sample['index'], dmso_cals)
    c_func, c_raw = build_concentration_profile(dmso, sample['concentration_M'])

    # --- Double referencing ---
    if blanks:
        signal_bl, blank_index = double_reference(sample, blanks)
    else:
        inj_time = sample['markers'].get('Injection', t[0])
        bl_mask = t < inj_time
        baseline = sample['signal'][bl_mask].mean() if bl_mask.any() else 0.0
        signal_bl = sample['signal'] - baseline
        blank_index = None

    # --- Full-sensorgram Direct Kinetics ---
    # Weight the entire Injection→RinseEnd window so the DK linear solver
    # sees both association (c > 0 → constrains ka) and dissociation (c = 0
    # → constrains kd) data simultaneously.
    inj_time = sample['markers'].get('Injection', t[0])
    rinse = sample['markers'].get('Rinse', t[-1])
    rinse_end = sample['markers'].get('RinseEnd', t[-1])

    w = np.zeros_like(t)
    w[(t >= inj_time) & (t <= rinse_end)] = 1.0

    R_smooth, dRdt, spline = smooth_and_differentiate(
        t, signal_bl, smoothing_factor)
    c = c_func(t)

    result = direct_kinetics_fit(t, R_smooth, dRdt, c, w=w,
                                 lambda_reg=lambda_reg)

    # Extract physical parameters directly from the solver
    ka = abs(result['k_vec'][1])   # k2 = ka, enforce positive
    kd = abs(result['k_vec'][2])   # k3 = kd, enforce positive

    # R0: peak binding response at dissociation onset
    near_rinse = (t >= rinse - 2) & (t <= rinse)
    R0 = R_smooth[near_rinse].mean() if near_rinse.any() else R_smooth.max()
    R0 = max(R0, 0.0)

    # Rmax from solver: k1 = ka·Rmax → Rmax = k1/ka
    if ka > 1e-30:
        Rmax = abs(result['k_vec'][0]) / ka
        Rmax = max(Rmax, R0 * 1.05)  # ensure Rmax >= R0
    else:
        Rmax = max(R0 + 5.0, R0 * 1.5)

    KD = kd / ka if ka > 1e-30 else np.inf

    result['ka'] = ka
    result['kd'] = kd
    result['Rmax'] = Rmax
    result['KD'] = KD
    result['Rmax_corrected'] = Rmax
    result['R0_dissoc'] = R0

    result['c_func'] = c_func
    result['c_raw'] = c_raw
    result['R_smooth'] = R_smooth
    result['dRdt'] = dRdt
    result['t'] = t
    result['signal'] = signal_bl
    result['signal_raw'] = sample['signal']
    result['blank_index'] = blank_index
    result['dmso_index'] = dmso['index']

    return result
