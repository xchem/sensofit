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
                     smooth_and_differentiate, build_weight_mask,
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
    3. Fits the dissociation phase only (Rinse→RinseEnd) using the
       Direct Kinetics linear solver to obtain kd.
    4. Estimates ka and Rmax from the dissociation onset and c(t).

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

    # --- Smooth the dissociation phase for Direct Kinetics ---
    # Use the dissociation-only weight mask: fit kd from clean data
    w = build_weight_mask(t, sample['markers'])

    R_smooth, dRdt, spline = smooth_and_differentiate(
        t, signal_bl, smoothing_factor)
    c = c_func(t)

    # Direct Kinetics on dissociation only → gives kd
    result = direct_kinetics_fit(t, R_smooth, dRdt, c, w=w,
                                 lambda_reg=lambda_reg)

    # --- Estimate ka and Rmax from signal shape ---
    # kd is reliable from the dissociation fit (k3)
    kd = result['k_vec'][2]

    # R0: peak binding response at dissociation onset
    rinse = sample['markers'].get('Rinse', t[-1])
    inj_time = sample['markers'].get('Injection', t[0])
    near_rinse = (t >= rinse - 2) & (t <= rinse)
    R0 = R_smooth[near_rinse].mean() if near_rinse.any() else R_smooth.max()

    C = sample['concentration_M']
    ka = None

    # --- Primary: estimate ka from observed association rate (kobs) ---
    # For 1:1 Langmuir: R(t) ≈ R_eq·(1 - exp(-kobs·t))
    # kobs = ka·c + kd  →  ka = (kobs - kd) / c
    # Estimate kobs from time to reach 50% of R0.
    if R0 > 0.5 and C > 0:
        assoc_mask = (t >= inj_time) & (t <= rinse)
        R_assoc = R_smooth[assoc_mask]
        t_assoc = t[assoc_mask]
        above_half = np.where(R_assoc >= 0.5 * R0)[0]
        if len(above_half) > 0:
            t_half = t_assoc[above_half[0]] - inj_time
            if t_half > 0.5:
                kobs = np.log(2) / t_half
                if kobs > abs(kd):
                    ka = (kobs - abs(kd)) / C

    # --- Fallback: steady-state heuristic with derivative-based saturation ---
    if ka is None or ka <= 0:
        c_plateau = c_func(rinse - 1)

        # Estimate saturation fraction from how close the signal is to
        # equilibrium at rinse: if dR/dt ≈ 0, saturation is high (Rmax ≈ R0);
        # if dR/dt is still large, saturation is low (Rmax >> R0).
        # f_sat = R0 / Rmax, clamped to [0.3, 0.95].
        if R0 > 1.0 and abs(kd) > 1e-8:
            dRdt_near = dRdt[near_rinse].mean() if near_rinse.any() else 0.0
            # At equilibrium: dR/dt = 0 = ka*c*(Rmax-R0) - kd*R0
            # Departure: dR/dt = ka*c*(Rmax-R0) - kd*R0
            # Fractional departure: η = dR/dt / (kd*R0)
            # η = 0 → equilibrium, η > 0 → still rising (far from saturation)
            eta = max(dRdt_near / (abs(kd) * R0), 0.0)
            f_sat = np.clip(1.0 / (1.0 + eta), 0.3, 0.95)
        else:
            f_sat = 0.5  # default: assume 50% saturation

        Rmax = R0 / f_sat if f_sat > 0 else R0 + 5.0
        Rmax = max(Rmax, R0 + 1.0)  # ensure Rmax > R0

        if c_plateau > 0 and abs(kd) > 0 and (Rmax - R0) > 0:
            ka = abs(kd) * R0 / (c_plateau * (Rmax - R0))
        else:
            ka = 1e4

    # Rmax from steady-state: R0 = Rmax · c/(c+KD) = Rmax · ka·c/(ka·c+kd)
    # → Rmax = R0 · (ka·c + kd) / (ka·c)
    if ka > 0 and C > 0:
        Rmax = R0 * (ka * C + abs(kd)) / (ka * C)
    else:
        Rmax = max(R0 * 1.2, R0 + 5.0)

    KD = abs(kd) / ka if ka > 0 else np.inf

    # Overwrite the linear solver's unreliable ka/Rmax with these estimates
    result['ka'] = ka
    result['kd'] = abs(kd)  # enforce positive
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
