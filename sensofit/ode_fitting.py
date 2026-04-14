"""ODE-based fitting for 1:1 Langmuir kinetics.

Simulates the full sensorgram (including pulsed association via c(t))
and computes weighted residuals over the entire time course:
  - w = 1 during buffer pulses in the association phase (clean signal)
  - w = 1 during final dissociation (Rinse → RinseEnd)
  - w = 0 during analyte pulses (RI bulk artefacts) and baseline

Uses multi-start TRF with median aggregation: the optimizer is seeded
from multiple perturbed starting points and the median estimate over
converged fits is taken as the final result.

Initialised from Direct Kinetics estimates; refines ka, kd, Rmax.
"""

import numpy as np
from scipy.optimize import least_squares
from .models import (build_pulsed_concentration_profile, select_dmso_cal,
                     build_full_weight_mask, simulate_sensorgram,
                     trim_to_fit_window)
from .direct_kinetics import fit_sample as dk_fit_sample


def _residuals(params, t_dissoc, signal_dissoc, t0):
    """Residuals for dissociation-only fit with baseline offset.

    During dissociation c(t)=0, the Langmuir ODE has an exact solution:
        R(t) = R0 · exp(-kd · (t - t0)) + Rss

    Parameters: (R0, kd, Rss) — R0 is amplitude at t0, Rss is baseline.
    t0 is the effective start of clean dissociation (after transport lag).
    """
    R0, kd, Rss = params
    R_model = R0 * np.exp(-kd * (t_dissoc - t0)) + Rss
    return signal_dissoc - R_model


def _residuals_full(params, t, signal, c_func, w):
    """Full ODE residuals (weighted).

    Optimises (ka, kd, Rmax) simultaneously.
    """
    ka, kd, Rmax = params
    R_sim = simulate_sensorgram(t, ka, kd, Rmax, c_func, R0=0.0)
    return w * (signal - R_sim)


def _solve_R0_Rss(kd, t_dissoc, signal_dissoc, t0):
    """Closed-form linear regression for R0 and Rss given fixed kd.

    Model: signal = R0 · exp(-kd·(t - t0)) + Rss
    This is linear in (R0, Rss) when kd is fixed.
    """
    X = np.exp(-kd * (t_dissoc - t0))
    A = np.column_stack([X, np.ones_like(X)])
    # Ordinary least squares: [R0, Rss] = (AᵀA)⁻¹ Aᵀ signal
    params, residuals_sum, _, _ = np.linalg.lstsq(A, signal_dissoc, rcond=None)
    R0, Rss = params
    return R0, Rss


def ode_fit(t, signal, c_func, w, markers, ka0, kd0, Rmax0,
            n_starts=1, rng_seed=None, skip_s=1.0):
    """Fit 1:1 Langmuir parameters via DK-seeded ODE refinement.

    Three-phase approach:
      1. Closed-form linear regression for (R0, Rss) using kd from DK,
         from the final dissociation phase (Rinse → RinseEnd).
         The first ``skip_s`` seconds are excluded for transport lag.
      2. Derive ka from steady-state at dissociation onset.
      3. Multi-start ODE refinement for (ka, kd, Rmax) seeded from DK.
         Residuals are computed over the full sensorgram using ``w``
         (buffer pulses during association + final dissociation).

    Parameters
    ----------
    t, signal, c_func, w : arrays / callable
        Time, double-referenced signal, concentration profile, weight mask.
        The weight mask should be 1 during buffer pulses (association) and
        during dissociation, 0 during analyte pulses and baseline.
    markers : dict
        Cycle markers with 'Rinse' and 'RinseEnd' for dissociation extraction.
    ka0, kd0, Rmax0 : float
        Initial parameter estimates (from Direct Kinetics).
    n_starts : int
        Number of starting points for ODE refinement.
    rng_seed : int or None
        Random seed for reproducibility.  None (default) = non-reproducible.
    skip_s : float
        Seconds to skip after rinse onset to avoid transport lag.
    """
    kd_final = max(kd0, 1e-5)  # kd pinned from DK

    # ---- Extract dissociation-only data (from markers, not weight mask) ----
    rinse = markers.get('Rinse', 0)
    rinse_end = markers.get('RinseEnd', t[-1])
    dissoc_mask = (t >= rinse) & (t <= rinse_end)
    t_dissoc_full = t[dissoc_mask]
    signal_dissoc_full = signal[dissoc_mask]
    t_rinse = rinse

    # Skip transport lag
    t0 = t_rinse + skip_s
    lag_mask = t_dissoc_full >= t0
    t_dissoc = t_dissoc_full[lag_mask]
    signal_dissoc = signal_dissoc_full[lag_mask]

    # ---- Phase 1: Closed-form R0, Rss with kd from DK ----
    R0_est, Rss_est = _solve_R0_Rss(kd_final, t_dissoc, signal_dissoc, t0)
    R0_est = max(R0_est, 1.0)

    # ---- Phase 2: Derive ka from steady-state at dissociation onset ----
    c_plateau = float(c_func(t_rinse - 2.0))
    Rmax_est = max(Rmax0, R0_est * 1.2)

    if c_plateau > 0 and Rmax_est > R0_est:
        ka_est = kd_final * R0_est / (c_plateau * (Rmax_est - R0_est))
    else:
        ka_est = max(ka0, 1.0)

    # ---- Phase 3: Multi-start ODE refinement for (ka, kd and Rmax) ----
    lb_full = np.array([1e-1, 1e-6, 1.0])
    ub_full = np.array([1e8, 1e1, 1e4])

    rng = np.random.default_rng(rng_seed)
    starts = [np.clip([ka_est, kd_final, Rmax_est], lb_full, ub_full)]
    # Also try DK's ka/kd/Rmax as a starting point
    starts.append(np.clip([max(ka0, 1.0), max(kd0, 1e-5), max(Rmax0, 2.0)], lb_full, ub_full))
    for _ in range(max(n_starts - 2, 0)):
        log_perturb = rng.normal(0, 0.5, size=3)
        p = np.array([ka_est, kd_final, Rmax_est]) * np.exp(log_perturb)
        starts.append(np.clip(p, lb_full, ub_full))

    fits = []
    for p0 in starts:
        try:
            opt = least_squares(
                _residuals_full, p0,
                args=(t, signal, c_func, w),
                bounds=(lb_full, ub_full),
                method='trf',
                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                max_nfev=200,
                diff_step=1e-2,
            )
            if opt.success:
                fits.append((opt.x, opt.cost, opt.jac, opt.nfev))
        except Exception:
            pass

    if not fits:
        # Fallback: use derived estimates
        R_fit = simulate_sensorgram(t, ka_est, kd_final, Rmax_est, c_func, R0=0.0)
        return {
            'ka': ka_est, 'kd': kd_final, 'Rmax': Rmax_est,
            'KD': kd_final / ka_est,
            'R0': R0_est, 'Rss': Rss_est,
            'ka_se': np.nan, 'kd_se': np.nan, 'Rmax_se': np.nan,
            'cov': np.full((3, 3), np.nan),
            'R_fit': R_fit,
            'residuals': w * (signal - R_fit),
            'sigma_residual': np.nan,
            'n_points': int((w > 0).sum()),
            'cost': np.nan, 'nfev': 0,
            'n_converged': 0, 'n_starts': n_starts,
            'success': False, 'message': 'All ODE fits failed',
        }

    # Median aggregation over converged ODE fits
    all_params = np.array([f[0] for f in fits])
    ka_final_val = float(np.median(all_params[:, 0]))
    kd_final_val = float(np.median(all_params[:, 1]))
    Rmax_final = float(np.median(all_params[:, 2]))
    total_nfev = sum(f[3] for f in fits)

    # IQR
    iqr_ka = float(np.percentile(all_params[:, 0], 75) -
                    np.percentile(all_params[:, 0], 25))
    iqr_kd = float(np.percentile(all_params[:, 1], 75) -
                    np.percentile(all_params[:, 1], 25))
    iqr_Rmax = float(np.percentile(all_params[:, 2], 75) -
                      np.percentile(all_params[:, 2], 25))

    KD = kd_final_val / ka_final_val

    # Confidence from best Jacobian (lowest cost)
    best_idx = np.argmin([f[1] for f in fits])
    best_jac = fits[best_idx][2]

    residuals = _residuals_full(
        [ka_final_val, kd_final_val, Rmax_final], t, signal, c_func, w)
    n = int((w > 0).sum())
    dof = max(n - 3, 1)
    sigma2 = np.sum(residuals ** 2) / dof

    ka_se, kd_se, Rmax_se = np.nan, np.nan, np.nan
    cov = np.full((3, 3), np.nan)
    try:
        JtJ_inv = np.linalg.inv(best_jac.T @ best_jac)
        cov = sigma2 * JtJ_inv
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        ka_se, kd_se, Rmax_se = se
    except np.linalg.LinAlgError:
        pass

    R_fit = simulate_sensorgram(t, ka_final_val, kd_final_val, Rmax_final,
                                c_func, R0=0.0)

    return {
        'ka': ka_final_val,
        'kd': kd_final_val,
        'Rmax': Rmax_final,
        'KD': KD,
        'R0': R0_est,
        'Rss': Rss_est,
        'ka_se': ka_se,
        'kd_se': kd_se,
        'Rmax_se': Rmax_se,
        'ka_iqr': iqr_ka,
        'kd_iqr': iqr_kd,
        'Rmax_iqr': iqr_Rmax,
        'cov': cov,
        'R_fit': R_fit,
        'residuals': residuals,
        'sigma_residual': np.sqrt(sigma2),
        'n_points': n,
        'cost': float(np.sum(residuals ** 2)),
        'n_converged': len(fits),
        'n_starts': n_starts,
        'nfev': total_nfev,
        'success': True,
        'message': f'{len(fits)}/{n_starts} ODE starts converged',
    }


def fit_sample(sample, dmso_cals, blanks=None, lambda_reg=0.0,
               smoothing_factor=None, n_starts=1):
    """Fit a single sample using Direct Kinetics → ODE refinement.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    dmso_cals : list[dict]
        DMSO calibration cycles.
    blanks : list[dict] or None
        Blank cycles for double referencing.
    lambda_reg : float
        Tikhonov regularisation for Direct Kinetics initial estimates.
    smoothing_factor : float or None
        Smoothing parameter for spline in Direct Kinetics.
    n_starts : int
        Number of starting points for ODE multi-start refinement.

    Returns
    -------
    result : dict
        Full ODE fit results plus Direct Kinetics initial estimates
        and preprocessed signal arrays.
    """
    # Step 1: Direct Kinetics for initial estimates
    dk = dk_fit_sample(sample, dmso_cals, blanks=blanks,
                       lambda_reg=lambda_reg,
                       smoothing_factor=smoothing_factor)

    t = dk['t']
    signal = dk['signal']

    # Build pulsed c(t) for ODE fitting (preserves pulse structure)
    dmso = select_dmso_cal(sample['index'], dmso_cals)
    c_func_pulsed, _ = build_pulsed_concentration_profile(
        dmso, sample['concentration_M'])

    # Full weight mask: buffer pulses during association + dissociation
    w = build_full_weight_mask(t, sample['markers'], dmso)

    # Trim to active fitting window (Injection → RinseEnd + margin)
    t_fit, sig_fit, w_fit, fit_mask = trim_to_fit_window(
        t, signal, w, sample['markers'])

    # Step 2: ODE fit on trimmed arrays
    ode = ode_fit(t_fit, sig_fit, c_func_pulsed, w_fit, sample['markers'],
                  ka0=dk['ka'], kd0=dk['kd'], Rmax0=dk['Rmax'],
                  n_starts=n_starts)

    # Map R_fit back to full time grid
    R_fit_full = np.full_like(signal, np.nan)
    R_fit_full[fit_mask] = ode['R_fit']
    ode['R_fit'] = R_fit_full

    residuals_full = np.zeros_like(signal)
    residuals_full[fit_mask] = ode['residuals']
    ode['residuals'] = residuals_full

    # Store envelope c_func for DK results / visualization
    ode['c_func'] = dk['c_func']

    # Combine results
    ode['dk_ka'] = dk['ka']
    ode['dk_kd'] = dk['kd']
    ode['dk_Rmax'] = dk['Rmax']
    ode['dk_KD'] = dk['KD']
    ode['R0_dissoc'] = dk['R0_dissoc']
    ode['t'] = t
    ode['signal'] = signal
    ode['c_raw'] = dk['c_raw']
    ode['dmso_index'] = dk['dmso_index']
    ode['blank_index'] = dk['blank_index']

    return ode
