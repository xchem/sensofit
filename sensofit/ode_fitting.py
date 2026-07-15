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
from .models import (build_pulsed_concentration_profile, double_reference, build_full_weight_mask, 
                     simulate_sensorgram, trim_to_fit_window, fit_last_disso, fit_last_asso, get_rmse)
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


def _residuals_full(params, t, signal, c_func, w, fast=True):
    """Full ODE residuals (weighted).

    Optimises (ka, kd, Rmax) simultaneously.
    """
    ka, kd, Rmax = params
    R_sim = simulate_sensorgram(t, ka, kd, Rmax, c_func, R0=0.0,
                                fast=fast)
    return w * (signal - R_sim)

def _chi2(residuals=None, w=None, n_params=None, R_sim=None, signal=None,
          params=None, t=None, c_func=None, sqrt=False, fast=True):
    """Calculate Chi2 from ODE residuals.
    
    Chi2 = sum((w * residuals)^2) / (N - n_params)
    Returns Chi2 or Sqrt(Chi2) if sqrt=True."""
    if residuals is None:
        if R_sim is not None:
            residuals = w * (signal - R_sim)
        else:
            residuals = _residuals_full(params, t, signal, c_func, w,
                                        fast=fast)

    n_points = int((w > 0).sum()) if w is not None else len(residuals)    
    chi2 = np.sum(residuals**2)/(n_points - max(n_params, 1))
    return np.sqrt(chi2) if sqrt else chi2


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
            n_starts=1, rng_seed=None, skip_s=1.0, fast=True):
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
    fast : bool
        Use the stable two-half-step exponential propagator when true
        (default), or the legacy adaptive RK45 solver when false.
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
                args=(t, signal, c_func, w, fast),
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
        R_fit = simulate_sensorgram(t, ka_est, kd_final, Rmax_est, c_func,
                                    R0=0.0, fast=fast)
        fit_mask = np.isfinite(R_fit)
        rmse = get_rmse(signal[fit_mask], R_fit[fit_mask])
        return {
            'ka': ka_est, 'kd': kd_final, 'Rmax': Rmax_est,
            'KD': kd_final / ka_est,
            'rmse': np.nan,
            'R0': R0_est, 'Rss': Rss_est,
            'ka_se': np.nan, 'kd_se': np.nan, 'Rmax_se': np.nan,
            'cov': np.full((3, 3), np.nan),
            'R_fit': R_fit,
            'residuals': w * (signal - R_fit),
            'sigma_residual': np.nan,
            'n_points': int((w > 0).sum()),
            'cost': np.nan, 'nfev': 0,
            'n_converged': 0, 'n_starts': n_starts,
            'success': False, 'message': 'All ODE fits failed', 'fast': fast,
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
    
    params = [ka_final_val, kd_final_val, Rmax_final]
    residuals = _residuals_full(
        params, t, signal, c_func, w, fast=fast)
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
                                c_func, R0=0.0, fast=fast)
    fit_mask = np.isfinite(R_fit)
    rmse = get_rmse(signal[fit_mask], R_fit[fit_mask])

    return {
        'ka': ka_final_val,
        'kd': kd_final_val,
        'Rmax': Rmax_final,
        'KD': KD,
        'rmse': rmse,
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
        'fast': fast,
    }


def fit_sample(sample, dmso, blank=None, lambda_reg=0.0, initial_estimates='LPF',
               smoothing_factor=None, neg_ss_correction=False,
               association_weight=0.0, n_starts=1, fast=True):
    """Fit a single sample using Direct Kinetics → ODE refinement.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    dmso : dict or None
        DMSO calibration cycle.
    blank : dict or None
        Blank cycle for double referencing.
    lambda_reg : float
        Tikhonov regularisation for Direct Kinetics initial estimates.
    initial_estimates : str
        Method for initial estimates: 'DK' (Direct Kinetics) or 'LPF' (last-pulse fit).
    smoothing_factor : float or None
        Smoothing parameter for spline in Direct Kinetics.
    neg_ss_correction : bool
        If True, apply a correction to the signal to ensure non-negative
        steady-state response during last dissociation (Rinse → RinseEnd).
    n_starts : int
        Number of starting points for ODE multi-start refinement.
    fast : bool
        Use the stable two-half-step exponential propagator when true
        (default), or the legacy adaptive RK45 solver when false.

    Returns
    -------
    result : dict
        Full ODE fit results plus Direct Kinetics initial estimates
        and preprocessed signal arrays.
    """
    # Step 1: Initial estimates
    if initial_estimates == 'DK':
        try:
            dk = dk_fit_sample(sample, dmso, blank=blank,
                            lambda_reg=lambda_reg,
                            smoothing_factor=smoothing_factor)
            t = dk['t']
            signal = dk['signal']
            blank_index = dk['blank_index']
            seed_method = 'DK'
            ka_seed = dk['ka']
            kd_seed = dk['kd']
            KD_seed = dk['KD']
            Rmax_seed = dk['Rmax']
        except Exception as e:
            print(f'WARNING! Direct Kinetics failed for sample {sample["index"]} (RK serie {sample.get("rk_serie_id", "")}, '
                  f'channel {sample.get("channel", "")}): {e}. Using default seeds for ODE fitting.')
            kd_seed = 1e-3
            ka_seed = 1e3
            Rmax_seed = 10.0
    else:
        if initial_estimates != 'LPF':
            print(f'WARNING! Unknown initial_estimates method "{initial_estimates}", defaulting to LPF (last-pulse fit).')
        try:
            t = sample['time']
            asso_mask = (t >= sample['markers'].get('Injection', 0)) & (t <= sample['markers'].get('Rinse', t[-1]))
            signal, blank_index = double_reference(sample, blank)
            seed_method = 'last_pulse_fit'
            kd_seed, _, _, _ = fit_last_disso(sample, channel="signal", blank=blank)
            ka_seed, _, _, _, _, _ = fit_last_asso(sample, blank=blank, koff=kd_seed)
            KD_seed = kd_seed / ka_seed if ka_seed > 0 else np.nan
            Rmax_seed = signal[asso_mask].max()*((ka_seed*sample['concentration_M']+kd_seed)/(ka_seed*sample['concentration_M']))
        except Exception as e:
            print(f'WARNING! Last-pulse fit failed for sample {sample["index"]} (RK serie {sample.get("rk_serie_id", "")}, '
                  f'channel {sample.get("channel", "")}): {e}. Using default seeds for ODE fitting.')
            kd_seed = 1e-3
            ka_seed = 1e3
            Rmax_seed = 10.0

    # Build pulsed c(t) for ODE fitting (preserves pulse structure)
    c_func_pulsed, _ = build_pulsed_concentration_profile(
        dmso, sample['concentration_M'])

    # Full weight mask: buffer pulses during association + dissociation
    assert 0 <= association_weight <= 1.0, "association_weight value must be between 0 and 1."
    w = build_full_weight_mask(t, sample['markers'], dmso, association_weight=association_weight)

    # Trim to active fitting window (Injection → RinseEnd + margin)
    t_fit, sig_fit, w_fit, fit_mask = trim_to_fit_window(
        t, signal, w, sample['markers'])
    if neg_ss_correction:
        last_diss_mask = (t_fit >= sample['markers'].get('Rinse', 0)) & (t_fit <= sample['markers'].get('RinseEnd', t[-1]))
        min_diss = sig_fit[last_diss_mask].min() if sig_fit[last_diss_mask].min() < 0 else 0.0
        sig_fit -= min_diss

    # Step 2: ODE fit on trimmed arrays
    ode = ode_fit(t_fit, sig_fit, c_func_pulsed, w_fit, sample['markers'],
                  ka0=ka_seed, kd0=kd_seed, Rmax0=Rmax_seed,
                  n_starts=n_starts, fast=fast)

    # Map R_fit back to full time grid
    R_fit_full = np.full_like(signal, np.nan)
    R_fit_full[fit_mask] = ode['R_fit']
    ode['R_fit'] = R_fit_full

    residuals_full = np.zeros_like(signal)
    residuals_full[fit_mask] = ode['residuals']
    ode['residuals'] = residuals_full

    # Store envelope c_func for DK results / visualization
    ode['c_func'] = c_func_pulsed

    # Combine results
    ode['seed_method'] = seed_method
    ode['ka_seed'] = ka_seed
    ode['kd_seed'] = kd_seed
    ode['Rmax_seed'] = Rmax_seed
    ode['KD_seed'] = KD_seed
    ode['t'] = t
    ode['signal'] = signal
    if neg_ss_correction:
        ode['signal'] -= min_diss
    ode['dmso_index'] = dmso['index'] if dmso else None
    ode['blank_index'] = blank_index
    ode['fast'] = fast

    return ode
