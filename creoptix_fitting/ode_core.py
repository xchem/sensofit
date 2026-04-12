"""Shared fitting engine for pure-ODE 1:1 Langmuir kinetics.

Fits all three parameters (ka, kd, Rmax) simultaneously via forward ODE
integration.  Supports multiple optimizer back-ends:

  least_squares: 'trf', 'dogbox', 'lm'
  minimize:      'Nelder-Mead', 'L-BFGS-B'

Used by ode_pure_heuristic and ode_pure_grid modules which differ only in
how starting points are generated.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from .models import simulate_sensorgram


# Physical bounds for the three parameters
BOUNDS_LO = np.array([1e-1, 1e-6, 1.0])      # ka, kd, Rmax
BOUNDS_HI = np.array([1e8,  10.0, 1e4])


def _residuals_3p(params, t, signal, c_func, w):
    """Weighted residuals for 3-parameter (ka, kd, Rmax) ODE fit."""
    ka, kd, Rmax = params
    R_sim = simulate_sensorgram(t, ka, kd, Rmax, c_func, R0=0.0)
    return w * (signal - R_sim)


def _scalar_objective(params, t, signal, c_func, w):
    """Sum-of-squares objective for minimize-based optimizers."""
    r = _residuals_3p(params, t, signal, c_func, w)
    return float(np.sum(r ** 2))


def _run_fit(p0, t, signal, c_func, w, method='trf'):
    """Run a single fit from starting point p0 using the given method.

    Parameters
    ----------
    p0 : array-like, shape (3,)
        Starting point [ka, kd, Rmax].
    t, signal, c_func, w : arrays / callable
        Time, signal, concentration function, weight mask.
    method : str
        One of 'trf', 'dogbox', 'lm', 'Nelder-Mead', 'L-BFGS-B'.

    Returns
    -------
    result : dict or None
        Dict with keys 'params', 'cost', 'jac', 'nfev', 'success'
        or None if the fit failed.
    """
    p0 = np.asarray(p0, dtype=float)

    try:
        if method in ('trf', 'dogbox'):
            opt = least_squares(
                _residuals_3p, p0,
                args=(t, signal, c_func, w),
                bounds=(BOUNDS_LO, BOUNDS_HI),
                method=method,
                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                max_nfev=300,
                diff_step=1e-2,
            )
            if not opt.success:
                return None
            return {
                'params': opt.x,
                'cost': opt.cost,
                'jac': opt.jac,
                'nfev': opt.nfev,
                'success': True,
            }

        elif method == 'lm':
            # Levenberg-Marquardt does not support bounds
            opt = least_squares(
                _residuals_3p, p0,
                args=(t, signal, c_func, w),
                method='lm',
                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                max_nfev=300,
            )
            if not opt.success:
                return None
            # Clip to physical range
            params = np.clip(opt.x, BOUNDS_LO, BOUNDS_HI)
            return {
                'params': params,
                'cost': opt.cost,
                'jac': opt.jac,
                'nfev': opt.nfev,
                'success': True,
            }

        elif method in ('Nelder-Mead', 'L-BFGS-B'):
            # Work in log-space for Nelder-Mead stability
            log_p0 = np.log10(np.clip(p0, BOUNDS_LO, BOUNDS_HI))

            if method == 'L-BFGS-B':
                log_bounds = list(zip(np.log10(BOUNDS_LO), np.log10(BOUNDS_HI)))

                def obj(log_params):
                    params = 10.0 ** log_params
                    return _scalar_objective(params, t, signal, c_func, w)

                opt = minimize(obj, log_p0, method='L-BFGS-B',
                               bounds=log_bounds,
                               options={'maxiter': 300, 'ftol': 1e-12})
            else:
                def obj(log_params):
                    params = 10.0 ** log_params
                    params = np.clip(params, BOUNDS_LO, BOUNDS_HI)
                    return _scalar_objective(params, t, signal, c_func, w)

                opt = minimize(obj, log_p0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6,
                                        'fatol': 1e-10})

            if not opt.success and opt.fun > 1e10:
                return None
            params = np.clip(10.0 ** opt.x, BOUNDS_LO, BOUNDS_HI)
            return {
                'params': params,
                'cost': float(opt.fun),
                'jac': None,   # minimize does not provide Jacobian
                'nfev': opt.nfev,
                'success': True,
            }

        else:
            raise ValueError(f"Unknown method: {method!r}")

    except Exception:
        return None


def aggregate_fits(fits, t, signal, c_func, w):
    """Aggregate multiple fit results via median and compute statistics.

    Parameters
    ----------
    fits : list[dict]
        Successful fit results from _run_fit.
    t, signal, c_func, w : arrays / callable
        For recomputing residuals at the median parameters.

    Returns
    -------
    result : dict
        Full result dictionary with ka, kd, Rmax, KD, R_fit, etc.
        Returns a failure dict if fits is empty.
    """
    n_starts = len(fits) if fits else 0

    if not fits:
        return {
            'ka': np.nan, 'kd': np.nan, 'Rmax': np.nan, 'KD': np.nan,
            'ka_se': np.nan, 'kd_se': np.nan, 'Rmax_se': np.nan,
            'ka_iqr': np.nan, 'kd_iqr': np.nan, 'Rmax_iqr': np.nan,
            'cov': np.full((3, 3), np.nan),
            'R_fit': np.full_like(signal, np.nan),
            'residuals': np.full_like(signal, np.nan),
            'sigma_residual': np.nan,
            'n_points': int((w > 0).sum()),
            'cost': np.nan, 'nfev': 0,
            'n_converged': 0, 'n_starts': 0,
            'success': False, 'message': 'All ODE fits failed',
        }

    all_params = np.array([f['params'] for f in fits])
    ka_final = float(np.median(all_params[:, 0]))
    kd_final = float(np.median(all_params[:, 1]))
    Rmax_final = float(np.median(all_params[:, 2]))
    total_nfev = sum(f['nfev'] for f in fits)

    # IQR
    iqr_ka = float(np.percentile(all_params[:, 0], 75) -
                   np.percentile(all_params[:, 0], 25))
    iqr_kd = float(np.percentile(all_params[:, 1], 75) -
                   np.percentile(all_params[:, 1], 25))
    iqr_Rmax = float(np.percentile(all_params[:, 2], 75) -
                     np.percentile(all_params[:, 2], 25))

    KD = kd_final / ka_final

    # Residuals at median parameters
    residuals = _residuals_3p([ka_final, kd_final, Rmax_final],
                              t, signal, c_func, w)
    n = int((w > 0).sum())
    dof = max(n - 3, 1)
    sigma2 = np.sum(residuals ** 2) / dof

    # Standard errors from best Jacobian (if available)
    ka_se, kd_se, Rmax_se = np.nan, np.nan, np.nan
    cov = np.full((3, 3), np.nan)
    jac_fits = [f for f in fits if f.get('jac') is not None]
    if jac_fits:
        best_idx = np.argmin([f['cost'] for f in jac_fits])
        best_jac = jac_fits[best_idx]['jac']
        try:
            JtJ_inv = np.linalg.inv(best_jac.T @ best_jac)
            cov = sigma2 * JtJ_inv
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            ka_se, kd_se, Rmax_se = se
        except np.linalg.LinAlgError:
            pass

    R_fit = simulate_sensorgram(t, ka_final, kd_final, Rmax_final,
                                c_func, R0=0.0)

    return {
        'ka': ka_final, 'kd': kd_final, 'Rmax': Rmax_final, 'KD': KD,
        'ka_se': ka_se, 'kd_se': kd_se, 'Rmax_se': Rmax_se,
        'ka_iqr': iqr_ka, 'kd_iqr': iqr_kd, 'Rmax_iqr': iqr_Rmax,
        'cov': cov,
        'R_fit': R_fit,
        'residuals': residuals,
        'sigma_residual': np.sqrt(sigma2),
        'n_points': n,
        'cost': float(np.sum(residuals ** 2)),
        'nfev': total_nfev,
        'n_converged': len(fits),
        'n_starts': n_starts,
        'success': True,
        'message': f'{len(fits)} starts converged',
        'all_params': all_params,
    }
