"""Batch fitting of all samples in a .cxw experiment file.

Provides two fitting tiers:
  - **DK-only** (Direct Kinetics): ~ms per sample, for rapid screening.
  - **DK → ODE** (full pipeline): DK initialisation + ODE refinement,
    ~15-20s per sample, for publication-quality parameters.

The main entry point is ``batch_fit()``, which returns a pandas DataFrame
with one row per sample and columns for all kinetic parameters.
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .data_loader import load_cxw
from .package_loader import load_experiment
from .models import (is_baseline_noisy, has_injection_error, is_reference_signal_negative,
                     is_sample_carried_over, has_low_signal_to_noise_reponse, is_nonspecific_binder,
                     double_reference, select_blank)
from .direct_kinetics import fit_sample as dk_fit_sample
from .ode_fitting import fit_sample as ode_fit_sample


def batch_fit(filepath, mode='dk', channels='all', progress=True,
              n_starts=3, n_parallel_jobs=None):
    """Fit all samples in a .cxw file (or exported package) and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to a ``.cxw`` experiment file, **or** to an exported
        SensoFit data package (a ``.zip`` produced by
        :func:`sensofit.dataexporter.export_package`, or an unzipped
        package directory).  The dispatcher
        :func:`sensofit.package_loader.load_experiment` picks the right
        loader by extension.
    mode : {'dk', 'ode'}
        Fitting mode:
        - ``'dk'``:  Direct Kinetics only (fast, ~ms/sample).
        - ``'ode'``: DK → ODE refinement (slower, ~15s/sample).
    channels : str or list[int]
        Which active flow cells to process.
        - ``'all'`` (default): every active channel in the file.
        - A list of FC numbers, e.g. ``[2]``: only that channel.
    progress : bool
        Print progress to stdout.
    n_starts : int
        Number of starting points for ODE multi-start refinement.
        Ignored when mode='dk'.

    Returns
    -------
    df : pd.DataFrame
        One row per sample per channel with kinetic parameters and metadata.
    data : dict
        Raw data dict (same shape as :func:`load_cxw`) for downstream use.
    results : list[dict or None]
        List of fit result dicts (one per sample), or None for failed fits.
    """
    data = load_experiment(filepath, channels=channels)
    samples = data['samples']
    dmso_cals = data['dmso_cals']
    blanks = data['blanks']

    if mode not in ('dk', 'ode'):
        raise ValueError(f"mode must be 'dk' or 'ode', got {mode!r}")

    n_parallel_jobs = n_parallel_jobs if mode != 'dk' else None

    fit_func = dk_fit_sample if mode == 'dk' else ode_fit_sample

    n = len(samples)
    if n == 0:
        print(f"No samples found in file: {filepath}.")
        return pd.DataFrame(), data
    t0 = time.time()

    if n_parallel_jobs:
        all_results = Parallel(n_jobs=n_parallel_jobs, backend="multiprocessing")(
            delayed(_batch_process)(i, t0, n, progress, sample, dmso_cals, blanks, mode,
                                    fit_func, n_starts) for i, sample in enumerate(samples)
        )
    else:
        all_results = [_batch_process(i, t0, n, progress, sample, dmso_cals, blanks, mode,
                                      fit_func, n_starts) for i, sample in enumerate(samples)]

    results = [r[0] for r in all_results]
    rows = [r[1] for r in all_results]

    if progress:
        elapsed = time.time() - t0
        print(f'\r  Done: {n} samples in {elapsed:.1f}s '
              f'({elapsed/n:.1f}s/sample) \n')

    df = pd.DataFrame(rows)

    # Sort by compound then concentration
    df.sort_values(['compound', 'concentration_M'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, data, results


def _batch_process(i, t0, n, progress, sample, dmso_cals, blanks, mode, fit_func, n_starts):
    """Process a single sample with error handling and NSB filtering."""
    if progress:
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n - i - 1) if i > 0 else 0
        ch_label = sample.get('channel', '')
        print(f'\r  [{i+1}/{n}] {sample["compound"]:20s} {ch_label:8s} '
                f'{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining',
                end='', flush=True)

    # Filter DMSO cals and blanks to same channel and same rk_serie
    rk_serie = sample.get('rk_serie_id')
    ch = sample.get('channel')
    ch_dmso = [d for d in dmso_cals if d.get('channel') == ch and d.get('rk_serie_id') == rk_serie]
    ch_blanks = [b for b in blanks if b.get('channel') == ch and b.get('rk_serie_id') == rk_serie]
    # Fallback: if no channel-matched cals, use all (single-channel files)
    if not ch_dmso:
        ch_dmso = dmso_cals
    if not ch_blanks:
        ch_blanks = blanks

    # Check for negative signal in reference channel before fitting
    heuristics = sensorgram_heuristics(sample, blanks=ch_blanks)
    if "negative_signal_in_reference_channel" in heuristics:
        row = _fallback_row(sample, mode)
        row['flag'] = True
        row['flag_reason'] = '; '.join(heuristics)
        row['success'] = np.nan
        return [None, row]

    try:
        kwargs = {'blanks': ch_blanks}
        if mode == 'ode':
            kwargs['n_starts'] = n_starts
        result = fit_func(sample, ch_dmso, **kwargs)
        row = _extract_row(sample, result, mode)
        row['flag'] = True if heuristics else False
        row['flag_reason'] = heuristics[0] if heuristics else np.nan
    except Exception as e:
        row = _fallback_row(sample, mode)
        row['flag'] = True
        row['flag_reason'] = str(e)
        return [None, row]

    return [result, row]

def _extract_row(sample, result, mode):
    """Build a flat dict from sample metadata + fit results."""
    row = {
        'compound_type':    sample['cycle_type'],
        'compound':         sample['compound'],
        'concentration_M':  sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'mw':               sample.get('mw'),
        'slot':             sample.get('slot'),
        'cycle_index':      sample['index'],
        'channel':          sample.get('channel', ''),
        'rk_serie_id':      sample.get('rk_serie_id'),
    }

    if mode == 'dk':
        row.update({
            'ka':           result['ka'],
            'kd':           result['kd'],
            'Rmax':         result['Rmax'],
            'KD':           result['KD'],
            'KD_uM':        result['KD'] * 1e6,
            'R0_dissoc':    result['R0_dissoc'],
            'sigma_res':    result['sigma_residual'],
            'n_points':     result['n_points'],
            'blank_index':  result['blank_index'],
            'dmso_index':   result['dmso_index'],
            'fit_mode':     'dk',
            'success':      True,
        })
    else:  # ode
        row.update({
            'ka':           result['ka'],
            'kd':           result['kd'],
            'Rmax':         result['Rmax'],
            'KD':           result['KD'],
            'KD_uM':        result['KD'] * 1e6,
            'sqrt_chi2':    result.get('sqrt_chi2', np.nan),
            'ka_se':        result.get('ka_se', np.nan),
            'kd_se':        result.get('kd_se', np.nan),
            'Rmax_se':      result.get('Rmax_se', np.nan),
            'R0':           result.get('R0', np.nan),
            'Rss':          result.get('Rss', np.nan),
            'R0_dissoc':    result.get('R0_dissoc', np.nan),
            'sigma_res':    result.get('sigma_residual', np.nan),
            'n_points':     result.get('n_points', 0),
            'n_converged':  result.get('n_converged', 0),
            'nfev':         result.get('nfev', 0),
            'blank_index':  result.get('blank_index'),
            'dmso_index':   result.get('dmso_index'),
            'dk_ka':        result.get('dk_ka', np.nan),
            'dk_kd':        result.get('dk_kd', np.nan),
            'dk_Rmax':      result.get('dk_Rmax', np.nan),
            'dk_KD':        result.get('dk_KD', np.nan),
            'fit_mode':     'ode',
            'success':      result.get('success', False),
            'message':      result.get('message', ''),
        })

    return row


def _fallback_row(sample, mode):
    """Return an all-NaN row when fitting raises an exception."""
    row = {
        'compound_type':    sample['cycle_type'],
        'compound':         sample['compound'],
        'concentration_M':  sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'mw':               sample.get('mw'),
        'slot':             sample.get('slot'),
        'cycle_index':      sample['index'],
        'channel':          sample.get('channel', ''),
        'rk_serie_id':      sample.get('rk_serie_id'),
        'ka':               np.nan,
        'kd':               np.nan,
        'Rmax':             np.nan,
        'KD':               np.nan,
        'KD_uM':            np.nan,
        'sqrt_chi2':        np.nan,
        'sigma_res':        np.nan,
        'n_points':         0,
        'fit_mode':         mode,
        'success':          False,
    }
    return row


def sensorgram_heuristics(sample, blanks=None):
    """Heuristic to flag sensorgram to check if they should be fitted or not.
    Criteria:
    - Noisy: baseline std > 5% of max abs(signal)
    - Injection issue: -10% of max(abs(signal)) > signal before injection > 10% of max(abs(signal))
    - Negative signal in reference channel: min signal < 1% of -max(abs(signal))
    - Low signal-to-noise response: binding response < 5% of max(abs(signal))
    - Sample carryover: steady-state signal > 10% of max(abs(signal))
    - Non-specific binding: signal after rinse in reference channel > 2.5% of max(abs(signal))
    """
    if blanks:
        blank = select_blank(sample['index'], blanks)
    signal, _ = double_reference(sample, blank)

    heuristics = []
    noisy, _ = is_baseline_noisy(sample, signal)
    if noisy:
        heuristics.append('noisy')
    inj_error, _ = has_injection_error(sample, signal)
    if inj_error:
        heuristics.append('injection_issue')
    neg_ref, _ = is_reference_signal_negative(sample)
    if neg_ref:
        heuristics.append('negative_signal_in_reference_channel')
    low_snr, _ = has_low_signal_to_noise_reponse(sample, signal)
    if low_snr:
        heuristics.append('low_signal_to_noise_response')
    carryover, _ = is_sample_carried_over(sample, signal)
    if carryover:
        heuristics.append('sample_carryover')
    nsb, _ = is_nonspecific_binder(sample)
    if nsb:
        heuristics.append('non_specific_interaction')
    return heuristics


def flag_poor_fits(df, kd_max=9.9, ka_min=0.5,
                   Rmax_min=1.1, sigma_max=2.0,
                   se_threshold=0.5, iqr_threshold=0.25):
    """Add a 'flag' column marking questionable fits.

    A fit is flagged if any of the following hold:
    - kd hit upper bound (>= kd_max)
    - ka hit lower bound (<= ka_min)
    - Rmax below noise floor (<= Rmax_min)
    - High residual (sigma_res > sigma_max)
    - Fit failed (success == False)

    Parameters
    ----------
    df : pd.DataFrame
        Output from ``batch_fit()``.
    kd_max, ka_min, Rmax_min, sigma_max : float
        Thresholds for flagging.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with 'flag' and 'flag_reason' columns added.
    """
    flags = []
    reasons = []

    for _, row in df.iterrows():
        r = []
        ka = row.get('ka')
        ka_se = row.get('ka_se')
        ka_iqr = row.get('ka_iqr')
        kd = row.get('kd')
        kd_se = row.get('kd_se')
        kd_iqr = row.get('kd_iqr')
        Rmax = row.get('Rmax')
        Rmax_se = row.get('Rmax_se')
        Rmax_iqr = row.get('Rmax_iqr')
        sigma_res = row.get('sigma_res')
        if row.get('flag', False):
            r.append(row.get('flag_reason', ''))  # Preserve existing flag reason
        if not row.get('success', False):
            r.append('fit_failed')
        if not np.isnan(ka) and ka <= ka_min:
            r.append('ka_at_bound')
        if not np.isnan(ka_se) and ka_se > se_threshold * abs(ka):
            r.append('ka_high_se')
        if not np.isnan(ka_iqr) and ka_iqr > iqr_threshold * abs(ka):
            r.append('ka_high_iqr')
        if not np.isnan(kd) and kd >= kd_max:
            r.append('kd_at_bound')
        if not np.isnan(kd_se) and kd_se > se_threshold * abs(kd):
            r.append('kd_high_se')
        if not np.isnan(kd_iqr) and kd_iqr > iqr_threshold * abs(kd):
            r.append('kd_high_iqr')
        if not np.isnan(Rmax) and Rmax <= Rmax_min:
            r.append('low_Rmax')
        if not np.isnan(Rmax_se) and Rmax_se > se_threshold * abs(Rmax):
            r.append('Rmax_high_se')
        if not np.isnan(Rmax_iqr) and Rmax_iqr > iqr_threshold * abs(Rmax):
            r.append('Rmax_high_iqr')
        if not np.isnan(sigma_res) and sigma_res > sigma_max:
            r.append('high_residual')

        flags.append(len(r) > 0)
        reasons.append('; '.join(r) if r else '')

    df = df.copy()
    df['flag'] = flags
    df['flag_reason'] = reasons
    return df
