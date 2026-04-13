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
from .data_loader import load_cxw
from .models import is_nonspecific_binder
from .direct_kinetics import fit_sample as dk_fit_sample
from .ode_fitting import fit_sample as ode_fit_sample


def batch_fit(filepath, mode='dk', include_nsb=False, channels='all',
              progress=True):
    """Fit all samples in a .cxw file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .cxw experiment file.
    mode : {'dk', 'ode'}
        Fitting mode:
        - ``'dk'``:  Direct Kinetics only (fast, ~ms/sample).
        - ``'ode'``: DK → ODE refinement (slower, ~15s/sample).
    include_nsb : bool
        If True, fit non-specific binders instead of skipping them.
        They are still flagged in the ``nonspecific`` column.
    channels : str or list[int]
        Which active flow cells to process.
        - ``'all'`` (default): every active channel in the file.
        - A list of FC numbers, e.g. ``[2]``: only that channel.
    progress : bool
        Print progress to stdout.

    Returns
    -------
    df : pd.DataFrame
        One row per sample per channel with kinetic parameters and metadata.
    data : dict
        Raw data from ``load_cxw()`` for downstream use.
    """
    data = load_cxw(filepath, channels=channels)
    samples = data['samples']
    dmso_cals = data['dmso_cals']
    blanks = data['blanks']

    if mode not in ('dk', 'ode'):
        raise ValueError(f"mode must be 'dk' or 'ode', got {mode!r}")

    fit_func = dk_fit_sample if mode == 'dk' else ode_fit_sample

    rows = []
    n = len(samples)
    t0 = time.time()

    for i, sample in enumerate(samples):
        if progress:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (n - i - 1) if i > 0 else 0
            ch_label = sample.get('channel', '')
            print(f'\r  [{i+1}/{n}] {sample["compound"]:20s} {ch_label:8s} '
                  f'{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining',
                  end='', flush=True)

        # Filter DMSO cals and blanks to same channel
        ch = sample.get('channel')
        ch_dmso = [d for d in dmso_cals if d.get('channel') == ch]
        ch_blanks = [b for b in blanks if b.get('channel') == ch]
        # Fallback: if no channel-matched cals, use all (single-channel files)
        if not ch_dmso:
            ch_dmso = dmso_cals
        if not ch_blanks:
            ch_blanks = blanks

        # Check for non-specific binding before fitting
        nsb, ref_dissoc = is_nonspecific_binder(sample)
        if nsb and not include_nsb:
            row = _fallback_row(sample, mode)
            row['fit_mode'] = 'nsb'
            row['fit_error'] = None
            row['nonspecific'] = True
            row['ref_dissoc'] = ref_dissoc
            rows.append(row)
            continue

        try:
            result = fit_func(sample, ch_dmso, blanks=ch_blanks)
            row = _extract_row(sample, result, mode)
            row['fit_error'] = None
        except Exception as e:
            row = _fallback_row(sample, mode)
            row['fit_error'] = str(e)

        row['nonspecific'] = False
        row['ref_dissoc'] = ref_dissoc

        rows.append(row)

    if progress:
        elapsed = time.time() - t0
        print(f'\r  Done: {n} samples in {elapsed:.1f}s '
              f'({elapsed/n:.1f}s/sample)' + ' ' * 30)

    df = pd.DataFrame(rows)

    # Sort by compound then concentration
    df.sort_values(['compound', 'concentration_M'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, data


def _extract_row(sample, result, mode):
    """Build a flat dict from sample metadata + fit results."""
    row = {
        'compound':        sample['compound'],
        'concentration_M': sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'mw':              sample.get('mw'),
        'slot':            sample.get('slot'),
        'cycle_index':     sample['index'],
        'channel':         sample.get('channel', ''),
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
        'compound':        sample['compound'],
        'concentration_M': sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'mw':              sample.get('mw'),
        'slot':            sample.get('slot'),
        'cycle_index':     sample['index'],
        'channel':         sample.get('channel', ''),
        'ka':              np.nan,
        'kd':              np.nan,
        'Rmax':            np.nan,
        'KD':              np.nan,
        'KD_uM':           np.nan,
        'sigma_res':       np.nan,
        'n_points':        0,
        'fit_mode':        mode,
        'success':         False,
    }
    return row


def flag_poor_fits(df, kd_max=10.0, ka_min=1.0,
                   Rmax_min=0.5, sigma_max=10.0):
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
        if row.get('nonspecific', False):
            r.append('nonspecific_binder')
        elif not row.get('success', False):
            r.append('fit_failed')
        if pd.notna(row.get('fit_error')) and row['fit_error']:
            r.append('exception')
        if pd.notna(row.get('kd')) and row['kd'] >= kd_max:
            r.append('kd_at_bound')
        if pd.notna(row.get('ka')) and row['ka'] <= ka_min:
            r.append('ka_at_bound')
        if pd.notna(row.get('Rmax')) and row['Rmax'] <= Rmax_min:
            r.append('low_Rmax')
        if pd.notna(row.get('sigma_res')) and row['sigma_res'] > sigma_max:
            r.append('high_residual')

        flags.append(len(r) > 0)
        reasons.append('; '.join(r) if r else '')

    df = df.copy()
    df['flag'] = flags
    df['flag_reason'] = reasons
    return df
