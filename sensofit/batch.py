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
from .data_loader import load_cxw, estimate_capture_levels
from .package_loader import load_experiment
from .models import (is_baseline_noisy, has_injection_issue, is_reference_response_negative,
                     is_sample_accumulated, has_low_signal_to_noise_reponse, is_nonspecific_binder,
                     double_reference, select_blank, select_dmso_cal,
                     prepare_blanks, get_weight_from_derivative)
from .direct_kinetics import fit_sample as dk_fit_sample
from .ode_fitting import fit_sample as ode_fit_sample


def batch_fit(filepath, mode='dk', channels='all', progress=True,
              n_starts=3, n_parallel_jobs=None, fast=True,
              blank_selection='current', rng_seed=None, 
              ligand_mw=None, subset_csv=None,
              ode_fit_variant='legacy', prefit_thresholds=None,
              fit_no_binding=False, max_cost_ratio=1.1):
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
    fast : bool
        Use the exponential midpoint propagator for ODE fitting (default),
        or the legacy adaptive RK45 solver when false. Ignored when
        mode='dk'.
    subset_csv : str or None
        Optional CSV containing ``rk_serie_id``, ``cycle_index``, and
        ``channel`` columns. Only matching samples are fitted.
    blank_selection : {'current', 'legacy'}
        Blank quality rules to use. ``'legacy'`` supports controlled
        comparisons with the rules used before the additional stability
        checks; all other fitting and exclusion logic is unchanged.
    rng_seed : int or None
        Base seed for reproducible ODE multi-start fits. Each sample receives
        a deterministic offset from this seed. ``None`` preserves the
        historical non-reproducible behavior.

    ode_fit_variant : {'legacy', 'joint_reference_offset_prefit_basin'}
        Use the legacy fitter or joint-reference fitting with pre-fit basins.
    prefit_thresholds : sequence of three floats or None
        Finite, increasing regime boundaries; default (0.80, 2.05, 2.97).
    fit_no_binding : bool
        Override only the pre-fit no-binding skip; default False.
    max_cost_ratio : float
        Finite constrained/unrestricted cost limit >= 1; default 1.1.

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
    if subset_csv is not None:
        subset = pd.read_csv(subset_csv)
        required = {'rk_serie_id', 'cycle_index', 'channel'}
        missing = required.difference(subset.columns)
        if missing:
            raise ValueError(
                f'subset CSV missing required columns: {sorted(missing)}')
        keys = {
            (str(row.rk_serie_id), int(row.cycle_index), str(row.channel))
            for row in subset.itertuples(index=False)
        }
        samples = [
            sample for sample in samples
            if (str(sample.get('rk_serie_id')), int(sample['index']),
                str(sample.get('channel', ''))) in keys
        ]
        data['samples'] = samples
    dmso_cals = data['dmso_cals']
    blanks = prepare_blanks(data['blanks'], selection=blank_selection)
    data['blanks'] = blanks
    ligand_mw_origin = "from_metadata" if ligand_mw is None else "user_defined"
    ligand_mw = ligand_mw if ligand_mw is not None else (data.get('project') or {}).get('ligand_mw_Da')

    if mode not in ('dk', 'ode'):
        raise ValueError(f"mode must be 'dk' or 'ode', got {mode!r}")

    if ode_fit_variant not in {'legacy', 'joint_reference_offset_prefit_basin'}:
        raise ValueError(f'unknown ODE fitting method: {ode_fit_variant}')
    
    n_parallel_jobs = n_parallel_jobs if mode != 'dk' else None

    fit_func = dk_fit_sample if mode == 'dk' else ode_fit_sample
    current_metadata = None
    if mode == 'ode' and ode_fit_variant == 'joint_reference_offset_prefit_basin':
        from .affinity_prior import _validate_prefit_thresholds
        from .current_fitting import fit_sample, _validate_max_cost_ratio
        fit_func = fit_sample
        current_metadata = (
            estimate_capture_levels(data),
            ligand_mw,
            _validate_prefit_thresholds(prefit_thresholds), fit_no_binding,
            _validate_max_cost_ratio(max_cost_ratio),
        )

    n = len(samples)
    if n == 0:
        if current_metadata is not None:
            return pd.DataFrame(), data, []
        print(f"No samples found in file: {filepath}.")
        return pd.DataFrame(), data
    t0 = time.time()

    if n_parallel_jobs:
        backend = None if current_metadata is not None else 'multiprocessing'
        all_results = Parallel(n_jobs=n_parallel_jobs, backend=backend)(
            delayed(_batch_process)(i, t0, n, progress, sample, dmso_cals, blanks, mode,
                                    fit_func, n_starts, fast, blank_selection,
                                    rng_seed, current_metadata, ligand_mw_origin)
            for i, sample in enumerate(samples)
        )
    else:
        all_results = [_batch_process(i, t0, n, progress, sample, dmso_cals, blanks, mode,
                                      fit_func, n_starts, fast, blank_selection,
                                      rng_seed, current_metadata, ligand_mw_origin)
                       for i, sample in enumerate(samples)]

    results = [r[0] for r in all_results]
    rows = [r[1] for r in all_results]
    for row in rows:
        row['blank_selection'] = blank_selection
        if current_metadata is not None:
            row['ode_fit_variant'] = ode_fit_variant

    if progress:
        elapsed = time.time() - t0
        print(f'\r  Done: {n} samples in {elapsed:.1f}s '
              f'({elapsed/n:.1f}s/sample) \n')

    df = pd.DataFrame(rows)

    # Sort by compound then concentration
    df.sort_values(['compound', 'concentration_M'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, data, results


def _batch_process(i, t0, n, progress, sample, dmso_cals, blanks, mode, fit_func,
                   n_starts, fast=True, blank_selection='current', rng_seed=None,
                   current_metadata=None, ligand_mw_origin="from_metadata"):
    """Process a single sample with error handling and NSB filtering."""
    current = current_metadata is not None
    if progress and not current:
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
    # Select blank and DMSO cal closest in time to the sample
    blank = (select_blank(sample['index'], ch_blanks,
                          selection=blank_selection)
             if ch_blanks else None)
    dmso = select_dmso_cal(sample['index'], ch_dmso) if ch_dmso else None

    # Check for negative signal in reference channel before fitting
    heuristics = sensorgram_heuristics(sample, blank=blank)
    kwargs, skip_fields = {}, {}
    if current and "negative_response_in_reference_channel" not in heuristics:
        from .current_fitting import prepare_sample
        kwargs, skip_fields = prepare_sample(sample, blank, dmso, heuristics, *current_metadata)
    if ("negative_response_in_reference_channel" in heuristics
            or (not current and "low_signal_to_noise_response" in heuristics)
            or skip_fields):
        row = _fallback_row(sample, mode)
        _add_blank_metadata(row, blank)
        row['binding'] = False
        row['non_specific'] = True if "non_specific_interaction" in heuristics else False
        row['noisy'] = True if "noisy" in heuristics else False
        row['injection_issue'] = True if "injection_issue" in heuristics else False
        row['accumulation'] = True if "sample_accumulation" in heuristics else False
        row['ligand_mw_Da'] = current_metadata[1] if current_metadata is not None else np.nan
        row['ligand_mw_origin'] = ligand_mw_origin
        row['error'] = np.nan
        row['success'] = np.nan
        row.update(skip_fields)
        return [None, row]

    try:
        kwargs['blank'] = blank
        if mode == 'ode':
            if not current:
                w = get_weight_from_derivative(sample, blank)
                kwargs["association_weight"] = w
            kwargs['n_starts'] = n_starts
            kwargs['fast'] = fast
            kwargs['rng_seed'] = (rng_seed + i if rng_seed is not None else None)
        result = fit_func(sample, dmso, **kwargs)
        _add_blank_metadata(result, blank)
        row = _extract_row(sample, result, mode)
        row['binding'] = True
        row['non_specific'] = True if "non_specific_interaction" in heuristics else False
        row['noisy'] = True if "noisy" in heuristics else False
        row['injection_issue'] = True if "injection_issue" in heuristics else False
        row['accumulation'] = True if "sample_accumulation" in heuristics else False
        row['ligand_mw_Da'] = current_metadata[1] if current_metadata is not None else np.nan
        row['ligand_mw_origin'] = ligand_mw_origin
        row['error'] = np.nan
    except Exception as e:
        row = _fallback_row(sample, mode)
        _add_blank_metadata(row, blank)
        row['binding'] = False
        row['non_specific'] = True if "non_specific_interaction" in heuristics else False
        row['noisy'] = True if "noisy" in heuristics else False
        row['injection_issue'] = True if "injection_issue" in heuristics else False
        row['accumulation'] = True if "sample_accumulation" in heuristics else False
        row['ligand_mw_Da'] = current_metadata[1] if current_metadata is not None else np.nan
        row['ligand_mw_origin'] = ligand_mw_origin
        row['error'] = str(e)
        if current:
            row['success'] = np.nan
        return [None, row]

    return [result, row]


def _add_blank_metadata(target, blank):
    """Record the blank selected for double referencing."""
    target['blank_index'] = blank.get('index') if blank else np.nan


def _extract_row(sample, result, mode):
    """Build a flat dict from sample metadata + fit results."""
    row = {
        'rk_serie_id':      sample.get('rk_serie_id'),
        'cycle_index':      sample['index'],
        'channel':          sample.get('channel', ''),
        'compound_type':    sample['cycle_type'],
        'compound':         sample['compound'],
        'concentration_M':  sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'analyte_mw_Da':    sample.get('mw'),
        'slot':             sample.get('slot'),
        'ka':               result['ka'],
        'kd':               result['kd'],
        'Rmax':             result['Rmax'],
        'Rmax_theory':      result.get('Rmax_theory', np.nan),
        'KD':               result['KD'],
        'KD_uM':            result['KD'] * 1e6,
        'rmse':             result.get('rmse', np.nan),
        'sigma_res':        result['sigma_residual'],
        'n_points':         result.get('n_points', 0),
        'blank_index':      result['blank_index'],
        'dmso_index':       result['dmso_index'],
    }

    if mode == 'dk':
        row.update({
            'fit_mode':     'dk',
            'success':      True,
        })
    else:  # ode
        row.update({
            'ka_se':        result.get('ka_se', np.nan),
            'kd_se':        result.get('kd_se', np.nan),
            'Rmax_se':      result.get('Rmax_se', np.nan),
            'R0':           result.get('R0', np.nan),
            'Rss':          result.get('Rss', np.nan),
            'n_converged':  result.get('n_converged', 0),
            'nfev':         result.get('nfev', 0),
            'seed_method':  result.get('seed_method', np.nan),
            'ka_seed':      result.get('ka_seed', np.nan),
            'kd_seed':      result.get('kd_seed', np.nan),
            'Rmax_seed':    result.get('Rmax_seed', np.nan),
            'KD_seed':      result.get('KD_seed', np.nan),
            'fit_mode':     'ode',
            'success':      result.get('success', False),
            'message':      result.get('message', ''),
        })

    if result.get('prefit_basin_selection_enabled', False):
        row.update({key: value for key, value in result.items()
                    if value is None or np.isscalar(value)})
        theory = result.get('Rmax_theory', np.nan)
        row['Rmax_ratio_theory'] = (
            result['Rmax'] / theory if np.isfinite(theory) and theory > 0 else np.nan)
        for key in ('prefit_basin_requested_bounds', 'reference_scale_bounds'):
            row[key] = result.get(key)
        row.update(kinetic_fit_skipped=False, kinetic_fit_skip_reason='')
    return row


def _fallback_row(sample, mode):
    """Return an all-NaN row when fitting raises an exception."""
    row = {
        'compound_type':    sample['cycle_type'],
        'compound':         sample['compound'],
        'concentration_M':  sample['concentration_M'],
        'concentration_uM': sample['concentration_M'] * 1e6,
        'analyte_mw_Da':    sample.get('mw'),
        'slot':             sample.get('slot'),
        'cycle_index':      sample['index'],
        'channel':          sample.get('channel', ''),
        'rk_serie_id':      sample.get('rk_serie_id'),
        'ka':               np.nan,
        'kd':               np.nan,
        'Rmax':             np.nan,
        'Rmax_theory':      np.nan,
        'KD':               np.nan,
        'KD_uM':            np.nan,
        'rmse':             np.nan,
        'sigma_res':        np.nan,
        'n_points':         0,
        'fit_mode':         mode,
        'success':          False,
    }
    return row


def sensorgram_heuristics(sample, blank=None):
    """Heuristic to flag sensorgram to check if they should be fitted or not.
    Criteria:
    - Noisy: baseline std > 5% of max abs(signal)
    - Injection issue: -10% of max(abs(signal)) > signal before injection > 10% of max(abs(signal))
    - Negative signal in reference channel: min signal < 5% of -max(abs(reference signal))
    - Low signal-to-noise response: binding response < 5% of max(abs(signal))
    - Sample accumulation: steady-state signal > 10% of max(abs(signal))
    - Non-specific binding: signal after rinse in reference channel > 2.5% of max(abs(reference signal))
    """
    signal, _ = double_reference(sample, blank)

    heuristics = []
    noisy, _ = is_baseline_noisy(sample, signal)
    if noisy:
        heuristics.append('noisy')
    inj_error, _ = has_injection_issue(sample)
    if inj_error:
        heuristics.append('injection_issue')
    neg_ref, _ = is_reference_response_negative(sample)
    if neg_ref:
        heuristics.append('negative_response_in_reference_channel')
    low_snr, _ = has_low_signal_to_noise_reponse(sample, signal)
    if low_snr:
        heuristics.append('low_signal_to_noise_response')
    accumulation, _ = is_sample_accumulated(sample, signal)
    if accumulation:
        heuristics.append('sample_accumulation')
    nsb, _ = is_nonspecific_binder(sample)
    if nsb:
        heuristics.append('non_specific_interaction')
    return heuristics


def flag_poor_fits(df, kd_max=9.9, ka_min=0.5,
                   Rmax_min=1.1, sigma_max=2.0,
                   se_threshold=0.5, iqr_threshold=0.25,
                   negligible_binding_amplitude=0.1,
                   bound_limited_binding_amplitude=2.0):
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
    affinity_identifiable = []
    affinity_reasons = []

    for _, row in df.iterrows():
        r = []
        ka = row.get('ka')
        ka_se = row.get('ka_se')
        ka_iqr = row.get('ka_iqr')
        kd = row.get('kd')
        kd_se = row.get('kd_se')
        kd_iqr = row.get('kd_iqr')
        Rmax = row.get('Rmax')
        Rmax_lower_bound = row.get('Rmax_lower_bound')
        Rmax_upper_bound = row.get('Rmax_upper_bound')
        Rmax_se = row.get('Rmax_se')
        Rmax_iqr = row.get('Rmax_iqr')
        sigma_res = row.get('sigma_res')
        concentration = row.get('concentration_M')
        binding_amplitude = row.get('binding_amplitude')
        kinetic_fit_skipped = row.get('kinetic_fit_skipped', False)
        kinetic_fit_skipped = bool(
            pd.notna(kinetic_fit_skipped) and kinetic_fit_skipped)
        skip_reason = row.get('kinetic_fit_skip_reason', '')
        skip_reason = str(skip_reason) if pd.notna(skip_reason) else ''
        if row.get('flag', False):
            r.append(row.get('flag_reason', ''))  # Preserve existing flag reason
        if kinetic_fit_skipped:
            r.append(f'kinetic_fit_skipped:{skip_reason}' if skip_reason else 'kinetic_fit_skipped')
        elif not row.get('success', False):
            r.append('fit_failed')
        if pd.notna(ka) and ka <= ka_min:
            r.append('ka_at_bound')
        if pd.notna(ka_se) and ka_se > se_threshold * abs(ka):
            r.append('ka_high_se')
        if pd.notna(ka_iqr) and ka_iqr > iqr_threshold * abs(ka):
            r.append('ka_high_iqr')
        if pd.notna(kd) and kd >= kd_max:
            r.append('kd_at_bound')
        if pd.notna(kd_se) and kd_se > se_threshold * abs(kd):
            r.append('kd_high_se')
        if pd.notna(kd_iqr) and kd_iqr > iqr_threshold * abs(kd):
            r.append('kd_high_iqr')
        if pd.notna(Rmax) and Rmax <= Rmax_min:
            r.append('low_Rmax')
        if (pd.notna(Rmax) and pd.notna(Rmax_lower_bound)
                and Rmax <= 1.01 * Rmax_lower_bound):
            r.append('Rmax_at_physical_lower_bound')
        if (pd.notna(Rmax) and pd.notna(Rmax_upper_bound)
                and Rmax >= 0.99 * Rmax_upper_bound):
            r.append('Rmax_at_physical_upper_bound')
        if pd.notna(Rmax_se) and Rmax_se > se_threshold * abs(Rmax):
            r.append('Rmax_high_se')
        if pd.notna(Rmax_iqr) and Rmax_iqr > iqr_threshold * abs(Rmax):
            r.append('Rmax_high_iqr')
        if pd.notna(sigma_res) and sigma_res > sigma_max:
            r.append('high_residual')
        early_mismatch = row.get('early_dissociation_mismatch', False)
        if pd.notna(early_mismatch) and bool(early_mismatch):
            r.append('early_dissociation_mismatch')
        prefit_bound_hit = row.get('prefit_basin_bound_hit', False)
        if pd.notna(prefit_bound_hit) and bool(prefit_bound_hit):
            r.append('pKD_at_prefit_basin_bound')

        flags.append(len(r) > 0)
        reasons.append('; '.join(r) if r else '')

        affinity_reason = ''
        positive_concentration = (
            pd.notna(concentration) and float(concentration) > 0)
        if kinetic_fit_skipped:
            affinity_reason = skip_reason or 'kinetic_fit_skipped'
        elif (positive_concentration and pd.notna(binding_amplitude)
                and binding_amplitude < negligible_binding_amplitude):
            affinity_reason = 'negligible_fitted_binding_amplitude'
        elif (
            positive_concentration
            and pd.notna(binding_amplitude)
            and binding_amplitude < bound_limited_binding_amplitude
            and pd.notna(kd)
            and kd >= kd_max
            and pd.notna(Rmax)
            and pd.notna(Rmax_lower_bound)
            and Rmax <= 1.01 * Rmax_lower_bound
        ):
            affinity_reason = 'bound_limited_low_binding_amplitude'
        affinity_identifiable.append(not affinity_reason)
        affinity_reasons.append(affinity_reason)

    df = df.copy()
    df['flag'] = flags
    df['flag_reason'] = reasons
    df['affinity_identifiable'] = affinity_identifiable
    df['affinity_unidentifiable_reason'] = affinity_reasons
    return df
