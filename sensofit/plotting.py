"""Plotting utilities for SensoFit kinetic fitting results.

Generates individual data-vs-model PNG plots with an information box
showing compound name, ka, kd, and KD values.  When the selected blank
cycle is supplied, the plot also shows the baseline-corrected raw active
and reference channels alongside the blank used for double referencing.
"""

import os
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from .affinity_prior import _score_to_affinity_label
from .models import double_reference


def _safe_float(value):
    """Coerce scalar metadata to float and keep invalid values as NaN.

    This is needed because plot metadata can come from mixed sources: XML, CSV,
    exported packages, and in-flight fit results. Some entries are numeric, some
    are strings, and some are missing entirely; converting them carefully avoids
    crashing the plotting code while preserving NaN for absent or malformed
    values.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _affinity_range_for_kd(KD):
    """Return the qualitative affinity range for a KD value."""
    KD = _safe_float(KD)
    if not np.isfinite(KD):
        return 'unknown', 'grey'
    if KD > 1e-4:
        return 'weak', 'yellowgreen'
    if KD >= 1e-6:
        return 'medium', 'wheat'
    return 'tight', 'lightcoral'


def _metric_lines_for_result(result, sample):
    """Return a list of metric strings for the info box."""
    if result is None:
        result = {}
    ka = _safe_float(result.get('ka', np.nan))
    kd = _safe_float(result.get('kd', np.nan))
    rmse = _safe_float(result.get('rmse', np.nan))
    sqrt_chi = _safe_float(result.get('sigma_residual', np.nan))

    lines = [
        f'ka  = {ka:.3e}' if np.isfinite(ka) else 'ka  = n/a',
        f'kd  = {kd:.3e}' if np.isfinite(kd) else 'kd  = n/a',
    ]
    if np.isfinite(rmse):
        lines.append(f'RMSE = {rmse:.3f}')
    if np.isfinite(sqrt_chi):
        lines.append(f'sqrt(chi2) = {sqrt_chi:.3f}')

    prior_score = None
    if sample is not None:
        for key in ('pre_fit_score', 'prefit_score', 'score', 'affinity_score'):
            value = sample.get(key)
            if value is None:
                value = result.get(key)
            if value is not None:
                prior_score = _safe_float(value)
                break
        if prior_score is None:
            try:
                from sensofit.affinity_prior import estimate_affinity_area_prior
                prior_score = estimate_affinity_area_prior(sample).score
            except Exception:
                prior_score = np.nan
    if np.isfinite(prior_score):
        label = _score_to_affinity_label(prior_score)
        lines.append(f'pre-fit score = {prior_score:.3f} ({label})')
    return lines


def _build_no_fit_reason(result, sample, row=None):
    """Return a human-readable reason why a fit is unavailable."""
    if row is not None:
        error_value = row.get('error')
        if error_value is not None and not (isinstance(error_value, float) and np.isnan(error_value)):
            return str(error_value)
        binding_value = row.get('binding')
        if binding_value is False:
            return 'No Binding'

    if result is not None and isinstance(result, dict):
        for key in ('reason', 'fit_reason', 'error', 'message', 'status'):
            value = result.get(key)
            if value:
                return str(value)
        if result.get('success') is False:
            return 'Fit failed without a recorded reason.'
    if sample is not None:
        return 'No fit result available for this sample.'
    return 'No fit result available.'


def _prefit_score_and_label(sample, result=None):
    """Return the pre-fit affinity score and human label for a sample."""
    score = None
    if sample is not None:
        score_map = (
            ('pre_fit_score', 'prefit_score', 'score', 'affinity_score')
        )
        for key in score_map:
            for candidate in key if isinstance(key, tuple) else (key,):
                if candidate in sample:
                    score = _safe_float(sample.get(candidate))
                    if np.isfinite(score):
                        return score, _score_to_affinity_label(score)
        if result is not None and isinstance(result, dict):
            for key in ('pre_fit_score', 'prefit_score', 'score', 'affinity_score'):
                score = _safe_float(result.get(key))
                if np.isfinite(score):
                    return score, _score_to_affinity_label(score)
        try:
            from sensofit.affinity_prior import estimate_affinity_area_prior
            score = estimate_affinity_area_prior(sample).score
            return score, _score_to_affinity_label(score)
        except Exception:
            return np.nan, 'unknown'
    if result is not None and isinstance(result, dict):
        for key in ('pre_fit_score', 'prefit_score', 'score', 'affinity_score'):
            score = _safe_float(result.get(key))
            if np.isfinite(score):
                return score, _score_to_affinity_label(score)
    return np.nan, 'unknown'


def _render_fit_panel(ax, sample, blank, result=None, mode='ode'):
    """Render the main fit panel for one repeat."""
    result = {} if result is None else result
    t = np.asarray(result.get('t', sample.get('time', [])), dtype=float)
    signal = np.asarray(result.get('signal', sample.get('signal', [])), dtype=float)
    if t.size and signal.size:
        if result.get('signal') is not None:
            signal_ref = signal
        elif blank is not None:
            signal_ref, _ = double_reference(sample, blank)
        else:
            signal_ref = signal
        if signal_ref.size:
            ax.plot(t, signal_ref, color='black', linewidth=0.8, label='Double reference')
    if result.get('R_fit') is not None:
        R_fit = np.asarray(result['R_fit'])
        mask = np.isfinite(R_fit)
        fit_failed = result.get('success') is False
        fit_color = 'orange' if fit_failed else ('red' if mode == 'ode' else 'blue')
        fit_label = 'Fallback estimate' if fit_failed else 'Fit'
        if mask.any():
            ax.plot(t[mask], R_fit[mask], color=fit_color, linewidth=1.2,
                    linestyle='--', label=fit_label)
    if 'R_smooth' in result:
        ax.plot(t, result['R_smooth'], color='grey', linewidth=0.8,
                linestyle='--', label='DK smooth')

    ax.set_title('Fit and double reference', fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response (pg/mm²)')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc='upper right', fontsize=9)

    info_lines = _metric_lines_for_result(result, sample)
    if result is None or result == {}:
        score, label = _prefit_score_and_label(sample, result)
        reason = _build_no_fit_reason(result, sample)
        info_lines = [
            f'pre-fit score = {score:.3f} ({label})' if np.isfinite(score) else 'pre-fit score = n/a',
            f'reason = {reason}',
        ]
    KD = _safe_float(result.get('KD', np.nan))
    Rmax = _safe_float(result.get('Rmax', np.nan))
    affinity_label, affinity_color = _affinity_range_for_kd(KD)
    affinity_lines = [
        f'KD   = {KD:.3e} M' if np.isfinite(KD) else 'KD   = n/a',
        f'Rmax = {Rmax:.2f} pg/mm²' if np.isfinite(Rmax) else 'Rmax = n/a',
        f'Affinity range = {affinity_label}',
    ]
    ax.text(0.02, 0.98, '\n'.join(affinity_lines), transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=affinity_color,
                      alpha=0.85, edgecolor='black'))

    ax.text(0.02, 0.80, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightgrey', alpha=0.85))

    if result.get('success') is False:
        ax.text(0.5, 0.98, 'FIT FAILED', transform=ax.transAxes, fontsize=10,
                fontweight='bold', color='white', horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='firebrick',
                          edgecolor='firebrick', alpha=0.9))


def plot_fit(result, sample, mode='ode', ax=None, title=None, blank=None):
    """Plot data vs model fit for a single sample.

    Parameters
    ----------
    result : dict
        Fit result from ``fit_sample()`` (ODE or DK mode).
    sample : dict
        Sample cycle dict from ``load_cxw()``.
    ax : matplotlib.axes.Axes or None
        If provided, plot on this axes. Otherwise create a new figure.
    title : str or None
        Override title. Default: compound name.
    blank : dict or None
        Blank cycle used for double referencing.  When provided, a second
        panel is added showing baseline-corrected ``raw_active``,
        ``raw_reference``, and blank ``signal`` traces.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure, or None if *ax* was provided.
    """
    t = np.asarray(result['t'])
    signal = np.asarray(result['signal'])

    fig = None
    if ax is None:
        if blank is not None:
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
            raw_ax, ax = axes
            _plot_raw_channels(raw_ax, sample, blank)
        else:
            # Keep the single-axis behaviour for callers that do not have
            # the selected blank available (and for minimal result fixtures).
            fig, ax = plt.subplots(figsize=(8, 5))

    # Data trace
    ax.plot(t, signal, color='black', linewidth=0.8, label='Sensorgram')

    # Model fit trace
    R_fit = result.get('R_fit')
    fit_failed = result.get('success') is False
    if R_fit is not None:
        # ODE mode: R_fit has NaN outside fit window
        mask = np.isfinite(R_fit)
        fit_color = 'orange' if fit_failed else ('red' if mode == 'ode' else 'blue')
        fit_label = 'Fallback estimate' if fit_failed else 'Fit'
        ax.plot(t[mask], R_fit[mask], color=fit_color, linewidth=1.2,
                linestyle='--', label=fit_label)
    if 'R_smooth' in result:
        # DK mode: plot smoothed signal
        ax.plot(t, result['R_smooth'], color='grey', linewidth=0.8,
                linestyle='--', label='DK smooth')

    # Labels
    compound = sample.get('compound', 'Unknown')
    conc_uM = sample.get('concentration_M', 0) * 1e6
    channel = sample.get('channel', '')
    if title is None:
        title = f'{compound}  ({conc_uM:.2f} µM)'
        if channel:
            title += f'  [{channel}]'
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response (pg/mm²)')
    ax.legend(loc='upper right', fontsize=10)

    # Primary metric box
    ka = result.get('ka', np.nan)
    kd = result.get('kd', np.nan)
    rmse = result.get('rmse', np.nan)

    info_lines = [
        f'ka  = {ka:.3e} M⁻¹s⁻¹',
        f'kd  = {kd:.3e} s⁻¹',
    ]
    if not fit_failed and np.isfinite(rmse):
        info_lines.append(f'RMSE = {rmse:.3f}')

    KD = result.get('KD', np.nan)
    Rmax = result.get('Rmax', np.nan)
    affinity_label, affinity_color = _affinity_range_for_kd(KD)
    affinity_lines = [
        f'KD   = {KD:.3e} M',
        f'Rmax = {Rmax:.2f} pg/mm²',
        f'Affinity range = {affinity_label}',
    ]
    ax.text(0.02, 0.98, '\n'.join(affinity_lines),
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontweight='bold',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=affinity_color,
                      alpha=0.85, edgecolor='black'))

    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.56, info_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgrey',
                      alpha=0.85))

    if fit_failed:
        ax.text(0.5, 0.98, 'FIT FAILED',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                color='white', horizontalalignment='center',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='firebrick',
                          edgecolor='firebrick', alpha=0.9))

    ax.grid(True, alpha=0.3)

    if fig is not None:
        fig.tight_layout()

    return fig


def _baseline_corrected_trace(cycle, value_key):
    """Return a cycle's time/value trace after baseline subtraction.

    The loader normally supplies ``baseline_duration_s``.  Older exported
    packages may not, so the Injection marker is used as a conservative
    fallback.  The returned arrays are truncated to their common length.
    """
    time = np.asarray(cycle.get('time', []), dtype=float)
    values = np.asarray(cycle.get(value_key, []), dtype=float)
    n = min(time.size, values.size)
    time = time[:n]
    values = values[:n]
    if n == 0:
        return time, values

    baseline_duration = cycle.get('baseline_duration_s')
    try:
        baseline_duration = float(baseline_duration)
    except (TypeError, ValueError):
        baseline_duration = np.nan
    if not np.isfinite(baseline_duration):
        baseline_duration = cycle.get('markers', {}).get('Injection', time[0])

    baseline_mask = time <= baseline_duration
    baseline = values[baseline_mask].mean() if baseline_mask.any() else values[0]
    return time, values - baseline


def _plot_raw_channels(ax, sample, blank):
    """Plot the raw channels and selected blank used by a sample."""
    t_ref, reference = _baseline_corrected_trace(sample, 'raw_reference')
    t_active, active = _baseline_corrected_trace(sample, 'raw_active')
    if t_ref.size:
        ax.plot(t_ref, reference, color='blue', linewidth=0.8,
                label='Ref. channel')
    if t_active.size:
        ax.plot(t_active, active, color='red', linewidth=0.8,
                label='Active channel')

    t_blank, blank_signal = _baseline_corrected_trace(blank, 'signal')
    if t_blank.size:
        ax.plot(t_blank, blank_signal, color='grey', linewidth=0.75,
                label='Blank')

    blank_index = blank.get('index')
    blank_title = 'Raw channels and selected blank'
    if blank_index is not None:
        blank_title += f' (cycle {blank_index})'
    ax.set_title(blank_title, fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response (pg/mm²)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def save_fit_plots(df, samples, results, output_dir, mode='ode',
                   n_parallel_jobs=None, blanks=None):
    """Save individual fit plots as PNGs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing fit results.
    samples : list[dict]
        Sample cycle dicts from ``load_cxw()``.
    results : list[dict or None]
        List of fit result dicts (one per sample), or None for failed fits.
    output_dir : str
        Directory to write PNG files into (created if needed).
    mode : str
        Label for the fit mode ('ode' or 'dk').
    blanks : list[dict] or None
        Blank cycles from the same experiment.  The blank whose index is
        stored in each fit result is passed to :func:`plot_fit`.

    Returns
    -------
    paths : list[str]
        File paths of the saved PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if n_parallel_jobs:
        paths = Parallel(n_jobs=n_parallel_jobs, backend="multiprocessing")(
            delayed(_save_fit_process)(i, row, results, samples, mode,
                                       output_dir, blanks)
            for i, row in df.iterrows())
    else:
        paths = [_save_fit_process(i, row, results, samples, mode,
                                   output_dir, blanks)
                 for i, row in df.iterrows()]

    return paths


def _resolve_sample_and_result_for_row(row, samples, results):
    """Resolve the matching sample + result for a DataFrame row."""
    row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    idx = row_dict.get('cycle_index', row_dict.get('index'))
    ch = row_dict.get('channel', '')
    rk_serie = row_dict.get('rk_serie_id', '')

    matches = [(sample_index, sample) for sample_index, sample in enumerate(samples)
               if sample.get('index') == idx
               and sample.get('channel', '') == ch
               and sample.get('rk_serie_id', '') == rk_serie]
    if len(matches) > 1:
        print(f'WARNING! Multiple samples with RK serie {rk_serie}, cycle number {idx} and channel {ch}, plotting only the first match.')
    if not matches:
        return None, None, None

    sample_index, sample = matches[0]
    result = results[sample_index] if sample_index < len(results) else None
    if result is not None and getattr(result, 'get', None) is None:
        result = None
    return sample, sample_index, result


def _resolve_blank_for_row(row, sample, result, blanks):
    """Resolve the blank using the row metadata before falling back to the result."""
    blank = None
    if row is not None:
        blank = _find_selected_blank_from_row(row, blanks)
    if blank is None and result is not None:
        blank = _find_selected_blank(result, sample, blanks)
    return blank


def _save_fit_process(i, row, results, samples, mode, output_dir, blanks=None):
    """Helper for multiprocessing save_fit_plots."""
    row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    idx = row_dict.get('cycle_index', row_dict.get('index'))
    ch = row_dict.get('channel', '')
    rk_serie = row_dict.get('rk_serie_id', '')

    sample, _, result = _resolve_sample_and_result_for_row(row, samples, results)
    if sample is None:
        print(f'WARNING! No sample found with RK serie {rk_serie}, cycle number {idx} and channel {ch}, skipping plot.')
        return None

    compound = sample.get('compound', 'Unknown')
    channel = sample.get('channel', ch)
    idx = sample.get('index', idx)
    rk_serie = sample.get('rk_serie_id', rk_serie)
    safe_name = _sanitise_filename(compound)
    safe_ch = _sanitise_filename(channel) if channel else ''
    parts = [f'RK{rk_serie:02d}', f'{idx:03d}', safe_name]
    if safe_ch:
        parts.append(safe_ch)
    fname = '_'.join(parts) + '_' + mode.upper() + '.png'
    fpath = os.path.join(output_dir, fname)

    if result is None:
        print(f'WARNING! No fit result for sample with RK serie {rk_serie}, cycle number {idx} and channel {ch}, skipping plot.')
        return None

    blank = _resolve_blank_for_row(row_dict, sample, result, blanks)

    fig = plot_fit(
        result,
        sample,
        mode=mode,
        blank=blank,
    )
    if fig is not None:
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fpath
    return None


def _save_plot_process(rows, samples, results, mode, output_dir, blanks=None):
    """Save a single triplicate subplot figure for one grouped DataFrame row set."""
    rows = list(rows)
    if not rows:
        return None

    first_row = rows[0]
    rk_serie = first_row.get('rk_serie_id', '')
    cycle_index = first_row.get('cycle_index', first_row.get('index', 0))
    compound = first_row.get('compound', 'Unknown')
    safe_name = _sanitise_filename(compound)
    fpath = os.path.join(output_dir, f'RK{rk_serie:02d}_{cycle_index:03d}_{safe_name}_{mode.upper()}_triplicate.png')

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), squeeze=False,
                             sharex='col', sharey='row')

    for row_idx in range(3):
        for col_idx in range(2):
            axes[row_idx, col_idx].set_visible(True)
            axes[row_idx, col_idx].grid(True, alpha=0.25)
            axes[row_idx, col_idx].set_axis_on()

    for row_idx, row in enumerate(rows[:3]):
        sample, _, result = _resolve_sample_and_result_for_row(row, samples, results)
        if sample is None:
            continue

        blank = _resolve_blank_for_row(row, sample, result, blanks)

        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        ax_left.set_visible(True)
        ax_right.set_visible(True)

        if blank is not None:
            _plot_raw_channels(ax_left, sample, blank)
        else:
            print(f'WARNING! No blank found for sample with RK serie {rk_serie}, cycle number {cycle_index} and channel {sample.get("channel", "")}, plotting raw signal only.')
            t = np.asarray(sample.get('time', []), dtype=float)
            signal = np.asarray(sample.get('raw_active', []), dtype=float)
            if t.size and signal.size:
                ax_left.plot(t, signal, color='black', linewidth=0.8, label='Signal')
            ax_left.set_title('Raw signal', fontsize=11)
            ax_left.set_xlabel('Time (s)')
            ax_left.set_ylabel('Response (pg/mm²)')
            ax_left.grid(True, alpha=0.3)
            ax_left.legend(loc='upper right', fontsize=9)

        if result is not None and result.get('success') is True:
            _render_fit_panel(ax_right, sample, blank, result=result, mode=mode)
            continue

        score, label = _prefit_score_and_label(sample, result)
        reason = _build_no_fit_reason(result, sample, row=row)
        if reason == 'No Binding':
            reason = 'No binding'

        if result is not None:
            t_double = np.asarray(result.get('t', sample.get('time', [])), dtype=float)
            signal_double = np.asarray(result.get('signal', sample.get('raw_active', [])), dtype=float)
        else:
            t_double = np.asarray(sample.get('time', []), dtype=float)
            signal_double = np.asarray(sample.get('raw_active', []), dtype=float)

        if blank is not None and t_double.size and signal_double.size:
            signal_double, _ = double_reference(sample, blank)
            if signal_double.size:
                ax_right.plot(t_double, signal_double, color='black', linewidth=0.8, label='Double reference')
        elif t_double.size and signal_double.size:
            ax_right.plot(t_double, signal_double, color='black', linewidth=0.8, label='Double reference')
        ax_right.set_title('Double reference', fontsize=11)
        ax_right.set_xlabel('Time (s)')
        ax_right.set_ylabel('Response (pg/mm²)')
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc='upper right', fontsize=9)

        text = f'pre-fit score = {score:.3f} ({label})\nreason = {reason}' if np.isfinite(score) else f'reason = {reason}'
        ax_right.text(0.02, 0.98, text, transform=ax_right.transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.35', facecolor='lightgrey', alpha=0.8))

    fig.suptitle(f'{compound} (RK{rk_serie:02d}, cycle {cycle_index})', fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def save_plot(df, samples, results, output_dir, mode='ode',
              n_parallel_jobs=None, blanks=None):
    """Save one grouped figure containing up to three replicate traces.

    Triplicates are first pooled from the DataFrame rows by the stable assay
    identity shared across repeat samples in different channels, then each group
    is resolved back to matching sample/result entries for plotting.
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped_rows = []
    for _, row in df.iterrows():
        key = (
            row.get('rk_serie_id'),
            row.get('cycle_index', row.get('index')),
            row.get('cycle_type'),
            row.get('compound'),
            row.get('concentration_M'),
            row.get('mw'),
        )
        for triplicate in grouped_rows:
            first_row = triplicate[0]
            existing_key = (
                first_row.get('rk_serie_id'),
                first_row.get('cycle_index', first_row.get('index')),
                first_row.get('cycle_type'),
                first_row.get('compound'),
                first_row.get('concentration_M'),
                first_row.get('mw'),
            )
            if existing_key == key:
                triplicate.append(row)
                break
        else:
            grouped_rows.append([row])

    total = len(grouped_rows)
    if total:
        print(f'Generating grouped triplicate plots: 0/{total}', end='\r')

    if n_parallel_jobs:
        paths = Parallel(n_jobs=n_parallel_jobs, backend="multiprocessing")(
            delayed(_save_plot_process)(rows, samples, results, mode, output_dir, blanks)
            for rows in grouped_rows
        )
    else:
        paths = []
        for progress_index, rows in enumerate(grouped_rows, start=1):
            paths.append(_save_plot_process(rows, samples, results, mode, output_dir, blanks))
            if total:
                print(f'Generating grouped triplicate plots: {progress_index}/{total}', end='\r')
        if total:
            print(f'Generating grouped triplicate plots: {total}/{total}  ')
    return paths


def _find_selected_blank_from_row(row, blanks):
    """Find the blank for a row using the blank metadata stored in the DataFrame."""
    if not blanks or row is None:
        return None
    blank_index = row.get('blank_index')
    try:
        blank_index_is_finite = bool(np.isfinite(blank_index))
    except (TypeError, ValueError):
        blank_index_is_finite = False
    if blank_index is None or not blank_index_is_finite:
        return None

    candidates = [blank for blank in blanks if blank.get('index') == blank_index]
    if not candidates:
        return None

    channel = row.get('channel')
    rk_serie_id = row.get('rk_serie_id')
    contextual = [blank for blank in candidates
                  if blank.get('channel') == channel
                  and blank.get('rk_serie_id') == rk_serie_id]
    return contextual[0] if contextual else candidates[0]


def _find_selected_blank(result, sample, blanks):
    """Find the blank selected by fitting for the sample being plotted."""
    if not blanks:
        return None
    blank_index = result.get('blank_index')
    try:
        blank_index_is_finite = bool(np.isfinite(blank_index))
    except (TypeError, ValueError):
        blank_index_is_finite = False
    if blank_index is None or not blank_index_is_finite:
        return None

    candidates = [blank for blank in blanks
                  if blank.get('index') == blank_index]
    if not candidates:
        return None

    channel = sample.get('channel')
    rk_serie_id = sample.get('rk_serie_id')
    contextual = [blank for blank in candidates
                  if blank.get('channel') == channel
                  and blank.get('rk_serie_id') == rk_serie_id]
    return contextual[0] if contextual else candidates[0]
    


def _sanitise_filename(name):
    """Replace characters unsafe for filenames."""
    keep = set('abcdefghijklmnopqrstuvwxyz'
               'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
               '0123456789_-.')
    return ''.join(c if c in keep else '_' for c in name)
