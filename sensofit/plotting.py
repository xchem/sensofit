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

    # Info box with kinetic parameters
    ka = result.get('ka', np.nan)
    kd = result.get('kd', np.nan)
    KD = result.get('KD', np.nan)
    Rmax = result.get('Rmax', np.nan)
    rmse = result.get('rmse', np.nan)

    info_lines = [
        f'ka  = {ka:.3e} M⁻¹s⁻¹',
        f'kd  = {kd:.3e} s⁻¹',
        f'KD  = {KD:.3e} M',
        f'Rmax = {Rmax:.2f} pg/mm²',
    ]
    if not fit_failed and np.isfinite(rmse):
        info_lines.append(f'RMSE = {rmse:.3f}')

    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                      alpha=0.8))

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


def _save_fit_process(i, row, results, samples, mode, output_dir, blanks=None):
    """Helper for multiprocessing save_fit_plots."""
    idx = row.get('cycle_index')
    ch = row.get('channel', '')
    rk_serie = row.get('rk_serie_id', '')
    match_sample = [(sample_index, sample) for sample_index, sample in enumerate(samples)
                    if sample['index'] == idx
                    and sample.get('channel', '') == ch
                    and sample.get('rk_serie_id', '') == rk_serie]
    if len(match_sample) > 1:
        print(f'WARNING! Multiple samples with RK serie {rk_serie}, cycle number {idx} and channel {ch}, plotting only the first match.')
    elif len(match_sample) == 0:
        print(f'WARNING! No sample found with RK serie {rk_serie}, cycle number {idx} and channel {ch}, skipping plot.')
        return None
    sample_index, sample = match_sample[0]
    result = results[sample_index] if sample_index < len(results) else None
    compound = sample.get('compound', 'Unknown')
    channel = sample.get('channel', ch)
    idx = sample.get('index', idx)
    rk_serie = sample.get('rk_serie_id', rk_serie)
    # Sanitise compound name for filename
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
    blank = _find_selected_blank(result, sample, blanks)
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
    else:
        return None


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
