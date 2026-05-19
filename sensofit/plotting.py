"""Plotting utilities for SensoFit kinetic fitting results.

Generates individual data-vs-model PNG plots with an information box
showing compound name, ka, kd, and KD values.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_fit(result, sample, mode='ode', ax=None, title=None):
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

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure, or None if *ax* was provided.
    """
    t = result['t']
    signal = result['signal']

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Data trace
    ax.plot(t, signal, color='black', linewidth=0.8, label='Data')

    # Model fit trace
    R_fit = result.get('R_fit')
    if R_fit is not None:
        # ODE mode: R_fit has NaN outside fit window
        mask = np.isfinite(R_fit)
        ax.plot(t[mask], R_fit[mask], color='red' if mode == 'ode' else 'blue', linewidth=1.2,
                linestyle='--', label='ODE fit')
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
    ax.legend(loc='upper right', fontsize=8)

    # Info box with kinetic parameters
    ka = result.get('ka', np.nan)
    kd = result.get('kd', np.nan)
    KD = result.get('KD', np.nan)
    Rmax = result.get('Rmax', np.nan)
    sigma = result.get('sigma_residual', np.nan)

    info_lines = [
        f'ka  = {ka:.3e} M⁻¹s⁻¹',
        f'kd  = {kd:.3e} s⁻¹',
        f'KD  = {KD:.3e} M',
        f'Rmax = {Rmax:.2f} pg/mm²',
    ]
    if np.isfinite(sigma):
        info_lines.append(f'σ_res = {sigma:.3f}')

    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                      alpha=0.8))

    ax.grid(True, alpha=0.3)

    if fig is not None:
        fig.tight_layout()

    return fig


def save_fit_plots(df, samples, results, output_dir, mode='ode'):
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

    Returns
    -------
    paths : list[str]
        File paths of the saved PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, row in df.iterrows():
        idx = row.get('cycle_index')
        ch = row.get('channel', '')
        rk_serie = row.get('rk_serie_id', '')
        match_sample = [s for s in samples
                 if s['index'] == idx and s.get('channel', '') == ch and s.get('rk_serie_id', '') == rk_serie]
        if len(match_sample) > 1:
            print(f'WARNING! Multiple samples with RK serie {rk_serie}, cycle number {idx} and channel {ch}, plotting only the first match.')
        elif len(match_sample) == 0:
            print(f'WARNING! No sample found with RK serie {rk_serie}, cycle number {idx} and channel {ch}, skipping plot.')
            paths.append(None)
            continue
        sample = match_sample[0]
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

        result = results[i]
        if result is None:
            print(f'WARNING! No fit result for sample with RK serie {rk_serie}, cycle number {idx} and channel {ch}, skipping plot.')
            paths.append(None)
            continue
        fig = plot_fit(result, sample, mode=mode)
        if fig is not None:
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            paths.append(fpath)
        else:
            paths.append(None)

    return paths


def _sanitise_filename(name):
    """Replace characters unsafe for filenames."""
    keep = set('abcdefghijklmnopqrstuvwxyz'
               'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
               '0123456789_-.')
    return ''.join(c if c in keep else '_' for c in name)
