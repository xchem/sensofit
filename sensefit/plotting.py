"""Plotting utilities for SenseFit kinetic fitting results.

Generates individual data-vs-model PNG plots with an information box
showing compound name, ka, kd, and KD values.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch use
import matplotlib.pyplot as plt


def plot_fit(result, sample, ax=None, title=None):
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
        ax.plot(t[mask], R_fit[mask], color='red', linewidth=1.2,
                label='ODE fit')
    elif 'R_smooth' in result:
        # DK mode: plot smoothed signal
        ax.plot(t, result['R_smooth'], color='blue', linewidth=1.0,
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


def save_fit_plots(results, samples, output_dir, mode='ode'):
    """Save individual fit plots as PNGs.

    Parameters
    ----------
    results : list[dict]
        Fit result dicts, one per sample (same order as *samples*).
    samples : list[dict]
        Sample cycle dicts from ``load_cxw()``.
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

    for i, (result, sample) in enumerate(zip(results, samples)):
        if result is None:
            paths.append(None)
            continue

        compound = sample.get('compound', 'Unknown')
        channel = sample.get('channel', '')
        # Sanitise compound name for filename
        safe_name = _sanitise_filename(compound)
        safe_ch = _sanitise_filename(channel) if channel else ''
        idx = sample.get('index', i)
        parts = [f'{idx:03d}', safe_name]
        if safe_ch:
            parts.append(safe_ch)
        fname = '_'.join(parts) + '.png'
        fpath = os.path.join(output_dir, fname)

        fig = plot_fit(result, sample)
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
