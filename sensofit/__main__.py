"""Command-line interface for SensoFit batch processing.

Usage
-----
    python -m sensofit /path/to/folder --mode ode --output results/
    python -m sensofit single_file.cxw --mode dk
"""

import argparse
import glob
import os
import sys
import time

import pandas as pd

from .batch import batch_fit, flag_poor_fits
from .plotting import save_fit_plots


def _find_cxw_files(path):
    """Return list of .cxw files from a path (file or directory)."""
    if os.path.isfile(path) and path.lower().endswith('.cxw'):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.cxw')))
        if not files:
            print(f'No .cxw files found in {path}', file=sys.stderr)
            sys.exit(1)
        return files
    print(f'{path} is not a .cxw file or directory', file=sys.stderr)
    sys.exit(1)


def _run_mode(filepath, mode, skip_nsb, output_dir, channels='all'):
    """Run batch_fit for one file in one mode, save plots and return df."""
    basename = os.path.splitext(os.path.basename(filepath))[0]

    print(f'\n{"=" * 60}')
    print(f'File: {os.path.basename(filepath)}')
    print(f'Mode: {mode.upper()}')
    print(f'{"=" * 60}')

    df, data = batch_fit(filepath, mode=mode, include_nsb=not skip_nsb,
                         channels=channels, progress=True)

    # Add source file info
    df.insert(0, 'source_file', os.path.basename(filepath))

    # Quality flags
    df = flag_poor_fits(df)

    # Filter NSB if requested
    if skip_nsb and 'nonspecific' in df.columns:
        n_nsb = df['nonspecific'].sum()
        if n_nsb > 0:
            print(f'  Skipping {n_nsb} non-specific binder(s)')

    # Save plots
    samples = data['samples']
    dmso_cals = data['dmso_cals']
    blanks = data['blanks']
    matched_results = []
    for _, row in df.iterrows():
        idx = row.get('cycle_index')
        ch = row.get('channel', '')
        match = [s for s in samples
                 if s['index'] == idx and s.get('channel', '') == ch]
        if not match:
            matched_results.append(None)
            continue
        sample = match[0]
        if row.get('nonspecific', False) or not row.get('success', False):
            matched_results.append(None)
        else:
            try:
                if mode == 'dk':
                    from .direct_kinetics import fit_sample as fit_fn
                else:
                    from .ode_fitting import fit_sample as fit_fn
                # Channel-matched DMSO/blanks
                ch_dmso = [d for d in dmso_cals if d.get('channel') == ch]
                ch_blanks = [b for b in blanks if b.get('channel') == ch]
                result = fit_fn(sample, ch_dmso or dmso_cals,
                                blanks=ch_blanks or blanks)
                matched_results.append(result)
            except Exception:
                matched_results.append(None)

    plot_dir = os.path.join(output_dir, f'{basename}_{mode}_plots')
    matched_samples = []
    for _, row in df.iterrows():
        idx = row.get('cycle_index')
        ch = row.get('channel', '')
        match = [s for s in samples
                 if s['index'] == idx and s.get('channel', '') == ch]
        if match:
            matched_samples.append(match[0])
        else:
            matched_samples.append({'compound': 'Unknown', 'concentration_M': 0,
                                    'index': idx, 'channel': ch})

    if matched_results:
        paths = save_fit_plots(matched_results, matched_samples,
                               plot_dir, mode=mode)
        n_plots = sum(1 for p in paths if p is not None)
        print(f'  Saved {n_plots} plot(s) → {plot_dir}/')

    return df


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='sensofit',
        description='Batch kinetic fitting of Creoptix GCI .cxw files.',
    )
    parser.add_argument(
        'input',
        help='Path to a .cxw file or a directory containing .cxw files.',
    )
    parser.add_argument(
        '--mode', choices=['dk', 'ode', 'both'], default='ode',
        help='Fitting mode: dk (fast), ode (full), or both. Default: ode.',
    )
    parser.add_argument(
        '--output', '-o', default='results',
        help='Output directory for CSV and plots. Default: results/',
    )
    parser.add_argument(
        '--include-nsb', action='store_true', default=False,
        help='Include non-specific binders in fitting (default: skip).',
    )
    parser.add_argument(
        '--channels', nargs='*', type=int, default=None,
        help='Active flow cell numbers to process (e.g. --channels 2 3). '
             'Default: all active channels.',
    )

    args = parser.parse_args(argv)

    cxw_files = _find_cxw_files(args.input)
    os.makedirs(args.output, exist_ok=True)
    skip_nsb = not args.include_nsb
    channels = args.channels if args.channels else 'all'

    modes = ['dk', 'ode'] if args.mode == 'both' else [args.mode]

    all_dfs = []
    t0 = time.time()

    for filepath in cxw_files:
        for mode in modes:
            df = _run_mode(filepath, mode, skip_nsb, args.output,
                          channels=channels)
            all_dfs.append(df)

    # Combine and save
    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns: source_file, cycle_index, compound first
    priority = ['source_file', 'cycle_index', 'channel', 'compound',
                'concentration_M', 'concentration_uM', 'fit_mode',
                'ka', 'kd', 'KD', 'KD_uM',
                'Rmax', 'sigma_res', 'flag', 'flag_reason']
    ordered = [c for c in priority if c in combined.columns]
    remaining = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + remaining]

    csv_path = os.path.join(args.output, 'batch_results.csv')
    combined.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'Complete: {len(cxw_files)} file(s), {len(combined)} rows')
    print(f'CSV  → {csv_path}')
    print(f'Time → {elapsed:.1f}s')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
