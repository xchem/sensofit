"""Command-line interface for SensoFit batch processing.

Usage
-----
    python -m sensofit /path/to/folder --mode ode --output results/
    python -m sensofit single_file.cxw --mode dk
    python -m sensofit data_package.zip --mode dk
"""

import argparse
import glob
import os
import sys
import time

import pandas as pd

from .batch import batch_fit, flag_poor_fits
from .plotting import save_fit_plots
from .dataexporter import export_package


def _find_cxw_files(path):
    """Return a list of inputs from a path.

    Accepted inputs:
    - A single ``.cxw`` file.
    - A single ``.zip`` SensoFit data package.
    - A directory containing one or more of the above (``.cxw`` files
      take precedence; if none are found, the directory is itself
      treated as an unzipped SensoFit package).
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.cxw', '.zip'):
            return [path]
        print(f'{path} is not a .cxw or .zip file', file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(path):
        cxws = sorted(glob.glob(os.path.join(path, '*.cxw')))
        zips = sorted(glob.glob(os.path.join(path, '*.zip')))
        if cxws or zips:
            return cxws + zips
        # Treat the directory itself as an unzipped data package.
        if os.path.exists(os.path.join(path, 'README.md')):
            return [path]
        print(f'No .cxw or .zip files found in {path}', file=sys.stderr)
        sys.exit(1)
    print(f'{path} is not a .cxw/.zip file or directory', file=sys.stderr)
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


def _expand_cxw_inputs(paths):
    """Expand a list of file/dir paths into a flat sorted list of .cxw files."""
    out = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith('.cxw'):
            out.append(p)
        elif os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, '*.cxw'))))
        else:
            print(f'Skipping {p}: not a .cxw file or directory',
                  file=sys.stderr)
    return out


def _run_export(argv):
    parser = argparse.ArgumentParser(
        prog='sensofit export',
        description='Package raw .cxw signal data into a self-describing zip '
                    'for data dissemination.',
    )
    parser.add_argument('paths', nargs='+',
                        help='One or more .cxw files or directories.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output .zip path. Default: '
                             'sensofit_package_{timestamp}.zip')
    parser.add_argument('--name', default=None,
                        help='Package name (used in README and as the top-'
                             'level folder inside the zip).')
    args = parser.parse_args(argv)

    cxw_files = _expand_cxw_inputs(args.paths)
    if not cxw_files:
        print('No .cxw files found.', file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        output = f'sensofit_package_{ts}.zip'

    print(f'Exporting {len(cxw_files)} .cxw file(s)...')
    for f in cxw_files:
        print(f'  - {os.path.basename(f)}')
    out_path = export_package(cxw_files, output, package_name=args.name)
    print(f'Wrote package: {out_path}')


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'export':
        _run_export(argv[1:])
        return

    parser = argparse.ArgumentParser(
        prog='sensofit',
        description='Batch kinetic fitting of Creoptix GCI .cxw files or '
                    'exported SensoFit data packages (.zip / unzipped '
                    'directory). Use `sensofit export ...` to package '
                    'raw data instead.',
    )
    parser.add_argument(
        'input',
        help='Path to a .cxw file, a SensoFit data package (.zip or '
             'directory), or a directory containing any of these.',
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
