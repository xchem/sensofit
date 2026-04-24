"""Re-ingest a SensoFit data package as if it were a ``.cxw`` file.

The data package produced by :func:`sensofit.dataexporter.export_package`
serialises every field that the SensoFit pipeline (batch / direct /
ode fitting, plotting, CLI) reads from a :func:`load_cxw` dict.  This
module reads such a package and returns the **same dict shape** so that
all downstream code works unchanged.

Two entry points are provided:

- :func:`list_experiments` — enumerate experiments in a package.
- :func:`load_package` — load one experiment (default: the first) and
  return a dict matching :func:`sensofit.data_loader.load_cxw`.

Packages may contain multiple experiments (one per CXW originally
exported); use ``name=...`` to pick a specific one.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile

import numpy as np


# ---------------------------------------------------------------------------
# Source abstraction (zip file or unzipped directory, transparently)
# ---------------------------------------------------------------------------

class _PackageSource:
    """Read JSON/CSV files from either a zip archive or a directory tree.

    Paths used internally are POSIX-style relative to the package root
    (the directory that holds ``README.md`` and one or more experiment
    folders).
    """

    def __init__(self, path: str):
        self.path = path
        if os.path.isdir(path):
            self._zip = None
            self._root = path
            # Auto-descend if user pointed at the parent of the package
            entries = os.listdir(path)
            if 'README.md' not in entries:
                subs = [e for e in entries
                        if os.path.isdir(os.path.join(path, e))
                        and os.path.exists(
                            os.path.join(path, e, 'README.md'))]
                if len(subs) == 1:
                    self._root = os.path.join(path, subs[0])
            self._names = self._list_dir_recursive(self._root)
        else:
            self._zip = zipfile.ZipFile(path, 'r')
            names = self._zip.namelist()
            roots = sorted({n.split('/', 1)[0] for n in names if '/' in n})
            self._root = ''
            for r in roots:
                if f'{r}/README.md' in names:
                    self._root = r
                    break
            self._names = [n for n in names if not n.endswith('/')]

    @staticmethod
    def _list_dir_recursive(root: str) -> list:
        out = []
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, root).replace(os.sep, '/')
            for f in files:
                out.append(f if rel == '.' else f'{rel}/{f}')
        return out

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _arc(self, rel: str) -> str:
        return f'{self._root}/{rel}' if self._root else rel

    def exists(self, rel: str) -> bool:
        if self._zip is None:
            return os.path.exists(os.path.join(self._root, rel))
        return self._arc(rel) in set(self._names)

    def read_bytes(self, rel: str) -> bytes:
        if self._zip is None:
            with open(os.path.join(self._root, rel), 'rb') as fh:
                return fh.read()
        return self._zip.read(self._arc(rel))

    def read_text(self, rel: str) -> str:
        return self.read_bytes(rel).decode('utf-8')

    def read_json(self, rel: str):
        return json.loads(self.read_text(rel))

    def list_subdirs(self, rel: str = '') -> list:
        """List immediate sub-directories under ``rel`` (POSIX names)."""
        prefix = (rel + '/') if rel else ''
        if self._zip is None:
            base = os.path.join(self._root, rel) if rel else self._root
            if not os.path.isdir(base):
                return []
            return sorted(e for e in os.listdir(base)
                          if os.path.isdir(os.path.join(base, e)))
        seen = set()
        arc_prefix = self._arc(prefix) if prefix else (
            (self._root + '/') if self._root else '')
        for n in self._names:
            if not n.startswith(arc_prefix):
                continue
            tail = n[len(arc_prefix):]
            if '/' in tail:
                seen.add(tail.split('/', 1)[0])
        return sorted(seen)


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def _experiment_dirs(src: _PackageSource) -> list:
    """Return CXW-folder names that contain an ``experiment.json``."""
    out = []
    for d in src.list_subdirs(''):
        if src.exists(f'{d}/experiment.json'):
            out.append(d)
    return out


def list_experiments(path: str) -> list:
    """List experiment folder names contained in a SensoFit data package.

    Parameters
    ----------
    path : str
        Path to the ``.zip`` package or to an unzipped package directory.

    Returns
    -------
    list[str]
        Folder names (one per CXW originally exported), alphabetical.
    """
    with _PackageSource(path) as src:
        return _experiment_dirs(src)


# ---------------------------------------------------------------------------
# Adapters: package JSON/CSV → load_cxw dict shape
# ---------------------------------------------------------------------------

# metadata.json fields injected by the exporter that have no counterpart
# in the original load_cxw cycle dict and must be dropped on round-trip.
_METADATA_DROP = ('mw_Da', 'concentration_pretty', 'channels')


def _cycle_meta_from_json(meta: dict) -> dict:
    """Convert exporter ``metadata.json`` → ``load_cxw`` cycle dict."""
    out = {k: v for k, v in meta.items() if k not in _METADATA_DROP}
    # Field rename: dataexporter writes mw_Da; loader uses mw.
    out['mw'] = float(meta.get('mw_Da', 0.0))
    if 'pulse_durations_s' in out and out['pulse_durations_s'] is not None:
        out['pulse_durations_s'] = [float(x) for x in out['pulse_durations_s']]
    if 'markers' in out and out['markers'] is not None:
        out['markers'] = {k: float(v) for k, v in out['markers'].items()}
    if 'enabled' in out:
        out['enabled'] = bool(out['enabled'])
    if 'index' in out:
        out['index'] = int(out['index'])
    return out


def _read_channel_csv(src: _PackageSource, rel: str):
    """Load one ``FCx-FCy.csv`` into four float64 numpy arrays."""
    text = src.read_text(rel)
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    cols = {name: [] for name in header}
    for row in reader:
        for name, val in zip(header, row):
            cols[name].append(float(val) if val != '' else np.nan)
    arr = {name: np.asarray(v, dtype=np.float64) for name, v in cols.items()}
    return (arr['time_s'], arr['signal'],
            arr['raw_active'], arr['raw_reference'])


def _config_from_experiment(exp: dict) -> dict:
    """Reconstruct the ``config`` dict, recomputing HDF5 channel numbers.

    The loader convention is ``channel = 5 + 2 * fc``
    (FC1→7, FC2→9, FC3→11, FC4→13).  We rebuild the integer keys so
    code reading ``config['active_channel']`` keeps working even if the
    exporter omitted them.
    """
    pairs_in = exp.get('channel_pairs') or []
    pairs = []
    for p in pairs_in:
        active_fc = int(p['active_fc'])
        reference_fc = int(p['reference_fc'])
        pairs.append({
            'active_fc': active_fc,
            'reference_fc': reference_fc,
            'active_ch': int(p.get('active_ch', 5 + 2 * active_fc)),
            'reference_ch': int(p.get('reference_ch',
                                      5 + 2 * reference_fc)),
        })
    if not pairs:
        return {'channel_pairs': [], 'active_fc': None,
                'reference_fc': None, 'active_channel': None,
                'reference_channel': None}
    first = pairs[0]
    return {
        'channel_pairs': pairs,
        'active_fc': first['active_fc'],
        'reference_fc': first['reference_fc'],
        'active_channel': first['active_ch'],
        'reference_channel': first['reference_ch'],
    }


# ---------------------------------------------------------------------------
# Kinetics (evaluations) reconstruction
# ---------------------------------------------------------------------------

def _evaluations_from_csv(src: _PackageSource, exp_dir: str) -> list:
    """Reconstruct the ``evaluations`` list from per-CXW kinetics.csv.

    Returns an empty list when the package carries no fits for this
    experiment.
    """
    rel = f'{exp_dir}/kinetics.csv'
    if not src.exists(rel):
        return []
    text = src.read_text(rel)
    reader = csv.DictReader(io.StringIO(text))
    out = []

    def _f(s):
        try:
            return float(s) if s not in (None, '') else None
        except (TypeError, ValueError):
            return None

    for row in reader:
        try:
            cycle_index = int(row['cycle_number']) if row.get('cycle_number') else None
        except ValueError:
            cycle_index = None
        ch = row.get('channel_label') or ''
        ka = _f(row.get('ka_M-1_s-1'))
        kd = _f(row.get('kd_s-1'))
        KD = _f(row.get('KD_M'))
        if KD is None and ka not in (None, 0.0) and kd is not None:
            KD = kd / ka
        active_fc = reference_fc = None
        if ch.startswith('FC') and '-' in ch:
            try:
                left, right = ch.split('-')
                active_fc = int(left[2:])
                reference_fc = int(right[2:])
            except ValueError:
                pass
        out.append({
            'cycle_index': cycle_index,
            'cycle_guid': '',  # not preserved in package CSV
            'compound': row.get('compound', '') or '',
            'channel': ch,
            'channel_str': row.get('channel', '') or '',
            'active_fc': active_fc,
            'reference_fc': reference_fc,
            'ka': ka,
            'ka_error': _f(row.get('ka_error')),
            'kd': kd,
            'kd_error': _f(row.get('kd_error')),
            'KD': KD,
            'Rmax': _f(row.get('Rmax_pg_per_mm2')),
            'Rmax_error': _f(row.get('Rmax_error')),
            'sqrt_chi2': _f(row.get('sqrt_chi2_pg_per_mm2')),
            'comment': row.get('comment', '') or '',
            'name': '',
            'eval_file': '',
        })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_package(path: str, name: str | None = None,
                 channels='all') -> dict:
    """Load one experiment from a SensoFit data package.

    The returned dict matches :func:`sensofit.data_loader.load_cxw` so
    that the rest of the SensoFit pipeline (batch, direct_kinetics,
    ode_fitting, plotting, CLI) works unchanged on packages.

    Parameters
    ----------
    path : str
        Path to a ``.zip`` data package (as produced by
        :func:`sensofit.dataexporter.export_package`) or to an
        unzipped package directory.
    name : str, optional
        Folder name of the experiment to load (one per CXW originally
        exported).  If omitted, the first experiment in alphabetical
        order is loaded.  See :func:`list_experiments`.
    channels : str or list[int]
        Same semantics as :func:`load_cxw`: ``'all'`` (default) or a
        list of active FC numbers (e.g. ``[2, 3]``) to filter to.

    Returns
    -------
    dict
        Keys: ``config``, ``project``, ``instrument``, ``buffers``,
        ``autosampler``, ``immobilization``, ``report_points``,
        ``samples``, ``dmso_cals``, ``blanks``, ``all_cycles``,
        ``evaluations``.
    """
    with _PackageSource(path) as src:
        exps = _experiment_dirs(src)
        if not exps:
            raise ValueError(
                f'No experiments found in package: {path!r} '
                '(no folders containing experiment.json)')
        if name is None:
            exp_dir = exps[0]
        else:
            if name not in exps:
                raise ValueError(
                    f'Experiment {name!r} not found in package. '
                    f'Available: {exps}')
            exp_dir = name

        exp = src.read_json(f'{exp_dir}/experiment.json')
        config = _config_from_experiment(exp)

        # Channel filter (mirrors load_cxw semantics)
        if channels != 'all':
            requested = set(channels)
            pairs = [p for p in config['channel_pairs']
                     if p['active_fc'] in requested]
            if not pairs:
                avail = [p['active_fc'] for p in config['channel_pairs']]
                raise ValueError(
                    f'No matching channels. Requested {channels}, '
                    f'available active FCs: {avail}')
            config['channel_pairs'] = pairs
            first = pairs[0]
            config['active_fc'] = first['active_fc']
            config['reference_fc'] = first['reference_fc']
            config['active_channel'] = first['active_ch']
            config['reference_channel'] = first['reference_ch']
        wanted_labels = {f"FC{p['active_fc']}-FC{p['reference_fc']}"
                         for p in config['channel_pairs']}

        # Walk cycle folders
        all_cycles = []
        samples = []
        dmso_cals = []
        blanks = []

        cycle_dirs = sorted(d for d in src.list_subdirs(exp_dir)
                            if src.exists(f'{exp_dir}/{d}/metadata.json'))

        # Stable order by cycle index, matching load_cxw
        ordered = []
        for d in cycle_dirs:
            try:
                meta = src.read_json(f'{exp_dir}/{d}/metadata.json')
            except json.JSONDecodeError:
                continue
            ordered.append((int(meta.get('index', 0)), d, meta))
        ordered.sort(key=lambda t: t[0])

        for _, d, meta in ordered:
            cyc = _cycle_meta_from_json(meta)
            all_cycles.append(cyc)

            ct = cyc.get('cycle_type', '')
            if ct not in ('Sample', 'ControlSample', 'DMSO Cal.', 'Blank'):
                continue

            for label in sorted(meta.get('channels') or []):
                if label not in wanted_labels:
                    continue
                csv_rel = f'{exp_dir}/{d}/{label}.csv'
                if not src.exists(csv_rel):
                    continue
                time, signal, raw_active, raw_reference = \
                    _read_channel_csv(src, csv_rel)
                entry = {**cyc,
                         'time': time,
                         'signal': signal,
                         'raw_active': raw_active,
                         'raw_reference': raw_reference,
                         'channel': label}
                if ct in ('Sample', 'ControlSample'):
                    samples.append(entry)
                elif ct == 'DMSO Cal.':
                    dmso_cals.append(entry)
                elif ct == 'Blank':
                    blanks.append(entry)

        evaluations = _evaluations_from_csv(src, exp_dir)

        return {
            'config': config,
            'project': exp.get('project') or {},
            'instrument': exp.get('instrument') or {},
            'buffers': exp.get('buffers') or [],
            'autosampler': exp.get('autosampler') or {},
            'immobilization': exp.get('immobilization') or {},
            'report_points': exp.get('report_points') or [],
            'samples': samples,
            'dmso_cals': dmso_cals,
            'blanks': blanks,
            'all_cycles': all_cycles,
            'evaluations': evaluations,
        }


def load_experiment(path: str, **kwargs) -> dict:
    """Dispatch loader: ``.cxw`` → :func:`load_cxw`, else :func:`load_package`.

    A ``path`` is treated as a SensoFit package when it is a directory
    or when its name ends in ``.zip``.  Anything else is delegated to
    :func:`sensofit.data_loader.load_cxw`.
    """
    from .data_loader import load_cxw  # local import to avoid cycle

    if os.path.isdir(path) or path.lower().endswith('.zip'):
        return load_package(path, **kwargs)
    kwargs.pop('name', None)
    return load_cxw(path, **kwargs)
