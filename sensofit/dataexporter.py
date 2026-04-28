"""Export raw .cxw signal data into a self-describing zipped package.

The package layout is::

    {package}.zip
    ├── README.md
    ├── all_creoptix_kinetics_evaluations.csv
    └── {cxw_basename}/
        ├── experiment.json
        ├── creoptix_kinetics_evaluations.csv
        └── {cpd}__{conc}__cyc{idx:03d}/
            ├── kinetics.json         
            ├── metadata.json
            ├── FC2-FC1.csv     # columns: time_s, signal, raw_active, raw_reference
            └── FC3-FC1.csv

Intended for data dissemination.  Only metadata that
:func:`sensofit.data_loader.load_cxw` already exposes is written.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

from .data_loader import load_cxw


_UNSAFE = re.compile(r'[^A-Za-z0-9._+-]+')


def _sanitize(name: str) -> str:
    """Make ``name`` filesystem- and zip-safe."""
    if name is None:
        return ''
    s = _UNSAFE.sub('_', str(name).strip())
    s = s.strip('_.')
    return s or 'unnamed'


def _format_concentration(M: float) -> str:
    """Pretty-print a molar concentration (e.g. 5e-7 -> ``500nM``)."""
    if M is None or M == 0:
        return '0M'
    a = abs(M)
    if a >= 1:
        val, unit = M, 'M'
    elif a >= 1e-3:
        val, unit = M * 1e3, 'mM'
    elif a >= 1e-6:
        val, unit = M * 1e6, 'uM'
    elif a >= 1e-9:
        val, unit = M * 1e9, 'nM'
    else:
        val, unit = M * 1e12, 'pM'
    if abs(val - round(val)) < 1e-6:
        return f'{int(round(val))}{unit}'
    return f'{val:g}{unit}'


def _cycle_label(cyc: dict) -> str:
    """Best-effort cycle label for the folder name.

    Non-Sample cycles (Blank / DMSO Cal. / ControlSample / Priming /
    Conditioning / Regeneration / …) are tagged by type so the folder
    is unambiguous.  The compound from the autosampler well is
    preserved only when it is genuinely informative: for Controls
    (real reference compound) and Samples.  Buffer / DMSO / priming /
    regeneration wells inherit a misleading "compound" string from
    the rack, so it is intentionally omitted in those cases.
    """
    ct = (cyc.get('cycle_type') or '').strip()
    cpd = (cyc.get('compound') or '').strip()
    if ct == 'Blank':
        return 'Blank'
    if ct == 'DMSO Cal.':
        return 'DMSOcal'
    if ct == 'ControlSample':
        # Real control compound name (e.g. literally 'control' on the
        # right rack, or an ASAP control identifier).
        return f'Control__{cpd}' if cpd else 'Control'
    if ct in ('Sample',):
        return cpd or ct
    # Non-fitting cycle types (Priming / Conditioning / Regeneration
    # / …): tag by type, ignore inherited well "compound".
    if ct:
        return _sanitize(ct)
    if cpd:
        return cpd
    return 'cycle'


def _cycle_folder_name(cyc: dict) -> str:
    """Folder name template ``{cpd}__{conc}__cyc{index:03d}``."""
    cpd = _sanitize(_cycle_label(cyc))
    conc = _sanitize(_format_concentration(cyc.get('concentration_M', 0.0)))
    idx = int(cyc.get('index', 0))
    return f'{cpd}__{conc}__cyc{idx:03d}'


def _write_cycle_csv(path: str, time, signal, raw_active, raw_reference) -> None:
    """Write a 4-column CSV for one channel of one cycle."""
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['time_s', 'signal', 'raw_active', 'raw_reference'])
        for row in zip(time, signal, raw_active, raw_reference):
            w.writerow([f'{row[0]:.6f}', f'{row[1]:.6f}',
                        f'{row[2]:.6f}', f'{row[3]:.6f}'])


def _group_cycles(data: dict) -> dict:
    """Group all loaded entries by cycle ``(index, guid)``.

    Each value is ``{'meta': cycle_meta_dict, 'channels': {label: entry}}``.
    """
    grouped: dict = {}
    for bucket in ('samples', 'dmso_cals', 'blanks', 'other_cycles'):
        for entry in data.get(bucket, []):
            key = (entry['index'], entry['guid'])
            if key not in grouped:
                meta = {k: v for k, v in entry.items()
                        if k not in ('time', 'signal', 'raw_active',
                                     'raw_reference', 'channel')}
                grouped[key] = {'meta': meta, 'channels': {}}
            grouped[key]['channels'][entry['channel']] = entry
    return grouped


def _experiment_summary(cxw_path: str, data: dict, grouped: dict) -> dict:
    """Build the per-CXW experiment.json structure."""
    cfg = data['config']
    type_counts: dict = defaultdict(int)
    for g in grouped.values():
        type_counts[g['meta'].get('cycle_type', '')] += 1
    return {
        'source_file': os.path.basename(cxw_path),
        'channel_pairs': cfg['channel_pairs'],
        'active_fc': cfg['active_fc'],
        'reference_fc': cfg['reference_fc'],
        'active_channel': cfg['active_channel'],
        'reference_channel': cfg['reference_channel'],
        'cycle_counts': dict(type_counts),
        'n_cycles_total': len(grouped),
        'project': data.get('project', {}),
        'instrument': data.get('instrument', {}),
        'buffers': data.get('buffers', []),
        'autosampler': data.get('autosampler', {}),
        'immobilization': data.get('immobilization', {}),
        'report_points': data.get('report_points', []),
    }


def _cycle_metadata(meta: dict, channels: list) -> dict:
    """JSON-safe per-cycle metadata sidecar."""
    def _opt(v):
        return None if v is None else float(v)
    return {
        'index': int(meta.get('index', -1)),
        'cycle_type': meta.get('cycle_type', ''),
        'guid': meta.get('guid', ''),
        'name': meta.get('name', ''),
        'slot': meta.get('slot', ''),
        'slot_side': meta.get('slot_side', ''),
        'compound': meta.get('compound', ''),
        'concentration_M': float(meta.get('concentration_M', 0.0)),
        'concentration_pretty': _format_concentration(
            meta.get('concentration_M', 0.0)),
        'mw_Da': float(meta.get('mw', 0.0)),
        'reagent_category': meta.get('reagent_category', ''),
        'reagent_volume_uL': float(meta.get('reagent_volume_uL', 0.0) or 0.0),
        'flow_rate_uLmin': _opt(meta.get('flow_rate_uLmin')),
        'contact_time_s': _opt(meta.get('contact_time_s')),
        'time_after_injection_s': _opt(meta.get('time_after_injection_s')),
        'baseline_duration_s': _opt(meta.get('baseline_duration_s')),
        'injection_mode': meta.get('injection_mode', ''),
        'pulse_durations_s': [float(x) for x in (meta.get('pulse_durations_s') or [])],
        'chip_prime_mode': meta.get('chip_prime_mode', ''),
        'wash_mode': meta.get('wash_mode', ''),
        'buffer_inlet': meta.get('buffer_inlet', ''),
        'block_id': meta.get('block_id'),
        'state': meta.get('state', ''),
        'enabled': bool(meta.get('enabled', True)),
        'markers': {k: float(v) for k, v in (meta.get('markers') or {}).items()},
        'channels': sorted(channels),
    }


def _build_eval_lookup(evaluations: list) -> dict:
    """Index evaluations by ``(cycle_index, channel_label)``.

    There can in principle be more than one evaluation per
    ``(cycle, channel)`` (e.g. a re-fit).  We keep the *last* one
    encountered, matching the way the WAVEcontrol UI shows the most
    recent evaluation.
    """
    out: dict = {}
    for e in evaluations or []:
        key = (e.get('cycle_index'), e.get('channel'))
        if key[0] is None or not key[1]:
            continue
        out[key] = e
    return out


_NA = 'not available'


def _kin_value(v):
    """JSON-friendly representation of a fit value (None → 'not available')."""
    if v is None:
        return _NA
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NA


def _cycle_kinetics_json(meta: dict, channels: list, eval_lookup: dict) -> dict:
    """Build the ``kinetics.json`` sidecar for one cycle folder.

    For every channel exported, report the 1:1 fit if WAVEcontrol stored
    one in the CXW; otherwise emit a placeholder marked
    ``'not available'``.  Errors are passed through verbatim as the
    instrument software reports them (no conversion to %, CI, …).
    """
    idx = meta.get('index')
    fits = {}
    for ch in sorted(channels):
        e = eval_lookup.get((idx, ch))
        if e is None:
            fits[ch] = {
                'available': False,
                'note': _NA,
            }
            continue
        fits[ch] = {
            'available': True,
            'model': '1:1 Kinetic',
            'ka_M-1_s-1': _kin_value(e.get('ka')),
            'ka_error': _kin_value(e.get('ka_error')),
            'kd_s-1': _kin_value(e.get('kd')),
            'kd_error': _kin_value(e.get('kd_error')),
            'KD_M': _kin_value(e.get('KD')),
            'Rmax_pg_per_mm2': _kin_value(e.get('Rmax')),
            'Rmax_error': _kin_value(e.get('Rmax_error')),
            'sqrt_chi2_pg_per_mm2': _kin_value(e.get('sqrt_chi2')),
            'comment': e.get('comment') or '',
        }
    return {
        'cycle_index': int(idx) if idx is not None else None,
        'compound': meta.get('compound', ''),
        'concentration_M': float(meta.get('concentration_M', 0.0)),
        'fits': fits,
    }


def _kinetics_csv_rows(cxw_path: str, data: dict, grouped: dict,
                       eval_lookup: dict) -> list:
    """Build kinetics.csv rows for one CXW (one row per fit found).

    Columns mirror the EV712A v1.0 release CSV layout.  Compound /
    concentration are taken from the cycle metadata so that fits whose
    Analyte string disagrees with the rack still resolve correctly.
    """
    instr = data.get('instrument') or {}
    run_date = (instr.get('measurement_start') or '').split('T')[0]
    rows = []
    for key in sorted(grouped.keys()):
        bundle = grouped[key]
        meta = bundle['meta']
        idx = meta.get('index')
        for ch in sorted(bundle['channels'].keys()):
            e = eval_lookup.get((idx, ch))
            if e is None:
                continue  # bulk CSV only carries actual fits
            rows.append({
                'source_file': os.path.basename(cxw_path),
                'run_date': run_date,
                'cycle_number': idx,
                'channel': e.get('channel_str') or ch,
                'channel_label': ch,
                'cycle_type': meta.get('cycle_type', ''),
                'compound': meta.get('compound', '') or e.get('compound', ''),
                'concentration_M': meta.get('concentration_M'),
                'ka_M-1_s-1': e.get('ka'),
                'ka_error': e.get('ka_error'),
                'kd_s-1': e.get('kd'),
                'kd_error': e.get('kd_error'),
                'KD_M': e.get('KD'),
                'Rmax_pg_per_mm2': e.get('Rmax'),
                'Rmax_error': e.get('Rmax_error'),
                'sqrt_chi2_pg_per_mm2': e.get('sqrt_chi2'),
                'comment': e.get('comment') or '',
            })
    return rows


_KINETICS_CSV_COLUMNS = [
    'source_file', 'run_date', 'cycle_number', 'channel', 'channel_label',
    'cycle_type', 'compound', 'concentration_M',
    'ka_M-1_s-1', 'ka_error', 'kd_s-1', 'kd_error', 'KD_M',
    'Rmax_pg_per_mm2', 'Rmax_error', 'sqrt_chi2_pg_per_mm2', 'comment',
]


def _write_kinetics_csv(path: str, rows: list) -> None:
    """Write a kinetics CSV.  Empty cells → empty string (never 'None')."""
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_KINETICS_CSV_COLUMNS,
                           extrasaction='ignore')
        w.writeheader()
        for r in rows:
            clean = {}
            for k in _KINETICS_CSV_COLUMNS:
                v = r.get(k)
                if v is None or v == '':
                    clean[k] = ''
                elif isinstance(v, float):
                    clean[k] = f'{v:.6g}'
                else:
                    clean[k] = v
            w.writerow(clean)


def export_cxw(cxw_path: str, out_dir: str) -> dict:
    """Export one .cxw file's raw data into ``out_dir/{basename}/``.

    Returns a small dict summarising what was written; it is the basis
    for the auto-generated README.
    """
    data = load_cxw(cxw_path, channels='all')
    basename = _sanitize(os.path.splitext(os.path.basename(cxw_path))[0])
    cxw_root = os.path.join(out_dir, basename)
    os.makedirs(cxw_root, exist_ok=True)

    grouped = _group_cycles(data)
    summary = _experiment_summary(cxw_path, data, grouped)
    eval_lookup = _build_eval_lookup(data.get('evaluations', []))

    cycle_rows = []
    for key in sorted(grouped.keys()):
        bundle = grouped[key]
        meta = bundle['meta']
        chans = bundle['channels']
        folder = _cycle_folder_name(meta)
        cyc_dir = os.path.join(cxw_root, folder)
        os.makedirs(cyc_dir, exist_ok=True)

        cmeta = _cycle_metadata(meta, list(chans.keys()))
        with open(os.path.join(cyc_dir, 'metadata.json'), 'w') as fh:
            json.dump(cmeta, fh, indent=2, sort_keys=True)

        # kinetics.json only makes sense for cycles that can carry a
        # 1:1 fit (Sample / ControlSample / DMSO Cal. / Blank).
        # Priming / conditioning / regeneration are signal-only.
        if cmeta['cycle_type'] in ('Sample', 'ControlSample',
                                    'DMSO Cal.', 'Blank'):
            kjson = _cycle_kinetics_json(meta, list(chans.keys()),
                                          eval_lookup)
            with open(os.path.join(cyc_dir, 'kinetics.json'), 'w') as fh:
                json.dump(kjson, fh, indent=2, sort_keys=True)

        for label, entry in chans.items():
            csv_path = os.path.join(cyc_dir, f'{_sanitize(label)}.csv')
            _write_cycle_csv(csv_path, entry['time'], entry['signal'],
                             entry['raw_active'], entry['raw_reference'])

        cycle_rows.append({
            'folder': folder,
            'index': cmeta['index'],
            'cycle_type': cmeta['cycle_type'],
            'compound': cmeta['compound'],
            'concentration_pretty': cmeta['concentration_pretty'],
            'concentration_M': cmeta['concentration_M'],
            'mw_Da': cmeta['mw_Da'],
            'slot': cmeta['slot'],
            'slot_side': cmeta['slot_side'],
            'injection_mode': cmeta['injection_mode'],
            'contact_time_s': cmeta['contact_time_s'],
            'channels': cmeta['channels'],
        })

    summary['cycles'] = cycle_rows
    summary['cxw_folder'] = basename
    with open(os.path.join(cxw_root, 'experiment.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # Per-CXW kinetics CSV (also used to build the package-wide bulk CSV).
    kin_rows = _kinetics_csv_rows(cxw_path, data, grouped, eval_lookup)
    _write_kinetics_csv(os.path.join(cxw_root, 'creoptix_kinetics_evaluations.csv'), kin_rows)
    summary['kinetics_rows'] = kin_rows
    summary['n_kinetic_fits'] = len(kin_rows)

    return summary


def _render_readme(summaries: list, package_name: str) -> str:
    """Render the top-level README.md describing the package."""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    n_cycles = sum(s['n_cycles_total'] for s in summaries)
    lines = []
    lines.append(f'# {package_name}')
    lines.append('')
    lines.append('Self-describing data package exported by '
                 '[SensoFit](https://github.com/xchem/sensofit) from '
                 'Creoptix WAVE `.cxw` experiment files.')
    lines.append('')
    lines.append('**About SensoFit.** SensoFit is an *exploratory* '
                 'open-source tool for parsing and analysing Creoptix '
                 'WAVE grating-coupled interferometry (GCI) data. It is '
                 'under active development and will be refined alongside '
                 'future data releases — both the data layout and the '
                 'analysis methods are expected to evolve. Feedback and '
                 'issues are very welcome at '
                 '<https://github.com/xchem/sensofit>.')
    lines.append('')
    lines.append('**Re-ingesting this package.** This data package can be '
                 'fed straight back into SensoFit — no `.cxw` file needed. '
                 'Use `sensofit.load_package(path)` (or the dispatcher '
                 '`sensofit.load_experiment(path)`) on either the `.zip` '
                 'or its unzipped directory; the returned dict matches '
                 '`sensofit.load_cxw(...)` so all batch / fitting / '
                 'plotting helpers work unchanged. The CLI also accepts '
                 'the `.zip` directly: `python -m sensofit package.zip '
                 '--mode dk`.')
    lines.append('')
    lines.append('**CEDRIC TODO NOTEBOOKS:** add list of URLS and notebooks for exploring the data')
    lines.append('')
    lines.append(f'- **Generated:** {now}')
    lines.append(f'- **Source files:** {len(summaries)}')
    lines.append(f'- **Total cycles:** {n_cycles}')
    lines.append('')

    lines.append('## Package layout')
    lines.append('')
    lines.append('```')
    lines.append(f'{package_name}.zip')
    lines.append('├── README.md')
    lines.append('├── all_creoptix_kinetics_evaluations.csv')
    for s in summaries:
        lines.append(f'└── {s["cxw_folder"]}/')
        lines.append('    ├── experiment.json')
        lines.append('    ├── creoptix_kinetics_evaluations.csv')
        for cyc in s['cycles'][:3]:
            lines.append(f'    ├── {cyc["folder"]}/')
            lines.append('    │   ├── metadata.json')
            ct = (cyc.get('cycle_type') or '').strip()
            if ct in ('Sample', 'ControlSample', 'DMSO Cal.', 'Blank'):
                lines.append('    │   ├── kinetics.json')
            for ch in cyc['channels']:
                lines.append(f'    │   └── {_sanitize(ch)}.csv')
        if len(s['cycles']) > 3:
            lines.append(f'    └── ... ({len(s["cycles"]) - 3} more cycle folders)')
    lines.append('```')
    lines.append('')

    lines.append('## Kinetic fits')
    lines.append('')
    lines.append('Per-channel 1:1 fits saved by WAVEcontrol in the source '
                 '`.cxw` (when present) are extracted as-is — values and '
                 'errors are reproduced verbatim from the instrument '
                 'software, with no re-conversion to %, confidence '
                 'interval, or χ² (vs. sqrt χ²) form.')
    lines.append('')
    lines.append('- Top-level `all_creoptix_kinetics_evaluations.csv` aggregates every fit found '
                 'across all source files (one row per cycle × channel).')
    lines.append('- Each per-CXW folder also contains a `creoptix_kinetics_evaluations.csv` '
                 'restricted to that experiment.')
    lines.append('- Inside each cycle folder, `kinetics.json` reports the '
                 'fit for every exported channel (or `"not available"` '
                 'when the CXW carried no fit for that channel).')
    lines.append('')
    lines.append('| Field | Units | Description |')
    lines.append('|---|---|---|')
    lines.append('| `ka_M-1_s-1` | M⁻¹ s⁻¹ | Association rate constant |')
    lines.append('| `kd_s-1` | s⁻¹ | Dissociation rate constant |')
    lines.append('| `KD_M` | M | Equilibrium dissociation constant '
                 '(= kd / ka) |')
    lines.append('| `Rmax_pg_per_mm2` | pg/mm² | Maximum response (saturation) |')
    lines.append('| `sqrt_chi2_pg_per_mm2` | pg/mm² | Square-root of χ² '
                 '(global fit residual) |')
    lines.append('| `*_error` | (as reported) | Error on the corresponding '
                 'parameter, exactly as stored in the CXW |')
    lines.append('| `comment` | — | `<UserComment>` annotation from the '
                 'evaluation file |')
    lines.append('')

    lines.append('## Data format')
    lines.append('')
    lines.append('Each cycle folder contains one CSV per active/reference '
                 'channel pair (e.g. `FC2-FC1.csv`).  The CSVs share the '
                 'same columns:')
    lines.append('')
    lines.append('| Column | Units | Description |')
    lines.append('|---|---|---|')
    lines.append('| `time_s` | seconds | Acquisition time |')
    lines.append('| `signal` | response units | Reference-subtracted '
                 'sensorgram (`raw_active − raw_reference`) |')
    lines.append('| `raw_active` | response units | Raw signal on the '
                 'ligand-bearing flow cell |')
    lines.append('| `raw_reference` | response units | Raw signal on the '
                 'reference flow cell |')
    lines.append('')
    lines.append('The channel label `FCx-FCy` denotes *active − reference*: '
                 '`FCx` is the ligand-bearing flow cell, `FCy` is the '
                 'unmodified reference.  Channel mapping inside the original '
                 '`.cxw` HDF5 store: FC*n* → channel `5 + 2n` '
                 '(FC1→7, FC2→9, FC3→11, FC4→13).')
    lines.append('')
    lines.append('**Non-fitting cycles** (Priming, Conditioning, '
                 'Regeneration, …) are exported for completeness with '
                 'their raw signal CSVs and `metadata.json`, but no '
                 '`kinetics.json` — they are not analyte injections and '
                 'WAVEcontrol does not store a 1:1 fit for them. The '
                 'SensoFit fitting pipeline ignores these cycles; they '
                 'are surfaced via `data["other_cycles"]` in '
                 '`load_cxw` / `load_package`.')
    lines.append('')

    lines.append('## Marker definitions')
    lines.append('')
    lines.append('Each `metadata.json` records phase markers (in seconds '
                 'on the same time axis as the CSV `time_s` column):')
    lines.append('')
    lines.append('- **Injection** — start of the analyte injection (association phase begins).')
    lines.append('- **Rinse** — end of injection / start of buffer rinse (dissociation phase begins).')
    lines.append('- **RinseEnd** — end of the rinse window.')
    lines.append('')

    lines.append('## Source experiments')
    lines.append('')
    for s in summaries:
        lines.append(f'### {s["source_file"]}')
        lines.append('')
        lines.append(f'- Folder: `{s["cxw_folder"]}/`')

        proj = s.get('project') or {}
        if proj:
            lines.append('')
            lines.append('**Project**')
            lines.append('')
            for label, key in [
                ('Name', 'name'),
                ('Created', 'creation_time'),
                ('Creator', 'creator'),
                ('Description', 'description'),
                ('Notes', 'notes'),
            ]:
                v = (proj.get(key) or '').strip()
                if v:
                    lines.append(f'- {label}: {v}')
            if proj.get('ligand_mw_Da'):
                lines.append(f'- Ligand MW: {proj["ligand_mw_Da"]:g} Da')

        instr = s.get('instrument') or {}
        if instr:
            lines.append('')
            lines.append('**Instrument & run**')
            lines.append('')
            for label, key, suffix in [
                ('Device', 'device_type', ''),
                ('Serial number', 'serial_number', ''),
                ('Hardware version', 'hardware_version', ''),
                ('Firmware', 'firmware_version', ''),
                ('WAVEcontrol (saved)', 'wave_control_version', ''),
                ('WAVEcontrol (recorded)', 'serie_recorded_version', ''),
                ('Flow-cell temperature', 'fc_temperature_C', ' °C'),
                ('Acquisition rate', 'acquisition_rate_Hz', ' Hz'),
                ('Max flow rate', 'max_flow_rate_uLmin', ' µL/min'),
                ('Measurement start', 'measurement_start', ''),
                ('Measurement end', 'measurement_end', ''),
            ]:
                v = instr.get(key)
                if v not in (None, ''):
                    lines.append(f'- {label}: {v}{suffix}')

        immob = s.get('immobilization') or {}
        if immob:
            lines.append('')
            lines.append('**Chip preparation (immobilization)**')
            lines.append('')
            if immob.get('name'):
                lines.append(f'- Serie name: {immob["name"]}')
            if immob.get('capture_fcs'):
                fcs = ', '.join(f'FC{n}' for n in immob['capture_fcs'])
                lines.append(f'- Capture flow cell(s): {fcs}')
            if immob.get('measurement_start'):
                lines.append(f'- Started: {immob["measurement_start"]}')
            if immob.get('measurement_end'):
                lines.append(f'- Ended: {immob["measurement_end"]}')

        buffers = s.get('buffers') or []
        if buffers:
            lines.append('')
            lines.append('**Buffers (inlet ports)**')
            lines.append('')
            lines.append('| Id | Inlet | Name |')
            lines.append('|---|---|---|')
            for b in buffers:
                lines.append(f'| {b.get("id","")} | {b.get("inlet","")} | '
                             f'{b.get("name","") or "—"} |')

        rps = s.get('report_points') or []
        if rps:
            lines.append('')
            lines.append('**Report points**')
            lines.append('')
            lines.append('| Name | Marker | Shift (s) | Averaging | Reference |')
            lines.append('|---|---|---|---|---|')
            for rp in rps:
                shift = rp.get('shift_s')
                avg = rp.get('averaging')
                lines.append(
                    f'| {rp.get("name","")} | {rp.get("marker","")} | '
                    f'{shift if shift is not None else ""} | '
                    f'{avg if avg is not None else ""} | '
                    f'{"yes" if rp.get("is_reference") else "no"} |')

        lines.append('')
        lines.append('**Channels & cycle counts**')
        lines.append('')
        lines.append(f'- Active flow cell: FC{s["active_fc"]} '
                     f'(channel {s["active_channel"]})')
        lines.append(f'- Reference flow cell: FC{s["reference_fc"]} '
                     f'(channel {s["reference_channel"]})')
        pair_strs = [f'FC{p["active_fc"]}-FC{p["reference_fc"]}'
                     for p in s['channel_pairs']]
        lines.append(f'- Channel pairs exported: {", ".join(pair_strs)}')
        cc = ', '.join(f'{k}={v}' for k, v in sorted(s['cycle_counts'].items()))
        lines.append(f'- Cycle counts: {cc}')
        lines.append('')
        lines.append('**Cycle table**')
        lines.append('')
        lines.append('| # | Type | Compound | Concentration | MW (Da) | '
                     'Slot | Inj. mode | Contact (s) | Channels | Folder |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for cyc in s['cycles']:
            chans = ', '.join(cyc['channels'])
            slot = cyc['slot'] or '—'
            if cyc.get('slot_side'):
                slot = f'{cyc["slot_side"][:1]}-{slot}'
            ct = cyc.get('contact_time_s')
            ct_s = f'{ct:g}' if ct is not None else '—'
            lines.append(
                f'| {cyc["index"]} | {cyc["cycle_type"]} | '
                f'{cyc["compound"] or "—"} | {cyc["concentration_pretty"]} | '
                f'{cyc["mw_Da"]:g} | {slot} | '
                f'{cyc.get("injection_mode") or "—"} | {ct_s} | {chans} | '
                f'`{cyc["folder"]}` |')
        lines.append('')

    lines.append('## Provenance')
    lines.append('')
    lines.append('Generated by `sensofit.dataexporter.export_package`. '
                 'Only metadata exposed by `sensofit.data_loader.load_cxw` '
                 'is included; see [docs/data_loader.md] in the SensoFit '
                 'repository (<https://github.com/xchem/sensofit>) for '
                 'the full data-loading specification.')
    lines.append('')

    return '\n'.join(lines)


def _zip_directory(src_dir: str, output_zip: str) -> None:
    with zipfile.ZipFile(output_zip, 'w',
                         compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, src_dir)
                zf.write(full, arc)


def export_package(cxw_paths, output_zip: str,
                   package_name: str | None = None) -> str:
    """Export N ``.cxw`` files into a single self-describing zip package.

    Parameters
    ----------
    cxw_paths : iterable of str
        Paths to ``.cxw`` files to include.
    output_zip : str
        Path to the output ``.zip`` file (created/overwritten).
    package_name : str, optional
        Human-readable name used in the README header and as the
        directory name *inside* the zip.  Defaults to the output zip's
        basename without extension.

    Returns
    -------
    str
        Absolute path to the written zip file.
    """
    cxw_paths = list(cxw_paths)
    if not cxw_paths:
        raise ValueError('export_package: no .cxw files supplied')

    if not output_zip.lower().endswith('.zip'):
        output_zip = output_zip + '.zip'

    if package_name is None:
        package_name = os.path.splitext(os.path.basename(output_zip))[0]
    package_name_safe = _sanitize(package_name)

    tmp_root = tempfile.mkdtemp(prefix='sensofit_export_')
    pkg_root = os.path.join(tmp_root, package_name_safe)
    os.makedirs(pkg_root, exist_ok=True)
    try:
        summaries = [export_cxw(p, pkg_root) for p in cxw_paths]
        readme = _render_readme(summaries, package_name)
        with open(os.path.join(pkg_root, 'README.md'), 'w') as fh:
            fh.write(readme)

        # Bulk kinetics CSV across every CXW in the package.
        bulk_rows = []
        for s in summaries:
            bulk_rows.extend(s.get('kinetics_rows', []))
        _write_kinetics_csv(os.path.join(pkg_root, 'all_creoptix_kinetics_evaluations.csv'),
                            bulk_rows)

        out_abs = os.path.abspath(output_zip)
        os.makedirs(os.path.dirname(out_abs) or '.', exist_ok=True)
        _zip_directory(tmp_root, out_abs)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return out_abs
