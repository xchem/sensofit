"""Export raw .cxw signal data into a self-describing zipped package.

The package layout is::

    {package}.zip
    ├── README.md
    └── {cxw_basename}/
        ├── experiment.json
        └── {cpd}__{conc}__cyc{idx:03d}/
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

    Non-Sample cycles (Blank / DMSO Cal. / ControlSample) are tagged by
    type so the folder is unambiguous.  The compound from the
    autosampler well is preserved only when it is genuinely informative:
    for Controls (real reference compound) and Samples.  Blanks and
    DMSO Cal. wells are buffer/DMSO and their inherited "compound" is
    misleading, so it is intentionally omitted.
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
    if cpd:
        return cpd
    return ct or 'cycle'


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
            w.writerow([f'{row[0]:.6g}', f'{row[1]:.6g}',
                        f'{row[2]:.6g}', f'{row[3]:.6g}'])


def _group_cycles(data: dict) -> dict:
    """Group all loaded entries by cycle ``(index, guid)``.

    Each value is ``{'meta': cycle_meta_dict, 'channels': {label: entry}}``.
    """
    grouped: dict = {}
    for bucket in ('samples', 'dmso_cals', 'blanks'):
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

    return summary


def _render_readme(summaries: list, package_name: str) -> str:
    """Render the top-level README.md describing the package."""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    n_cycles = sum(s['n_cycles_total'] for s in summaries)
    lines = []
    lines.append(f'# {package_name}')
    lines.append('')
    lines.append('Self-describing data package exported by '
                 '[SensoFit](https://github.com/) from Creoptix WAVE '
                 '`.cxw` experiment files.')
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
    for s in summaries:
        lines.append(f'└── {s["cxw_folder"]}/')
        lines.append('    ├── experiment.json')
        for cyc in s['cycles'][:3]:
            lines.append(f'    ├── {cyc["folder"]}/')
            lines.append('    │   ├── metadata.json')
            for ch in cyc['channels']:
                lines.append(f'    │   └── {_sanitize(ch)}.csv')
        if len(s['cycles']) > 3:
            lines.append(f'    └── ... ({len(s["cycles"]) - 3} more cycle folders)')
    lines.append('```')
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
                 'repository for the full data-loading specification.')
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

        out_abs = os.path.abspath(output_zip)
        os.makedirs(os.path.dirname(out_abs) or '.', exist_ok=True)
        _zip_directory(tmp_root, out_abs)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return out_abs
