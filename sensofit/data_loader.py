"""Load Creoptix .cxw experiment files.

A .cxw file is a ZIP archive containing:
  - _project.cx3   : XML with cycle metadata, flow cell config, markers
  - cyclesData.h5   : HDF5 with raw channel data keyed by cycle GUID
  - Wizard/*.cx3    : XML with reagent/sample definitions per serie
  - Evaluations/    : XML with per-compound kinetic fit results (not used here)
  - Correctors/     : XML with timing/offset corrections (not used here)
"""

import re
import zipfile
import io
from xml.etree import ElementTree as ET

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _clean_xml(raw: str) -> str:
    """Strip BOM and namespace artifacts that break ElementTree."""
    raw = raw.lstrip('\ufeff')
    raw = re.sub(r' xmlns:version="[^"]*"', '', raw)
    raw = re.sub(r' xmlns:p\d+="[^"]*"', '', raw)
    raw = re.sub(r' p\d+:(\w+)=', r' xsi_\1=', raw)
    return raw


def _parse_xml(zf: zipfile.ZipFile, path: str) -> ET.Element:
    raw = zf.read(path).decode('utf-8', errors='replace')
    return ET.fromstring(_clean_xml(raw))


# ---------------------------------------------------------------------------
# Concentration / MW parsing
# ---------------------------------------------------------------------------

_CONC_UNITS = [
    ('mM', 1e-3),
    ('µM', 1e-6), ('μM', 1e-6), ('uM', 1e-6),
    ('nM', 1e-9),
    ('pM', 1e-12),
    ('M',  1.0),
]


def _parse_concentration(s: str) -> float:
    """Parse '500.000 nM' → 5e-7 (in Molar)."""
    s = s.strip()
    for unit, factor in _CONC_UNITS:
        if s.endswith(unit):
            return float(s[:len(s) - len(unit)].strip()) * factor
    return float(s)


def _parse_mw(s: str) -> float:
    """Parse '455 Da' → 455.0."""
    s = s.strip()
    if s.endswith('Da'):
        s = s[:-2].strip()
    return float(s) if s else 0.0


# ---------------------------------------------------------------------------
# Flow-cell / channel detection
# ---------------------------------------------------------------------------

def _fc_to_channel(fc_num: int) -> int:
    """Map flow cell number (1-4) to HDF5 channel index.

    FC1 → ch7, FC2 → ch9, FC3 → ch11, FC4 → ch13.
    """
    return 5 + 2 * fc_num


def _capture_fcs(project_root: ET.Element) -> set[int]:
    """Return the set of FC numbers that have a ``CycleType=Capture`` cycle.

    These are the flow cells where ligand was actually immobilised.
    """
    fcs = set()
    for cycle in project_root.findall('.//Cycle'):
        if cycle.findtext('CycleType') != 'Capture':
            continue
        for sfc in cycle.findall('.//SelectableFlowCells'):
            if sfc.findtext('Selected') != 'true':
                continue
            fc_el = sfc.find('FlowCell')
            if fc_el is None:
                continue
            desg = fc_el.findtext('Designation', '')
            if desg.startswith('FC'):
                fcs.add(int(desg[2:]))
    return fcs


def _detect_channels(project_root: ET.Element):
    """Determine active and reference channels from ChannelReferencings.

    The RAPID Kinetics (Pulse) serie stores ``ChannelReferencings``
    entries whose ``ChannelDto.Id`` encodes an active/reference pair as
    ``active_ch * 100 + reference_ch`` (e.g. 907 = ch9−ch7 = FC2−FC1).

    Only channels whose active flow cell had ligand immobilised (i.e. a
    ``CycleType=Capture`` cycle with that FC selected) are returned.
    This avoids fitting empty/control channels.

    Returns
    -------
    channel_pairs : list[dict]
        One dict per active channel with keys:
        ``active_ch``, ``reference_ch``, ``active_fc``, ``reference_fc``.
    """
    def _channel_to_fc(ch: int) -> int:
        return (ch - 5) // 2  # inverse of _fc_to_channel

    ligand_fcs = _capture_fcs(project_root)

    for serie in project_root.findall('.//Serie'):
        if serie.findtext('Type') != 'Pulse':
            continue
        pairs = []
        for cr in serie.findall('.//ChannelReferencings'):
            dto = cr.find('ChannelDto')
            if dto is None:
                continue
            cid = int(dto.findtext('Id', '0'))
            if cid == 0:
                continue
            ref_ch = cid % 100
            act_ch = cid // 100
            pairs.append({
                'active_ch': act_ch,
                'reference_ch': ref_ch,
                'active_fc': _channel_to_fc(act_ch),
                'reference_fc': _channel_to_fc(ref_ch),
            })
        if pairs:
            # Filter to only channels with ligand, if Capture info exists
            if ligand_fcs:
                filtered = [p for p in pairs if p['active_fc'] in ligand_fcs]
                if filtered:
                    return filtered
            return pairs

    # Fallback: use the old heuristic (single channel)
    return _detect_channels_legacy(project_root)


def _detect_channels_legacy(project_root: ET.Element):
    """Legacy single-channel detection via Immobilization Capture cycle."""
    capture_fc = None
    for serie in project_root.findall('.//Serie'):
        if serie.findtext('Type') != 'Immobilization':
            continue
        for cyc in serie.find('Cycles').findall('Cycle'):
            ct = cyc.find('CycleType')
            if ct is not None and ct.text == 'Capture':
                fp2 = cyc.find('FlowPath2')
                if fp2 is not None:
                    for i, fcs in enumerate(fp2.find('FlowCells'), start=1):
                        if fcs.findtext('Selected') == 'true':
                            fc_inner = fcs.find('FlowCell')
                            desg = fc_inner.findtext('Designation', '') if fc_inner is not None else ''
                            capture_fc = int(desg.replace('FC', ''))
                            break
                break

    rk_fcs = []
    for serie in project_root.findall('.//Serie'):
        if serie.findtext('Type') != 'Pulse':
            continue
        for cyc in serie.find('Cycles').findall('Cycle'):
            ct = cyc.find('CycleType')
            if ct is not None and ct.text == 'Sample':
                fp2 = cyc.find('FlowPath2')
                if fp2 is not None:
                    for i, fcs in enumerate(fp2.find('FlowCells'), start=1):
                        if fcs.findtext('Selected') == 'true':
                            fc_inner = fcs.find('FlowCell')
                            desg = fc_inner.findtext('Designation', '') if fc_inner is not None else ''
                            rk_fcs.append(int(desg.replace('FC', '')))
                break
        break

    active_fc = capture_fc
    reference_fc = [fc for fc in rk_fcs if fc != capture_fc][0]
    return [{
        'active_ch': _fc_to_channel(active_fc),
        'reference_ch': _fc_to_channel(reference_fc),
        'active_fc': active_fc,
        'reference_fc': reference_fc,
    }]


# ---------------------------------------------------------------------------
# Wizard (reagent) lookup
# ---------------------------------------------------------------------------

def _build_reagent_lookup(zf: zipfile.ZipFile, serie_guid: str) -> dict:
    """Parse the RAPID Kinetics Wizard to build a ``(side, slot) → reagent``
    lookup.

    The Wizard stores its reagents inside ``<LeftRack>`` and
    ``<RightRack>`` containers; the same ``Slot`` (e.g. ``D2``) may exist
    on **both** racks but refer to completely different wells (a Sample
    compound on the left, the control or a blank/DMSO well on the
    right).  Keying the lookup on ``(side, slot)`` is required so that
    Control / Blank / DMSO cycles — which typically run from the Right
    rack — pick up the correct reagent.
    """
    lookup = {}
    for fname in zf.namelist():
        if not fname.startswith('Wizard/'):
            continue
        wiz = _parse_xml(zf, fname)
        if wiz.findtext('SerieId') != serie_guid:
            continue
        for rack_tag, side in (('LeftRack', 'Left'), ('RightRack', 'Right')):
            rack = wiz.find(rack_tag)
            if rack is None:
                continue
            for r in rack.findall('.//Reagent'):
                slot = r.get('Slot', '')
                if not slot:
                    continue
                try:
                    vol = float(r.get('Volume', '0') or '0')
                except ValueError:
                    vol = 0.0
                try:
                    conc_M = _parse_concentration(
                        r.get('Concentration', '0'))
                except ValueError:
                    conc_M = 0.0
                lookup[(side, slot)] = {
                    'compound': r.get('Designation', ''),
                    'concentration_M': conc_M,
                    'mw': _parse_mw(r.get('MW', '0')),
                    'category': r.get('Category', ''),
                    'volume_uL': vol,
                }
        break  # found the right wizard
    return lookup


# ---------------------------------------------------------------------------
# Cycle metadata extraction
# ---------------------------------------------------------------------------

def _extract_cycles(serie_el: ET.Element, reagent_lookup: dict,
                    autosampler_lookup: dict | None = None) -> list[dict]:
    """Extract ordered cycle metadata from a Serie element.

    ``autosampler_lookup`` (optional) is keyed by ``(side, slot)`` and
    supplies the populated ``category`` and ``volume_uL`` values that the
    Wizard placeholders lack.
    """
    autosampler_lookup = autosampler_lookup or {}
    cycles = []
    for cyc in serie_el.find('Cycles').findall('Cycle'):
        ct_el = cyc.find('CycleType')
        cycle_type = ct_el.text if ct_el is not None else ''

        # Autosampler slot + side
        slot_el = cyc.find('.//AutosamplerLocation')
        slot = slot_el.get('Slot', '') if slot_el is not None else ''
        slot_side = slot_el.get('Side', '') if slot_el is not None else ''

        # Reagent info: wizard provides compound/conc/MW per (side, slot);
        # the autosampler reagent table carries the real category/volume
        # (the wizard ones are placeholders).
        reagent = dict(reagent_lookup.get((slot_side, slot), {}))
        as_info = autosampler_lookup.get((slot_side, slot), {})
        # Prefer autosampler designation if wizard is empty (e.g. side
        # mismatch in legacy files where lookup keyed only on slot).
        if not reagent.get('compound') and as_info.get('designation'):
            reagent['compound'] = as_info['designation']
            if as_info.get('mw_Da'):
                reagent['mw'] = as_info['mw_Da']
            if as_info.get('concentration_M') is not None:
                reagent['concentration_M'] = as_info['concentration_M']
        category = as_info.get('category') or reagent.get('category', '')
        volume_uL = as_info.get('volume_uL', reagent.get('volume_uL', 0.0))

        # Markers (Baseline, Injection, Rinse, RinseEnd, ...)
        markers = {}
        for marker in cyc.findall('.//Marker'):
            mtype = marker.findtext('Type', '')
            mx = float(marker.findtext('X', '0'))
            if mtype:
                markers[mtype] = mx

        # Pulse durations (kinetic-titration injections)
        pulse_durations = []
        pd_el = cyc.find('PulseDurations')
        if pd_el is not None:
            for d in pd_el.findall('double'):
                try:
                    pulse_durations.append(float(d.text or '0'))
                except ValueError:
                    pass

        def _opt_float(tag):
            v = cyc.findtext(tag, '')
            try:
                return float(v) if v else None
            except ValueError:
                return None

        def _opt_int(tag):
            v = cyc.findtext(tag, '')
            try:
                return int(v) if v else None
            except ValueError:
                return None

        cycles.append({
            'index': int(cyc.findtext('Index', '-1')),
            'cycle_type': cycle_type,
            'guid': cyc.findtext('Guid', ''),
            'name': cyc.findtext('Name', ''),
            'slot': slot,
            'slot_side': slot_side,
            'compound': reagent.get('compound', ''),
            'concentration_M': reagent.get('concentration_M', 0.0),
            'mw': reagent.get('mw', 0.0),
            'reagent_category': category,
            'reagent_volume_uL': volume_uL,
            'markers': markers,
            'flow_rate_uLmin': _opt_float('FlowRate'),
            'contact_time_s': _opt_float('ContactTime'),
            'time_after_injection_s': _opt_float('TimeAfterInjection'),
            'baseline_duration_s': _opt_float('BaselineDuration'),
            'injection_mode': cyc.findtext('InjectionModeLabel', ''),
            'pulse_durations_s': pulse_durations,
            'chip_prime_mode': cyc.findtext('ChipPrimeModeLabel', ''),
            'wash_mode': cyc.findtext('WashModeLabel', ''),
            'buffer_inlet': cyc.findtext('Buffer', ''),
            'block_id': _opt_int('BlockId'),
            'state': cyc.findtext('State', ''),
            'enabled': cyc.findtext('IsEnabled', '') == 'true',
        })
    return cycles


# ---------------------------------------------------------------------------
# Project / instrument / buffers / autosampler / report points
# ---------------------------------------------------------------------------

def _extract_project_meta(zf: zipfile.ZipFile) -> dict:
    """Parse ``_projectmeta.cx3`` for project-level context."""
    if '_projectmeta.cx3' not in zf.namelist():
        return {}
    try:
        root = _parse_xml(zf, '_projectmeta.cx3')
    except ET.ParseError:
        return {}

    def _t(tag):
        return (root.findtext(tag) or '').strip()

    def _f(tag):
        v = _t(tag)
        try:
            return float(v) if v else None
        except ValueError:
            return None

    log_entries = []
    for le in root.findall('.//LogEntry'):
        log_entries.append({
            'time': (le.findtext('LogTime') or '').strip(),
            'type': (le.findtext('LogType') or '').strip(),
            'visibility': (le.findtext('LogVisibility') or '').strip(),
            'message': (le.findtext('LogMessage') or '').strip(),
        })

    return {
        'name': _t('Name'),
        'id': _t('Id'),
        'creator': _t('Creator'),
        'creation_time': _t('CreationTime'),
        'number': _t('Number'),
        'journal': _t('Journal'),
        'description': _t('Description'),
        'objectives': _t('Objectives'),
        'notes': _t('Notes'),
        'results': _t('Results'),
        'ligand_mw_Da': _f('LigandMolecularWeight'),
        'log_entries': log_entries,
    }


def _extract_instrument(project_root: ET.Element,
                        rk_serie: ET.Element) -> dict:
    """Instrument / firmware / software / run timing for the Pulse serie."""
    info = {
        'device_type': (project_root.findtext('DeviceType') or '').strip(),
        'wave_control_version': (
            project_root.findtext('WAVEcontrolLastSavedVersion') or '').strip(),
    }
    if rk_serie is None:
        return info
    meta = rk_serie.find('Meta')
    if meta is not None:
        def _t(tag):
            return (meta.findtext(tag) or '').strip()
        fw = '.'.join(filter(None, [
            _t('FirmwareMajorVersion'), _t('FirmwareMinorVersion'),
            _t('FirmwareBuild'), _t('FirmwareRevision')]))
        info.update({
            'serial_number': _t('SerialNumber'),
            'hardware_version': _t('HardwareVersion'),
            'firmware_version': fw,
            'serie_recorded_version': _t('WAVEcontrolSerieRecordedVersion'),
            'measurement_start': _t('MeasurementStartTime'),
            'measurement_end': _t('MeasurementEndTime'),
        })
    fc_temp = rk_serie.findtext('FCTemperature')
    if fc_temp:
        try:
            info['fc_temperature_C'] = float(fc_temp)
        except ValueError:
            pass
    rate = rk_serie.findtext('AcquisitionRate')
    if rate:
        try:
            info['acquisition_rate_Hz'] = float(rate)
        except ValueError:
            pass
    mfr = rk_serie.findtext('MaxFlowRate')
    if mfr:
        try:
            info['max_flow_rate_uLmin'] = float(mfr)
        except ValueError:
            pass
    return info


def _extract_buffers(rk_serie: ET.Element) -> list[dict]:
    """List of buffer port definitions for the Pulse serie."""
    out = []
    if rk_serie is None:
        return out
    for b in rk_serie.findall('.//Buffers/Buffer'):
        out.append({
            'id': (b.findtext('Id') or '').strip(),
            'inlet': (b.findtext('Inlet') or '').strip(),
            'name': (b.findtext('Name') or '').strip(),
        })
    return out


def _extract_autosampler(rk_serie: ET.Element) -> dict:
    """Autosampler racks + reagent table (with category & volume)."""
    out = {'racks': [], 'reagents': []}
    if rk_serie is None:
        return out
    asv = rk_serie.find('AutosamplerVM')
    if asv is None:
        return out
    for rack_tag in ('LeftRack', 'RightRack'):
        rack = asv.find(rack_tag)
        if rack is None:
            continue
        rc = rack.find('.//RackConfig')
        rack_info = {
            'side': rack_tag.replace('Rack', ''),
            'use_single_well': rack.get('UseSingleWell', ''),
            'columns': rc.get('Columns', '') if rc is not None else '',
            'rows': rc.get('Rows', '') if rc is not None else '',
            'rack_volume_uL': rc.get('Volume', '') if rc is not None else '',
            'well_volume_uL': (rack.findtext('WellVolume') or '').strip(),
        }
        out['racks'].append(rack_info)
        for r in rack.findall('.//Reagent'):
            try:
                vol = float(r.get('Volume', '0') or '0')
            except ValueError:
                vol = 0.0
            conc_raw = r.get('Concentration', '')
            try:
                conc_M = _parse_concentration(conc_raw or '0')
            except ValueError:
                conc_M = None  # e.g. '0.5 %' DMSO
            out['reagents'].append({
                'side': rack_tag.replace('Rack', ''),
                'slot': r.get('Slot', ''),
                'designation': r.get('Designation', ''),
                'category': r.get('Category', ''),
                'concentration_M': conc_M,
                'concentration_raw': conc_raw,
                'mw_Da': _parse_mw(r.get('MW', '0')),
                'volume_uL': vol,
            })
    return out


def _extract_immobilization(project_root: ET.Element) -> dict:
    """Summary of the Immobilization serie (chip prep)."""
    serie = next((s for s in project_root.findall('.//Serie')
                  if s.findtext('Type') == 'Immobilization'), None)
    if serie is None:
        return {}
    meta = serie.find('Meta')
    info = {
        'name': (serie.findtext('Name') or '').strip(),
        'guid': (serie.findtext('Guid') or '').strip(),
        'measurement_start': (
            meta.findtext('MeasurementStartTime') or '').strip()
            if meta is not None else '',
        'measurement_end': (
            meta.findtext('MeasurementEndTime') or '').strip()
            if meta is not None else '',
        'capture_fcs': sorted(_capture_fcs(project_root)),
        'cycle_count': len(serie.findall('.//Cycle')),
    }
    return info


def _extract_report_points(rk_serie: ET.Element) -> list[dict]:
    """Report-point definitions from the Pulse serie."""
    out = []
    if rk_serie is None:
        return out
    for rp in rk_serie.findall('.//ReportPointConfiguration'):
        def _t(tag):
            return (rp.findtext(tag) or '').strip()
        try:
            shift = float(_t('Shift'))
        except ValueError:
            shift = None
        try:
            avg = float(_t('Averaging'))
        except ValueError:
            avg = None
        out.append({
            'name': _t('Name'),
            'marker': _t('MarkerType'),
            'shift_s': shift,
            'averaging': avg,
            'is_reference': _t('IsReference') == 'true',
            'active': _t('Active') == 'true',
        })
    return out


# ---------------------------------------------------------------------------
# Wizard reagent lookup — extended with category and volume
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# HDF5 data loading
# ---------------------------------------------------------------------------

def _load_cycle_data(h5f: h5py.File, guid: str, active_ch: int, reference_ch: int):
    """Load time, active, and reference signals for one cycle.

    Returns (time, signal, raw_active, raw_reference) as numpy arrays.
    signal = raw_active - raw_reference (reference-subtracted).
    """
    time_key = f'{guid}.0'
    act_key = f'{guid}.{active_ch}'
    ref_key = f'{guid}.{reference_ch}'

    if time_key not in h5f:
        return None

    time = h5f[time_key][:]
    raw_active = h5f[act_key][:]
    raw_reference = h5f[ref_key][:]
    signal = raw_active - raw_reference

    return time, signal, raw_active, raw_reference


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cxw(filepath: str, channels='all') -> dict:
    """Load a Creoptix .cxw experiment file.

    Parameters
    ----------
    filepath : str
        Path to the .cxw file.
    channels : str or list[int]
        Which active flow cells to load:
        - ``'all'`` (default): load every active channel found in the file.
        - A list of FC numbers, e.g. ``[2, 3]``: load only those channels.
        Each sample cycle produces one entry per active channel.  The
        ``channel`` field (e.g. ``'FC2-FC1'``) identifies which pair was
        used.

    Returns
    -------
    dict with keys:
        config : dict
            ``channel_pairs``: list of dicts with active/reference FC info.
            Legacy keys ``active_fc``, ``reference_fc``, ``active_channel``,
            ``reference_channel`` are set to the **first** pair for
            backwards compatibility.
        samples : list[dict]
            Sample and ControlSample cycles.  Each dict has:
            index, cycle_type, guid, name, compound, concentration_M, mw,
            markers, time, signal, raw_active, raw_reference, channel
        dmso_cals : list[dict]
            DMSO Cal. cycles (same fields).
        blanks : list[dict]
            Blank cycles.
        all_cycles : list[dict]
            Full ordered list of all RAPID Kinetics cycle metadata
            (without signal data).
    """
    with zipfile.ZipFile(filepath, 'r') as zf:
        project = _parse_xml(zf, '_project.cx3')

        # Channel config — returns list of {active_ch, reference_ch, ...}
        all_pairs = _detect_channels(project)

        # Filter to requested channels
        if channels != 'all':
            requested = set(channels)
            all_pairs = [p for p in all_pairs if p['active_fc'] in requested]
            if not all_pairs:
                avail = [p['active_fc'] for p in _detect_channels(project)]
                raise ValueError(
                    f'No matching channels. Requested {channels}, '
                    f'available active FCs: {avail}')

        # Find RAPID Kinetics serie
        rk_serie = None
        rk_guid = None
        for serie in project.findall('.//Serie'):
            if serie.findtext('Type') == 'Pulse':
                rk_serie = serie
                rk_guid = serie.findtext('Guid', '')
                break

        # Reagent lookup from wizard
        reagent_lookup = _build_reagent_lookup(zf, rk_guid)

        # Extended project / instrument / buffers / autosampler / report points
        project_meta = _extract_project_meta(zf)
        instrument = _extract_instrument(project, rk_serie)
        buffers = _extract_buffers(rk_serie)
        autosampler = _extract_autosampler(rk_serie)
        immobilization = _extract_immobilization(project)
        report_points = _extract_report_points(rk_serie)

        # (side, slot) → {category, volume_uL, ...}  for cycle merging
        as_lookup = {(r['side'], r['slot']): r
                     for r in autosampler.get('reagents', [])}

        # Cycle metadata (with autosampler enrichment)
        all_cycles = _extract_cycles(rk_serie, reagent_lookup, as_lookup)

        # Load HDF5 data
        h5_bytes = zf.read('cyclesData.h5')

    h5f = h5py.File(io.BytesIO(h5_bytes), 'r')

    samples = []
    dmso_cals = []
    blanks = []

    for cyc in all_cycles:
        if cyc['cycle_type'] not in ('Sample', 'ControlSample', 'DMSO Cal.', 'Blank'):
            continue

        for pair in all_pairs:
            result = _load_cycle_data(
                h5f, cyc['guid'], pair['active_ch'], pair['reference_ch'])
            if result is None:
                continue

            time, signal, raw_active, raw_reference = result
            channel_label = f"FC{pair['active_fc']}-FC{pair['reference_fc']}"
            entry = {**cyc, 'time': time, 'signal': signal,
                     'raw_active': raw_active, 'raw_reference': raw_reference,
                     'channel': channel_label}

            if cyc['cycle_type'] in ('Sample', 'ControlSample'):
                samples.append(entry)
            elif cyc['cycle_type'] == 'DMSO Cal.':
                dmso_cals.append(entry)
            elif cyc['cycle_type'] == 'Blank':
                blanks.append(entry)

    h5f.close()

    # Backwards-compatible config uses first pair
    first = all_pairs[0]

    return {
        'config': {
            'channel_pairs': all_pairs,
            'active_fc': first['active_fc'],
            'reference_fc': first['reference_fc'],
            'active_channel': first['active_ch'],
            'reference_channel': first['reference_ch'],
        },
        'project': project_meta,
        'instrument': instrument,
        'buffers': buffers,
        'autosampler': autosampler,
        'immobilization': immobilization,
        'report_points': report_points,
        'samples': samples,
        'dmso_cals': dmso_cals,
        'blanks': blanks,
        'all_cycles': all_cycles,
    }
