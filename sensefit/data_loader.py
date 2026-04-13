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
    """Parse the RAPID Kinetics Wizard to build slot → reagent info."""
    lookup = {}
    for fname in zf.namelist():
        if not fname.startswith('Wizard/'):
            continue
        wiz = _parse_xml(zf, fname)
        if wiz.findtext('SerieId') != serie_guid:
            continue
        for r in wiz.findall('.//Reagent'):
            slot = r.get('Slot', '')
            if not slot:
                continue
            lookup[slot] = {
                'compound': r.get('Designation', ''),
                'concentration_M': _parse_concentration(r.get('Concentration', '0')),
                'mw': _parse_mw(r.get('MW', '0')),
            }
        break  # found the right wizard
    return lookup


# ---------------------------------------------------------------------------
# Cycle metadata extraction
# ---------------------------------------------------------------------------

def _extract_cycles(serie_el: ET.Element, reagent_lookup: dict) -> list[dict]:
    """Extract ordered cycle metadata from a Serie element."""
    cycles = []
    for cyc in serie_el.find('Cycles').findall('Cycle'):
        ct_el = cyc.find('CycleType')
        cycle_type = ct_el.text if ct_el is not None else ''

        # Autosampler slot
        slot_el = cyc.find('.//AutosamplerLocation')
        slot = slot_el.get('Slot', '') if slot_el is not None else ''

        # Reagent info from wizard
        reagent = reagent_lookup.get(slot, {})

        # Markers (Injection, Rinse, etc.)
        markers = {}
        for marker in cyc.findall('.//Marker'):
            mtype = marker.findtext('Type', '')
            mx = float(marker.findtext('X', '0'))
            if mtype:
                markers[mtype] = mx

        cycles.append({
            'index': int(cyc.findtext('Index', '-1')),
            'cycle_type': cycle_type,
            'guid': cyc.findtext('Guid', ''),
            'name': cyc.findtext('Name', ''),
            'slot': slot,
            'compound': reagent.get('compound', ''),
            'concentration_M': reagent.get('concentration_M', 0.0),
            'mw': reagent.get('mw', 0.0),
            'markers': markers,
        })
    return cycles


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

        # Cycle metadata
        all_cycles = _extract_cycles(rk_serie, reagent_lookup)

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
        'samples': samples,
        'dmso_cals': dmso_cals,
        'blanks': blanks,
        'all_cycles': all_cycles,
    }
