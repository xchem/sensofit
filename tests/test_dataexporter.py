"""Tests for sensofit.dataexporter — packaged data export."""

import csv
import io
import json
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensofit.dataexporter import (
    _cycle_folder_name,
    _format_concentration,
    _sanitize,
    export_cxw,
    export_package,
)

CXW = os.path.join(os.path.dirname(__file__), '..',
                   '20260323_EV71 2A Binding assay.cxw')


class TestHelpers:
    def test_format_concentration_units(self):
        assert _format_concentration(0) == '0M'
        assert _format_concentration(5e-7) == '500nM'
        assert _format_concentration(25e-6) == '25uM'
        assert _format_concentration(2e-3) == '2mM'
        assert _format_concentration(1e-12) == '1pM'

    def test_sanitize_strips_unsafe_chars(self):
        assert _sanitize('FC2-FC1') == 'FC2-FC1'
        assert _sanitize('hello world') == 'hello_world'
        assert _sanitize('a/b\\c?d') == 'a_b_c_d'
        assert _sanitize('') == 'unnamed'
        assert _sanitize('   ') == 'unnamed'
        # No path separators leak through
        assert '/' not in _sanitize('foo/bar')

    def test_cycle_folder_name_sample(self):
        cyc = {'compound': 'ASAP-0001', 'concentration_M': 25e-6, 'index': 7}
        assert _cycle_folder_name(cyc) == 'ASAP-0001__25uM__cyc007'

    def test_cycle_folder_name_blank(self):
        cyc = {'compound': '', 'cycle_type': 'Blank',
               'concentration_M': 0, 'index': 1}
        assert _cycle_folder_name(cyc) == 'Blank__0M__cyc001'

    def test_cycle_folder_name_blank_with_inherited_compound(self):
        # Blank cycles may carry an inherited compound name from a
        # mismatched-side wizard lookup; the folder must NOT include it.
        cyc = {'compound': 'ASAP-0001', 'cycle_type': 'Blank',
               'concentration_M': 25e-6, 'index': 4}
        name = _cycle_folder_name(cyc)
        assert name.startswith('Blank__')
        assert 'ASAP' not in name
        assert name.endswith('__cyc004')

    def test_cycle_folder_name_dmso(self):
        cyc = {'compound': '', 'cycle_type': 'DMSO Cal.',
               'concentration_M': 0, 'index': 2}
        assert _cycle_folder_name(cyc) == 'DMSOcal__0M__cyc002'

    def test_cycle_folder_name_control(self):
        cyc = {'compound': 'ASAP-CTRL', 'cycle_type': 'ControlSample',
               'concentration_M': 5e-6, 'index': 9}
        name = _cycle_folder_name(cyc)
        assert name.startswith('Control__')
        assert '__cyc009' in name


@pytest.mark.skipif(not os.path.isfile(CXW),
                    reason='Sample .cxw not present')
class TestExportCxw:
    def test_export_cxw_creates_structure(self, tmp_path):
        summary = export_cxw(CXW, str(tmp_path))
        cxw_root = tmp_path / summary['cxw_folder']
        assert cxw_root.is_dir()
        assert (cxw_root / 'experiment.json').is_file()
        assert summary['n_cycles_total'] > 0
        assert summary['cycles'], 'no cycles recorded in summary'

        first = summary['cycles'][0]
        cyc_dir = cxw_root / first['folder']
        assert cyc_dir.is_dir()
        assert (cyc_dir / 'metadata.json').is_file()

        # CSV per channel
        for ch in first['channels']:
            csv_path = cyc_dir / f'{_sanitize(ch)}.csv'
            assert csv_path.is_file()
            with open(csv_path) as fh:
                reader = csv.reader(fh)
                header = next(reader)
                assert header == ['time_s', 'signal',
                                  'raw_active', 'raw_reference']
                rows = list(reader)
                assert len(rows) > 10  # sanity: real sensorgram
                # 4 numeric columns
                for r in rows[:5]:
                    assert len(r) == 4
                    for v in r:
                        float(v)

    def test_metadata_json_fields(self, tmp_path):
        summary = export_cxw(CXW, str(tmp_path))
        cyc = summary['cycles'][0]
        meta_path = (tmp_path / summary['cxw_folder'] / cyc['folder']
                     / 'metadata.json')
        meta = json.loads(meta_path.read_text())
        for key in ('index', 'cycle_type', 'guid', 'compound',
                    'concentration_M', 'concentration_pretty',
                    'mw_Da', 'markers', 'channels',
                    'slot_side', 'flow_rate_uLmin', 'contact_time_s',
                    'injection_mode', 'pulse_durations_s',
                    'reagent_category', 'buffer_inlet'):
            assert key in meta

    def test_experiment_json_enriched(self, tmp_path):
        summary = export_cxw(CXW, str(tmp_path))
        exp_path = tmp_path / summary['cxw_folder'] / 'experiment.json'
        exp = json.loads(exp_path.read_text())
        for key in ('project', 'instrument', 'buffers',
                    'autosampler', 'immobilization', 'report_points'):
            assert key in exp, f'missing top-level key: {key}'
        # Instrument basics from the Pulse serie
        assert exp['instrument'].get('device_type')
        assert exp['instrument'].get('serial_number')
        # Report points list non-empty for a real run
        assert exp['report_points'], 'no report points captured'


@pytest.mark.skipif(not os.path.isfile(CXW),
                    reason='Sample .cxw not present')
class TestExportPackage:
    def test_export_package_zip(self, tmp_path):
        out_zip = tmp_path / 'pkg.zip'
        result = export_package([CXW], str(out_zip), package_name='demo_pkg')
        assert os.path.isfile(result)

        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
            # Top-level package folder + README
            assert any(n.endswith('demo_pkg/README.md') for n in names), names
            # CXW subfolder + experiment.json
            assert any(n.endswith('experiment.json') for n in names)
            # At least one CSV
            assert any(n.endswith('.csv') for n in names)

            readme_name = next(n for n in names
                               if n.endswith('demo_pkg/README.md'))
            readme = zf.read(readme_name).decode('utf-8')
            assert '# demo_pkg' in readme
            assert 'time_s' in readme
            assert 'FC' in readme  # channel label rendered

    def test_export_package_requires_input(self, tmp_path):
        with pytest.raises(ValueError):
            export_package([], str(tmp_path / 'empty.zip'))
