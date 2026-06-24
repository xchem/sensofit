"""GUI to run different SensoFit module interactively.

You can use 4 different types of modules:
    - Protocol development analysis
    - Check sensorgrams
    - Fit sensorgrams
    - Export data package
"""

import io
import os

try:
    from kivy.app import App
    from kivy.core.image import Image as CoreImage
    from kivy.core.window import Window
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
    from kivy.uix.image import Image
    from kivy.uix.spinner import Spinner
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.textinput import TextInput
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.gridlayout import GridLayout
    from kivy.clock import Clock
except ImportError as exc:
    raise ImportError(
        "Kivy is required to run the SensoFit GUI. "
        "Install it with `pip install kivy`."
    ) from exc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .package_loader import load_experiment
from .models import select_blank, _get_binding_response, fit_last_disso, _disso_rate_equation, double_reference

MODULES = {
    'protocol_dev': {
        'title': 'Protocol Development',
        'description': 'Run automated protocol development analysis for capture, buffer-screen, or stability experiments.',
    },
    'check_sensorgrams': {
        'title': 'Check Sensorgrams',
        'description': 'Inspect raw sensorgrams and identify quality issues or experimental artifacts.',
    },
    'fit_sensorgrams': {
        'title': 'Fit Sensorgrams',
        'description': 'Fit sensorgrams using DK or ODE fitting and review the results graphically.',
    },
    'export_data': {
        'title': 'Export Data Package',
        'description': 'Export raw .cxw files into a self-describing SensoFit package for data release.',
    },
}


class ModuleSelectionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=12, padding=20)

        title = Label(
            text='Select a SensoFit module',
            font_size='22sp',
            size_hint=(1, 0.18),
            halign='center',
            valign='middle',
        )
        title.bind(size=title.setter('text_size'))
        layout.add_widget(title)

        for key, module in MODULES.items():
            button = Button(
                text=module['title'],
                size_hint=(1, 0.18),
                bold=True,
            )
            button.bind(on_release=lambda btn, key=key: self._open_module_screen(key))
            layout.add_widget(button)

        footer = Label(
            text='Run `python -m sensofit gui` to open this interface.',
            size_hint=(1, 0.12),
            halign='center',
            valign='middle',
            font_size='14sp',
        )
        footer.bind(size=footer.setter('text_size'))
        layout.add_widget(footer)

        self.add_widget(layout)

    def _open_module_screen(self, module_key):
        if module_key == 'check_sensorgrams':
            self.manager.current = 'check'
        else:
            module_screen = self.manager.get_screen('module')
            module_screen.set_module(module_key)
            self.manager.current = 'module'


class PlaceholderModuleScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_key = None
        self.layout = BoxLayout(orientation='vertical', spacing=12, padding=20)

        self.title_label = Label(
            text='',
            font_size='24sp',
            size_hint=(1, 0.16),
            halign='center',
            valign='middle',
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        self.layout.add_widget(self.title_label)

        self.description_label = Label(
            text='',
            font_size='16sp',
            size_hint=(1, 0.36),
            halign='center',
            valign='middle',
        )
        self.description_label.bind(size=self.description_label.setter('text_size'))
        self.layout.add_widget(self.description_label)

        run_button = Button(
            text='Select this module',
            size_hint=(1, 0.18),
            bold=True,
        )
        run_button.bind(on_release=self._run_module)
        self.layout.add_widget(run_button)

        back_button = Button(
            text='Back to selection',
            size_hint=(1, 0.18),
        )
        back_button.bind(on_release=self._go_back)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def set_module(self, module_key):
        self.module_key = module_key
        module = MODULES.get(module_key, {})
        self.title_label.text = module.get('title', 'Module')
        self.description_label.text = module.get('description', '')

    def _run_module(self, _):
        popup = Popup(
            title='Not implemented',
            content=Label(
                text=f'The "{MODULES[self.module_key]["title"]}" module is not implemented in GUI yet.',
                halign='center',
                valign='middle',
            ),
            size_hint=(0.8, 0.4),
        )
        popup.content.bind(size=popup.content.setter('text_size'))
        popup.open()

    def _go_back(self, _):
        self.manager.current = 'selection'


class CheckSensorgramsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.samples_info = []
        self.selected_item = None
        self.selected_filepath = None

        # Main window layout
        container = BoxLayout(orientation='horizontal', spacing=12, padding=12)

        # Left and right handside layouts
        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.72, spacing=8)
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.28, spacing=8)

        ### LEFT HAND SIDE (LHS)
        # LHS - Top: Import files layouts
        control_bar = BoxLayout(orientation='vertical', size_hint_y=None, height=88, spacing=8)
        
        # First row --> import file, activate processing
        first_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=8)
        browse_button = Button(text='Browse .cxw or .zip file', size_hint_x=None, width=180)
        browse_button.bind(on_release=self.open_file_browser)
        self.process_button = Button(text='Process file', size_hint_x=None, width=180, disabled=True)
        self.process_button.bind(on_release=self.start_processing)
        self.status_label = Label(text='No file selected', halign='left', valign='middle')
        self.status_label.bind(size=self.status_label.setter('text_size'))
        first_row.add_widget(browse_button)
        first_row.add_widget(self.process_button)
        first_row.add_widget(self.status_label)
        
        # Second row --> add pre-existing checks (to continue a check after saving)
        second_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=8)
        self.import_button = Button(text='Import previous checks to data', size_hint_x=None, width=368, disabled=True)
        self.import_button.bind(on_release=self.on_import_button)
        second_row.add_widget(self.import_button)
        
        control_bar.add_widget(first_row)
        control_bar.add_widget(second_row)

        self.selected_file_label = Label(
            text='Selected file: none',
            size_hint_y=None,
            height=24,
            halign='left',
            valign='middle',
            font_size='14sp',
        )
        self.selected_file_label.bind(size=self.selected_file_label.setter('text_size'))

        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        self.progress_label = Label(
            text='',
            size_hint_y=None,
            height=24,
            halign='left',
            valign='middle',
            font_size='13sp',
        )
        self.progress_label.bind(size=self.progress_label.setter('text_size'))

        # LHS - Middle: Split figures layout
        self.plot_image_1 = Image(size_hint_y=0.5)
        self.plot_image_2 = Image(size_hint_y=0.5)

        # LHS - Undeneath figure: Flags and Comment container (horizontal layout)
        flags_comment_container = BoxLayout(orientation='horizontal', size_hint_y=None, height=80, spacing=8)
        
        # Left --> Good and Bad checkboxes (vertical)
        flags_container = BoxLayout(orientation='vertical', size_hint_x=0.3, spacing=6)
        good_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=28, spacing=8)
        self.good_checkbox = CheckBox(size_hint_x=None, width=24)
        good_label = Label(text='Good', size_hint_x=None, width=40, halign='left', valign='middle')
        good_label.bind(size=good_label.setter('text_size'))
        good_row.add_widget(self.good_checkbox)
        good_row.add_widget(good_label)
        
        bad_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=28, spacing=8)
        self.bad_checkbox = CheckBox(size_hint_x=None, width=24)
        bad_label = Label(text='Bad', size_hint_x=None, width=40, halign='left', valign='middle')
        bad_label.bind(size=bad_label.setter('text_size'))
        bad_row.add_widget(self.bad_checkbox)
        bad_row.add_widget(bad_label)
        
        flags_container.add_widget(good_row)
        flags_container.add_widget(bad_row)
        self.good_checkbox.bind(active=self.on_flag_change)
        self.bad_checkbox.bind(active=self.on_flag_change)
        
        # Right --> Comment box
        comment_subcontainer = BoxLayout(orientation='vertical', size_hint_x=0.7, spacing=2)
        comment_label = Label(text='Comment:', size_hint_y=None, height=20, halign='left', valign='middle')
        comment_label.bind(size=comment_label.setter('text_size'))
        self.comment_input = TextInput(
            text='',
            multiline=True,
            size_hint_y=None,
            height=56,
            halign='left',
        )
        self.comment_input.bind(text=self.on_comment_change)
        comment_subcontainer.add_widget(comment_label)
        comment_subcontainer.add_widget(self.comment_input)
        
        flags_comment_container.add_widget(flags_container)
        flags_comment_container.add_widget(comment_subcontainer)
        
        # LHS - Bottom: navigation buttons (Previous/Next)
        nav_container = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=8)
        prev_button = Button(text='<-- Previous', size_hint_x=0.5)
        prev_button.bind(on_release=self.on_prev_cycle)
        next_button = Button(text='Next -->', size_hint_x=0.5)
        next_button.bind(on_release=self.on_next_cycle)
        nav_container.add_widget(prev_button)
        nav_container.add_widget(next_button)

        left_panel.add_widget(control_bar)
        left_panel.add_widget(self.selected_file_label)
        left_panel.add_widget(self.progress_bar)
        left_panel.add_widget(self.progress_label)
        left_panel.add_widget(self.plot_image_1)
        left_panel.add_widget(self.plot_image_2)
        left_panel.add_widget(flags_comment_container)
        left_panel.add_widget(nav_container)

        ### RIGHT HAND SIDE (RHS)
        right_title = Label(
            text='Sorted compounds',
            size_hint_y=None,
            height=30,
            font_size='16sp',
            bold=True,
            halign='center',
            valign='middle',
        )
        right_title.bind(size=right_title.setter('text_size'))

        # RHS - Top: sort cycles
        sort_bar = BoxLayout(orientation='vertical', size_hint_y=None, height=80, spacing=4)
        top_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=44, spacing=8)
        sort_label = Label(text='Sort by:', size_hint_x=None, width=55, halign='left', valign='middle')
        sort_label.bind(size=sort_label.setter('text_size'))
        self.sort_field_spinner = Spinner(text='', values=('cycle type', 'koff (active)', 'bind. response'), size_hint_x=0.45)
        # Descending tick box (default = ascending)
        desc_box = BoxLayout(orientation='horizontal', size_hint_x=0.55, spacing=2)
        self.desc_checkbox = CheckBox(active=False, size_hint_x=None, width=24)
        desc_label = Label(text='Descending', size_hint_x=1, halign='left', valign='middle')
        desc_label.bind(size=desc_label.setter('text_size'))
        desc_box.add_widget(self.desc_checkbox)
        desc_box.add_widget(desc_label)
        top_row.add_widget(sort_label)
        top_row.add_widget(self.sort_field_spinner)
        top_row.add_widget(desc_box)

        bottom_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=36)
        self.filter_button = Button(text='Apply filter', size_hint_x=1, disabled= True)
        self.filter_button.bind(on_release=self.on_filter_button)
        bottom_row.add_widget(self.filter_button)

        sort_bar.add_widget(top_row)
        sort_bar.add_widget(bottom_row)

        # RHS - Middle: List of cycles
        self.list_container = GridLayout(cols=1, spacing=4, size_hint_y=None)
        self.list_container.bind(minimum_height=self.list_container.setter('height'))
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.list_container)

        # RHS - Bottom: Export checks or go back to Main GUI Window
        self.export_button = Button(text='Export to CSV/XLSX', size_hint_y=None, height=40, disabled=True)
        self.export_button.bind(on_release=self.on_export_button)
        back_button = Button(text='Back to selection', size_hint_y=None, height=40)
        back_button.bind(on_release=self._go_back)

        right_panel.add_widget(right_title)
        right_panel.add_widget(sort_bar)
        right_panel.add_widget(scroll)
        right_panel.add_widget(self.export_button)
        right_panel.add_widget(back_button)

        container.add_widget(left_panel)
        container.add_widget(right_panel)
        self.add_widget(container)

    def _go_back(self, _):
        self.manager.current = 'selection'

    def open_file_browser(self, _):
        content = BoxLayout(orientation='vertical', spacing=8, padding=8)
        chooser = FileChooserIconView(filters=['*.cxw', '*.zip'], size_hint=(1, 0.9))
        chooser.path = os.getcwd()
        chooser.filter_pattern = '*.cxw'

        select_button = Button(text='Select file', size_hint_y=None, height=38)
        cancel_button = Button(text='Cancel', size_hint_y=None, height=38)

        control = BoxLayout(size_hint_y=None, height=38, spacing=8)
        control.add_widget(select_button)
        control.add_widget(cancel_button)

        content.add_widget(chooser)
        content.add_widget(control)

        popup = Popup(title='Choose a .cxw or .zip file', content=content, size_hint=(0.9, 0.9))

        def select_file(_):
            if chooser.selection:
                filepath = chooser.selection[0]
                if filepath.lower().endswith(('.cxw', '.zip')):
                    popup.dismiss()
                    self.set_selected_file(filepath)
                    return
            self.show_error('Please select a valid .cxw or .zip file.')

        select_button.bind(on_release=select_file)
        cancel_button.bind(on_release=lambda _: popup.dismiss())
        popup.open()

    def set_selected_file(self, filepath):
        self.selected_filepath = filepath
        self.selected_file_label.text = f'Selected file: {os.path.basename(filepath)}'
        self.process_button.disabled = False
        self.status_label.text = 'Ready to process'
        self.progress_bar.value = 0
        self.progress_label.text = ''
        self.samples_info = []
        self.selected_item = None
        self.list_container.clear_widgets()
        self.plot_image_1.texture = None
        self.plot_image_2.texture = None
        self.good_checkbox.active = False
        self.bad_checkbox.active = False
        self.comment_input.text = ''
        self.import_button.disabled = True
        self.filter_button.disabled = True
        self.export_button.disabled = True

    def start_processing(self, _):
        if not self.selected_filepath:
            self.show_error('No file selected. Please choose a .cxw or a .zip file first.')
            return
        self.process_button.disabled = True
        self.progress_bar.value = 1
        self.progress_label.text = 'Starting processing...'
        Clock.schedule_once(self._load_selected_file, 0)

    def _load_selected_file(self, dt):
        filepath = self.selected_filepath
        self._update_progress(1, f'Loading {os.path.basename(filepath)}...')
        try:
            data = load_experiment(filepath)
        except Exception as exc:
            self.show_error(f'Failed to load file: {exc}')
            self._reset_after_error('File load failed')
            return

        samples = data.get('samples', [])
        blanks = data.get('blanks', [])
        if not samples:
            self.show_error('No sample cycles found in this file.')
            self._reset_after_error('No samples found')
            return

        self.samples_info = []
        self._processing_samples = samples
        self._processing_blanks = blanks
        self._processing_total = len(samples)
        self._processing_index = 0
        self.list_container.clear_widgets()
        self.plot_image_1.texture = None
        self.plot_image_2.texture = None
        self._update_progress(1, 'Processing cycles...')
        Clock.schedule_once(self._process_next_sample, 0)

    def _process_next_sample(self, dt):
        if self._processing_index >= self._processing_total:
            self._finalize_processing()
            return

        idx = self._processing_index + 1
        sample = self._processing_samples[self._processing_index]
        blanks = self._processing_blanks
        blank = select_blank(sample['index'], [b for b in blanks if b['channel'] == sample['channel']])
        sample_bl = sample.copy()
        t = sample_bl['time']
        t_inj = sample_bl['markers'].get('Injection', t[0])
        bl_mask = t < t_inj
        sample_bl['raw_active_bl'] = sample['raw_active'] - sample['raw_active'][bl_mask].mean() if bl_mask.any() else sample['raw_active'] - sample['raw_active'][0]
        sample_bl['raw_reference_bl'] = sample['raw_reference'] - sample['raw_reference'][bl_mask].mean() if bl_mask.any() else sample['raw_reference'] - sample['raw_reference'][0]
        sample_bl['sensorgram'], _ = double_reference(sample, blank)

        popt_act, perr_act = fit_last_disso(sample_bl, channel='raw_active')
        popt_ref, perr_ref = fit_last_disso(sample_bl, channel='raw_reference')
        popt_senso, perr_senso = fit_last_disso(sample_bl, channel='signal', blank=blank)
        bind_resp = _get_binding_response(sample_bl, blank)

        compound = sample_bl.get('compound', 'Unknown')
        cycle_id = sample_bl.get('index', 'n/a')
        channel = sample_bl.get('channel', 'n/a')
        label = f'{compound} (cycle {cycle_id} - channel {channel})'

        self.samples_info.append({
            'label': label,
            'compound': compound,
            'cycle_id': cycle_id,
            'channel': channel,
            'koff_active': float(popt_act[0]),
            'koff_active_error': float(perr_act[0]),
            'koff_reference': float(popt_ref[0]),
            'koff_reference_error': float(perr_ref[0]),
            'koff_sensorgram': float(popt_senso[0]),
            'koff_sensorgram_error': float(perr_senso[0]),
            'binding_response': float(bind_resp),
            'sample': sample_bl,
            'blank': blank,
            'popt_active': popt_act,
            'perr_active': perr_act,
            'popt_reference': popt_ref,
            'perr_reference': perr_ref,
            'popt_sensorgram': popt_senso,
            'perr_sensorgram': perr_senso,
            'good_flag': False,
            'bad_flag': False,
            'comment': '',
        })

        self._processing_index += 1
        progress_value = int(100 * idx / self._processing_total)
        self._update_progress(progress_value, f'Processed cycle {self._processing_index} of {self._processing_total}...')
        Clock.schedule_once(self._process_next_sample, 0)

    def _finalize_processing(self):
        # Apply user-selected sorting and refresh the UI
        self.apply_sort_and_refresh()
        if self.samples_info:
            self._update_progress(100, f'Loaded {len(self.samples_info)} cycles from {os.path.basename(self.selected_filepath)}')
            self.status_label.text = 'Successfully processed'
            self.process_button.disabled = True
            # Select and highlight first cycle
            self.selected_item = self.samples_info[0]
            self.populate_sample_list()
            self.update_plots()
            # Enable import, filter, and export button
            self.import_button.disabled = False
            self.filter_button.disabled = False
            self.export_button.disabled = False
        else:
            self._reset_after_error('No valid sample cycles available')

    def _reset_after_error(self, status_text):
        self.progress_bar.value = 0
        self.progress_label.text = ''
        self.status_label.text = status_text
        self.process_button.disabled = False

    def _update_progress(self, value, text):
        self.progress_bar.value = value
        self.progress_label.text = text

    def show_message(self, message):
        popup = Popup(
            title='Message',
            content=Label(text=message, halign='center', valign='middle'),
            size_hint=(0.7, 0.35),
        )
        popup.content.bind(size=popup.content.setter('text_size'))
        popup.open()

    def show_error(self, message):
        popup = Popup(
            title='Error',
            content=Label(text=message, halign='center', valign='middle'),
            size_hint=(0.7, 0.35),
        )
        popup.content.bind(size=popup.content.setter('text_size'))
        popup.open()

    def populate_sample_list(self):
        self.list_container.clear_widgets()
        for item in self.samples_info:
            button = Button(
                text=item['label'],
                size_hint_y=None,
                height=40,
                halign='left',
                valign='middle',
            )
            # Highlight selected item
            if item == self.selected_item:
                button.background_color = (0.3, 0.7, 1.0, 1.0)  # Light blue
            else:
                button.background_color = (1, 1, 1, 1)  # White
            
            button.bind(size=button.setter('text_size'))
            button.bind(on_release=lambda btn, item=item: self.select_item(item))
            self.list_container.add_widget(button)

    def on_prev_cycle(self, _):
        """Navigate to previous cycle in the list."""
        if not self.samples_info or not self.selected_item:
            return
        try:
            current_index = self.samples_info.index(self.selected_item)
            if current_index > 0:
                self.select_item(self.samples_info[current_index - 1])
                self.populate_sample_list()
        except (ValueError, IndexError):
            pass

    def on_next_cycle(self, _):
        """Navigate to next cycle in the list."""
        if not self.samples_info or not self.selected_item:
            return
        try:
            current_index = self.samples_info.index(self.selected_item)
            if current_index < len(self.samples_info) - 1:
                self.select_item(self.samples_info[current_index + 1])
                self.populate_sample_list()
        except (ValueError, IndexError):
            pass

    def on_filter_button(self, _):
        # User pressed Filter — apply user-selected primary sorting
        self.apply_sort_and_refresh(use_user_field=True)

    def apply_sort_and_refresh(self, use_user_field: bool = False):
        """Sort `self.samples_info` and refresh the UI.

        - If `use_user_field` is False: apply the default sort (index asc, channel asc).
        - If True: sort by the chosen primary field (ascending by default),
          using cycle index then channel as stable secondary/tertiary keys.
        """
        if not self.samples_info:
            return

        # Always ensure stable tiebreakers: tertiary = channel (asc), secondary = index (asc)
        try:
            # tertiary: channel asc
            self.samples_info.sort(key=lambda r: (r.get('sample', {}).get('channel') or ''))
            # secondary: index asc
            self.samples_info.sort(key=lambda r: r.get('sample', {}).get('index', 0))
        except Exception:
            pass

        if not use_user_field:
            # Default behavior already applied (index, channel)
            self.populate_sample_list()
            if self.samples_info:
                self.selected_item = self.samples_info[0]
                self.update_plots()
            return

        # User-chosen primary field
        key_field = getattr(self, 'sort_field_spinner', None).text if getattr(self, 'sort_field_spinner', None) else ''
        descending = getattr(self, 'desc_checkbox', None).active if getattr(self, 'desc_checkbox', None) else False

        if key_field == 'cycle type':
            primary_key = lambda r: (r.get('sample', {}).get('cycle_type') or '')
        elif key_field == 'koff (active)':
            primary_key = lambda r: r.get('koff_active', float('inf'))
        elif key_field == 'bind. response':
            primary_key = lambda r: r.get('binding_response', 0.0)
        else:
            primary_key = lambda r: r.get('sample', {}).get('index', 0)

        try:
            # primary sort (may be descending)
            self.samples_info.sort(key=primary_key, reverse=descending)
        except Exception:
            pass

        # Refresh UI
        self.populate_sample_list()
        if self.samples_info:
            self.selected_item = self.samples_info[0]
            self.update_plots()

    def select_item(self, item):
        # Save previous item's flags/comment if any
        if self.selected_item and self.selected_item != item:
            self.selected_item['good_flag'] = self.good_checkbox.active
            self.selected_item['bad_flag'] = self.bad_checkbox.active
            self.selected_item['comment'] = self.comment_input.text
        
        # Load new item and populate UI
        self.selected_item = item
        if 'good_flag' not in item:
            item['good_flag'] = False
        if 'bad_flag' not in item:
            item['bad_flag'] = False
        if 'comment' not in item:
            item['comment'] = ''
        
        # Populate flags and comment UI (without triggering change handlers)
        self.good_checkbox.active = item.get('good_flag', False)
        self.bad_checkbox.active = item.get('bad_flag', False)
        self.comment_input.text = item.get('comment', '')
        
        # Refresh highlighting and plots
        self.populate_sample_list()
        self.update_plots()

    def on_flag_change(self, instance, value):
        """Handle good/bad flag changes."""
        if self.selected_item:
            self.selected_item['good_flag'] = self.good_checkbox.active
            self.selected_item['bad_flag'] = self.bad_checkbox.active

    def on_comment_change(self, instance, value):
        """Handle comment text changes."""
        if self.selected_item:
            self.selected_item['comment'] = value

    def on_export_button(self, _):
        """Open file browser and export data to CSV or XLSX."""
        # Save current item's flags/comment
        if self.selected_item:
            self.selected_item['good_flag'] = self.good_checkbox.active
            self.selected_item['bad_flag'] = self.bad_checkbox.active
            self.selected_item['comment'] = self.comment_input.text
        
        if not self.samples_info:
            self.show_error('No data to export. Please process a file first.')
            return
        
        content = BoxLayout(orientation='vertical', spacing=8, padding=8)
        chooser = FileChooserListView(size_hint=(1, 0.9))
        chooser.path = os.path.expanduser('~')
        
        filename_input = TextInput(
            text='export',
            multiline=False,
            size_hint_y=None,
            height=40,
            halign='left',
        )
        
        ext_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=8)
        ext_label = Label(text='Format:', size_hint_x=None, width=80, halign='left', valign='middle')
        ext_label.bind(size=ext_label.setter('text_size'))
        format_spinner = Spinner(
            text='CSV',
            values=('CSV', 'XLSX'),
            size_hint_x=None,
            width=100,
        )
        ext_box.add_widget(ext_label)
        ext_box.add_widget(format_spinner)
        
        control = BoxLayout(size_hint_y=None, height=40, spacing=8)
        save_button = Button(text='Save', size_hint_x=0.5)
        cancel_button = Button(text='Cancel', size_hint_x=0.5)
        control.add_widget(save_button)
        control.add_widget(cancel_button)
        
        content.add_widget(Label(text='Select save location and enter filename:', size_hint_y=None, height=30))
        content.add_widget(chooser)
        content.add_widget(Label(text='Filename:', size_hint_y=None, height=24))
        content.add_widget(filename_input)
        content.add_widget(ext_box)
        content.add_widget(control)
        
        popup = Popup(title='Export Data', content=content, size_hint=(0.85, 0.95))
        
        def save_file(_):
            dirname = chooser.path
            filename = filename_input.text.strip()
            if not filename:
                self.show_error('Please enter a filename.')
                return
            
            ext = format_spinner.text.lower()
            filepath = os.path.join(dirname, f'{filename}.{ext}')
            
            try:
                self._export_dataframe(filepath, ext)
                popup.dismiss()
                self.show_message(f'Data exported to:\n{filepath}')
            except Exception as e:
                self.show_error(f'Export failed: {str(e)}')
        
        save_button.bind(on_release=save_file)
        cancel_button.bind(on_release=lambda _: popup.dismiss())
        popup.open()

    def _export_dataframe(self, filepath: str, ext: str):
        """Export samples_info to CSV or XLS file."""
        rows = []
        for item in self.samples_info:
            sample = item.get('sample', {})
            row = {
                'file_name': os.path.basename(self.selected_filepath) if self.selected_filepath else '',
                'rk_serie_id': sample.get('rk_serie_id', ''),
                'cycle_id': item.get('cycle_id', ''),
                'channel': item.get('channel', ''),
                'cycle_type': sample.get('cycle_type', ''),
                'compound': item.get('compound', ''),
                'concentration': sample.get('concentration_M', ''),
                'koff_active': item.get('koff_active', ''),
                'koff_act_error': item.get('koff_active_error', ''),
                'koff_reference': item.get('koff_reference', ''),
                'koff_ref_err': item.get('koff_reference_error', ''),
                'koff_sensorgram': item.get('koff_sensorgram', ''),
                'koff_senso_err': item.get('koff_sensorgram_error', ''),
                'bind_response': item.get('binding_response', ''),
                'good_flag': item.get('good_flag', False),
                'bad_flag': item.get('bad_flag', False),
                'comments': item.get('comment', ''),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if ext.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif ext.lower() == 'xls':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f'Unsupported format: {ext}')


    def update_plots(self):
        if not self.selected_item:
            return
        self.plot_image_1.texture = self.make_plot_texture(self.make_first_figure(self.selected_item))
        self.plot_image_2.texture = self.make_plot_texture(self.make_second_figure(self.selected_item))

    def make_first_figure(self, item):
        sample = item['sample']
        t = sample['time']
        t_rinse = sample['markers'].get('Rinse')
        disso_mask = t > t_rinse
        blank = item['blank']

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, sample['raw_active_bl'], color='red', linewidth=1.5, label='Active (baseline-subtracted)')
        ax.plot(t, sample['raw_reference_bl'], color='blue', linewidth=1.5, label='Reference (baseline-subtracted)')

        koff, R0, t0  = item['popt_active']
        active_fit = _disso_rate_equation(t[disso_mask], koff, R0, t0)
        ax.plot(t[disso_mask], active_fit, color='orange', linestyle='--', linewidth=1.2, label=f'Disso. fit (active)\nkoff = {koff:.2f}')

        koff, R0, t0 = item['popt_reference']
        ref_fit = _disso_rate_equation(t[disso_mask], koff, R0, t0)
        ax.plot(t[disso_mask], ref_fit, color='cyan', linestyle='--', linewidth=1.2, label=f'Disso. fit (reference)\nkoff = {koff:.2f}')

        if blank is not None:
            t_blk = blank['time']
            t_blk_inj = blank['markers'].get('Injection')
            blk_bl_mask = t_blk < t_blk_inj
            blank_signal = blank['signal'] - blank['signal'][blk_bl_mask].mean() if blk_bl_mask.any() else blank['signal'] - blank['signal'][0]
            ax.plot(blank['time'], blank_signal, color='grey', linewidth=1.0, label='Blank (baseline-subtracted)')

        ax.set_title(item['label'], fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response (pg/mm²)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def make_second_figure(self, item):
        sample = item['sample']
        t = sample['time']
        t_rinse = sample['markers'].get('Rinse')
        disso_mask = t > t_rinse
        signal = sample['sensorgram']

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, signal, color='black', linewidth=1.5, label='Sensorgram')

        koff, R0, t0 = item['popt_sensorgram']
        disso_fit = _disso_rate_equation(t[disso_mask], koff, R0, t0)
        ax.plot(t[disso_mask], disso_fit, color='purple', linestyle='--', linewidth=1.2, label=f'Disso. fit\nkoff={koff:.2f}')
        ax.plot([], [], ' ', label=f"Bind. response = {item['binding_response']:.2e}")

        ax.set_title(f'Double-referenced signal for {item['label']}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response (pg/mm²)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def make_plot_texture(self, fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        return CoreImage(buffer, ext='png').texture

    def on_import_button(self, _):
        """Open file browser to import previous checks from CSV/XLSX files."""
        if not self.samples_info:
            self.show_error('No data to import into. Please process a file first.')
            return
        
        content = BoxLayout(orientation='vertical', spacing=8, padding=8)
        chooser = FileChooserListView(filters=['*.csv', '*.xlsx'], size_hint=(1, 0.9))
        chooser.path = os.path.expanduser('~')
        chooser.filter_pattern = '*.csv|*.xlsx'
        
        select_button = Button(text='Import file', size_hint_y=None, height=38)
        cancel_button = Button(text='Cancel', size_hint_y=None, height=38)
        
        control = BoxLayout(size_hint_y=None, height=38, spacing=8)
        control.add_widget(select_button)
        control.add_widget(cancel_button)
        
        content.add_widget(chooser)
        content.add_widget(control)
        
        popup = Popup(title='Import previous checks', content=content, size_hint=(0.9, 0.9))
        
        def import_file(_):
            if chooser.selection:
                filepath = chooser.selection[0]
                if filepath.lower().endswith(('.csv', '.xlsx')):
                    popup.dismiss()
                    self._import_checks_from_file(filepath)
                    return
            self.show_error('Please select a valid CSV or XLSX file.')
        
        select_button.bind(on_release=import_file)
        cancel_button.bind(on_release=lambda _: popup.dismiss())
        popup.open()

    def _import_checks_from_file(self, filepath: str):
        """Import good_flag, bad_flag, and comments from a CSV or XLSX file.
        
        Matches records by compound, index, and channel. Only updates if filename matches
        the currently processed file.
        """
        try:
            if filepath.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.lower().endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                self.show_error('Unsupported file format. Use CSV or XLSX.')
                return
        except Exception as e:
            self.show_error(f'Failed to read file: {str(e)}')
            return
        
        # Check if filename matches
        if 'file_name' in df.columns:
            imported_filename = df['file_name'].iloc[0] if len(df) > 0 else None
            current_filename = os.path.basename(self.selected_filepath) if self.selected_filepath else None
            
            if imported_filename != current_filename:
                self.show_error(f'Filename mismatch!\n'
                               f'Imported: {imported_filename}\n'
                               f'Current: {current_filename}\n'
                               f'File ignored.')
                return
        
        # Match and update records
        matches_found = 0
        for _, row in df.iterrows():
            imported_compound = row.get('compound', '') if 'compound' in row else ''
            imported_index = row.get('cycle_id', '') if 'cycle_id' in row else ''
            imported_channel = row.get('channel', '') if 'channel' in row else ''
            imported_good = row.get('good_flag', False) if 'good_flag' in row else False
            imported_bad = row.get('bad_flag', False) if 'bad_flag' in row else False
            imported_comment = row.get('comments', '') if 'comments' in row else ''
            
            # Find matching sample
            for sample in self.samples_info:
                if (sample.get('compound') == imported_compound and
                    sample.get('cycle_id') == imported_index and
                    sample.get('channel') == imported_channel):
                    sample['good_flag'] = bool(imported_good)
                    sample['bad_flag'] = bool(imported_bad)
                    sample['comment'] = str(imported_comment) if pd.notna(imported_comment) else ''
                    matches_found += 1
                    break
        
        # Refresh UI
        self.populate_sample_list()
        if self.selected_item:
            self.good_checkbox.active = self.selected_item.get('good_flag', False)
            self.bad_checkbox.active = self.selected_item.get('bad_flag', False)
            self.comment_input.text = self.selected_item.get('comment', '')
        
        self.show_message(f'Import complete!\n{matches_found} records updated.')


class SensoFitApp(App):
    def build(self):
        Window.size = (1100, 720)
        manager = ScreenManager(transition=FadeTransition())
        manager.add_widget(ModuleSelectionScreen(name='selection'))
        manager.add_widget(PlaceholderModuleScreen(name='module'))
        manager.add_widget(CheckSensorgramsScreen(name='check'))
        return manager
