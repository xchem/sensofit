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
    from kivy.uix.filechooser import FileChooserListView
    from kivy.uix.image import Image
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.gridlayout import GridLayout
except ImportError as exc:
    raise ImportError(
        "Kivy is required to run the SensoFit GUI. "
        "Install it with `pip install kivy`."
    ) from exc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .package_loader import load_experiment
from .models import select_blank, _get_binding_response, fit_last_disso, double_reference

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

        container = BoxLayout(orientation='horizontal', spacing=12, padding=12)

        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.72, spacing=8)
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.28, spacing=8)

        control_bar = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=8)
        load_button = Button(text='Browse .cxw file', size_hint_x=None, width=180)
        load_button.bind(on_release=self.open_file_browser)
        self.status_label = Label(text='No file loaded', halign='left', valign='middle')
        self.status_label.bind(size=self.status_label.setter('text_size'))
        control_bar.add_widget(load_button)
        control_bar.add_widget(self.status_label)

        self.plot_image_1 = Image(size_hint_y=0.55)
        self.plot_image_2 = Image(size_hint_y=0.45)

        left_panel.add_widget(control_bar)
        left_panel.add_widget(self.plot_image_1)
        left_panel.add_widget(self.plot_image_2)

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

        self.list_container = GridLayout(cols=1, spacing=4, size_hint_y=None)
        self.list_container.bind(minimum_height=self.list_container.setter('height'))
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.list_container)

        back_button = Button(text='Back to selection', size_hint_y=None, height=40)
        back_button.bind(on_release=self._go_back)

        right_panel.add_widget(right_title)
        right_panel.add_widget(scroll)
        right_panel.add_widget(back_button)

        container.add_widget(left_panel)
        container.add_widget(right_panel)
        self.add_widget(container)

    def _go_back(self, _):
        self.manager.current = 'selection'

    def open_file_browser(self, _):
        content = BoxLayout(orientation='vertical', spacing=8, padding=8)
        chooser = FileChooserListView(filters=['*.cxw'], size_hint=(1, 0.9))
        chooser.path = os.getcwd()
        chooser.filter_pattern = '*.cxw'

        select_button = Button(text='Open selected file', size_hint_y=None, height=38)
        cancel_button = Button(text='Cancel', size_hint_y=None, height=38)

        control = BoxLayout(size_hint_y=None, height=38, spacing=8)
        control.add_widget(select_button)
        control.add_widget(cancel_button)

        content.add_widget(chooser)
        content.add_widget(control)

        popup = Popup(title='Choose a .cxw file', content=content, size_hint=(0.9, 0.9))

        def select_file(_):
            if chooser.selection:
                filepath = chooser.selection[0]
                if filepath.lower().endswith('.cxw'):
                    popup.dismiss()
                    self.load_and_process_cxw(filepath)
                    return
            self.show_error('Please select a .cxw file.')

        select_button.bind(on_release=select_file)
        cancel_button.bind(on_release=lambda _: popup.dismiss())
        popup.open()

    def show_error(self, message):
        popup = Popup(
            title='Error',
            content=Label(text=message, halign='center', valign='middle'),
            size_hint=(0.7, 0.35),
        )
        popup.content.bind(size=popup.content.setter('text_size'))
        popup.open()

    def load_and_process_cxw(self, filepath):
        self.status_label.text = f'Loading {os.path.basename(filepath)}...'
        try:
            data = load_experiment(filepath)
        except Exception as exc:
            self.show_error(f'Failed to load file: {exc}')
            self.status_label.text = 'File load failed'
            return

        samples = data.get('samples', [])
        blanks = data.get('blanks', [])
        if not samples:
            self.show_error('No sample cycles found in this file.')
            self.status_label.text = 'No samples found'
            return

        self.samples_info = []
        for sample in samples:
            blank = select_blank(sample['index'], [b for b in blanks if b["channel"] == sample["channel"]])
            sample_bl = sample.copy()
            t = sample_bl["time"]
            t_inj = sample_bl["markers"].get("Injection")
            bl_mask = t < t_inj
            sample_bl["raw_active_bl"] = sample["raw_active"] - sample["raw_active"][bl_mask].mean() if bl_mask.any() else sample["raw_active"] - sample["raw_active"][0]
            sample_bl["raw_reference_bl"] = sample["raw_reference"] - sample["raw_reference"][bl_mask].mean() if bl_mask.any() else sample["raw_reference"] - sample["raw_reference"][0]
            sample_bl["sensorgram"], _ = double_reference(sample, blank)

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
            })

        self.samples_info.sort(key=lambda row: (row['koff_active'], -row['binding_response']))
        self.populate_sample_list()
        if self.samples_info:
            self.selected_item = self.samples_info[0]
            self.update_plots()
            self.status_label.text = f'Loaded {len(self.samples_info)} cycles from {os.path.basename(filepath)}'
        else:
            self.status_label.text = 'No valid sample cycles available'

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
            button.bind(size=button.setter('text_size'))
            button.bind(on_release=lambda btn, item=item: self.select_item(item))
            self.list_container.add_widget(button)

    def select_item(self, item):
        self.selected_item = item
        self.update_plots()

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

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, sample['raw_active_bl'], color='red', linewidth=1.5, label='Active (baseline-subtracted)')
        ax.plot(t, sample['raw_reference_bl'], color='blue', linewidth=1.5, label='Reference (baseline-subtracted)')

        popt_act = item['popt_active']
        active_fit = self.exponential_fit(t[disso_mask], popt_act)
        ax.plot(t[disso_mask], active_fit, color='orange', linestyle='--', linewidth=1.2, label=f'Disso. fit (active)\nkoff = {popt_act[0]:.2f}')

        popt_ref = item['popt_reference']
        ref_fit = self.exponential_fit(t[disso_mask], popt_ref)
        ax.plot(t[disso_mask], ref_fit, color='cyan', linestyle='--', linewidth=1.2, label=f'Disso. fit (reference)\nkoff = {popt_ref[0]:.2f}')

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

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, signal, color='black', linewidth=1.5, label='Sensorgram')

        popt_senso = item['popt_sensorgram']
        disso_fit = self.exponential_fit(t[disso_mask], popt_senso)
        ax.plot(t[disso_mask], disso_fit, color='purple', linestyle='--', linewidth=1.2, label=f'Disso. fit\nkoff={popt_senso[0]:.2f}')
        ax.plot([], [], ' ', label=f"Bind. response = {item['binding_response']:.2e}")

        ax.set_title(f'Double-referenced signal for {item["compound"]}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response (pg/mm²)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def exponential_fit(self, t, popt):
        koff, R0, t0 = popt
        return R0 * np.exp(-koff * (t - t0))

    def make_plot_texture(self, fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        return CoreImage(buffer, ext='png').texture


class SensoFitApp(App):
    def build(self):
        Window.size = (1100, 720)
        manager = ScreenManager(transition=FadeTransition())
        manager.add_widget(ModuleSelectionScreen(name='selection'))
        manager.add_widget(PlaceholderModuleScreen(name='module'))
        manager.add_widget(CheckSensorgramsScreen(name='check'))
        return manager
