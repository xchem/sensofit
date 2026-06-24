"""GUI to run different SensoFit module interactively.

You can use 4 different types of modules:
    - Protocol development analysis
    - Check sensorgrams
    - Fit sensorgrams
    - Export data package
"""

try:
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
except ImportError as exc:
    raise ImportError(
        "Kivy is required to run the SensoFit GUI. "
        "Install it with `pip install kivy`."
    ) from exc

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
        'description': 'Export raw .cxw data into a self-describing SensoFit package for sharing.',
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
        module_screen = self.manager.get_screen('module')
        module_screen.set_module(module_key)
        self.manager.current = 'module'


class ModuleScreen(Screen):
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
        module = MODULES.get(self.module_key)
        if not module:
            return
        popup = Popup(
            title='Module selected',
            content=Label(
                text=f'You selected "{module["title"]}".\n' \
                     'This interface can be extended to launch the corresponding workflow.',
                halign='center',
                valign='middle',
            ),
            size_hint=(0.8, 0.4),
        )
        popup.content.bind(size=popup.content.setter('text_size'))
        popup.open()

    def _go_back(self, _):
        self.manager.current = 'selection'


class SensoFitApp(App):
    def build(self):
        Window.size = (520, 460)
        manager = ScreenManager(transition=FadeTransition())
        manager.add_widget(ModuleSelectionScreen(name='selection'))
        manager.add_widget(ModuleScreen(name='module'))
        return manager
