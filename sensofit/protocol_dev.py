"""Analysis functions for protocol development experiments.

You can analyse 3 types of experiments with this module (selected by the
--mode argument when running `sensofit protocol-dev` with the CLI):
    - capture: Compare per-sample binding responses with different constructs
    captured on the chips & analyse % of non-specific binding (NSB) across samples
    for each construct.
    - buffer-screen: Compare per-sample binding responses and kinetic parameters
    in different buffer conditions & analyse % of NSB across samples for each buffer.
    - stability: Analyse binding responses and kinetic parameters across multiple
    cycles of the same sample in different conditions.
"""

def run_capture():
    """Analyse capture protocol development experiments."""
    pass

def run_buffer_screen():
    """Analyse buffer screen protocol development experiments."""
    pass

def run_stability():
    """Analyse stability protocol development experiments."""
    pass
