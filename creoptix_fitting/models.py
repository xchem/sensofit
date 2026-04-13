"""1:1 Langmuir binding model, concentration profile construction, and signal preprocessing.

Core model:
    dR/dt = ka * c(t) * (Rmax - R) - kd * R

Where:
    R(t)   = binding response (pg/mm², reference-subtracted)
    c(t)   = analyte concentration at the sensor surface (M)
    ka     = association rate constant (M⁻¹ s⁻¹)
    kd     = dissociation rate constant (s⁻¹)
    Rmax   = maximum binding capacity (pg/mm²)
    KD     = kd / ka  (equilibrium dissociation constant, M)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Concentration profile from DMSO calibration
# ---------------------------------------------------------------------------

def build_concentration_profile(dmso_cycle: dict, C_analyte: float,
                                pulse_window: float = 3.5):
    """Build c(t) from a DMSO calibration cycle.

    The Creoptix GCI uses pulsed injections — the raw signal alternates
    between analyte and buffer pulses.  We extract the upper envelope
    (rolling maximum over one pulse period) to recover the smooth
    transport/dispersion profile, then baseline-subtract, normalize to
    [0, 1], and scale by C_analyte.

    Parameters
    ----------
    dmso_cycle : dict
        A cycle dict from load_cxw with keys 'time', 'raw_reference', 'markers'.
    C_analyte : float
        Injected analyte concentration (M).
    pulse_window : float
        Approximate pulse period in seconds (default 3.5 s).  The rolling
        maximum window is set to this value.

    Returns
    -------
    c_func : callable
        c(t) → concentration in M.  Interpolates the envelope profile.
    c_raw : np.ndarray
        Envelope concentration array on the DMSO time grid.
    """
    t = dmso_cycle['time']
    sig = dmso_cycle['raw_reference'].copy()

    # --- Extract upper envelope via rolling maximum ---
    dt = np.median(np.diff(t))
    half_win = int(np.ceil(pulse_window / dt / 2))
    envelope = np.array([sig[max(0, i - half_win):i + half_win + 1].max()
                         for i in range(len(sig))])

    # Baseline-subtract using pre-injection envelope
    inj_time = dmso_cycle['markers'].get('Injection', t[0])
    baseline_mask = t < inj_time
    if baseline_mask.any():
        envelope -= envelope[baseline_mask].mean()

    # Normalize to [0, 1] then scale
    env_max = envelope.max()
    if env_max > 0:
        c_raw = envelope / env_max * C_analyte
    else:
        c_raw = np.zeros_like(envelope)

    # Clamp negative values (noise below baseline)
    c_raw = np.clip(c_raw, 0, None)

    # Build interpolator (extrapolate as 0 outside domain)
    from scipy.interpolate import interp1d
    c_func = interp1d(t, c_raw, kind='linear', bounds_error=False, fill_value=0.0)

    return c_func, c_raw


def build_pulsed_concentration_profile(dmso_cycle: dict, C_analyte: float,
                                       pulse_window: float = 3.5):
    """Build pulsed c(t) from a DMSO calibration cycle.

    Unlike ``build_concentration_profile``, this preserves the pulse
    structure: c(t) is high during analyte pulses and drops toward zero
    during buffer pulses.  Suitable for ODE fitting over the full
    sensorgram where buffer-pulse intervals are weighted.

    The raw DMSO reference signal directly measures the RI bulk
    (proportional to analyte concentration at the sensor).  We
    baseline-subtract, normalize by the envelope peak, and scale
    by C_analyte.

    Parameters
    ----------
    dmso_cycle : dict
        DMSO cal dict with 'time', 'raw_reference', 'markers'.
    C_analyte : float
        Nominal analyte concentration (M).
    pulse_window : float
        Approximate pulse period (s) for envelope peak estimation.

    Returns
    -------
    c_func : callable
        c(t) → concentration (M).  Preserves pulse structure.
    c_raw : np.ndarray
        Pulsed concentration array on the DMSO time grid.
    """
    t = dmso_cycle['time']
    sig = dmso_cycle['raw_reference'].copy()

    # Baseline-subtract
    inj_time = dmso_cycle['markers'].get('Injection', t[0])
    baseline_mask = t < inj_time
    if baseline_mask.any():
        sig -= sig[baseline_mask].mean()

    # Normalize by the envelope peak (= peak analyte concentration)
    dt = np.median(np.diff(t))
    half_win = int(np.ceil(pulse_window / dt / 2))
    envelope = np.array([sig[max(0, i - half_win):i + half_win + 1].max()
                         for i in range(len(sig))])
    env_max = envelope.max()

    if env_max > 0:
        c_raw = sig / env_max * C_analyte
    else:
        c_raw = np.zeros_like(sig)

    c_raw = np.clip(c_raw, 0, None)

    from scipy.interpolate import interp1d
    c_func = interp1d(t, c_raw, kind='linear', bounds_error=False, fill_value=0.0)

    return c_func, c_raw


def select_dmso_cal(sample_index: int, dmso_cals: list[dict]) -> dict:
    """Select the nearest DMSO Cal cycle for a given sample index.

    Uses the DMSO Cal. cycle with the closest index (preferring the
    preceding one in case of ties).
    """
    best = min(dmso_cals,
               key=lambda d: (abs(d['index'] - sample_index),
                              d['index'] > sample_index))
    return best


# ---------------------------------------------------------------------------
# Blank selection and double referencing
# ---------------------------------------------------------------------------

def select_blank(sample_index: int, blanks: list[dict]) -> dict:
    """Select the nearest preceding blank cycle for double referencing.

    If no blank precedes the sample, returns the closest overall.
    """
    preceding = [b for b in blanks if b['index'] < sample_index]
    if preceding:
        return max(preceding, key=lambda b: b['index'])
    return min(blanks, key=lambda b: abs(b['index'] - sample_index))


def double_reference(sample: dict, blanks: list[dict]):
    """Apply double referencing: subtract nearest preceding blank.

    If the subtraction yields a negative peak response near the Rinse
    marker, iterates through preceding blanks until a valid one is found.
    Handles array length mismatches by truncating to the shorter length.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    blanks : list[dict]
        Blank cycles from load_cxw().

    Returns
    -------
    signal_corrected : np.ndarray
        Baseline-subtracted, blank-subtracted binding signal.
    blank_index : int or None
        Index of the blank used, or None if fallback (no blank subtraction).
    """
    t = sample['time']
    inj_time = sample['markers'].get('Injection', t[0])
    rinse_time = sample['markers'].get('Rinse', t[-1])
    n = len(t)

    # Baseline-subtract sample
    bl_mask = t < inj_time
    s_baseline = sample['signal'][bl_mask].mean() if bl_mask.any() else 0.0
    s_bl = sample['signal'] - s_baseline

    # Sort blanks: preceding first (nearest first), then following
    candidates = sorted(
        blanks,
        key=lambda b: (b['index'] > sample['index'],
                       abs(b['index'] - sample['index'])))

    for blank in candidates:
        n_b = len(blank['signal'])
        n_min = min(n, n_b)
        bl_mask_b = t[:n_min] < inj_time
        b_baseline = blank['signal'][:n_min][bl_mask_b].mean() if bl_mask_b.any() else 0.0
        b_bl = blank['signal'][:n_min] - b_baseline
        corrected_short = s_bl[:n_min] - b_bl

        # Pad back to original length if blank was shorter
        if n_min < n:
            corrected = np.empty(n)
            corrected[:n_min] = corrected_short
            corrected[n_min:] = s_bl[n_min:]
        else:
            corrected = corrected_short

        # Check: peak response near Rinse should be positive
        peak_mask = (t >= rinse_time - 5) & (t <= rinse_time)
        if peak_mask.any() and corrected[peak_mask].mean() > 0:
            return corrected, blank['index']

    # Fallback: no blank subtraction, just baseline-subtract
    return s_bl, None


# ---------------------------------------------------------------------------
# Non-specific binder detection
# ---------------------------------------------------------------------------

def is_nonspecific_binder(sample: dict, threshold: float = 2.0):
    """Detect non-specific binding from the reference channel.

    Non-specific binders show significant analyte retention on the
    reference surface (FC1) after rinse.  This is measured as a
    positive baseline-subtracted raw_reference signal 2-5 s into
    the dissociation phase (after RI bulk has been rinsed away).

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    threshold : float
        Reference dissociation signal (pg/mm²) above which the sample
        is classified as a non-specific binder.  Default 2.0.

    Returns
    -------
    nsb : bool
        True if the sample is a non-specific binder.
    ref_dissoc : float
        Baseline-subtracted reference signal at dissociation onset.
    """
    t = sample['time']
    ref = sample['raw_reference']
    markers = sample['markers']
    inj = markers.get('Injection', t[0])
    rinse = markers.get('Rinse', t[-1])

    bl_mask = t < inj
    ref_bl = ref[bl_mask].mean() if bl_mask.any() else ref[0]

    # 2-5 s after rinse: RI bulk gone, only true binding remains
    diss_mask = (t >= rinse + 2) & (t <= rinse + 5)
    if not diss_mask.any():
        return False, 0.0

    ref_dissoc = float((ref[diss_mask] - ref_bl).mean())
    return ref_dissoc > threshold, ref_dissoc


# ---------------------------------------------------------------------------
# Pulse mask from DMSO calibration
# ---------------------------------------------------------------------------

def build_pulse_mask(dmso_cycle: dict, threshold_frac: float = 0.3):
    """Identify analyte-pulse vs buffer-pulse intervals from DMSO cal.

    During pulsed injection, the raw reference signal alternates between
    high (analyte flowing → RI bulk) and low (buffer flowing → no RI).
    This mask labels each time point.

    Parameters
    ----------
    dmso_cycle : dict
        DMSO cal cycle with 'time', 'raw_reference', 'markers'.
    threshold_frac : float
        Fraction of (peak − baseline) above baseline to use as threshold.

    Returns
    -------
    is_buffer : np.ndarray (bool)
        True where buffer is flowing (safe for fitting — no RI bulk).
        During baseline and dissociation, always True.
    """
    t = dmso_cycle['time']
    ref = dmso_cycle['raw_reference']
    inj_time = dmso_cycle['markers'].get('Injection', t[0])
    rinse_time = dmso_cycle['markers'].get('Rinse', t[-1])

    baseline = ref[t < inj_time].mean()
    inj_mask = (t >= inj_time) & (t <= rinse_time)
    peak = ref[inj_mask].max() if inj_mask.any() else baseline
    threshold = baseline + threshold_frac * (peak - baseline)

    is_buffer = np.ones(len(t), dtype=bool)
    # During injection window, mark analyte pulses
    is_buffer[inj_mask & (ref > threshold)] = False
    return is_buffer


# ---------------------------------------------------------------------------
# Signal smoothing and differentiation
# ---------------------------------------------------------------------------

def smooth_and_differentiate(t: np.ndarray, R: np.ndarray,
                             smoothing_factor: float | None = None):
    """Fit a smoothing spline and compute dR/dt analytically.

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    R : np.ndarray
        Binding response array.
    smoothing_factor : float or None
        Smoothing parameter for UnivariateSpline.  None = automatic (GCV).

    Returns
    -------
    R_smooth : np.ndarray
        Smoothed response.
    dRdt : np.ndarray
        Time derivative of the smoothed response.
    spline : UnivariateSpline
        The fitted spline object.
    """
    spline = UnivariateSpline(t, R, s=smoothing_factor)
    R_smooth = spline(t)
    dRdt = spline.derivative()(t)
    return R_smooth, dRdt, spline


# ---------------------------------------------------------------------------
# Weight mask
# ---------------------------------------------------------------------------

def build_weight_mask(t: np.ndarray, markers: dict) -> np.ndarray:
    """Build a weight mask that zeros association phases.

    Weights = 1 during dissociation (Rinse → RinseEnd),
    weights = 0 elsewhere (baseline + association).

    For waveRAPID with multiple sub-pulses, the association phase
    runs from Injection to Rinse, and dissociation from Rinse to RinseEnd.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    markers : dict
        Must contain 'Rinse' and 'RinseEnd'.

    Returns
    -------
    w : np.ndarray
        Weight mask, same shape as t.
    """
    rinse = markers.get('Rinse', 0)
    rinse_end = markers.get('RinseEnd', t[-1])

    w = np.zeros_like(t)
    w[(t >= rinse) & (t <= rinse_end)] = 1.0
    return w


def build_full_weight_mask(sample_time: np.ndarray, sample_markers: dict,
                           dmso_cycle: dict) -> np.ndarray:
    """Weight mask using buffer pulses during association + full dissociation.

    During pulsed GCI injection, RI bulk artifacts contaminate the signal
    when analyte is flowing.  Buffer-pulse intervals (analyte not flowing)
    show clean binding signal suitable for fitting.

    This mask weights:
      - w = 0  during baseline (before injection)
      - w = 0  during analyte pulses (RI artifacts)
      - w = 1  during buffer pulses in association phase (clean signal)
      - w = 1  during dissociation (Rinse → RinseEnd)

    Parameters
    ----------
    sample_time : np.ndarray
        Time array for the sample cycle.
    sample_markers : dict
        Markers dict with 'Injection', 'Rinse', 'RinseEnd'.
    dmso_cycle : dict
        Nearest DMSO cal cycle (used to detect pulse timing).

    Returns
    -------
    w : np.ndarray
        Weight mask, same shape as sample_time.
    """
    from scipy.interpolate import interp1d

    t = sample_time
    inj_time = sample_markers.get('Injection', t[0])
    rinse_time = sample_markers.get('Rinse', t[-1])
    rinse_end = sample_markers.get('RinseEnd', t[-1])

    # Detect buffer vs analyte pulse intervals from DMSO cal
    is_buffer = build_pulse_mask(dmso_cycle)

    # Transfer pulse timing to sample time grid (nearest-neighbour)
    f = interp1d(dmso_cycle['time'], is_buffer.astype(float),
                 kind='nearest', bounds_error=False, fill_value=1.0)
    is_buffer_sample = f(t) > 0.5

    w = np.zeros_like(t)
    # Buffer pulses during injection phase
    inj_mask = (t >= inj_time) & (t < rinse_time)
    w[inj_mask & is_buffer_sample] = 1.0
    # Full dissociation window
    w[(t >= rinse_time) & (t <= rinse_end)] = 1.0

    return w


# ---------------------------------------------------------------------------
# Fitting window trimming
# ---------------------------------------------------------------------------

def trim_to_fit_window(t, signal, w, markers, pre_s=0.5, post_s=2.0):
    """Trim arrays to the active fitting window [Injection-pre_s, RinseEnd+post_s].

    The ODE only needs to integrate from just before injection (where
    R≈0 and c(t) is about to rise) through dissociation.  Excluding the
    pre-injection baseline and post-dissociation tail speeds up fitting
    and avoids modelling irrelevant regions.

    Parameters
    ----------
    t, signal, w : np.ndarray
        Full-cycle time, signal, and weight mask.
    markers : dict
        Cycle markers with 'Injection' and 'RinseEnd'.
    pre_s : float
        Seconds before Injection to include (default 0.5).
    post_s : float
        Seconds after RinseEnd to include (default 2.0).

    Returns
    -------
    t_trim, signal_trim, w_trim : np.ndarray
        Trimmed arrays.
    fit_mask : np.ndarray (bool)
        Boolean mask on the original arrays so that
        ``t[fit_mask] == t_trim``.
    """
    inj = markers.get('Injection', t[0])
    rinse_end = markers.get('RinseEnd', t[-1])
    t_start = inj - pre_s
    t_end = rinse_end + post_s
    fit_mask = (t >= t_start) & (t <= t_end)
    return t[fit_mask], signal[fit_mask], w[fit_mask], fit_mask


# ---------------------------------------------------------------------------
# 1:1 Langmuir ODE
# ---------------------------------------------------------------------------

def langmuir_ode(t, R, ka, kd, Rmax, c_func):
    """1:1 Langmuir binding ODE.

    dR/dt = ka * c(t) * (Rmax - R) - kd * R
    """
    c = c_func(t)
    return ka * c * (Rmax - R) - kd * R


def simulate_sensorgram(t: np.ndarray, ka: float, kd: float, Rmax: float,
                        c_func, R0: float = 0.0) -> np.ndarray:
    """Simulate a 1:1 Langmuir sensorgram via ODE integration.

    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate.
    ka, kd, Rmax : float
        Kinetic parameters.
    c_func : callable
        c(t) → concentration (M).
    R0 : float
        Initial response at t[0].

    Returns
    -------
    R : np.ndarray
        Simulated binding response at each time point.
    """
    sol = solve_ivp(
        langmuir_ode,
        t_span=(t[0], t[-1]),
        y0=[R0],
        t_eval=t,
        args=(ka, kd, Rmax, c_func),
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        max_step=0.5,
    )
    return sol.y[0]
