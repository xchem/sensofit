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
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Root Mean Square Error (RMSE) for model fitting
# ---------------------------------------------------------------------------

def get_rmse(y_true, y_pred):
    """Calculate the root mean square error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    rmse : float
        The RMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


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
        DMSO cal dict with 'time', 'raw_active', 'markers'.
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
    sig = dmso_cycle['raw_active'].copy()

    # Baseline-subtract
    bl_time = dmso_cycle.get('baseline_duration_s', 45)
    baseline_mask = t <= bl_time
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


def select_dmso_cal(sample_index: int, dmso_cals: list[dict], verbose=False) -> dict:
    """Select the nearest preceding DMSO Cal cycle for a given sample index.

    If no DMSO Cal precedes the sample, returns the closest overall.
    """
    valid = [d for d in dmso_cals if _is_dmso_cal_valid(d, verbose=verbose)]
    if not valid:
        channel = dmso_cals[0]['channel'][:2] if dmso_cals else 'unknown'
        rk_serie = dmso_cals[0].get('rk_serie_id', 'unknown') if dmso_cals else 'unknown'
        raise ValueError(f"No valid DMSO Cal found in RK serie {rk_serie} for channel {channel}. "
                         f"Please retry batch processing by excluding this channel.")
    preceding = [d for d in valid if d['index'] < sample_index]
    if preceding:
        return max(preceding, key=lambda d: d['index'])
    return min(valid, key=lambda d: abs(d['index'] - sample_index))


def _is_dmso_cal_valid(dmso_cycle: dict, verbose=False) -> bool:
    """Check if a DMSO Cal cycle is valid for concentration profile construction.

    Criteria:
    - Baseline signal std >= 5.0 ==> BAD (indicates a noisy baseline before injection).
    - max(Response) <= 50 ==> BAD (indicates a weak or failed pulse).
    - min(raw signal in Active channel) <= -10 ==> BAD (indicates a problem with the raw signal, e.g. sensor error or incorrect channel).
    - Otherwise ==> GOOD (valid for c(t) construction).
    """
    t = dmso_cycle['time']
    markers = dmso_cycle['markers']
    bl_time = dmso_cycle.get('baseline_duration_s', 45)
    inj_time = markers.get('Injection', t[0])
    rinse_time = markers.get('Rinse', t[-1])
    bl_mask = t <= bl_time
    response_mask = (t >= inj_time) & (t <= rinse_time)
    baseline = dmso_cycle['raw_active'][bl_mask].mean() if bl_mask.any() else dmso_cycle['raw_active'][:int(bl_time)].mean()
    dmso_signal = dmso_cycle['raw_active'] - baseline
    baseline_std = dmso_signal[bl_mask].std() if bl_mask.any() else 0.0
    max_response = dmso_signal[response_mask].max() if response_mask.any() else dmso_signal.max()
    min_signal = dmso_signal.min()
    if verbose and not (baseline_std <= 5.0 and max_response >= 50.0 and min_signal >= -10.0):
        print(f"WARNING! DMSO Cal cycle {dmso_cycle['index']} ({dmso_cycle['channel']}) failed validity check "
              f"(std={baseline_std:.2f}, max_response={max_response:.2f}, "
              f"min_signal={min_signal:.2f}). This cycle will be excluded from "
              "concentration profile construction.")
    return baseline_std < 5.0 and max_response > 50.0 and min_signal > -10.0


# ---------------------------------------------------------------------------
# Blank selection and double referencing
# ---------------------------------------------------------------------------

def select_blank(sample_index: int, blanks: list[dict], verbose=False) -> dict:
    """Select the nearest preceding blank cycle for double referencing.

    If no blank precedes the sample, returns the closest overall.
    """
    valid = [b for b in blanks if _is_blank_valid(b, verbose=verbose)]
    if not valid:
        if verbose:
            channel = blanks[0]['channel'][:2] if blanks else 'unknown'
            rk_serie = blanks[0].get('rk_serie_id', 'unknown') if blanks else 'unknown'
            print(f"WARNING: No valid blank cycles found in RK serie {rk_serie} for channel {channel}.  "
                  f"Proceeding without blank subtraction...")
        return None
    preceding = [b for b in valid if b['index'] < sample_index]
    if preceding:
        return max(preceding, key=lambda b: b['index'])
    return min(valid, key=lambda b: abs(b['index'] - sample_index))


def _is_blank_valid(blank: dict, verbose=False) -> bool:
    """Check if a blank cycle is valid for double referencing.
    Criteria:
    - Baseline signal std > 2.5 ==> BAD (indicates a noisy baseline before injection).
    - -5 >= Steady-state baseline-subtracted signal (currently mean of last 10 points) >= 5 ==> BAD (indicates a strong binding response in the blank).
    - max(Baseline-subtracted signal) > 50 ==> BAD (indicates a strong binding response in the blank).
    - Response between injection and rinse <= -5 ==> BAD (indicates a problem during injection)
    - Otherwise ==> GOOD (valid for double referencing).
    """
    t = blank['time']
    markers = blank['markers']
    bl_time = blank.get('baseline_duration_s', 45)
    inj_time = markers.get('Injection', t[0])
    rinse_time = markers.get('Rinse', t[-1])
    bl_mask = t <= bl_time
    response_mask = (t >= inj_time) & (t <= rinse_time)
    baseline = blank['signal'][bl_mask].mean() if bl_mask.any() else blank['signal'][:int(bl_time)].mean()
    blank_signal = blank['signal'] - baseline
    baseline_std = blank_signal[bl_mask].std() if bl_mask.any() else 0.0
    steady_state = blank_signal[-10:].mean()
    max_signal = blank_signal.max()
    min_response = blank_signal[response_mask].min() if response_mask.any() else blank_signal.min()
    if verbose and not (baseline_std <= 2.5 and -5.0 < steady_state < 5.0 and max_signal <= 50.0 and min_response > -5.0):
        print(f"WARNING! Blank cycle {blank['index']} ({blank['channel']}) failed validity check "
              f"(std={baseline_std:.2f}, steady_state={steady_state:.2f}, "
              f"max_signal={max_signal:.2f}, min_response={min_response:.2f}). "
              "This blank will be excluded from double referencing.")
    return baseline_std <= 2.5 and -5.0 < steady_state < 5.0 and max_signal <= 50.0 and min_response > -5.0

def double_reference(sample: dict, blank: dict):
    """Apply double referencing: subtract nearest preceding blank.

    If the subtraction yields a negative peak response near the Rinse
    marker, iterates through preceding blanks until a valid one is found.
    Handles array length mismatches by truncating to the shorter length.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    blank : dict
        Blank cycle from load_cxw().

    Returns
    -------
    signal_corrected : np.ndarray
        Baseline-subtracted, blank-subtracted binding signal.
    blank_index : int or None
        Index of the blank used, or None if fallback (no blank subtraction).
    """
    t = sample['time']
    bl_time = sample.get('baseline_duration_s', 45)
    n = len(t)

    # Baseline-subtract sample
    bl_mask = t <= bl_time
    s_baseline = sample['signal'][bl_mask].mean() if bl_mask.any() else sample['signal'][:int(bl_time)].mean()
    s_bl = sample['signal'] - s_baseline

    if blank:
        # Baseline-subtract blank
        n_b = len(blank['signal'])
        n_min = min(n, n_b)
        bl_mask_b = t[:n_min] <= bl_time
        b_baseline = blank['signal'][:n_min][bl_mask_b].mean() if bl_mask_b.any() else blank['signal'][:n_min].mean()
        b_bl = blank['signal'][:n_min] - b_baseline
        corrected_short = s_bl[:n_min] - b_bl

        # Pad back to original length if blank was shorter
        if n_min < n:
            corrected = np.empty(n)
            corrected[:n_min] = corrected_short
            corrected[n_min:] = s_bl[n_min:]
        else:
            corrected = corrected_short

        return corrected, blank['index']

    # Fallback: no blank subtraction, just baseline-subtract
    print(f"WARNING: No valid blank found for sample {sample['index']} (channel {sample['channel']}). "
          f"Proceeding with baseline-subtracted signal without blank subtraction...")
    return s_bl, None


# ---------------------------------------------------------------------------
# Sensorgram heuristics and quality control.
# ---------------------------------------------------------------------------

def _get_binding_response(sample: dict, signal: np.ndarray):
    """Calculate the binding response for a sample cycle. 
    The binding response is defined as the mean double-referenced signal 
    in the window 2 seconds before the Rinse marker.

    This function also checks for a negative response immediately after injection, 
    which may indicate an injection error. If such an error is detected, the function 
    returns 0.0 as the binding response.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    signal : np.ndarray
        Double-referenced (or baseline-subtracted) signal from the sample.

    Returns
    -------
    bind_resp : float
        The calculated binding response (pg/mm²). 
        If an injection error is detected, returns 0.0.
    """
    t = sample['time']
    markers = sample['markers']
    rinse_time = markers.get('Rinse', t[-1])
    
    # 2-5 s before rinse: RI bulk gone, only true binding remains
    resp_mask = (t >= rinse_time - 2) & (t <= rinse_time)
    if not resp_mask.any():
        print(f"WARNING! Binding response cannot be calculated for sample {sample['index']} (channel {sample['channel']}). "
              "Returning 0.0 as binding response.")
        return 0.0

    bind_resp = float(signal[resp_mask].mean())
    return bind_resp if bind_resp >= 0 else 0.0  # Return 0.0 if negative response detected


def is_baseline_noisy(sample: dict, signal: np.ndarray, percent_threshold: float = 10):
    """Detect noisy baseline in a sample cycle.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    signal : np.ndarray
        Double-referenced (or baseline-subtracted) signal from sample.
    percent_threshold : float
        Threshold for the standard deviation of the signal above which it is considered noisy. 
        Default 10.0% of the maximum abs(signal) value.

    Returns
    -------
    noisy : bool
        True if baseline_std > percent_threshold * max_abs_signal / 100.
    baseline_std : float
        Standard deviation of the double-referenced signal, used for assessment.
    """
    t = sample['time']
    bl_time = sample.get('baseline_duration_s', 45)
    bl_mask = t <= bl_time
    baseline_std = signal[bl_mask].std() if bl_mask.any() else 0.0
    return baseline_std > (percent_threshold / 100.0) * np.max(np.abs(signal)), baseline_std


def has_injection_issue(sample: dict, percent_threshold: float = 15.0):
    """Detect injection issues in a sample cycle.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    percent_threshold : float
        Threshold for maximum abs(delta) value between the baseline_mean and injection_mean 
        for either reference or active channel, above which there is probably an injection issue.
        Default 15.0% of maximum signal value.
    
    Returns
    -------
    inj_issue : bool
        True if abs(delta) > percent_threshold * max_signal for either reference.
    deltas : tuple
        Delta values between baseline and injection responses for reference and 
        active channels, used for error assessment.
    """
    t = sample['time']
    bl_time = sample.get('baseline_duration_s', 45)
    inj_time = sample['markers'].get('Injection', bl_time + sample.get('contact_time_s', 25))
    bl_mask = t <= bl_time
    inj_mask = (t > bl_time) & (t < inj_time)

    ref_bl_mean = sample['raw_reference'][bl_mask].mean() if bl_mask.any() else sample['raw_reference'][:int(bl_time)].mean()
    ref_bl = sample['raw_reference'][bl_mask] - ref_bl_mean if bl_mask.any() else sample['raw_reference'][:int(bl_time)] - ref_bl_mean
    ref_inj = sample['raw_reference'][inj_mask] - ref_bl_mean if inj_mask.any() else sample['raw_reference'][int(bl_time):int(inj_time)] - ref_bl_mean
    ref_max = np.abs(sample['raw_reference'] - ref_bl_mean).max()

    active_bl_mean = sample["raw_active"][bl_mask].mean() if bl_mask.any() else sample["raw_active"][:int(bl_time)].mean()
    active_bl = sample['raw_active'][bl_mask] - active_bl_mean if bl_mask.any() else sample['raw_active'][:int(bl_time)] - active_bl_mean
    active_inj = sample["raw_active"][inj_mask] - active_bl_mean if inj_mask.any() else sample["raw_active"][int(bl_time):int(inj_time)] - active_bl_mean
    active_max = np.abs(sample['raw_active'] - active_bl_mean).max()

    delta_ref = np.abs(ref_bl.mean() - ref_inj.mean())
    delta_active = np.abs(active_bl.mean() - active_inj.mean())
    inj_issue = (delta_ref >= percent_threshold * ref_max / 100) or (delta_active >= percent_threshold * active_max / 100)
    return inj_issue, (delta_ref, delta_active)


def is_reference_response_negative(sample: dict, percent_threshold: float = 10.0):
    """Detect negative response in reference channel, 
    which will affect signal interpretation.
    
    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    percent_threshold : float
        Threshold for the negative response in the reference channel.
        Default 10.0% times of -max_abs_signal.

    Returns
    -------
    negative : bool
        True if min(reference_response) < injection_mean - (percent_threshold / 100.0 * max_abs_signal).
    min_ref : float
        Minimum value of the raw_reference signal, used for assessment.
    """
    t = sample['time']
    bl_time = sample.get('baseline_duration_s', 45)
    inj_time = sample['markers'].get('Injection', t[0])
    rinse_time = sample['markers'].get('Rinse', t[-1])
    bl_mask = t <= bl_time
    inj_mask = (t > bl_time) & (t < inj_time)
    response_mask = (t >= inj_time) & (t <= rinse_time)
    s_baseline = sample['raw_reference'][bl_mask] if bl_mask.any() else sample['raw_reference'][:int(bl_time)]
    ref_signal = sample['raw_reference'] - s_baseline.mean()
    min_response = ref_signal[response_mask].min() if response_mask.any() else ref_signal.min()
    max_abs_signal = np.abs(ref_signal).max()
    ref_inj_mean = ref_signal[inj_mask].mean() if inj_mask.any() else ref_signal[0]
    return min_response < ref_inj_mean - (percent_threshold / 100.0 * max_abs_signal), ref_inj_mean


def is_sample_carried_over(sample: dict, signal: np.ndarray, percent_threshold: float = 10.0, time_window: float = 10.0):
    """Detect sample carryover at the end of the cycle.
    Get the mean of the baseline-subtracted signal in the last X seconds of the cycle, 
    and if it is above the threshold, there is probably carryover.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    signal : np.ndarray
        Double-referenced (or baseline-subtracted) signal from sample.
    percent_threshold : float
        Threshold for the signal at the end of the cycle above which there is probably carryover. 
        Default 10.0% of the maximum abs(signal) value.
    time_window : float
        The duration (in seconds) of the time window at the end of the cycle to consider for carryover detection. 
        Default 10.0.

    Returns
    -------
    carryover : bool
        True if signal at the end of the cycle > threshold.
    end_signal : float
        Baseline-subtracted signal at the end of the cycle, used for assessment.    
    """
    t = sample['time']
    rinseend_time = sample['markers'].get('RinseEnd', t[-1])
    ss_mask = (t >= rinseend_time - time_window) & (t <= rinseend_time) # last time_window seconds of the cycle
    end_signal = signal[ss_mask].mean() if ss_mask.any() else signal[-1]
    return end_signal > ((percent_threshold / 100.0) * np.max(np.abs(signal))), end_signal


def has_low_signal_to_noise_reponse(sample: dict, signal: np.ndarray, snr_threshold: float = 50.0):
    """Detect low signal-to-noise response in sensorgram.

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    signal : np.ndarray
        Double-referenced (or baseline-subtracted) signal from sample.
    snr_threshold: float
        Threshold of the signal to noise ratio between the binding response 
        and the noise (`snr = bind_resp/contact_std`).
        Default 50.0.

    Returns
    -------
    low_snr : bool
        True if the sample has a signal-to-noise ratio lower 
        than the threshold.
    bind_resp : float
        Binding response (pg/mm²), used for assessment.
    """
    t = sample['time']
    bl_time = sample.get('baseline_duration_s', 45)
    inj_time = sample['markers'].get('Injection', (bl_time + sample.get('contact_time_s', 25)))
    contact_mask = (t > bl_time) & (t <= inj_time)
    contact_std = signal[contact_mask].std() if contact_mask.any() else 0.0
    bind_resp = _get_binding_response(sample, signal)
    return (bind_resp / contact_std) <= snr_threshold if contact_std != 0.0 else False, bind_resp


def is_nonspecific_binder(sample: dict, koff_threshold: float = 1.25, percent_threshold: float = 5.0):
    """Detect non-specific binding from the reference channel.

    Non-specific binders show significant analyte retention on the
    reference surface after rinse.  This is measured as a
    low disssociation constant in the reference channel.
    If dissociation fit fails, non-specific binding is detected based 
    on a positive baseline-subtracted raw_reference signal 2-5 s into
    the dissociation phase (after RI bulk has been rinsed away).

    Parameters
    ----------
    sample : dict
        Sample cycle from load_cxw().
    koff_threshold : float
        Threshold for the dissociation constant of the reference channel.
        Default 1.25 s⁻¹.
     percent_threshold : float
        Threshold for the baseline-subtracted signal in the raw_reference channel 
        above which the sample is classified as a non-specific binder (in case 
        fitting fails).  
        Default 5.0% of the maximum abs(signal) value.

    Returns
    -------
    nsb : bool
        True if the koff_ref < koff_active OR koff_ref <= koff_threshold.
    koffs : tuple
        Fitted dissociation constants for the reference and active channels.
    OR
    nsb : bool
        True if ref_disso > ((percent_threshold / 100.0) * max_ref_signal).
    ref_disso : float
        Baseline-subtracted raw_reference signal 2-5 s into the dissociation phase, 
        used for assessment if fitting fails.
    """
    try:
        koff_ref, _, _, _ = fit_last_disso(sample, channel="raw_reference")
        koff_active, _, _, _ = fit_last_disso(sample, channel="raw_active")
        if koff_ref <= koff_threshold:
            return True, (koff_ref, koff_active)  # Reference channel shows some interactions
        return koff_ref < koff_active, (koff_ref, koff_active)
    except Exception as e: 
        print(f"WARNING: Could not fit last dissociation for sample {sample['index']} (channel {sample['channel']}). "
              f"Exception: {e}. Checking non-specific based on response in reference channel only.")
        t = sample['time']
        ref = sample['raw_reference']
        markers = sample['markers']
        bl_time = sample.get('baseline_duration_s', 45)
        rinse = markers.get('Rinse', t[-1])

        bl_mask = t <= bl_time
        ref_bl = ref[bl_mask].mean() if bl_mask.any() else ref[:int(bl_time)].mean()

        # 2-5 s after rinse: RI bulk gone, only true binding remains
        diss_mask = (t >= rinse + 2) & (t <= rinse + 5)
        if not diss_mask.any():
            return False, 0.0

        ref_disso = float((ref[diss_mask] - ref_bl).mean())
        max_ref_signal = np.max(np.abs(ref - ref_bl))
        return ref_disso > ((percent_threshold / 100.0) * max_ref_signal), ref_disso


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
    bl_time = dmso_cycle.get('baseline_duration_s', 45)
    inj_time = dmso_cycle['markers'].get('Injection', t[0])
    rinse_time = dmso_cycle['markers'].get('Rinse', t[-1])

    baseline = ref[t <= bl_time].mean()
    pulse_mask = (t >= inj_time) & (t <= rinse_time)
    peak = ref[pulse_mask].max() if pulse_mask.any() else baseline
    threshold = baseline + threshold_frac * (peak - baseline)

    is_buffer = np.ones(len(t), dtype=bool)
    # During injection window, mark analyte pulses
    is_buffer[pulse_mask & (ref > threshold)] = False
    return is_buffer


# ---------------------------------------------------------------------------
# Signal smoothing and differentiation
# ---------------------------------------------------------------------------

def smooth_and_differentiate(t: np.ndarray, R: np.ndarray,
                             smoothing_factor: float | None = None,
                             window_sec: float = 2.0, polyorder: int = 3):
    """Smooth binding signal and compute dR/dt using Savitzky-Golay filter.

    Unlike a global spline, SG filtering is local — it preserves pulse
    structure and sharp transitions in pulsed GCI data without imposing
    a sigmoidal shape.

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    R : np.ndarray
        Binding response array.
    smoothing_factor : float or None
        Ignored (kept for API compatibility with callers that pass it).
    window_sec : float
        Smoothing window width in seconds (default 2.0 s).
    polyorder : int
        Polynomial order for the SG filter (default 3).

    Returns
    -------
    R_smooth : np.ndarray
        Smoothed response.
    dRdt : np.ndarray
        Time derivative of the smoothed response.
    spline : None
        Placeholder for API compatibility (no spline object).
    """
    dt = np.median(np.diff(t))
    window_len = int(round(window_sec / dt))
    # Must be odd and >= polyorder + 2
    if window_len % 2 == 0:
        window_len += 1
    window_len = max(window_len, polyorder + 2)

    R_smooth = savgol_filter(R, window_length=window_len, polyorder=polyorder)
    # SG derivative: deriv=1 gives dR/dt when delta=dt
    dRdt = savgol_filter(R, window_length=window_len, polyorder=polyorder,
                         deriv=1, delta=dt)

    return R_smooth, dRdt, None


# ---------------------------------------------------------------------------
# Association pulses mask
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
                           dmso_cycle: dict, association_weight: float = 0.0) -> np.ndarray:
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

    if association_weight > 0:
        w[(w != 1) & inj_mask] = association_weight

    return w


def get_weight_from_derivative(sample, blank):
    t = sample['time']
    t_inj = sample['markers'].get("Injection")
    t_rinse = sample['markers'].get("Rinse")
    asso_mask = (t > t_inj) & (t <= t_rinse)
    signal, _ = double_reference(sample, blank)
    signal_scaled = (signal - signal.min()) / (signal.max() - signal.min())
    dRdt_asso = np.diff(signal_scaled[asso_mask]) / np.diff(t[asso_mask]) 
    return round(np.mean(np.abs(dRdt_asso)), 2) if dRdt_asso.any() else 0.0

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
# Last dissociation fit
# ---------------------------------------------------------------------------

def _disso_rate_equation(t, koff, R0, t0):
    return R0 * np.exp(-koff * (t-t0))
    

def fit_last_disso(sample: dict = {}, channel: str = "raw_active", blank: dict = None, debug=False):
    t = sample["time"]
    bl_time = sample.get("baseline_duration_s", 45)
    t_rinse = sample["markers"].get("Rinse")
    bl_mask = t <= bl_time
    disso_mask = t > t_rinse
    t = t[disso_mask]
    t0 = t[0]
    if channel != "signal":
        signal = sample[channel] - sample[channel][bl_mask].mean() if bl_mask.any() else sample[channel] - sample[channel][:int(bl_time)].mean()
    else:
        signal, _ = double_reference(sample, blank)
    signal = signal[disso_mask]
    R0 = signal[0]
    try:
        popt, pcov = curve_fit(lambda t, kon: _disso_rate_equation(t, kon, R0, t0), xdata=t, ydata=signal, p0=[1], bounds=(0, 10))
    except Exception as e:
        print(f"WARNING! Couldn't fit last dissociation of sample {sample['compound']} (cycle {sample['index']} - channel {sample['channel']}).\n"
              f"Error: {e}")
        if debug:
            return t, signal, np.nan, np.nan, np.array([]), np.nan, R0, t0
        else:
            return  np.nan, np.nan, np.array([]), np.nan
    perr = np.sqrt(np.diag(pcov))
    fit = _disso_rate_equation(t, popt[0], R0, t0)
    rmse = get_rmse(signal, fit)
    if debug:
        return t, signal, popt[0], perr[0], fit, rmse, R0, t0
    return popt[0], perr[0], fit, rmse


# ---------------------------------------------------------------------------
# Last dissociation fit
# ---------------------------------------------------------------------------

def _asso_rate_equation(t, kon, koff, C, Req, t0):
    return Req * (1 - np.exp(-(kon * C + koff) * (t-t0)))
    

def fit_last_asso(sample: dict = {}, channel: str = "signal", blank: dict = None, koff: float = 0.0, debug=False):
    from scipy.optimize import curve_fit
    t = sample["time"]
    bl_time = sample.get("baseline_duration_s", 45)
    bl_mask = t <= bl_time
    if channel != "signal":
        signal = sample[channel] - sample[channel][bl_mask].mean() if bl_mask.any() else sample[channel] - sample[channel][:int(bl_time)].mean()
    else:
        signal, _ = double_reference(sample, blank)
    C = sample['concentration_M']
    t_inj = sample['markers'].get('Injection')
    t0 = t_inj
    for t_pulse in sample['pulse_durations_s']:
        t1 = t0 + t_pulse
        if t_pulse == 3.0312500000000004:
            t0 = t1
            continue
        pulse_mask = (t >= t0) & (t <= t1)
        t0 = t1
    signal = signal[pulse_mask] # Get last asso. pulse only
    t = t[pulse_mask]
    t0 = t[0]
    Req = signal.max()
    try:
        if koff:
            popt, pcov = curve_fit(lambda t, kon: _asso_rate_equation(t, kon, koff, C, Req, t0), xdata=t, ydata=signal, p0=[1e3], bounds=(1e2, 1e9))
        else:
            popt, pcov = curve_fit(lambda t, kon, koff: _asso_rate_equation(t, kon, koff, C, Req, t0), xdata=t, ydata=signal, p0=[1e3, 1], bounds=([1e2, 0], [1e9, 10]))
            koff = popt[1]
    except Exception as e:
        print(f"WARNING! Couldn't fit association pulses of sample {sample['compound']} (cycle {sample['index']} - channel {sample['channel']}).\n"
              f"Error: {e}")
        koff = koff if koff else np.nan
        if debug:
            return t, signal, np.nan, np.nan, koff, np.nan, np.array([]), np.nan, C, Req, t0
        else:
            return np.nan, np.nan, koff, np.nan, np.array([]), np.nan
    perr = np.sqrt(np.diag(pcov))
    koff_err = perr[1] if len(perr) == 2 else np.nan
    fit = _asso_rate_equation(t, popt[0], koff, C, Req, t0)
    rmse = get_rmse(signal, fit)
    if debug:
        return t, signal, popt[0], perr[0], koff, koff_err, fit, rmse, C, Req, t0
    return popt[0], perr[0], koff, koff_err, fit, rmse

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
                        c_func, R0: float = 0.0,
                        fast: bool = True) -> np.ndarray:
    """Simulate a 1:1 Langmuir sensorgram.

    When ``fast=True``, each measured time interval is propagated as two
    half-intervals.  Concentration is evaluated at the midpoint of each
    half-interval and held constant there, so each update has an exact
    exponential solution.  This is a stable, higher-accuracy approximation
    for varying ``c(t)`` while avoiding the overhead of an adaptive solver
    inside every residual evaluation.  When ``fast=False``, the legacy
    adaptive RK45 integration path is used.

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
    fast : bool
        Select the two half-step exponential propagator (default) or the
        slower adaptive RK45 solver used by the original implementation.

    Returns
    -------
    R : np.ndarray
        Simulated binding response at each time point.
    """
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return np.empty_like(t)

    if not fast:
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

    R = np.empty_like(t)

    R[0] = R0
    if len(t) == 1:
        return R

    dt = np.diff(t)
    half_dt = 0.5 * dt
    # Two midpoint evaluations per measured interval retain the unconditional
    # stability of the exponential update, but resolve changes in the pulsed
    # concentration profile more accurately than one midpoint evaluation.
    c_eval_t = np.concatenate((
        t[:-1] + 0.25 * dt,
        t[:-1] + 0.75 * dt,
    ))
    c_eval = np.asarray(c_func(c_eval_t), dtype=float)
    if c_eval.ndim == 0:
        c_eval = np.full(c_eval_t.shape, float(c_eval))
    else:
        c_eval = np.broadcast_to(c_eval, c_eval_t.shape)
    c_first = c_eval[:len(dt)]
    c_second = c_eval[len(dt):]

    # Vectorise the interval coefficients.  Only the state recurrence itself
    # has to remain serial; evaluating rates, equilibria, and exponentials in
    # the Python loop would erase much of the fast-path benefit.
    rate_first = ka * c_first + kd
    rate_second = ka * c_second + kd
    decay_first = np.exp(-rate_first * half_dt)
    decay_second = np.exp(-rate_second * half_dt)
    equilibrium_first = np.divide(
        ka * c_first * Rmax,
        rate_first,
        out=np.zeros_like(rate_first),
        where=rate_first != 0,
    )
    equilibrium_second = np.divide(
        ka * c_second * Rmax,
        rate_second,
        out=np.zeros_like(rate_second),
        where=rate_second != 0,
    )

    for i in range(1, len(t)):
        j = i - 1
        r_half = (equilibrium_first[j]
                  + (R[i - 1] - equilibrium_first[j]) * decay_first[j])
        R[i] = (equilibrium_second[j]
                + (r_half - equilibrium_second[j]) * decay_second[j])

    return R
