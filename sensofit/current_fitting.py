"""Per-channel joint-reference fitting with pre-fit basin selection."""

from functools import partial
import numpy as np
from scipy.optimize import brentq, least_squares, lsq_linear
from .models import (
    build_pulsed_concentration_profile,
    double_reference,
    blank_correct_raw_channels,
    build_full_weight_mask,
    simulate_sensorgram_zoh as simulate_sensorgram,
    trim_to_fit_window,
    get_rmse,
    build_pulse_mask,
)
from .ode_fitting import _solve_R0_Rss
from .raw_review import calculate_raw_identifiability
from .affinity_prior import PREFIT_BASIN_PKD_BOUNDS, estimate_affinity_area_prior


def _prefit_basin_skip_reason(area_prior, raw_diagnostics, *, injection_issue=False,
                             fit_no_binding=False):
    """Skip no-binding traces, or raw unidentifiability with an injection issue."""
    if (not fit_no_binding and getattr(area_prior, "usable", False)
            and getattr(area_prior, "regime", None) == "no_binding"):
        return "prefit_no_binding"
    if not raw_diagnostics.get("raw_affinity_identifiable", True) and injection_issue:
        return "raw_unidentifiable_with_injection_issue"
    return ""


def prepare_sample(sample, blank, dmso, heuristics, capture_levels, ligand_mw,
                   prefit_thresholds=None, fit_no_binding=False, max_cost_ratio=1.1):
    """Supply current-method fit arguments or diagnostic fields for a skipped fit."""
    prior = estimate_affinity_area_prior(
        sample, prefit_thresholds=prefit_thresholds,
        end_guard_s=1.0, tail_endpoint_window_s=2.0)
    raw = calculate_raw_identifiability(sample, blank=blank, dmso=dmso)
    reason = _prefit_basin_skip_reason(
        prior, raw, injection_issue="injection_issue" in heuristics,
        fit_no_binding=fit_no_binding)
    kwargs = dict(area_prior=prior, raw_diagnostics=raw, ligand_mw_Da=ligand_mw,
                  max_cost_ratio=max_cost_ratio,
                  capture_level=capture_levels.get(str(sample.get("channel")).split("-")[0]))
    skipped = {}
    if reason:
        skipped.update(raw, **{f"affinity_area_{key}": value for key, value in prior.as_dict().items()})
        skipped.update(
            kinetic_fit_skipped=True, kinetic_fit_skip_reason=reason,
            kinetics_reportable=False, prefit_basin_selection_enabled=True,
            prefit_basin_attempted=False, prefit_basin_selected=False,
            prefit_basin_selection_reason=f"fit_skipped_{reason}",
            prefit_basin_cost_ratio=np.nan, prefit_basin_cost_ratio_max=max_cost_ratio,
            prefit_basin_bound_hit=False, prefit_basin_effective_n_starts=0,
            message=f"kinetic fit skipped: {reason}",
        )
    return kwargs, skipped


def _project_signal_nuisance(params, t, signal, c_func, w, *, fast=True,
                            reference_signal=None, reference_scale_bounds=(0.0, 2.0),
                            reference_scale_prior_fraction=0.0):
    """Project a bounded reference scale and constant offset for fixed kinetics.

    The scale prior is centered at one and scaled by reference information.
    The inner problem uses bounded linear least squares; the outer kinetic
    optimizer uses soft-L1 loss on the resulting signal residuals.
    """
    binding = simulate_sensorgram(t, *params, c_func, R0=0.0, fast=fast)
    signal = np.asarray(signal, dtype=float)
    reference = np.asarray(reference_signal, dtype=float)
    if reference.shape != signal.shape:
        raise ValueError("reference signal must match the active signal")
    residual_scale = np.asarray(w, dtype=float)
    valid = (residual_scale > 0) & np.isfinite(signal) & np.isfinite(binding)
    design = np.column_stack([reference, -np.ones_like(t)])
    coefficients = np.zeros(2)
    if valid.sum() >= 2:
        scaled_design = design[valid] * residual_scale[valid, None]
        rhs = (signal[valid] - binding[valid]) * residual_scale[valid]
        if reference_scale_prior_fraction > 0:
            information = float(np.sum(np.square(scaled_design[:, 0])))
            prior_scale = np.sqrt(reference_scale_prior_fraction * information)
            scaled_design = np.vstack([scaled_design, [prior_scale, 0.0]])
            rhs = np.append(rhs, prior_scale)
        coefficients = lsq_linear(
            scaled_design,
            rhs,
            bounds=(
                [reference_scale_bounds[0], -np.inf],
                [reference_scale_bounds[1], np.inf],
            ),
            method="trf",
        ).x
    return {
        "fit": binding + design @ coefficients,
        "binding": binding,
        "reference_scale": float(coefficients[0]),
        "reference_component": coefficients[0] * reference,
        "offset": float(coefficients[1]),
    }


def _residuals_projected_log(params, t, signal, c_func, w, fast=True,
                             reference_signal=None, reference_scale_bounds=(0.0, 2.0),
                             reference_scale_prior_fraction=0.0, pKD_coordinates=False):
    """Residuals with linear nuisance projection in log kinetic coordinates."""
    ka, kd, Rmax = _kinetic_parameters(params, pKD_coordinates)
    projection = _project_signal_nuisance(
        (ka, kd, Rmax), t, signal, c_func, w, fast=fast,
        reference_signal=reference_signal,
        reference_scale_bounds=reference_scale_bounds,
        reference_scale_prior_fraction=reference_scale_prior_fraction,
    )
    return w * (signal - projection["fit"])


def _log_parameters(params, pKD_coordinates=False):
    """Convert positive kinetics to independent log coordinates or log10(KD, kd, Rmax)."""
    params = np.asarray(params, dtype=float)
    if params.shape != (3,) or not np.isfinite(params).all() or np.any(params <= 0):
        raise ValueError("kinetic parameters must be positive and finite")
    if pKD_coordinates:
        params = params.copy()
        params[0] = params[1] / params[0]
    return np.log10(params)


def _kinetic_parameters(log_params, pKD_coordinates=False):
    """Decode one start or a collection of optimizer results into (ka, kd, Rmax)."""
    log_params = np.asarray(log_params, dtype=float)
    if pKD_coordinates:
        log_params = log_params.copy()
        log_params[..., 0] = log_params[..., 1] - log_params[..., 0]
    return np.power(10.0, log_params[..., :3])


def theoretical_rmax(capture_level, ligand_mw_Da, analyte_mw_Da, stoichiometry=1.0):
    """Calculate the mass-based theoretical maximum response."""
    try:
        values = tuple(map(float, (capture_level, ligand_mw_Da, analyte_mw_Da, stoichiometry)))
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            "capture level, ligand MW, analyte MW and stoichiometry must all be positive"
        ) from None
    capture_level, ligand_mw_Da, analyte_mw_Da, stoichiometry = values
    return capture_level * analyte_mw_Da / ligand_mw_Da * stoichiometry


def _estimate_early_dissociation_seed(t, signal, markers, skip_s=1.0, window_s=20.0):
    """Fit ``A exp(-kd t) + offset`` to a short dissociation segment."""
    rinse = markers.get("Rinse", t[0])
    rinse_end = markers.get("RinseEnd", t[-1])
    start = rinse + skip_s
    end = min(start + window_s, rinse_end)
    mask = (t >= start) & (t <= end) & np.isfinite(signal)
    td = np.asarray(t[mask], dtype=float)
    yd = np.asarray(signal[mask], dtype=float)
    if td.size < 8:
        raise ValueError("insufficient early dissociation data")
    elapsed = td - td[0]
    tail = float(np.median(yd[-max(3, yd.size // 5) :]))
    amplitude = max(float(np.median(yd[: max(3, yd.size // 10)])) - tail, 0.001)
    positive = yd - tail
    valid = positive > max(1e-06, 0.01 * amplitude)
    if valid.sum() >= 4:
        slope = np.polyfit(elapsed[valid], np.log(positive[valid]), 1)[0]
        kd0 = float(np.clip(-slope, 1e-05, 2.0))
    else:
        kd0 = 0.01
    noise = np.median(np.abs(yd - np.median(yd))) * 1.4826

    def residuals(params):
        amp, kd, offset = params
        return yd - (amp * np.exp(-kd * elapsed) + offset)

    opt = least_squares(
        residuals,
        x0=np.array([amplitude, kd0, tail]),
        bounds=(np.array([0.0, 1e-06, -np.inf]), np.array([np.inf, 10.0, np.inf])),
        loss="soft_l1",
        f_scale=max(float(noise), 0.05),
        max_nfev=500,
    )
    if not opt.success or not np.isfinite(opt.x).all():
        raise ValueError("early dissociation exponential fit failed")
    return float(opt.x[1]), float(opt.x[2])


def _nearest_boolean_transfer(source_t, source_values, target_t):
    """Transfer a boolean time series to another grid by nearest neighbour."""
    source_t = np.asarray(source_t, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 0, source_t.size - 1)
    left = np.clip(right - 1, 0, source_t.size - 1)
    choose_left = np.abs(target_t - source_t[left]) <= np.abs(
        target_t - source_t[right]
    )
    indices = np.where(choose_left, left, right)
    return np.asarray(source_values, dtype=bool)[indices]


def _analyte_pulse_intervals(t, markers, dmso):
    """Return analyte-on intervals observed at the sensor surface."""
    is_buffer_dmso = build_pulse_mask(dmso)
    is_buffer = _nearest_boolean_transfer(dmso["time"], is_buffer_dmso, t)
    injection = markers.get("Injection", t[0])
    rinse = markers.get("Rinse", t[-1])
    analyte = (t >= injection) & (t < rinse) & ~is_buffer
    indices = np.flatnonzero(analyte)
    if not indices.size:
        return ([], is_buffer)
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    dt = float(np.median(np.diff(t)))
    intervals = []
    for run in runs:
        start = float(t[run[0]])
        end = float(t[run[-1]] + dt)
        if end - start >= max(0.1, 0.5 * dt):
            intervals.append((start, end))
    return (intervals, is_buffer)


def _window_median(t, signal, is_buffer, start, end):
    mask = (t >= start) & (t <= end) & is_buffer & np.isfinite(signal)
    if mask.sum() < 3:
        return (np.nan, np.nan)
    return (float(np.median(signal[mask])), float(np.median(t[mask])))


def _pulse_transition_KD_estimates(t, signal, markers, dmso, concentration, kd, Rmax):
    """Estimate KD analytically from clean responses around analyte pulses."""
    if concentration <= 0 or kd <= 0 or Rmax <= 0:
        return []
    intervals, is_buffer = _analyte_pulse_intervals(t, markers, dmso)
    estimates = []
    kobs_low = kd * (1.0 + 1e-08)
    kobs_high = kd + 100000000.0 * concentration
    if kobs_high <= kobs_low:
        return estimates
    for start, end in intervals:
        before, before_t = _window_median(
            t, signal, is_buffer, start - 0.9, start - 0.2
        )
        after, after_t = _window_median(t, signal, is_buffer, end + 0.25, end + 0.95)
        if not np.isfinite([before, before_t, after, after_t]).all():
            continue
        before_boundary = before * np.exp(-kd * max(start - before_t, 0.0))
        after_boundary = after * np.exp(kd * max(after_t - end, 0.0))
        if (
            after_boundary <= before_boundary
            or before_boundary < 0
            or after_boundary >= 0.995 * Rmax
        ):
            continue
        duration = end - start

        def equation(kobs):
            equilibrium = Rmax * (1.0 - kd / kobs)
            predicted = equilibrium + (before_boundary - equilibrium) * np.exp(
                -kobs * duration
            )
            return predicted - after_boundary

        low_value = equation(kobs_low)
        high_value = equation(kobs_high)
        if not np.isfinite([low_value, high_value]).all():
            continue
        if low_value == 0:
            kobs = kobs_low
        elif low_value * high_value >= 0:
            continue
        else:
            try:
                kobs = brentq(equation, kobs_low, kobs_high, maxiter=100)
            except ValueError:
                continue
        ka = (kobs - kd) / concentration
        KD = kd / ka if ka > 0 else np.nan
        if np.isfinite(KD) and KD > 0:
            estimates.append(float(KD))
    return estimates


def _occupancy_KD_seed(t, signal, markers, concentration, kd, Rmax):
    """Fallback KD seed from response near the start of dissociation."""
    rinse = markers.get("Rinse", t[-1])
    mask = (t >= rinse + 1.0) & (t <= rinse + 2.0)
    if mask.sum() < 3:
        raise ValueError("insufficient rinse response for occupancy seed")
    response_time = float(np.median(t[mask]))
    response = float(np.median(signal[mask]))
    response_at_rinse = response * np.exp(kd * (response_time - rinse))
    fraction = float(np.clip(response_at_rinse / Rmax, 0.01, 0.95))
    return concentration * (1.0 - fraction) / fraction


def physical_pulse_seed_candidates(sample, signal, dmso, capture_level, ligand_mw_Da,
                                   rmax_factors=(1.0, 0.5, 2.0), rmax_bounds=(0.2, 5.0)):
    """Seed kinetics from pulse transitions, falling back to rinse occupancy."""
    t = np.asarray(sample["time"], dtype=float)
    concentration = float(sample["concentration_M"])
    analyte_mw_Da = float(sample.get("mw", np.nan))
    Rmax_theory = theoretical_rmax(capture_level, ligand_mw_Da, analyte_mw_Da)
    lower = float(rmax_bounds[0] * Rmax_theory)
    upper = float(rmax_bounds[1] * Rmax_theory)
    if lower <= 0 or upper <= lower:
        raise ValueError("invalid physical Rmax bounds")
    kd, offset = _estimate_early_dissociation_seed(t, signal, sample["markers"])
    binding_signal = np.asarray(signal, dtype=float) - offset
    _, is_buffer = _analyte_pulse_intervals(t, sample["markers"], dmso)
    injection = sample["markers"].get("Injection", t[0])
    rinse_end = sample["markers"].get("RinseEnd", t[-1])
    clean = (
        (t >= injection) & (t <= rinse_end) & is_buffer & np.isfinite(binding_signal)
    )
    clean_peak = float(np.nanmax(binding_signal[clean])) if clean.any() else 0.0
    candidate_floor = min(max(lower, 1.05 * max(clean_peak, 0.0)), upper)
    rmax_candidates = _distinct_values(
        np.clip(factor * Rmax_theory, candidate_floor, upper) for factor in rmax_factors)
    if len(rmax_candidates) < len(rmax_factors) and candidate_floor < upper:
        rmax_candidates = _distinct_values([
            *rmax_candidates, *np.geomspace(candidate_floor, upper, len(rmax_factors)),
        ])[:len(rmax_factors)]
    starts = []
    pulse_counts = []
    seed_sources = []
    for Rmax in rmax_candidates:
        pulse_KDs = _pulse_transition_KD_estimates(
            t, binding_signal, sample["markers"], dmso, concentration, kd, Rmax
        )
        if pulse_KDs:
            KD = float(np.exp(np.median(np.log(pulse_KDs))))
            source = "pulse_transition"
        else:
            KD = _occupancy_KD_seed(
                t, binding_signal, sample["markers"], concentration, kd, Rmax
            )
            source = "rinse_occupancy"
        ka = float(np.clip(kd / KD, 0.1, 100000000.0))
        starts.append(np.array([ka, kd, Rmax], dtype=float))
        pulse_counts.append(len(pulse_KDs))
        seed_sources.append(source)
    if not starts:
        raise ValueError("could not construct physical seed candidates")
    return {
        "starts": starts, "Rmax_theory": Rmax_theory, "Rmax_lower": lower, "Rmax_upper": upper,
        "capture_level": float(capture_level), "ligand_mw_Da": float(ligand_mw_Da),
        "analyte_mw_Da": analyte_mw_Da, "kd": kd, "offset": offset,
        "pulse_counts": pulse_counts, "seed_sources": seed_sources,
    }


def _distinct_values(values):
    """Keep candidate order while collapsing numerically equivalent seeds."""
    distinct = []
    for value in values:
        if not any(np.isclose(value, previous) for previous in distinct):
            distinct.append(float(value))
    return distinct


def prefit_basin_seed_candidates(physical_seed, regime):
    """Cross three feasible pKD starts with three physical capacities."""
    bounds = PREFIT_BASIN_PKD_BOUNDS.get(str(regime))
    if bounds is None:
        return {"starts": [], "pKD_bounds": None, "pKD_values": [], "Rmax_values": []}
    kd = float(physical_seed["kd"])
    if not np.isfinite(kd) or kd <= 0:
        raise ValueError("physical seed must contain a positive kd")
    lower, upper = map(float, bounds)
    nominal = ([lower, lower + 2.0, lower + 4.0] if str(regime) == "tight"
               else [lower, 0.5 * (lower + upper), upper])
    feasible_lower = max(lower, float(np.log10(0.1 / kd)))
    feasible_upper = min(nominal[-1], float(np.log10(1e8 / kd)))
    if feasible_upper <= feasible_lower:
        raise ValueError("prefit pKD seeds do not overlap feasible ka bounds at seeded kd")
    pKD_values = [float(np.clip(p, feasible_lower, feasible_upper)) for p in nominal]
    if len(_distinct_values(pKD_values)) != 3:
        pKD_values = np.linspace(feasible_lower, feasible_upper, 3).tolist()
    rmax_values = _distinct_values(
        physical_seed[key] for key in ("Rmax_lower", "Rmax_theory", "Rmax_upper"))
    if len(rmax_values) != 3:
        raise ValueError("prefit basin requires three distinct physical Rmax seeds")
    starts = [np.array([float(kd * 10.0**pKD), kd, rmax])
              for pKD in pKD_values for rmax in rmax_values]
    return {"starts": starts, "pKD_bounds": (lower, upper),
            "pKD_values": pKD_values, "Rmax_values": rmax_values}


def _validate_max_cost_ratio(max_cost_ratio):
    max_cost_ratio = float(max_cost_ratio)
    if not np.isfinite(max_cost_ratio) or max_cost_ratio < 1.0:
        raise ValueError("prefit basin cost ratio must be at least 1")
    return max_cost_ratio


def select_prefit_basin_fit(unrestricted, constrained, *, max_cost_ratio=1.1):
    """Select a score-compatible basin only when the data cost supports it."""
    max_cost_ratio = _validate_max_cost_ratio(max_cost_ratio)

    def fit_cost(result):
        cost = float(result.get("cost", np.nan))
        return cost, bool(result.get("success", False)) and np.isfinite(cost) and cost >= 0

    unrestricted_cost, unrestricted_valid = fit_cost(unrestricted)
    constrained_cost, constrained_valid = fit_cost(constrained)
    if unrestricted_valid and unrestricted_cost == 0:
        ratio = 1.0 if constrained_valid and constrained_cost == 0 else np.inf
    elif unrestricted_valid and constrained_valid:
        ratio = constrained_cost / unrestricted_cost
    else:
        ratio = np.nan
    selected_basin = constrained_valid and unrestricted_valid and ratio <= max_cost_ratio
    if selected_basin:
        reason = "prefit_basin_within_cost_tolerance"
    elif not unrestricted_valid:
        reason = "unrestricted_failed_no_cost_comparison"
    elif not constrained_valid:
        reason = "constrained_failed"
    else:
        reason = "prefit_basin_exceeds_cost_tolerance"
    return dict(constrained if selected_basin else unrestricted), {
        "prefit_basin_selected": bool(selected_basin), "prefit_basin_selection_reason": reason,
        "prefit_basin_cost_ratio": float(ratio),
        "prefit_basin_unrestricted_cost": unrestricted_cost,
        "prefit_basin_constrained_cost": constrained_cost,
    }


def ode_fit(t, signal, c_func, w, markers, ka0, kd0, Rmax0, n_starts=1,
            rng_seed=None, skip_s=1.0, fast=True, initial_starts=None, Rmax_bounds=None,
            reference_signal=None, reference_scale_bounds=(0.0, 2.0),
            reference_scale_prior_fraction=1.0, pKD_bounds=None):
    """Fit log kinetic parameters and return the lowest-cost converged start.

    With pKD bounds, optimize (log10 KD, log10 kd, log10 Rmax) and
    convert to (log10 ka, log10 kd, log10 Rmax) inside the residual.
    Otherwise use independent log kinetic coordinates. The reference scale
    and offset are projected at every evaluation. Standard errors are local
    Jacobian approximations; physical and pre-fit boundaries are reported."""
    if pKD_bounds is not None:
        pKD_lower, pKD_upper = map(float, pKD_bounds)
        if not np.isfinite([pKD_lower, pKD_upper]).all() or pKD_lower >= pKD_upper:
            raise ValueError("invalid pKD bounds")
    else:
        pKD_lower, pKD_upper = (np.nan, np.nan)
    pKD_constrained_kinetic = pKD_bounds is not None
    if reference_signal is None:
        raise ValueError("joint-reference fitting requires a reference signal")
    reference_signal = np.asarray(reference_signal, dtype=float)
    if reference_signal.shape != np.asarray(signal).shape:
        raise ValueError("reference signal must match the active signal shape")
    try:
        reference_scale_bounds = tuple(map(float, reference_scale_bounds))
    except (TypeError, ValueError):
        raise ValueError("reference scale bounds must contain two numbers")
    if (len(reference_scale_bounds) != 2 or not np.isfinite(reference_scale_bounds).all()
            or reference_scale_bounds[0] >= reference_scale_bounds[1]):
        raise ValueError("invalid reference scale bounds")
    resolved_loss = "soft_l1"
    kd_final = max(kd0, 1e-05)
    w = np.asarray(w, dtype=float)
    rinse = markers.get("Rinse", 0)
    rinse_end = markers.get("RinseEnd", t[-1])
    t0 = rinse + skip_s
    dissoc_mask = (t >= rinse) & (t <= rinse_end) & (t >= t0)
    R0_est, Rss_est = _solve_R0_Rss(kd_final, t[dissoc_mask], signal[dissoc_mask], t0)
    R0_est = max(R0_est, 1.0)
    c_plateau = float(c_func(rinse - 2.0))
    Rmax_est = max(Rmax0, R0_est * 1.2)
    if c_plateau > 0 and Rmax_est > R0_est:
        ka_est = kd_final * R0_est / (c_plateau * (Rmax_est - R0_est))
    else:
        ka_est = max(ka0, 1.0)
    if Rmax_bounds is None:
        rmax_lower, rmax_upper = (1.0, 10000.0)
    else:
        rmax_lower, rmax_upper = map(float, Rmax_bounds)
        if (not np.isfinite([rmax_lower, rmax_upper]).all()
                or rmax_lower <= 0 or rmax_upper <= rmax_lower):
            raise ValueError("invalid Rmax bounds")
    lb_full = np.array([0.1, 1e-06, rmax_lower])
    ub_full = np.array([100000000.0, 10.0, rmax_upper])
    Rmax_est = float(np.clip(Rmax_est, rmax_lower, rmax_upper))
    rng = np.random.default_rng(rng_seed)
    if initial_starts is not None:
        starts = [
            np.clip(np.asarray(start, dtype=float), lb_full, ub_full)
            for start in initial_starts
            if np.asarray(start).shape == (3,) and np.isfinite(start).all()
        ][: max(n_starts, 1)]
        if not starts:
            raise ValueError("no valid explicit ODE starts")
        while len(starts) < max(n_starts, 1):
            perturb = rng.normal(0, 0.35, size=3)
            starts.append(np.clip(starts[0] * np.exp(perturb), lb_full, ub_full))
    else:
        starts = [np.clip([ka_est, kd_final, Rmax_est], lb_full, ub_full)]
        starts.append(np.clip([max(ka0, 1.0), max(kd0, 1e-05), max(Rmax0, rmax_lower)],
                              lb_full, ub_full))
        for _ in range(max(n_starts - 2, 0)):
            log_perturb = rng.normal(0, 0.5, size=3)
            p = np.array([ka_est, kd_final, Rmax_est]) * np.exp(log_perturb)
            starts.append(np.clip(p, lb_full, ub_full))
    lb_optim, ub_optim = np.log10(lb_full), np.log10(ub_full)
    if pKD_constrained_kinetic:
        lb_optim[0], ub_optim[0] = -pKD_upper, -pKD_lower
    optim_starts = [np.clip(_log_parameters(start, pKD_constrained_kinetic), lb_optim, ub_optim)
                    for start in starts]

    residual_function = partial(
        _residuals_projected_log, reference_signal=reference_signal,
        reference_scale_bounds=reference_scale_bounds,
        reference_scale_prior_fraction=reference_scale_prior_fraction,
        pKD_coordinates=pKD_constrained_kinetic)

    fits = []
    for p0 in optim_starts:
        try:
            opt = least_squares(
                residual_function,
                p0,
                args=(t, signal, c_func, w, fast),
                bounds=(lb_optim, ub_optim),
                method="trf",
                ftol=1e-06,
                xtol=1e-06,
                gtol=1e-06,
                max_nfev=200,
                diff_step=0.01,
                loss=resolved_loss,
            )
            if opt.success:
                fits.append((opt.x, opt.cost, opt.jac, opt.nfev))
        except Exception:
            pass
    metadata = {
        "R0": R0_est, "Rss": Rss_est, "n_starts": n_starts, "fast": fast,
        "dissociation_weighting": "uniform", "dissociation_weight_half_life_s": None,
        "Rmax_lower_bound": rmax_lower, "Rmax_upper_bound": rmax_upper,
        "parameterization": "log10_ka_kd_Rmax", "fit_weight": w, "objective_weight": w,
        "optimizer_loss": resolved_loss, "pKD_lower_bound": pKD_lower,
        "pKD_upper_bound": pKD_upper, "dissociation_start_offset_s": float(skip_s),
        "dissociation_end_s": float(rinse_end - rinse),
    }
    if not fits:
        R_fit = simulate_sensorgram(t, ka_est, kd_final, Rmax_est, c_func, R0=0.0, fast=fast)
        return {
            **metadata, **dict.fromkeys(("ka_se", "kd_se", "Rmax_se", "KD_se", "ka_iqr",
                                        "kd_iqr", "Rmax_iqr", "jac_condition", "rmse",
                                        "sigma_residual", "sqrt_chi2", "objective_sqrt_chi2",
                                        "cost"), np.nan),
            "ka": ka_est, "kd": kd_final, "Rmax": Rmax_est, "KD": kd_final / ka_est,
            "R_fit": R_fit, "residuals": w * (signal - R_fit),
            "n_parameters": 3, "cov": np.full((3, 3), np.nan),
            "n_points": int((w > 0).sum()), "nfev": 0, "n_converged": 0,
            "success": False, "message": "All ODE fits failed",
        }
    all_optim_params = np.array([fit[0] for fit in fits])
    best_idx = int(np.argmin([fit[1] for fit in fits]))
    all_params = _kinetic_parameters(all_optim_params, pKD_constrained_kinetic)
    ka_final_val, kd_final_val, Rmax_final = map(float, all_params[best_idx])
    total_nfev = sum((f[3] for f in fits))
    iqr_ka, iqr_kd, iqr_Rmax = np.subtract(
        *np.percentile(all_params, [75, 25], axis=0)
    )
    KD = kd_final_val / ka_final_val
    best_jac = fits[best_idx][2]
    try:
        jac_condition = float(np.linalg.cond(best_jac))
    except (np.linalg.LinAlgError, ValueError, TypeError):
        jac_condition = np.nan
    params = [ka_final_val, kd_final_val, Rmax_final]
    projection = _project_signal_nuisance(
        params, t, signal, c_func, w, fast=fast,
        reference_signal=reference_signal,
        reference_scale_bounds=reference_scale_bounds,
        reference_scale_prior_fraction=reference_scale_prior_fraction,
    )
    R_fit = projection["fit"]
    residuals = w * (signal - R_fit)
    n = int((w > 0).sum())
    n_parameters = 5
    dof = max(n - n_parameters, 1)
    sigma2 = np.sum(residuals**2) / dof
    ka_se, kd_se, Rmax_se, KD_se = (np.nan, np.nan, np.nan, np.nan)
    cov = np.full((3, 3), np.nan)
    try:
        cov_optim = sigma2 * np.linalg.inv(best_jac.T @ best_jac)
        cov_log = cov_optim[:3, :3]
        scale = np.log(10.0) * np.array(params)
        transform = np.diag(scale)
        if pKD_constrained_kinetic:
            transform[0, :2] = [-scale[0], scale[0]]
        cov = transform @ cov_log @ transform.T
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        ka_se, kd_se, Rmax_se = se
        kd_gradient = np.array(
            [-kd_final_val / ka_final_val**2, 1.0 / ka_final_val, 0.0]
        )
        KD_variance = float(kd_gradient @ cov @ kd_gradient)
        if np.isfinite(KD_variance) and KD_variance >= 0:
            KD_se = float(np.sqrt(KD_variance))
    except np.linalg.LinAlgError:
        pass
    fit_mask = np.isfinite(R_fit)
    rmse = get_rmse(signal[fit_mask], R_fit[fit_mask])
    valid = (w > 0) & np.isfinite(R_fit)
    binding_amplitude = float(np.ptp(projection["binding"][valid])) if valid.any() else np.nan
    bound_tolerance = 0.0001
    params_array = np.asarray(params)
    kinetic_bound_hits = int(np.sum(
        (np.abs(params_array - lb_full) <= bound_tolerance * np.maximum(lb_full, 1.0))
        | (np.abs(params_array - ub_full) <= bound_tolerance * np.maximum(ub_full, 1.0))))
    reference_scale = projection["reference_scale"]
    ref_lower, ref_upper = reference_scale_bounds
    reference_bound_hit = bool(
        reference_scale <= ref_lower + bound_tolerance * max(abs(ref_lower), 1.0)
        or reference_scale >= ref_upper - bound_tolerance * max(abs(ref_upper), 1.0))
    return {
        **metadata,
        "ka": ka_final_val, "kd": kd_final_val, "Rmax": Rmax_final, "KD": KD,
        "R_fit": R_fit, "rmse": rmse, "residuals": residuals,
        "ka_se": ka_se, "kd_se": kd_se, "Rmax_se": Rmax_se, "KD_se": KD_se,
        "ka_iqr": iqr_ka, "kd_iqr": iqr_kd, "Rmax_iqr": iqr_Rmax,
        "jac_condition": jac_condition, "n_parameters": n_parameters, "cov": cov,
        "sigma_residual": np.sqrt(sigma2), "sqrt_chi2": np.sqrt(sigma2),
        "objective_sqrt_chi2": np.sqrt(sigma2), "cost": float(np.sum(residuals**2)),
        "n_points": n, "n_converged": len(fits), "nfev": total_nfev,
        "success": True, "message": f"{len(fits)}/{n_starts} ODE starts converged",
        "aggregation": "best", "variable_nuisance": False,
        "projected_nuisance": "joint_reference_offset", "offset": projection["offset"],
        "drift": 0.0, "binding_amplitude": binding_amplitude,
        "reference_scale": reference_scale, "reference_scale_bound_hit": reference_bound_hit,
        "reference_scale_bounds": tuple(reference_scale_bounds),
        "reference_scale_prior_fraction": float(reference_scale_prior_fraction),
        "kinetic_bound_hits": kinetic_bound_hits,
        "binding_fit": projection["binding"], "offset_fit": -projection["offset"] * np.ones_like(t),
        "drift_fit": np.zeros_like(t), "reference_component_fit": projection["reference_component"],
    }


def fit_sample(sample, dmso, blank=None, initial_estimates="PHYSICAL", n_starts=1,
               fast=True, rng_seed=None, capture_level=None, ligand_mw_Da=None,
               reference_scale_bounds=(0.0, 2.0), area_prior=None, raw_diagnostics=None,
               prefit_thresholds=None, max_cost_ratio=1.1):
    """Fit unrestricted and pre-fit basins, selecting by data cost."""
    if initial_estimates != "PHYSICAL":
        raise ValueError("prefit basin selection requires PHYSICAL initial estimates")
    max_cost_ratio = _validate_max_cost_ratio(max_cost_ratio)
    if area_prior is None:
        area_prior = estimate_affinity_area_prior(
            sample, prefit_thresholds=prefit_thresholds,
            end_guard_s=1.0, tail_endpoint_window_s=2.0)
    t = sample["time"]
    signal, blank_index = double_reference(sample, blank)
    physical_seed = prefit_basin = None
    Rmax_theory_estimated = False
    try:
        Rmax_theory = theoretical_rmax(capture_level, ligand_mw_Da, float(sample.get("mw", np.nan)))
        Rmax_theory_estimated = True if Rmax_theory > 0 else False
    except Exception:
        print(f'WARNING! Could not estimate theoretical Rmax for sample {sample["index"]} (RK serie {sample.get("rk_serie_id", "")}, '
              f'channel {sample.get("channel", "")}). Using fallback seeds instead...')
        Rmax_theory = np.nan
    fallback = float(sample.get("concentration_M", 0.0)) <= 0 or not Rmax_theory_estimated
    if fallback:
        seed_method = "control_defaults"
        ka_seed, kd_seed, Rmax_seed = 1e3, 1e-3, 10.0
    else:
        physical_seed = physical_pulse_seed_candidates(
            sample, signal, dmso, capture_level, ligand_mw_Da,
            rmax_factors=(0.2, 1.0, 5.0), rmax_bounds=(0.2, 5.0))
        ka_seed, kd_seed, Rmax_seed = physical_seed["starts"][0]
        seed_method = "physical_pulse_transition"
        if area_prior.usable:
            prefit_basin = prefit_basin_seed_candidates(physical_seed, area_prior.regime)
    KD_seed = kd_seed / ka_seed

    signal, reference_signal = blank_correct_raw_channels(sample, blank)
    c_func, _ = build_pulsed_concentration_profile(dmso, sample["concentration_M"])
    w = build_full_weight_mask(
        t, sample["markers"], dmso, association_weight=0.0, transition_window_s=0.5)
    t_fit, sig_fit, w_fit, fit_mask = trim_to_fit_window(t, signal, w, sample["markers"])
    effective_n_starts = max(int(n_starts), 9)
    options = dict(
        ka0=ka_seed, kd0=kd_seed, Rmax0=Rmax_seed, n_starts=effective_n_starts,
        rng_seed=rng_seed, fast=fast, reference_signal=reference_signal[fit_mask],
        reference_scale_bounds=reference_scale_bounds,
        Rmax_bounds=(physical_seed["Rmax_lower"], physical_seed["Rmax_upper"])
        if physical_seed else None,
    )
    unrestricted = ode_fit(
        t_fit, sig_fit, c_func, w_fit, sample["markers"], **options,
        initial_starts=physical_seed["starts"] if physical_seed else None)
    basin_fields = dict(
        prefit_basin_attempted=False, prefit_basin_selected=False,
        prefit_basin_selection_reason=("prefit_unusable" if not area_prior.usable else
                                       "prefit_no_binding" if getattr(area_prior, "regime", None) == "no_binding"
                                       else "disabled"),
        prefit_basin_cost_ratio=np.nan, prefit_basin_requested_bounds=None,
        prefit_basin_bound_hit=False, prefit_basin_selection_enabled=True,
        prefit_basin_cost_ratio_max=max_cost_ratio, prefit_basin_effective_n_starts=effective_n_starts,
    )
    ode = unrestricted
    if prefit_basin and prefit_basin["starts"]:
        constrained = ode_fit(
            t_fit, sig_fit, c_func, w_fit, sample["markers"], **options,
            initial_starts=prefit_basin["starts"], pKD_bounds=prefit_basin["pKD_bounds"])
        ode, selection = select_prefit_basin_fit(
            unrestricted, constrained, max_cost_ratio=max_cost_ratio)
        basin_fields.update(selection, prefit_basin_attempted=True,
                            prefit_basin_requested_bounds=prefit_basin["pKD_bounds"])
        for output, key in (("pKDs", "pKD_values"), ("Rmax_values", "Rmax_values")):
            basin_fields[f"prefit_basin_start_{output}"] = ",".join(
                f"{value:.6g}" for value in prefit_basin[key])
        for prefix, candidate in (("unrestricted", unrestricted), ("constrained", constrained)):
            for key in ("ka", "kd", "KD", "Rmax", "cost", "sqrt_chi2",
                        "objective_sqrt_chi2", "success"):
                basin_fields[f"prefit_basin_{prefix}_{key}"] = candidate.get(key, np.nan)
        pKD = -np.log10(float(ode["KD"])) if np.isfinite(ode["KD"]) and ode["KD"] > 0 else np.nan
        lower, upper = prefit_basin["pKD_bounds"]
        basin_fields["prefit_basin_bound_hit"] = bool(
            np.isfinite(pKD) and basin_fields["prefit_basin_selected"]
            and (pKD <= lower + 0.001 or pKD >= upper - 0.001))
    ode.update(basin_fields)

    # Expand every fitted component once, preserving the residual's zero padding.
    for key in ("R_fit", "residuals", "fit_weight", "objective_weight", "binding_fit",
                "offset_fit", "drift_fit", "reference_component_fit"):
        if key in ode:
            values = np.full_like(signal, 0.0 if key == "residuals" else np.nan)
            values[fit_mask] = ode[key]
            ode[key] = values
    if raw_diagnostics is None:
        raw_diagnostics = calculate_raw_identifiability(sample, blank=blank, dmso=dmso)
    ode.update(raw_diagnostics)
    ode.update({f"affinity_area_{key}": value for key, value in area_prior.as_dict().items()})
    ode.update(
        kinetics_reportable=True, c_func=c_func, concentration_profile=np.asarray(c_func(t), dtype=float),
        concentration_profile_mode="dmso_nearest", seed_method=seed_method,
        ka_seed=ka_seed, kd_seed=kd_seed, Rmax_seed=Rmax_seed, KD_seed=KD_seed,
        analyte_pulse_weight=0.0, transition_window_s=0.5, post_rinse_exclusion_s=0.0,
        dissociation_end_s=ode.get("dissociation_end_s", np.nan), reference_scale_fixed=np.nan,
        physical_seed_fallback_reason="non_positive_concentration" if fallback else None,
        t=t, signal=signal, reference_signal=reference_signal, joint_channel_projection=True,
        dmso_index=dmso["index"] if dmso else None, blank_index=blank_index, fast=fast,
        aggregation="best", variable_nuisance=False, optimizer_loss="auto",
        requested_projected_nuisance="joint_reference_offset",
        projected_nuisance=ode.get("projected_nuisance", "joint_reference_offset"),
    )
    if physical_seed:
        ode.update({key: physical_seed[key] for key in
                    ("Rmax_theory", "capture_level", "ligand_mw_Da", "analyte_mw_Da")})
        ode.update(
            physical_seed_offset=physical_seed["offset"],
            physical_seed_pulse_count=int(max(physical_seed["pulse_counts"], default=0)),
            physical_seed_sources=",".join(physical_seed["seed_sources"]),
            physical_seed_KDs=",".join(f"{start[1] / start[0]:.6g}" for start in physical_seed["starts"]),
            physical_seed_Rmax_values=",".join(f"{start[2]:.6g}" for start in physical_seed["starts"]),
        )
    ode.update(_early_dissociation_diagnostics(t, signal, ode["R_fit"], sample["markers"]))
    return ode


def _early_dissociation_diagnostics(t, signal, fitted_signal, markers,
                                   start_offset_s=1.0, window_s=5.0,
                                   minimum_residual_drop=0.2, maximum_slope_ratio=0.75):
    """Detect a fitted dissociation that is visibly flatter than the data."""
    t, signal, fitted_signal = [np.asarray(values, dtype=float)
                               for values in (t, signal, fitted_signal)]
    rinse = float(markers.get("Rinse", np.nan))
    rinse_end = float(markers.get("RinseEnd", np.nan))
    diagnostics = dict(data_slope=np.nan, model_slope=np.nan, slope_ratio=np.nan,
                       residual_drop=np.nan, mismatch=False)
    end = min(rinse + start_offset_s + window_s, rinse_end)
    mask = ((t >= rinse + start_offset_s) & (t <= end)
            & np.isfinite(signal) & np.isfinite(fitted_signal))
    if (np.isfinite([rinse, rinse_end]).all()
            and np.count_nonzero(mask) >= 5 and np.ptp(t[mask]) > 0):
        tw = t[mask]
        data_slope, model_slope, residual_slope = [
            float(np.polyfit(tw, values[mask], 1)[0])
            for values in (signal, fitted_signal, signal - fitted_signal)
        ]
        ratio = abs(model_slope) / abs(data_slope) if data_slope < 0 else np.nan
        residual_drop = max(0.0, -residual_slope * float(np.ptp(tw)))
        mismatch = bool(data_slope < 0 and np.isfinite(ratio)
                        and ratio < maximum_slope_ratio and residual_slope < 0
                        and residual_drop >= minimum_residual_drop)
        diagnostics.update(data_slope=data_slope, model_slope=model_slope,
                           slope_ratio=ratio, residual_drop=residual_drop, mismatch=mismatch)
    return {f"early_dissociation_{key}": value for key, value in diagnostics.items()}
