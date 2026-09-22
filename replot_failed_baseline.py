from pathlib import Path

import numpy as np
import pandas as pd

from sensofit.package_loader import load_experiment
from sensofit.direct_kinetics import fit_sample as dk_fit_sample
from sensofit.models import (
    build_pulsed_concentration_profile,
    select_dmso_cal,
    simulate_sensorgram,
    trim_to_fit_window,
)
from sensofit.plotting import save_fit_plots


BASELINE_DIR = Path(
    "runs/pre_characterisation_baseline_20260713_213302"
)
DATA_DIR = Path("data")

baseline = pd.read_csv(BASELINE_DIR / "all_results.csv")
failed = baseline[baseline["success"] != True].copy()  # noqa: E712

print(f"Reconstructing {len(failed)} failed-fit plots...")

for source_file, source_rows in failed.groupby("source_file"):
    cxw_path = DATA_DIR / source_file
    data = load_experiment(str(cxw_path))

    plot_samples = []
    plot_results = []
    plot_rows = []

    for _, row in source_rows.iterrows():
        cycle = int(row["cycle_index"])
        channel = row["channel"]
        rk_series = int(row["rk_serie_id"])

        sample = next(
            sample for sample in data["samples"]
            if sample["index"] == cycle
            and sample.get("channel") == channel
            and sample.get("rk_serie_id") == rk_series
        )

        channel_dmso = [
            dmso for dmso in data["dmso_cals"]
            if dmso.get("channel") == channel
            and dmso.get("rk_serie_id") == rk_series
        ]
        channel_blanks = [
            blank for blank in data["blanks"]
            if blank.get("channel") == channel
            and blank.get("rk_serie_id") == rk_series
        ]

        if not channel_dmso:
            channel_dmso = data["dmso_cals"]
        if not channel_blanks:
            channel_blanks = data["blanks"]

        # Reconstruct the same double-referenced signal used by fitting.
        dk = dk_fit_sample(
            sample,
            channel_dmso,
            blanks=channel_blanks,
        )

        t = dk["t"]
        signal = dk["signal"]

        dmso = select_dmso_cal(sample["index"], channel_dmso)
        concentration_func, _ = build_pulsed_concentration_profile(
            dmso,
            sample["concentration_M"],
        )

        # Recreate the original ODE fitting window.
        _, _, _, fit_mask = trim_to_fit_window(
            t,
            signal,
            np.zeros_like(t),
            sample["markers"],
        )
        t_fit = t[fit_mask]

        # Use the fallback parameters already recorded in the baseline CSV.
        ka = float(row["ka"])
        kd = float(row["kd"])
        Rmax = float(row["Rmax"])

        fitted = np.full_like(signal, np.nan)
        fitted[fit_mask] = simulate_sensorgram(
            t_fit,
            ka,
            kd,
            Rmax,
            concentration_func,
            R0=0.0,
        )

        result = {
            "t": t,
            "signal": signal,
            "R_fit": fitted,
            "ka": ka,
            "kd": kd,
            "KD": float(row["KD"]),
            "Rmax": Rmax,
            "sqrt_chi2": np.nan,
            "sigma_residual": np.nan,
            "success": False,
            "message": row.get("message", "All ODE fits failed"),
            "blank_index": dk.get("blank_index"),
        }

        plot_samples.append(sample)
        plot_results.append(result)
        plot_rows.append({
            "cycle_index": cycle,
            "channel": channel,
            "rk_serie_id": rk_series,
        })

    plot_dir = BASELINE_DIR / Path(source_file).stem / "plots"

    paths = save_fit_plots(
        pd.DataFrame(plot_rows),
        plot_samples,
        plot_results,
        str(plot_dir),
        mode="ode",
        n_parallel_jobs=None,
        blanks=data["blanks"],
    )

    written = sum(path is not None for path in paths)
    print(f"Wrote {written}/{len(source_rows)} plots to {plot_dir}")

print("Finished.")
