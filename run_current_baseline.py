from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import pandas as pd

from sensofit.batch import batch_fit, flag_poor_fits
from sensofit.plotting import save_fit_plots


DATA_DIR = Path("data")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("runs") / f"pre_characterisation_baseline_{STAMP}"

N_STARTS = 3
N_PARALLEL_JOBS = None  # Serial gives a clean timing baseline.


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
cxw_files = sorted(DATA_DIR.glob("*.cxw"))

if not cxw_files:
    raise RuntimeError(f"No .cxw files found in {DATA_DIR.resolve()}")

all_frames = []
file_summaries = []
run_start = time.perf_counter()

for cxw in cxw_files:
    print(f"\nFitting {cxw}...")
    file_start = time.perf_counter()

    df, data, results = batch_fit(
        str(cxw),
        mode="ode",
        progress=True,
        n_starts=N_STARTS,
        n_parallel_jobs=N_PARALLEL_JOBS,
    )

    if df.empty:
        print("No samples found; skipping.")
        continue

    df.insert(0, "source_file", cxw.name)
    df = flag_poor_fits(df)

    file_dir = OUTPUT_DIR / cxw.stem
    plot_dir = file_dir / "plots"
    file_dir.mkdir(parents=True)

    csv_path = file_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    paths = save_fit_plots(
        df,
        data["samples"],
        results,
        str(plot_dir),
        mode="ode",
        n_parallel_jobs=None,
        blanks=data["blanks"],
    )

    elapsed = time.perf_counter() - file_start
    successful = int(df["success"].fillna(False).astype(bool).sum())
    flagged = int(df["flag"].fillna(False).astype(bool).sum())
    plots_written = sum(path is not None for path in paths)

    file_summaries.append({
        "source_file": cxw.name,
        "rows": len(df),
        "successful_fits": successful,
        "flagged_fits": flagged,
        "plots_written": plots_written,
        "elapsed_seconds": elapsed,
    })
    all_frames.append(df)

combined = pd.concat(all_frames, ignore_index=True)
combined.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

manifest = {
    "created": datetime.now().isoformat(),
    "git_commit": git_commit(),
    "data_directory": str(DATA_DIR.resolve()),
    "fit_mode": "ode",
    "n_starts": N_STARTS,
    "n_parallel_jobs": N_PARALLEL_JOBS,
    "total_rows": len(combined),
    "total_elapsed_seconds": time.perf_counter() - run_start,
    "files": file_summaries,
}

with open(OUTPUT_DIR / "manifest.json", "w") as handle:
    json.dump(manifest, handle, indent=2)

print(f"\nFinished. Results written to:\n{OUTPUT_DIR.resolve()}")
