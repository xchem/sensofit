"""Evaluate the pre-fit area score against experimentalist affinity labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from sensofit.affinity_prior import estimate_affinity_area_prior
from sensofit.data_loader import load_cxw


ORDINAL_LABEL = {"no_binding": 0, "weak": 1, "medium": 2, "tight": 3}
# Reporting cutoffs for comparison with the annotated benchmark only.
SCORE_BINS = [-np.inf, 0.80, 2.00, 2.97, np.inf]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default="data/experimentalist_benchmark/affinity_regime.csv",
    )
    parser.add_argument(
        "--concentration-overrides",
        default=(
            "data/experimentalist_benchmark/"
            "affinity_concentration_overrides.csv"
        ),
        help=(
            "Optional CSV with source_file, cycle_index and concentration_M "
            "for corrections to missing or incorrect source metadata."
        ),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations)
    concentration_overrides = {}
    overrides_path = Path(args.concentration_overrides)
    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path)
        expected = ["source_file", "cycle_index", "concentration_M"]
        if list(overrides.columns) != expected:
            raise ValueError(
                f"{overrides_path} columns must be exactly {expected}"
            )
        concentration_overrides = {
            (Path(row.source_file).name, int(row.cycle_index)):
                float(row.concentration_M)
            for row in overrides.itertuples(index=False)
        }
    data_dir = Path(args.data_dir)
    experiments = {}
    rows = []
    for annotation in annotations.itertuples(index=False):
        source = Path(annotation.source_file).name
        if source not in experiments:
            matches = list(data_dir.rglob(source))
            if not matches:
                raise FileNotFoundError(
                    f"could not find {source} under {data_dir}"
                )
            selected = min(matches, key=lambda path: len(path.parts))
            experiments[source] = load_cxw(selected, channels="all")
        sample = next(
            sample for sample in experiments[source]["samples"]
            if int(sample["index"]) == int(annotation.cycle_index)
            and str(sample.get("rk_serie_id")) == str(annotation.rk_serie_id)
            and sample["channel"] == annotation.channel
        )
        override = concentration_overrides.get(
            (source, int(annotation.cycle_index))
        )
        if override is not None:
            sample = {
                **sample,
                "concentration_M": float(override),
                "concentration_overridden": True,
            }
        prior = estimate_affinity_area_prior(sample)
        rows.append({
            "source_file": source,
            "rk_serie_id": annotation.rk_serie_id,
            "cycle_index": annotation.cycle_index,
            "channel": annotation.channel,
            "compound": annotation.compound,
            "experimentalist_label": annotation.experimentalist_label,
            **prior.as_dict(),
        })

    result = pd.DataFrame(rows)
    result["predicted_regime"] = pd.cut(
        result["score"],
        bins=SCORE_BINS,
        labels=list(ORDINAL_LABEL),
        right=False,
    )
    result["experimentalist_ordinal"] = result[
        "experimentalist_label"].map(ORDINAL_LABEL)
    rho = result["score"].corr(
        result["experimentalist_ordinal"], method="spearman")
    print(f"scored: {len(result)}")
    print(f"Spearman score vs affinity ordinal: {rho:.3f}")
    design = np.column_stack([
        np.ones(len(result)),
        -np.log10(result["concentration_M"].to_numpy()),
        np.log10(result["association_selectivity"].to_numpy()),
        np.log10(result["retention_ratio"].to_numpy()),
        np.log10(result["tail_survival_ratio"].to_numpy()),
        (
            np.maximum(result["early_decay_delta_bic"].to_numpy(), 0.0)
            / (
                np.maximum(
                    result["early_decay_delta_bic"].to_numpy(), 0.0,
                )
                + 100.0
            )
        ),
        np.sqrt(result["retained_tail_fraction"].to_numpy()),
    ])
    target = result["experimentalist_ordinal"].to_numpy(dtype=float)
    coefficient_fit = lsq_linear(
        design,
        target,
        bounds=(
            [-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )
    print(
        "Monotonic ordinal coefficients "
        "(intercept, concentration, selectivity, retention, tail survival, "
        "bounded early decay, retained-tail persistence): "
        + ", ".join(f"{value:.3f}" for value in coefficient_fit.x)
    )
    grouped_prediction = np.full(len(result), np.nan)
    compounds = result["compound"].fillna("").astype(str).to_numpy()
    for compound in np.unique(compounds):
        test = compounds == compound
        train = ~test
        fold_fit = lsq_linear(
            design[train],
            target[train],
            bounds=(
                [-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        grouped_prediction[test] = design[test] @ fold_fit.x
    grouped_rho = pd.Series(grouped_prediction).corr(
        pd.Series(target), method="spearman")
    grouped_mae = float(np.mean(np.abs(grouped_prediction - target)))
    print(
        "Leave-one-compound-out: "
        f"Spearman={grouped_rho:.3f}, ordinal MAE={grouped_mae:.3f}"
    )
    print(
        result.groupby("experimentalist_label")["score"]
        .agg(["count", "min", "median", "max"])
        .reindex(ORDINAL_LABEL)
        .round(3)
        .to_string()
    )
    print("\nPredicted regime by experimentalist label:")
    print(pd.crosstab(
        result["experimentalist_label"],
        result["predicted_regime"],
    ).reindex(index=ORDINAL_LABEL, columns=ORDINAL_LABEL, fill_value=0))
    if args.output:
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
