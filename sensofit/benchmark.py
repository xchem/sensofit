"""Experimentalist-labelled benchmark extraction, fitting, and scoring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .batch import batch_fit, flag_poor_fits


TRACE_KEY = ["source_file", "rk_serie_id", "cycle_index", "channel"]
QUALITY_TARGETS = ["noisy", "injection_issue", "carryover"]
PREDICTION_INPUT_COLUMNS = [
    "binding",
    "non_specific",
    "nonspecific",
    "binding_regime",
    "KD_uM",
    "KD",
    *QUALITY_TARGETS,
]
REQUIRED_SOURCE_FILES = {
    "20260323_EV712A_Binding_assay.cxw",
    "20260526_ZIKV-RdRp_Binding_assay.cxw",
    "20260626_ZIKV-NS2B-NS3_Binding_assay.cxw",
}
BENCHMARK_FILES = {
    "binding": "binding.csv",
    "non_specific": "non_specific_binding.csv",
    "affinity_regime": "affinity_regime.csv",
    "sensorgram_quality": "sensorgram_quality.csv",
}


@dataclass(frozen=True)
class BenchmarkTasks:
    """Normalized annotations for the four benchmark tasks."""

    binding: pd.DataFrame
    non_specific: pd.DataFrame
    affinity_regime: pd.DataFrame
    sensorgram_quality: pd.DataFrame

    def trace_keys(self) -> pd.DataFrame:
        frames = [
            self.binding[TRACE_KEY],
            self.non_specific[TRACE_KEY],
            self.affinity_regime[TRACE_KEY],
            self.sensorgram_quality[TRACE_KEY],
        ]
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates()
            .sort_values(TRACE_KEY)
            .reset_index(drop=True)
        )

    def annotations_long(self) -> pd.DataFrame:
        frames = []
        task_targets = [
            ("binding", self.binding, ["binding"]),
            ("non_specific", self.non_specific, ["non_specific"]),
            ("affinity_regime", self.affinity_regime, ["binding_regime"]),
            (
                "sensorgram_quality",
                self.sensorgram_quality,
                QUALITY_TARGETS,
            ),
        ]
        for task, frame, targets in task_targets:
            for target in targets:
                part = frame[
                    TRACE_KEY + ["compound", "experimentalist_label"]
                ].copy()
                part.insert(len(TRACE_KEY), "task", task)
                part.insert(len(TRACE_KEY) + 1, "target", target)
                part["label"] = frame[target].to_numpy()
                frames.append(part)
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Tables produced by benchmark scoring."""

    summary: pd.DataFrame
    per_class: pd.DataFrame
    scored_predictions: pd.DataFrame


def _canonical_integerish(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if np.isfinite(number) and number.is_integer() else text


def normalize_trace_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize trace identifiers so independent CSVs join reliably."""
    missing = [column for column in TRACE_KEY if column not in frame.columns]
    if missing:
        raise ValueError(f"missing trace-key columns: {missing}")
    normalized = frame.copy()
    normalized["source_file"] = normalized["source_file"].map(
        lambda value: Path(str(value)).name
    )
    normalized["rk_serie_id"] = normalized["rk_serie_id"].map(
        _canonical_integerish
    )
    normalized["cycle_index"] = normalized["cycle_index"].map(
        _canonical_integerish
    )
    normalized["channel"] = normalized["channel"].astype(str).str.strip()
    return normalized


def _load_task(directory: Path, filename: str) -> pd.DataFrame:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"benchmark task file not found: {path}")
    frame = normalize_trace_keys(pd.read_csv(path))
    expected = TRACE_KEY + ["compound", "experimentalist_label"]
    if list(frame.columns) != expected:
        raise ValueError(
            f"{path.name} columns must be exactly {expected}, "
            f"got {list(frame.columns)}"
        )
    if frame.duplicated(TRACE_KEY).any():
        raise ValueError(f"{path.name} contains duplicate trace keys")
    return frame


def _map_labels(
    frame: pd.DataFrame,
    mapping: dict[str, object],
    output_column: str,
) -> pd.DataFrame:
    labels = (
        frame["experimentalist_label"].fillna("").astype(str).str.strip().str.lower()
    )
    unknown = sorted(set(labels).difference(mapping))
    if unknown:
        raise ValueError(f"unknown labels for {output_column}: {unknown}")
    result = frame.copy()
    result[output_column] = labels.map(mapping)
    return result


def load_benchmark_tasks(
    directory: str | Path = "data/experimentalist_benchmark",
) -> BenchmarkTasks:
    """Load and validate the four compact benchmark task definitions."""
    directory = Path(directory)
    binding = _map_labels(
        _load_task(directory, BENCHMARK_FILES["binding"]),
        {"binder": True, "non_binder": False},
        "binding",
    )
    non_specific = _map_labels(
        _load_task(directory, BENCHMARK_FILES["non_specific"]),
        {"specific": False, "non_specific": True},
        "non_specific",
    )
    affinity_regime = _map_labels(
        _load_task(directory, BENCHMARK_FILES["affinity_regime"]),
        {
            "tight": "tight",
            "medium": "medium",
            "weak": "weak",
            "no_binding": "no_binding",
        },
        "binding_regime",
    )
    quality = _load_task(directory, BENCHMARK_FILES["sensorgram_quality"])
    quality_labels = (
        quality["experimentalist_label"].fillna("").astype(str).str.strip().str.lower()
    )
    known_quality = {"clean", "noisy", "injection_issue", "carryover"}
    unknown_quality = sorted(set(quality_labels).difference(known_quality))
    if unknown_quality:
        raise ValueError(f"unknown sensorgram quality labels: {unknown_quality}")
    quality = quality.copy()
    quality["noisy"] = quality_labels.eq("noisy")
    quality["injection_issue"] = quality_labels.eq("injection_issue")
    quality["carryover"] = quality_labels.eq("carryover")

    tasks = BenchmarkTasks(
        binding=binding,
        non_specific=non_specific,
        affinity_regime=affinity_regime,
        sensorgram_quality=quality,
    )
    source_files = set(tasks.trace_keys()["source_file"])
    if source_files != REQUIRED_SOURCE_FILES:
        raise ValueError(
            "benchmark source files do not match the required dataset set: "
            f"{sorted(source_files)}"
        )
    return tasks


def _coerce_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def normalize_affinity_regime(value):
    """Normalize a predicted affinity class to the benchmark vocabulary."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tight*": "tight",
        "nonbinder": "no_binding",
        "non_binder": "no_binding",
        "no_bind": "no_binding",
    }
    text = aliases.get(text, text)
    return text if text in {"tight", "medium", "weak", "no_binding"} else None


def classify_affinity_regime(
    kd_uM,
    binding=True,
    *,
    tight_upper_uM: float = 1.0,
    medium_upper_uM: float = 100.0,
):
    """Derive an affinity class from binding and KD predictions."""
    if _coerce_bool(binding) is False:
        return "no_binding"
    try:
        kd_uM = float(kd_uM)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(kd_uM) or kd_uM < 0:
        return None
    if kd_uM < tight_upper_uM:
        return "tight"
    if kd_uM < medium_upper_uM:
        return "medium"
    return "weak"


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction keys/labels and derive affinity when necessary."""
    prepared = normalize_trace_keys(predictions)
    if prepared.duplicated(TRACE_KEY).any():
        raise ValueError("prediction table contains duplicate trace keys")
    if "non_specific" not in prepared and "nonspecific" in prepared:
        prepared["non_specific"] = prepared["nonspecific"]
    if "binding_regime" in prepared:
        prepared["binding_regime"] = prepared["binding_regime"].map(
            normalize_affinity_regime
        )
    else:
        if "KD_uM" in prepared:
            kd_uM = pd.to_numeric(prepared["KD_uM"], errors="coerce")
        elif "KD" in prepared:
            kd_uM = pd.to_numeric(prepared["KD"], errors="coerce") * 1e6
        else:
            kd_uM = pd.Series(np.nan, index=prepared.index)
        binding = prepared.get("binding", pd.Series(True, index=prepared.index))
        prepared["binding_regime"] = [
            classify_affinity_regime(kd, binds)
            for kd, binds in zip(kd_uM, binding)
        ]
    return prepared


def validate_evaluation_input(
    predictions: pd.DataFrame,
    tasks: BenchmarkTasks,
    *,
    source: str = "prediction CSV",
) -> pd.DataFrame:
    """Validate and normalize an external prediction table before scoring."""
    supplied = [
        column for column in PREDICTION_INPUT_COLUMNS
        if column in predictions.columns
    ]
    if not supplied:
        raise ValueError(
            f"{source} contains no recognized prediction columns. "
            "Expected at least one of: "
            f"{', '.join(PREDICTION_INPUT_COLUMNS)}. "
            "benchmark_trace_keys.csv contains identifiers only; pass a "
            "predictions.csv file or add predictions to the exported keys."
        )

    prepared = prepare_predictions(predictions)
    usable = pd.Series(False, index=prepared.index)
    for target in ["binding", "non_specific", *QUALITY_TARGETS]:
        if target in prepared:
            usable |= prepared[target].map(_coerce_bool).notna()
    if "binding_regime" in prepared:
        usable |= prepared["binding_regime"].notna()
    if not usable.any():
        raise ValueError(
            f"{source} has recognized prediction columns "
            f"({', '.join(supplied)}), but none contain usable values."
        )

    benchmark_keys = tasks.trace_keys()
    matched = prepared[TRACE_KEY].merge(
        benchmark_keys,
        on=TRACE_KEY,
        how="inner",
    )
    if matched.empty:
        raise ValueError(
            f"{source} contains no trace keys matching this benchmark. "
            "Check source_file, rk_serie_id, cycle_index, channel, and "
            "--benchmark-dir."
        )
    return prepared


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _class_metrics(
    truth: pd.Series,
    prediction: pd.Series,
    classes: Iterable,
    *,
    task: str,
    target: str,
) -> pd.DataFrame:
    rows = []
    for class_label in classes:
        truth_positive = truth.eq(class_label)
        predicted_positive = prediction.eq(class_label)
        tp = int((truth_positive & predicted_positive).sum())
        fp = int((~truth_positive & predicted_positive).sum())
        fn = int((truth_positive & ~predicted_positive).sum())
        tn = int((~truth_positive & ~predicted_positive).sum())
        support = int(truth_positive.sum())
        precision = (
            _safe_divide(tp, tp + fp)
            if tp + fp
            else (0.0 if support else np.nan)
        )
        rows.append(
            {
                "task": task,
                "target": target,
                "class": class_label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": _safe_divide(tp, tp + fn),
                "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
            }
        )
    return pd.DataFrame(rows)


def _score_target(
    expected: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    task: str,
    target: str,
    classes: list,
    boolean: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    truth = expected[TRACE_KEY + [target]].rename(columns={target: "_truth"})
    if target in predictions:
        predicted = predictions[TRACE_KEY + [target]].rename(
            columns={target: "_prediction"}
        )
    else:
        predicted = predictions[TRACE_KEY].copy()
        predicted["_prediction"] = None
    merged = truth.merge(predicted, on=TRACE_KEY, how="left", validate="one_to_one")
    if boolean:
        merged["truth"] = merged["_truth"].astype(bool)
        merged["prediction"] = merged["_prediction"].map(_coerce_bool)
    else:
        merged["truth"] = merged["_truth"]
        merged["prediction"] = merged["_prediction"].map(
            normalize_affinity_regime
        )
    merged["covered"] = merged["prediction"].notna()
    merged["correct"] = (
        merged["covered"] & merged["prediction"].eq(merged["truth"])
    )
    covered = merged[merged["covered"]]
    n_expected = len(merged)
    n_predicted = len(covered)
    summary = {
        "task": task,
        "target": target,
        "kind": "binary" if boolean else "multiclass",
        "n_expected": n_expected,
        "n_predicted": n_predicted,
        "coverage": _safe_divide(n_predicted, n_expected),
        "accuracy": float(covered["correct"].mean()) if n_predicted else np.nan,
        "coverage_adjusted_accuracy": _safe_divide(
            int(merged["correct"].sum()), n_expected
        ),
    }
    per_class = _class_metrics(
        covered["truth"], covered["prediction"], classes, task=task, target=target
    )
    if boolean:
        positive = per_class[per_class["class"].eq(True)].iloc[0]
        summary.update(
            precision=positive["precision"],
            recall=positive["recall"],
            f1=positive["f1"],
        )
    else:
        summary.update(
            macro_precision=per_class["precision"].mean(),
            macro_recall=per_class["recall"].mean(),
            macro_f1=per_class["f1"].mean(),
        )
    scored = merged[TRACE_KEY + ["truth", "prediction", "covered", "correct"]]
    scored = scored.copy()
    scored.insert(len(TRACE_KEY), "task", task)
    scored.insert(len(TRACE_KEY) + 1, "target", target)
    return summary, per_class, scored


def evaluate_predictions(
    predictions: pd.DataFrame,
    tasks: BenchmarkTasks,
) -> BenchmarkEvaluation:
    """Score predictions against all benchmark tasks."""
    predictions = prepare_predictions(predictions)
    summaries = []
    per_class_tables = []
    scored_tables = []
    specs = [
        ("binding", tasks.binding, "binding", [False, True], True),
        (
            "non_specific",
            tasks.non_specific,
            "non_specific",
            [False, True],
            True,
        ),
        (
            "affinity_regime",
            tasks.affinity_regime,
            "binding_regime",
            ["no_binding", "weak", "medium", "tight"],
            False,
        ),
    ]
    for task, expected, target, classes, boolean in specs:
        summary, per_class, scored = _score_target(
            expected,
            predictions,
            task=task,
            target=target,
            classes=classes,
            boolean=boolean,
        )
        summaries.append(summary)
        per_class_tables.append(per_class)
        scored_tables.append(scored)

    quality_summaries = []
    quality_scores = []
    for target in QUALITY_TARGETS:
        summary, per_class, scored = _score_target(
            tasks.sensorgram_quality,
            predictions,
            task="sensorgram_quality",
            target=target,
            classes=[False, True],
            boolean=True,
        )
        quality_summaries.append(summary)
        quality_scores.append(scored)
        per_class_tables.append(per_class)
        scored_tables.append(scored)

    exact = tasks.sensorgram_quality[TRACE_KEY].copy()
    exact["covered"] = True
    exact["correct"] = True
    for scored in quality_scores:
        target = scored["target"].iloc[0]
        part = scored[TRACE_KEY + ["covered", "correct"]].rename(
            columns={
                "covered": f"{target}_covered",
                "correct": f"{target}_correct",
            }
        )
        exact = exact.merge(part, on=TRACE_KEY, validate="one_to_one")
        exact["covered"] &= exact[f"{target}_covered"]
        exact["correct"] &= exact[f"{target}_correct"]
    covered = exact[exact["covered"]]
    summaries.append(
        {
            "task": "sensorgram_quality",
            "target": "all_labels",
            "kind": "multilabel",
            "n_expected": len(exact),
            "n_predicted": len(covered),
            "coverage": _safe_divide(len(covered), len(exact)),
            "accuracy": (
                float(covered["correct"].mean()) if len(covered) else np.nan
            ),
            "coverage_adjusted_accuracy": _safe_divide(
                int(exact["correct"].sum()), len(exact)
            ),
            "macro_precision": pd.Series(
                [row["precision"] for row in quality_summaries]
            ).mean(),
            "macro_recall": pd.Series(
                [row["recall"] for row in quality_summaries]
            ).mean(),
            "macro_f1": pd.Series(
                [row["f1"] for row in quality_summaries]
            ).mean(),
        }
    )
    return BenchmarkEvaluation(
        summary=pd.DataFrame(summaries),
        per_class=pd.concat(per_class_tables, ignore_index=True),
        scored_predictions=pd.concat(scored_tables, ignore_index=True),
    )


def _find_source_file(data_directory: Path, source_file: str) -> Path:
    matches = sorted(data_directory.rglob(source_file))
    if not matches:
        raise FileNotFoundError(
            f"could not find {source_file!r} under {data_directory}"
        )
    if len(matches) > 1:
        raise ValueError(f"found multiple copies of {source_file!r}: {matches}")
    return matches[0]


def run_current_sensofit(
    tasks: BenchmarkTasks,
    data_directory: str | Path,
    *,
    mode: str = "dk",
    n_starts: int = 3,
    n_parallel_jobs: int | None = None,
) -> pd.DataFrame:
    """Run the current SensoFit implementation on benchmark traces only."""
    if mode not in {"dk", "ode"}:
        raise ValueError("mode must be 'dk' or 'ode'")
    frames = []
    with tempfile.TemporaryDirectory(prefix="sensofit-benchmark-") as temp:
        for source_file, keys in tasks.trace_keys().groupby("source_file"):
            cxw_path = _find_source_file(Path(data_directory), source_file)
            subset_path = Path(temp) / f"{cxw_path.stem}.csv"
            keys[["rk_serie_id", "cycle_index", "channel"]].to_csv(
                subset_path, index=False
            )
            frame, _, _ = batch_fit(
                str(cxw_path),
                mode=mode,
                progress=True,
                n_starts=n_starts,
                n_parallel_jobs=n_parallel_jobs,
                subset_csv=str(subset_path),
            )
            if not frame.empty:
                frame.insert(0, "source_file", source_file)
                frames.append(flag_poor_fits(frame))
    if not frames:
        return pd.DataFrame(columns=TRACE_KEY)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(TRACE_KEY)
        .reset_index(drop=True)
    )


def _records(frame: pd.DataFrame) -> list[dict]:
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
    return records


def write_benchmark_outputs(
    output_directory: str | Path,
    *,
    tasks: BenchmarkTasks,
    evaluation: BenchmarkEvaluation,
    predictions: pd.DataFrame | None = None,
    manifest: dict | None = None,
) -> Path:
    """Write annotations, predictions, metrics, and run metadata."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tasks.annotations_long().to_csv(output / "annotations.csv", index=False)
    tasks.trace_keys().to_csv(output / "benchmark_trace_keys.csv", index=False)
    evaluation.summary.to_csv(output / "summary.csv", index=False)
    evaluation.per_class.to_csv(output / "per_class_metrics.csv", index=False)
    evaluation.scored_predictions.to_csv(
        output / "scored_predictions.csv", index=False
    )
    if predictions is not None:
        predictions.to_csv(output / "predictions.csv", index=False)
    with (output / "metrics.json").open("w") as handle:
        json.dump(
            {
                "summary": _records(evaluation.summary),
                "per_class": _records(evaluation.per_class),
            },
            handle,
            indent=2,
            allow_nan=False,
        )
    if manifest is not None:
        with (output / "manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
    return output


def _default_output(prefix: str) -> Path:
    return Path("runs") / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def _print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "task",
        "n_expected",
        "coverage",
        "accuracy",
        "coverage_adjusted_accuracy",
        "precision",
        "recall",
        "macro_f1",
    ]
    print(
        summary[[column for column in columns if column in summary]]
        .to_string(index=False, float_format="%.4f")
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sensofit.benchmark",
        description="Run and score the experimentalist-labelled benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "evaluate", "run"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--benchmark-dir",
            default="data/experimentalist_benchmark",
        )
        command.add_argument("--output", "-o", default=None)
    evaluate_parser = subparsers.choices["evaluate"]
    evaluate_parser.add_argument("predictions")
    run_parser = subparsers.choices["run"]
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--mode", choices=["dk", "ode"], default="dk")
    run_parser.add_argument("--n-starts", type=int, default=3)
    run_parser.add_argument("--n-parallel-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    tasks = load_benchmark_tasks(args.benchmark_dir)

    if args.command == "extract":
        if not args.output:
            parser.error("extract requires --output")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        tasks.annotations_long().to_csv(output / "annotations.csv", index=False)
        tasks.trace_keys().to_csv(
            output / "benchmark_trace_keys.csv", index=False
        )
        print(f"Wrote normalized benchmark to {output.resolve()}")
        return

    if args.command == "evaluate":
        try:
            predictions = validate_evaluation_input(
                pd.read_csv(args.predictions),
                tasks,
                source=str(args.predictions),
            )
        except ValueError as error:
            parser.error(str(error))
        evaluation = evaluate_predictions(predictions, tasks)
        output = Path(args.output) if args.output else _default_output(
            "experimentalist_benchmark_evaluation"
        )
        write_benchmark_outputs(
            output,
            tasks=tasks,
            evaluation=evaluation,
            predictions=predictions,
            manifest={
                "created": datetime.now(timezone.utc).isoformat(),
                "prediction_source": str(Path(args.predictions).resolve()),
                "benchmark_directory": str(Path(args.benchmark_dir).resolve()),
            },
        )
        _print_summary(evaluation.summary)
        print(f"\nResults written to {output.resolve()}")
        return

    output = Path(args.output) if args.output else _default_output(
        f"experimentalist_benchmark_{args.mode}"
    )
    start = time.perf_counter()
    predictions = run_current_sensofit(
        tasks,
        args.data_dir,
        mode=args.mode,
        n_starts=args.n_starts,
        n_parallel_jobs=args.n_parallel_jobs,
    )
    evaluation = evaluate_predictions(predictions, tasks)
    elapsed = time.perf_counter() - start
    write_benchmark_outputs(
        output,
        tasks=tasks,
        evaluation=evaluation,
        predictions=prepare_predictions(predictions),
        manifest={
            "created": datetime.now(timezone.utc).isoformat(),
            "approach": "sensofit.batch_fit",
            "mode": args.mode,
            "n_starts": args.n_starts,
            "n_parallel_jobs": args.n_parallel_jobs,
            "benchmark_directory": str(Path(args.benchmark_dir).resolve()),
            "data_directory": str(Path(args.data_dir).resolve()),
            "n_trace_keys": len(tasks.trace_keys()),
            "elapsed_seconds": elapsed,
        },
    )
    _print_summary(evaluation.summary)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
