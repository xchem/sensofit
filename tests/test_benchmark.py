"""Tests for the experimentalist-labelled benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from sensofit import batch as batch_module
from sensofit.benchmark import (
    BENCHMARK_FILES,
    REQUIRED_SOURCE_FILES,
    TRACE_KEY,
    classify_affinity_regime,
    evaluate_predictions,
    load_benchmark_tasks,
)


BENCHMARK_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "experimentalist_benchmark"
)


def test_extracts_expected_task_sizes_and_trace_count():
    tasks = load_benchmark_tasks(BENCHMARK_DIR)
    assert len(tasks.binding) == 21
    assert len(tasks.non_specific) == 16
    assert len(tasks.affinity_regime) == 48
    assert len(tasks.sensorgram_quality) == 21
    assert len(tasks.trace_keys()) == 106
    assert set(tasks.trace_keys()["source_file"]) == REQUIRED_SOURCE_FILES


def test_task_definition_csvs_are_compact():
    expected_columns = TRACE_KEY + ["compound", "experimentalist_label"]
    for filename in BENCHMARK_FILES.values():
        frame = pd.read_csv(BENCHMARK_DIR / filename)
        assert list(frame.columns) == expected_columns


def test_labels_are_normalized():
    tasks = load_benchmark_tasks(BENCHMARK_DIR)
    assert set(tasks.binding["experimentalist_label"]) == {
        "binder",
        "non_binder",
    }
    assert set(tasks.affinity_regime["binding_regime"]) == {
        "tight",
        "medium",
        "weak",
        "no_binding",
    }
    clean = tasks.sensorgram_quality[
        tasks.sensorgram_quality["experimentalist_label"].eq("clean")
    ]
    assert not clean[
        ["noisy", "injection_issue", "carryover"]
    ].to_numpy().any()


def test_affinity_regime_thresholds_and_no_binding_class():
    assert classify_affinity_regime(0.999, True) == "tight"
    assert classify_affinity_regime(1.0, True) == "medium"
    assert classify_affinity_regime(99.999, True) == "medium"
    assert classify_affinity_regime(100.0, True) == "weak"
    assert classify_affinity_regime(np.nan, False) == "no_binding"


def test_perfect_predictions_score_all_four_tasks():
    tasks = load_benchmark_tasks(BENCHMARK_DIR)
    predictions = tasks.trace_keys()
    frames = [
        tasks.binding[TRACE_KEY + ["binding"]],
        tasks.non_specific[TRACE_KEY + ["non_specific"]],
        tasks.affinity_regime[TRACE_KEY + ["binding_regime"]],
        tasks.sensorgram_quality[
            TRACE_KEY + ["noisy", "injection_issue", "carryover"]
        ],
    ]
    for frame in frames:
        predictions = predictions.merge(frame, on=TRACE_KEY, how="left")
    evaluation = evaluate_predictions(predictions, tasks)
    assert evaluation.summary["coverage"].eq(1.0).all()
    assert evaluation.summary["accuracy"].eq(1.0).all()
    assert evaluation.summary["coverage_adjusted_accuracy"].eq(1.0).all()


def test_missing_predictions_reduce_coverage_adjusted_accuracy():
    tasks = load_benchmark_tasks(BENCHMARK_DIR)
    predictions = tasks.binding.iloc[:-1][TRACE_KEY + ["binding"]]
    evaluation = evaluate_predictions(predictions, tasks)
    binding = evaluation.summary[evaluation.summary["task"].eq("binding")].iloc[0]
    assert binding["n_expected"] == 21
    assert binding["n_predicted"] == 20
    assert binding["accuracy"] == 1.0
    assert binding["coverage_adjusted_accuracy"] == 20 / 21


def test_batch_subset_csv_selects_only_requested_trace(tmp_path, monkeypatch):
    samples = [
        {
            "rk_serie_id": 1,
            "index": 10,
            "channel": "FC2-FC1",
            "compound": "keep",
            "concentration_M": 1e-6,
        },
        {
            "rk_serie_id": 1,
            "index": 11,
            "channel": "FC2-FC1",
            "compound": "drop",
            "concentration_M": 1e-6,
        },
    ]
    experiment = {"samples": samples, "dmso_cals": [], "blanks": []}
    monkeypatch.setattr(
        batch_module, "load_experiment", lambda filepath, channels: experiment
    )

    def fake_process(*args, **kwargs):
        sample = args[4]
        return [
            None,
            {
                "compound": sample["compound"],
                "concentration_M": sample["concentration_M"],
            },
        ]

    monkeypatch.setattr(batch_module, "_batch_process", fake_process)
    subset = tmp_path / "subset.csv"
    pd.DataFrame(
        [{"rk_serie_id": 1, "cycle_index": 10, "channel": "FC2-FC1"}]
    ).to_csv(subset, index=False)
    result, data, _ = batch_module.batch_fit(
        "unused.cxw", mode="dk", progress=False, subset_csv=str(subset)
    )
    assert result["compound"].tolist() == ["keep"]
    assert [sample["compound"] for sample in data["samples"]] == ["keep"]
