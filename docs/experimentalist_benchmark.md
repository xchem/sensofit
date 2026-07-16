# Experimentalist benchmark

The benchmark is defined by four compact CSVs in
`data/experimentalist_benchmark/`. Every row contains only the trace
identifiers, a human-readable compound identifier, and the normalized
experimentalist label:

`source_file`, `rk_serie_id`, `cycle_index`, `channel`, `compound`,
`experimentalist_label`

## Tasks

| Task | Source CSV | Ground truth | Size |
|---|---|---|---:|
| Binding detection | `binding.csv` | `binder`, `non_binder` | 21 |
| Non-specific binding detection | `non_specific_binding.csv` | `specific`, `non_specific` | 16 |
| Affinity regime | `affinity_regime.csv` | `no_binding`, `weak`, `medium`, `tight` | 48 |
| Sensorgram quality | `sensorgram_quality.csv` | `clean`, `noisy`, `injection_issue`, `carryover` | 21 |

There are 106 labelled benchmark traces across 43 cycles. The task files do
not overlap at present.

The committed task definitions are already normalized. Unlabelled review rows
are excluded, alternate spellings and review notation have been resolved, and
each CSV uses one documented label vocabulary.

## Required experimental datasets

Running a fitting approach on the benchmark requires:

- `20260323_EV712A_Binding_assay.cxw`
- `20260526_ZIKV-RdRp_Binding_assay.cxw`
- `20260626_ZIKV-NS2B-NS3_Binding_assay.cxw`

The files are not committed to Git. Pass a directory containing them with
`--data-dir`. The benchmark runner searches that directory recursively, so the
current layout under `data/data_for_diamond_meeting_2026_07_14/` works with
`--data-dir data`.

`20250826_DENV-2 NS2B3 Binding Assay.cxw` is not used by this benchmark.

When an approach supplies KD rather than a direct regime prediction, the
benchmark adapter uses:

- `tight`: KD < 1 µM
- `medium`: 1 µM <= KD < 100 µM
- `weak`: KD >= 100 µM
- `no_binding`: the approach predicts no binding

## Evaluate any fitting approach

Produce a CSV with the four trace-key columns and whichever prediction columns
the approach supports:

- `binding`
- `non_specific`
- `binding_regime`, or `binding` plus `KD_uM`/`KD`
- `noisy`
- `injection_issue`
- `carryover`

Then run:

```bash
python -m sensofit.benchmark evaluate \
  runs/experimentalist_benchmark_dk/predictions.csv \
  --benchmark-dir data/experimentalist_benchmark \
  --output runs/my_approach_benchmark
```

`evaluate` scores existing predictions without running a fitter. This is
useful for comparing another fitting implementation, rescoring an older run,
or evaluating hand-produced predictions. The `benchmark_trace_keys.csv` file
is an identifier template, not a prediction file; add prediction columns to
it first or pass the `predictions.csv` written by `run`.

The command stops with an explanatory error if the input has no recognized
prediction columns, all supplied predictions are unusable, or none of its
trace keys match the selected benchmark.

Missing predictions are reported through coverage. Accuracy is reported both
on covered rows and as coverage-adjusted accuracy, where missing predictions
count as incorrect.

The current quality subset has no experimentally labelled `noisy` positives,
so noisy recall and F1 are undefined.

## Run the current SensoFit implementation

The convenience adapter fits only the 106 labelled traces:

```bash
python -m sensofit.benchmark run \
  --benchmark-dir data/experimentalist_benchmark \
  --data-dir data \
  --mode dk \
  --output runs/experimentalist_benchmark_dk
```

Use `--mode ode` for the ODE fitting pipeline. The output directory contains:

- `annotations.csv`
- `benchmark_trace_keys.csv`
- `predictions.csv`
- `scored_predictions.csv`
- `summary.csv`
- `per_class_metrics.csv`
- `metrics.json`
- `manifest.json`
