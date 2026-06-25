# SensoFit

Kinetic fitting for Creoptix GCI (Grating-Coupled Interferometry) biosensor data.

SensoFit extracts 1:1 Langmuir binding kinetics (ka, kd, KD) from `.cxw` experiment files
using a two-stage pipeline: **Direct Kinetics** for fast initial estimates, followed by
**ODE-based refinement** with multi-start optimisation for publication-quality parameters.

## Features

- **`.cxw` parser** — reads ZIP/XML/HDF5 experiment files directly
- **Direct Kinetics (DK)** — millisecond-scale linear fit from dR/dt vs R
- **ODE fitting** — full 1:1 Langmuir ODE solved with `scipy.integrate.solve_ivp`
- **Batch processing** — fit all samples in a file with one call
- **Sensorgram heuristics** — flags samples with expert-defined annotation (e.g. noisy, non-specific-binder, etc...)
- **Goodness-of-fit** — automatically assert the quality of the fit
- **Plotting** — annotated data-vs-model overlays saved as individual PNGs
- **CLI** — command-line batch processing of one or many `.cxw` files
- **GUI** — Graphical User Interface for differents modules (protocol development, check sensorgrams, fit sensorgrams, export data package)

## Installation

```bash
# Create a conda environment with dependencies
conda create -n sensofit python=3.12
conda activate sensofit
```
Clone/copy this repository, cd to the root, then install the package using the following command:
```bash
pip install -e .
``` 

## Quick Start

### Python API

#### Batch processing

```python
from sensofit import load_cxw, batch_fit, flag_poor_fits
from sensofit.plotting import save_fit_plots

# Load and fit all samples
df, data = batch_fit('experiment.cxw', mode='ode')
df = flag_poor_fits(df)

# Or with more starting points for robust estimates
df, data = batch_fit('experiment.cxw', mode='ode', n_starts=10)

# Save results
df.to_csv('results.csv', index=False)
```

#### Export package
```python
from sensofit import export_package

export_package(['file1.cxw', 'file2.cxw'],
               '/tmp/my_dataset.zip',
               package_name='my_dataset')
```

### Command Line

The CLI has 4 modes: **batch fitting** (default) and **data export** (`export` subcommand) **protocol development** (`protocol-dev` subcommand) and **graphical user interface** (`gui` subcommand).

#### Batch fitting

```bash
# Fit a single file (ODE mode, default)
python -m sensofit experiment.cxw -o results/

# Fit all .cxw files in a folder using Direct Kinetics
python -m sensofit data_folder/ --mode dk -o results/

# Run both DK and ODE fits
python -m sensofit data_folder/ --mode both -o results/

# Restrict to specific active flow cells
python -m sensofit experiment.cxw --channels 2 3 -o results/
```

**Outputs:**
- `results/batch_results.csv` — all fits with source file, cycle number, sample ID, ka, kd, KD, and quality flags
- `results/<filename>_<mode>_plots/` — individual PNG plots per sample

#### Data export (dissemination package)

Package raw signal data from one or more `.cxw` files into a self-describing zip
(per-CXW folders → per-cycle folders → one CSV per `FCx-FCy` channel pair, plus
`metadata.json` sidecars, `experiment.json`, and an auto-generated `README.md`).

```bash
# Single file → auto-named zip (sensofit_package_<timestamp>.zip)
python -m sensofit export experiment.cxw

# Multiple files / directories → custom output zip and package name
python -m sensofit export file1.cxw file2.cxw data_folder/ \
    -o /tmp/my_dataset.zip --name my_dataset
```

### CSV columns

| Column | Description |
|--------|-------------|
| `source_file` | Original `.cxw` filename |
| `cycle_index` | Cycle number within the experiment |
| `compound` | Sample / compound identifier |
| `ka` | Association rate constant (M⁻¹s⁻¹) |
| `kd` | Dissociation rate constant (s⁻¹) |
| `KD` | Equilibrium dissociation constant (M) |
| `KD_uM` | KD in micromolar |
| `Rmax` | Maximum binding response (pg/mm²) |
| `sigma_res` | Residual standard deviation |
| `flag` | `True` if fit quality is questionable |
| `flag_reason` | Reason(s) for flagging |
| `nonspecific` | `True` if non-specific binder detected |

## Package Structure

```
sensofit/
├── __init__.py          # Public API: load_cxw, batch_fit, flag_poor_fits, export_package
├── __main__.py          # CLI entry point (python -m sensofit [export])
├── batch.py             # Batch processing and quality flagging
├── data_loader.py       # .cxw file parser (ZIP → XML + HDF5)
├── dataexporter.py      # Package raw .cxw data into a self-describing zip
├── models.py            # Preprocessing, concentration profiles, ODE model
├── direct_kinetics.py   # Fast DK fitting (dR/dt linear regression)
├── ode_fitting.py       # Full ODE fitting (DK → multi-start TRF)
├── package_loader.py    # .zip/dir parser
├── plotting.py          # Data-vs-model fit plots
└── protocol_dev.py      # Currently a placeholder for future protocol development analysis tool
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_data_release.ipynb` | SensoFit walkthrough (for OpenBind [first data release](https://openbind.uk/news/blog-affinity-and-kinetics-data-in-the-ev-a71-2a-openbind-release/))
| `01_explore.ipynb` | Data exploration and signal inspection |
| `02_fitting_demo.ipynb` | Single-sample fitting walkthrough |
| `03_concentration_error.ipynb` | Concentration error analysis |
| `04_batch_screening.ipynb` | Batch DK screening |
| `05_batch_ode.ipynb` | Full ODE batch fitting with plots |

## Fitting Approach

1. **Direct Kinetics**: Linearise the 1:1 Langmuir ODE as dR/dt = k₁·c·Rmax - (k₁·c + k₃)·R, solve by weighted least squares on the dissociation phase to get kd, then estimate ka from association kinetics.

2. **ODE Refinement**: Use DK estimates as seeds for multi-start `scipy.optimize.least_squares` (TRF) against the full numerical ODE solution. The number of random starts is controlled by `n_starts` (default 3; use 1 for fast screening, 10–20 for robust estimates). The fit window is trimmed to [Injection, RinseEnd + margin] to exclude baseline artefacts.

See [docs/fitting_approach.md](docs/fitting_approach.md) for details.
