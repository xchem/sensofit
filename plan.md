# Creoptix GCI Kinetic Fitting — Project Plan

## Goal

Build a Python library to fit 1:1 Langmuir binding kinetics from
Creoptix WAVE/WAVEsystem GCI data (.cxw files), suitable for
high-throughput fragment screening campaigns.

## Architecture

```
creoptix_fitting/
    __init__.py          # Public API: load_cxw, batch_fit, flag_poor_fits
    data_loader.py       # .cxw (ZIP/XML/HDF5) parser
    models.py            # c(t) construction, double referencing, ODE, weight masks
    direct_kinetics.py   # Closed-form linear solver (DK) for initial estimates
    ode_fitting.py       # ODE refinement with multi-start TRF
    batch.py             # Batch fitting of all samples in an experiment
```

## Fitting Pipeline

1. **Data Loading** (`data_loader.py`)
   - Parse .cxw ZIP → XML metadata + HDF5 signal data
   - Auto-detect active/reference flow cells from XML
   - Extract cycle metadata (markers, reagents, concentrations)
   - Reference-subtract (active − reference channel)

2. **Preprocessing** (`models.py`)
   - Double referencing: subtract nearest preceding blank
   - Build c(t) from DMSO calibration (envelope for DK, pulsed for ODE)
   - Smooth signal via UnivariateSpline; compute dR/dt analytically
   - Build weight masks (dissociation-only for DK; buffer-pulse + dissociation for ODE)
   - Non-specific binder detection from reference channel dissociation

3. **Direct Kinetics** (`direct_kinetics.py`)
   - Reparameterize Langmuir ODE as linear system: X·k + b = 0
   - Closed-form solve with optional Tikhonov regularisation
   - Estimate ka from kobs (association half-time) or steady-state heuristic
   - Provides deterministic initial estimates in ~ms per sample

4. **ODE Refinement** (`ode_fitting.py`)
   - Three-phase approach:
     1. Closed-form (R0, Rss) with kd fixed from DK
     2. Derive ka from steady-state at dissociation onset
     3. Multi-start TRF optimisation for (ka, Rmax) with kd fixed
   - Uses pulsed c(t) and full weight mask (buffer pulses + dissociation)
   - Median aggregation over converged starts for robustness

5. **Batch Processing** (`batch.py`)
   - Fit all samples in a .cxw file (DK-only or DK→ODE modes)
   - Flag poor fits: boundary hits, high residuals, NSB, failures
   - Returns DataFrame with kinetic parameters + QC flags

## Notebooks

- `01_explore.ipynb` — Data exploration and signal visualisation
- `02_fitting_demo.ipynb` — Walk-through of the full fitting pipeline
- `03_concentration_error.ipynb` — KD sensitivity to concentration errors
- `04_batch_screening.ipynb` — Batch fit + hit ranking

## Key Decisions

- **kd from DK, ka/Rmax from ODE**: kd is well-determined from
  dissociation-only data (c=0). ka and Rmax require the full
  association/dissociation profile and benefit from ODE refinement.
- **Pulsed c(t)**: Creoptix uses alternating analyte/buffer pulses.
  The ODE fitter uses the raw pulsed c(t) and weights only buffer-pulse
  intervals during association, avoiding RI bulk artefacts.
- **Multi-start median**: ODE cost surface can be multimodal; median
  aggregation over multiple converged starts is more robust than
  single-seed optimisation.
- **Non-specific binder detection**: Reference channel dissociation
  signal > 2 pg/mm² flags compounds retained on the unmodified surface.

## Data Format

The .cxw file is a ZIP archive containing:
- `_project.cx3`: XML with cycle metadata, flow cell config, markers
- `cyclesData.h5`: HDF5 with raw channel data (keyed by cycle GUID)
- `Wizard/*.cx3`: XML with reagent/sample definitions per serie
- `Evaluations/`, `Correctors/`: additional XML (not used)

## Dependencies

- numpy, scipy, pandas, h5py, matplotlib
- Standard library: zipfile, io, xml.etree.ElementTree, re
