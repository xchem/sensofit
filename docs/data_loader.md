# Data Loader — Detailed Documentation

## Overview

`creoptix_fitting.data_loader.load_cxw()` parses Creoptix WAVE .cxw
experiment files and returns structured Python dicts with all cycle
metadata and signal data ready for fitting.

---

## 1. CXW File Format

A `.cxw` file is a **ZIP archive** containing:

```
myexperiment.cxw (ZIP)
├── _project.cx3           # XML: cycle metadata, flow cell config, markers
├── cyclesData.h5          # HDF5: raw channel data (keyed by cycle GUID)
├── Wizard/
│   └── *.cx3              # XML: reagent/sample definitions per serie
├── Evaluations/           # XML: Creoptix's own fit results (not used)
└── Correctors/            # XML: timing/offset corrections (not used)
```

### 1a. _project.cx3 (XML)

The master metadata file. Key structure:

```xml
<Project>
  <Series>
    <Serie>
      <Type>Immobilization | Pulse | ...</Type>
      <Guid>...</Guid>
      <Cycles>
        <Cycle>
          <Index>5</Index>
          <CycleType>Sample | Blank | DMSO Cal. | ControlSample</CycleType>
          <Guid>abc-123-...</Guid>
          <Name>...</Name>
          <FlowPath2>
            <FlowCells>
              <Selected>true</Selected>
              <FlowCell><Designation>FC2</Designation></FlowCell>
            </FlowCells>
          </FlowPath2>
          <Markers>
            <Marker>
              <Type>Injection</Type>
              <X>20.5</X>     <!-- time in seconds -->
            </Marker>
            <Marker>
              <Type>Rinse</Type>
              <X>85.3</X>
            </Marker>
            <Marker>
              <Type>RinseEnd</Type>
              <X>155.7</X>
            </Marker>
          </Markers>
          <AutosamplerLocation Slot="A1"/>
        </Cycle>
      </Cycles>
    </Serie>
  </Series>
</Project>
```

### 1b. cyclesData.h5 (HDF5)

Raw signal data keyed by `{GUID}.{channel}`:

| Key pattern     | Content                          |
|-----------------|----------------------------------|
| `{GUID}.0`      | Time array (seconds)             |
| `{GUID}.7`      | FC1 signal (reference, if FC1)   |
| `{GUID}.9`      | FC2 signal (active, if FC2)      |
| `{GUID}.11`     | FC3 signal                       |
| `{GUID}.13`     | FC4 signal                       |

Channel mapping: FC *n* → HDF5 channel `5 + 2n`
(FC1→7, FC2→9, FC3→11, FC4→13).

### 1c. Wizard/*.cx3 (XML)

Per-serie reagent definitions with slot-to-compound mapping:

```xml
<Wizard>
  <SerieId>...</SerieId>
  <Reagent Slot="A1" Designation="ASAP-0044216"
           Concentration="25.000 µM" MW="455 Da"/>
</Wizard>
```

---

## 2. Loading Pipeline

### Step 1: Open ZIP and Parse XML

```python
data = load_cxw('experiment.cxw')
```

1. Open `.cxw` as ZIP
2. Parse `_project.cx3` → ElementTree root
3. Clean XML: strip BOM, remove namespace prefixes that break ET

### Step 2: Detect Flow Cell Configuration

The loader automatically determines which flow cells are active (ligand)
and reference (unmodified) by examining the XML:

1. Find the **Capture** cycle in the **Immobilization** serie → this is
   the active flow cell (where ligand was immobilised)
2. Find which FCs are used in **Pulse** (RAPID Kinetics) sample cycles
3. The non-capture FC is the reference

Typical result: FC2 = active (channel 9), FC1 = reference (channel 7).

### Step 3: Build Reagent Lookup

Parse Wizard XML to build a slot → compound/concentration/MW mapping.
This links autosampler slot positions to chemical identity.

### Step 4: Extract Cycle Metadata

For each cycle in the RAPID Kinetics serie, extract:
- `index`: Cycle number (ordering within the serie)
- `cycle_type`: Sample, ControlSample, Blank, or DMSO Cal.
- `guid`: Unique identifier (maps to HDF5 keys)
- `compound`, `concentration_M`, `mw`: From Wizard lookup
- `markers`: Dict of named time points (Injection, Rinse, RinseEnd)

### Step 5: Load HDF5 Signal Data

For each cycle of interest (Sample, ControlSample, Blank, DMSO Cal.):
1. Read time array from `{GUID}.0`
2. Read active channel from `{GUID}.{active_ch}`
3. Read reference channel from `{GUID}.{ref_ch}`
4. Compute reference-subtracted signal: `signal = raw_active - raw_reference`

---

## 3. Output Structure

```python
data = load_cxw('experiment.cxw')

data['config']
# {
#     'active_fc': 2,
#     'reference_fc': 1,
#     'active_channel': 9,
#     'reference_channel': 7,
# }

data['samples']      # List of sample/control dicts
data['dmso_cals']    # List of DMSO calibration dicts
data['blanks']       # List of blank dicts
data['all_cycles']   # Full ordered cycle metadata (no signal data)
```

Each sample/blank/dmso dict contains:

| Key              | Type         | Description                        |
|------------------|--------------|------------------------------------|
| `index`          | int          | Cycle index                        |
| `cycle_type`     | str          | 'Sample', 'Blank', etc.            |
| `guid`           | str          | HDF5 key prefix                    |
| `compound`       | str          | Compound name from Wizard          |
| `concentration_M`| float        | Injected concentration (M)         |
| `mw`             | float        | Molecular weight (Da)              |
| `markers`        | dict         | {Injection, Rinse, RinseEnd: time} |
| `time`           | np.ndarray   | Time array (seconds)               |
| `signal`         | np.ndarray   | Reference-subtracted signal        |
| `raw_active`     | np.ndarray   | Raw active channel signal          |
| `raw_reference`  | np.ndarray   | Raw reference channel signal       |

---

## 4. Experiment Structure (RAPID Kinetics)

A typical RAPID Kinetics experiment has this cycle ordering:

```
Blank → DMSO Cal → Sample → Sample → Sample → Blank → DMSO Cal → ...
```

Each sample sensorgram has three phases:

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│  Baseline │    Association (pulsed)    │ Dissociation │
│  (buffer) │ ┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐     │  (buffer)    │
│           │ │└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘│     │              │
│───────────┘                         └──│──────────    │
│                                        │              │
│  t=0     Injection              Rinse    RinseEnd     │
└───────────────────────────────────────────────────────┘
```

- **Baseline**: Buffer flowing, no analyte. Establishes zero level.
- **Association**: Analyte injected in rapid alternating pulses with buffer.
  Signal rises as analyte binds to ligand.
- **Dissociation**: Buffer-only rinse. Signal decays exponentially as
  analyte dissociates from ligand.

---

## 5. XML Cleaning

The .cx3 XML files contain artefacts that break Python's ElementTree:
- UTF-8 BOM (`\ufeff`)
- Versioned namespace declarations (`xmlns:version="..."`)
- Prefixed attributes (`p1:type="..."`)

The `_clean_xml()` helper strips these before parsing.

---

## 6. Concentration Parsing

Reagent concentrations in Wizard XML use various formats:
- `"25.000 µM"` → 25e-6 M
- `"500.000 nM"` → 5e-7 M
- `"1.000 mM"` → 1e-3 M

Supported units: mM, µM/μM/uM, nM, pM, M.

Molecular weights: `"455 Da"` → 455.0
