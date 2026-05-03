# GeoPINONet

**Sparse-Supervised Physics-Informed Operator Learning for Full-Field Prediction in Variable 3D Elastic Domains**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19816527-blue)](https://doi.org/10.5281/zenodo.19816527)

GeoPINONet is a geometry-conditioned physics-informed neural operator for full-field displacement and stress prediction over variable 3D geometries in the linear-elastic regime. The framework combines PointNet-based geometry encoding, multiscale Fourier positional features, mixed displacement–stress prediction, and Active Critical Selective Validation (ActiveCSV) for sparse supervision in mechanically critical regions.

This repository accompanies the manuscript:

> **GeoPINONet: Sparse-Supervised Physics-Informed Operator Learning for Full-Field Prediction in Variable 3D Elastic Domains**
> Vicente Urrutia Valdés, Iván La Fé-Perdomo, Angel Rodríguez Soto — *Pontificia Universidad Católica de Valparaíso, Chile*

---

## Table of Contents

- [Main Features](#main-features)
- [Model Overview](#model-overview)
- [Training Strategy](#training-strategy)
- [Supported Geometry Families](#supported-geometry-families)
- [Repository Structure](#repository-structure)
- [Data, Models & Archive](#data-models--archive)
- [Installation](#installation)
- [Quick Repository Checks](#quick-repository-checks)
- [Running Lightweight Examples](#running-lightweight-examples)
- [Trained Models](#trained-models)
- [Real Inference Demo](#real-inference-demo)
- [Training Sanity Check](#training-sanity-check)
- [Full Dataset Structure](#full-dataset-structure)
- [Outputs](#outputs)
- [Reproducing Paper Tables](#reproducing-paper-tables)
- [CAD Automation](#cad-automation)
- [Data Generation](#data-generation)
- [Requirements](#requirements)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Main Features

- Geometry-aware neural operator based on **PointNet** geometry encoding and **Fourier positional features**
- **Dual-branch architecture** with independent branches per load case (axial compression / lateral bending)
- **Physics-informed training** with PDE equilibrium, constitutive (Hooke), and boundary-condition losses
- **Thickness-mode parameterization** for out-of-plane field reconstruction
- **ActiveCSV adaptive sampling** focused on von Mises stress concentration hotspots
- **Curriculum training** schedule from data-only fitting to full physics enforcement
- **Homoscedastic uncertainty** weighting for automatic loss balancing (`log_vars`)
- **Nested dataset subsets** (4 / 8 / 16 / 32 / 64) for reproducible scaling studies
- Automated **CAD-to-dataset** workflow via Autodesk Inventor iLogic and ANSYS

---

## Model Overview

GeoPINONet predicts the full 3D displacement field `(ux, uy, uz)` and the full Cauchy stress tensor `(sxx, syy, szz, sxy, syz, sxz)` for parametric structural geometries. The architecture is composed of three main blocks:

### 1. PointNetEncoder

Encodes the input STL point cloud into a fixed latent representation of the geometry, enabling the model to generalize across geometry families.

### 2. PositionalEncoder

Maps spatial coordinates into multi-scale random Fourier features, enabling the decoder to represent both smooth global deformations and high-frequency stress concentration fields.

### 3. PhysicsDecoder

Maps geometry-aware latent features and encoded coordinates into displacement and stress predictions using a thickness-aware parameterization.

Two independent branches are used, one per canonical load case:

| Branch | Load Case         |
|--------|-------------------|
| `COMP` | Axial compression |
| `LAT`  | Lateral bending   |

Combined load responses are obtained by linear superposition in the linear-elastic regime.

---

## Training Strategy

The training loss combines supervised FEM fitting with physics-informed regularization through a three-stage curriculum.

### Supervised Term

- FEM displacement and stress matching from ANSYS CSV data

### Physics Terms

- **PDE equilibrium residual:** `∇·σ = 0`
- **Hooke consistency loss**
- **Dirichlet BC loss** on the fixed support surface
- **Neumann BC loss** on the load surface

### Training Schedule

1. **Data-only phase** — supervised fitting only
2. **Linear ramp** — gradual introduction of physics terms
3. **Full coupled training** — data + physics jointly enforced

### Loss Balancing

All major loss terms are weighted automatically using learnable **homoscedastic uncertainty parameters** (`log_vars`), eliminating the need for manual tuning of physics weights.

### Adaptive Supervision

The `ActiveCSV` sampler (implemented in `src/active_sampler.py`) refreshes the supervised subset during training and increases sampling probability near **von Mises stress concentration regions**.

---

## Supported Geometry Families

### Plate with a Central Hole

| Parameter | Description   | Range        |
|-----------|---------------|--------------|
| `W`       | Plate width   | 100 – 180 mm |
| `H`       | Plate height  | 140 – 240 mm |
| `D`       | Hole diameter | 20 – 70 mm   |

> Thickness fixed at `T = 20 mm`. Minimum edge clearance enforced between the hole boundary and each plate edge.

### Lug (3D)

| Parameter | Description                 | Range     |
|-----------|-----------------------------|-----------|
| `W/D`     | Width-to-diameter ratio     | 2.0 – 3.0 |
| `e/D`     | Edge-to-diameter ratio      | 1.5 – 2.5 |
| `t/D`     | Thickness-to-diameter ratio | 0.4 – 0.8 |

> Pin-hole diameter fixed at `D = 20 mm`. Physical dimensions also stored as `W_mm`, `e_mm`, and `t_mm`.

---

## Repository Structure

```text
GeoPINONet/
├── cad_automation/          # Autodesk Inventor iLogic rules and export scripts
├── data_generation/         # LHS generation and nested geometry subsets
│   ├── Lug_3D/
│   └── Plate_with_a_hole/
├── examples/                # Lightweight example geometries and FEM CSV files
│   ├── Lug_3D/case_001/
│   └── Plate_with_a_hole/case_001/
├── results/                 # Reported metrics, ablation outputs, and reproduced tables
│   ├── ablation_study/
│   ├── demo_inference/
│   ├── final_model/
│   ├── reproduced_tables/
│   └── scaling_study/
├── scripts/                 # Reproducibility, smoke tests, and inference demos
├── src/                     # Main GeoPINONet implementation
├── trained_models/          # Model download utility and checkpoint placeholders
└── zenodo/                  # Zenodo archive documentation
```

The full FEM datasets and trained model weights are not stored in the GitHub repository due to file size. They are archived on Zenodo.

---

## Data, Models & Archive

The complete FEM datasets, trained model weights, archived source-code snapshot, and full result files are available through Zenodo:

**[https://doi.org/10.5281/zenodo.19816527](https://doi.org/10.5281/zenodo.19816527)**

The GitHub repository contains the source code, lightweight example cases, reproducibility scripts, LHS parameter files, CAD automation utilities, and result files.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/v-urrutiav/GeoPINONet.git
cd GeoPINONet
```

Create and activate a Python environment:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately according to your CUDA version:

```bash
# Example only — choose the wheel matching your CUDA setup
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> Python 3.11 is recommended.

---

## Quick Repository Checks

Before running training or inference, verify that all modules compile and import correctly:

```bash
python -m compileall src scripts
python scripts/smoke_imports.py
```

These commands confirm that source files are syntactically valid and that the main GeoPINONet modules can be imported without errors.

---

## Running Lightweight Examples

The `examples/` directory contains one lightweight case for each geometry family:

```text
examples/
├── Lug_3D/case_001/
│   ├── 1_body.stl
│   ├── 1_load.stl
│   ├── 1_fixed.stl
│   ├── 1_compression.csv
│   └── 1_lateral_bending.csv
└── Plate_with_a_hole/case_001/
    ├── 1_body.stl
    ├── 1_load.stl
    ├── 1_fixed.stl
    ├── 1_compression.csv
    └── 1_lateral_bending.csv
```

Run the example checks with:

```bash
python scripts/run_example_lug.py
python scripts/run_example_plate.py
```

These scripts verify that the example STL and FEM CSV files can be loaded and processed correctly by the GeoPINONet data pipeline.

---

## Trained Models

The trained model checkpoints are available in the Zenodo archive:

**[https://doi.org/10.5281/zenodo.19816527](https://doi.org/10.5281/zenodo.19816527)**

After downloading, place the checkpoints as follows:

```text
trained_models/
├── Lug_3D/
│   └── 64_lug_final.pth
└── Plate_with_a_hole/
    └── 64_hole_final.pth
```

Alternatively, use the download utility:

```bash
python trained_models/download_models.py
```

---

## Real Inference Demo

A real inference demo is provided for the 3D lug geometry. It loads the trained checkpoint, samples surface points, predicts the canonical load responses, applies linear superposition, computes von Mises stress, and writes a PNG visualization.

Run with a single load case:

```bash
python scripts/demo_inference_lug.py --f-comp 1000 --f-lat 0
```

Or with a combined load:

```bash
python scripts/demo_inference_lug.py --f-comp 700 --f-lat 300
```

Output is written to:

```text
results/demo_inference/Lug_3D/
```

---

## Training Sanity Check

The main training loop is implemented in `src/train.py`. The active geometry and domain configuration is controlled through `src/domain.py`.

For lightweight sanity testing, configure `domain.py` to use the example 3D lug case, then run:

```bash
python -m src.train
```

This verifies that the full training pipeline runs end-to-end with a small example case. Full training requires the complete FEM datasets available through Zenodo.

Edit `src/config.py` to define:

- Data paths
- Physical constants (Young's modulus, Poisson's ratio)
- Hyperparameters (learning rate, batch size, epochs)
- Training budget and device configuration

Edit `src/domain.py` to select the active subset or geometry family:

```python
ACTIVE_SET = SUBSET_16   # choose from SUBSET_4 / 8 / 16 / 32 / 64
```

---

## Full Dataset Structure

The complete datasets archived on Zenodo use the same naming convention as the repository. Each geometry case contains STL boundary files and FEM CSV solutions:

```text
{i}_body.stl
{i}_load.stl
{i}_fixed.stl
{i}_compression.csv
{i}_lateral_bending.csv
```

**Expected CSV columns:**

```text
x, y, z, ux, uy, uz, sxx, syy, szz, sxy, syz, sxz
```

---

## Outputs

Training generates checkpoints, logs, ActiveCSV diagnostic plots, and metric
files in the configured output path.

| File / Pattern | Description |
|---|---|
| `geo_pino_model.pth` | Trained model checkpoint |
| `training_log.csv` | Per-epoch loss history |
| `scan_log.csv` | ActiveCSV scan metrics |
| `squad_*_epoch*.pdf` | ActiveCSV critical/dynamic anchor visualization |

Additional experiment outputs are organized under `results/`:

| Folder | Contents |
|---|---|
| `ablation_study/` | Sensitivity analyses and component-wise evaluations |
| `scaling_study/` | Performance trends across nested training subsets |
| `final_model/` | Final trained-model metrics for each geometry family |
| `reproduced_tables/` | LaTeX and CSV tables reproduced from paper metrics |
| `demo_inference/` | PNG and CSV outputs from the inference demo |

---

## Reproducing Paper Tables

The summary tables reported in the manuscript can be regenerated from the raw per-geometry metric files stored in `results/`:

```bash
python scripts/reproduce_tables.py
```

Reproduced tables are written to:

```text
results/reproduced_tables/
```

This includes final-model tables (training and validation, for both geometry families), the compact ablation table, and the compact comparison table between full 3D von Mises and in-plane von Mises metrics.

---

## CAD Automation

The `cad_automation/` folder contains the automation workflow used to generate CAD exports for batch FEM preprocessing:

- `iLogic_Lug_3D_rule.txt` / `iLogic_Plate_rule.txt` — Autodesk Inventor iLogic rules
- `Macro_Inventor.mrf` — macro-based export automation
- `collect_stl_files.py` / `collect_step_file.py` — STL and STEP collection and renaming scripts

This pipeline reads parametric values from CSV files, updates the CAD model, and exports the body STL, load STL, fixed STL, and STEP geometry for each case. All outputs are stored in indexed case folders for downstream ANSYS simulation.

> These scripts are provided for reproducibility and may require local path configuration depending on the Autodesk Inventor installation.

---

## Data Generation

The `data_generation/` folder contains scripts and CSV files used to generate the training and validation datasets for both geometry families.

### Sampling Strategy

- A **64-sample master dataset** is generated using **Latin Hypercube Sampling (LHS)**.
- Nested subsets of size 4, 8, 16, and 32 are extracted using **Farthest Point Sampling (FPS)**, so each smaller subset is strictly contained in the larger ones.
- This design supports reproducible scaling studies.

### Validation Split

For each geometry family, the validation set contains **16 samples**:

- 12 inside the convex hull of the training set (interpolation)
- 4 outside the training domain (mild extrapolation)

### Regenerating the Datasets

```bash
cd data_generation/Plate_with_a_hole
python plate_hole_lhs_generation.py

cd ../Lug_3D
python lug_lhs_generation.py

cd ..
python scaling_study.py
```

---

## Requirements

Main dependencies are listed in `requirements.txt`. Core packages used in this repository include:

- `torch` — neural network framework (install separately with the correct CUDA wheel)
- `numpy` — numerical computing
- `pandas` — data handling and CSV processing
- `matplotlib` — visualization
- `trimesh` — STL mesh loading and surface sampling
- `rtree` — spatial indexing
- `openpyxl` — Excel output support
- `scipy` — LHS and FPS sampling utilities

Install all non-PyTorch dependencies with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use GeoPINONet in your research, please cite the manuscript and the Zenodo archive:

```bibtex
@misc{urrutia2026geopinonet_archive,
  author       = {Urrutia Vald{\'e}s, Vicente, La F{\'e}-Perdomo, Iv{\'a}n and Rodríguez Soto, Angel},
  title        = {GeoPINONet datasets, trained models, and source code for geometry-conditioned 3D elasticity},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19816527},
  url          = {https://doi.org/10.5281/zenodo.19816527}
}
```

A `CITATION.cff` file is also provided in the root of the repository for automatic citation metadata.

---

## Authors

- **Vicente Urrutia Valdés**
  ORCID: [0009-0001-6349-5747](https://orcid.org/0009-0001-6349-5747)

- **Iván La Fé-Perdomo**
  ORCID: [0000-0002-4042-1534](https://orcid.org/0000-0002-4042-1534)

- **Angel Rodríguez Soto**
  ORCID: [0000-0002-5512-9288](https://orcid.org/0000-0002-5512-9288)

*Pontificia Universidad Católica de Valparaíso, Chile*

---

## License

The source code is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

You are free to **Share** (copy and redistribute) and **Adapt** (remix, transform, and build upon) the material for any purpose, even commercially, provided appropriate **Attribution** is given.

Datasets, trained models, and result archives are provided through Zenodo under the license specified in the Zenodo record.

See the [LICENSE](LICENSE) file for the full legal text.
