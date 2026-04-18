# GeoPINONet
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
**GeoPINONet** is a geometry-aware **Physics-Informed Neural Operator (PINO)** for **3D structural mechanics**.  
It predicts full-field **displacements** and **stresses** on parametric 3D geometries under two load cases:

- **Axial compression**
- **Lateral bending**

The model is trained using **ANSYS FEM supervision** together with **physics-based constraints**, including:

- Equilibrium residuals
- Constitutive consistency through Hooke's law
- Dirichlet boundary conditions
- Neumann boundary conditions

GeoPINONet is designed as a surrogate modeling framework for structural analysis across geometry families, with emphasis on **stress concentration regions**, **scaling studies**, and **geometry-aware generalization**.

---

## Table of Contents

- [Main Features](#main-features)
- [Model Overview](#model-overview)
- [Training Strategy](#training-strategy)
- [Supported Geometry Families](#supported-geometry-families)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Training](#training)
- [Outputs](#outputs)
- [Data Generation](#data-generation)
- [CAD Automation](#cad-automation)
- [Notebook](#notebook)
- [Results](#results)
- [Requirements](#requirements)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Main Features

- Geometry-aware neural operator based on **PointNet** geometry encoding and **Fourier positional features**
- **Dual-branch architecture** with independent branches for each load case
- **Physics-informed training** with PDE, constitutive, and boundary-condition losses
- **Thickness-mode parameterization** for out-of-plane field reconstruction
- **ActiveCSV adaptive sampling** focused on Von Mises stress hotspots
- **Curriculum training** schedule from data-only fitting to full physics enforcement
- **Nested dataset subsets** for scaling studies (4 / 8 / 16 / 32 / 64)
- Automated **CAD-to-dataset** workflow for reproducible simulation preprocessing

---

## Model Overview

GeoPINONet predicts the full 3D displacement field `(UX, UY, UZ)` and the full Cauchy stress tensor `(Sxx, Syy, Szz, Sxy, Syz, Sxz)` for parametric structural geometries.

The architecture is composed of three main blocks:

### 1. PointNetEncoder

Encodes the input STL point cloud into a fixed latent representation of the geometry.

### 2. PositionalEncoder

Maps spatial coordinates into multi-scale random Fourier features, enabling the decoder to represent both smooth and high-frequency field variations.

### 3. PhysicsDecoder

Maps geometry-aware latent features and encoded coordinates into displacement and stress predictions using a thickness-aware parameterization.

Two independent branches are used, one per load case:

| Branch | Load Case         |
|--------|-------------------|
| `COMP` | Axial compression |
| `LAT`  | Lateral bending   |

---

## Training Strategy

The training loss combines supervised FEM fitting with physics-informed regularization.

### Supervised Term

- FEM displacement and stress matching from ANSYS CSV data

### Physics Terms

- **PDE equilibrium residual:** `∇·σ = 0`
- **Hooke consistency loss**
- **Dirichlet BC loss** on the fixed support surface
- **Neumann BC loss** on the load surface

### Training Schedule

A curriculum strategy is used in three stages:

1. **Data-only phase** — supervised fitting only
2. **Linear ramp** — gradual introduction of physics terms
3. **Full coupled training** — data + physics jointly

### Loss Balancing

All major loss terms are weighted automatically using learnable **homoscedastic uncertainty parameters** (`log_vars`).

### Adaptive Supervision

The `ActiveCSV` sampler refreshes the supervised subset during training and increases sampling probability near **Von Mises stress concentration regions**.

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
├── cad_automation/
├── data_generation/
│   ├── Lug_3D/
│   └── Plate_with_a_hole/
├── notebooks/
├── results/
│   ├── ablation study/
│   ├── final model/
│   │   ├── lug 3d/
│   │   └── plate with a hole/
│   └── scaling study/
└── src/
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/v-urrutiav/Geo-PINO-Net.git
cd GeoPINONet
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately according to your CUDA version:

```bash
# Example only — choose the wheel matching your CUDA setup
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> Python 3.11 is recommended.

---

## Data Format

Each geometry case requires the following files inside the path specified in `src/config.py`:

| File                      | Description                        |
|---------------------------|------------------------------------|
| `{i}_body.stl`            | Full body mesh                     |
| `{i}_load.stl`            | Load surface mesh                  |
| `{i}_fixed.stl`           | Fixed support surface mesh         |
| `{i}_compression.csv`     | FEM solution for axial compression |
| `{i}_lateral_bending.csv` | FEM solution for lateral bending   |

**Expected CSV columns:**

```
X, Y, Z, UX, UY, UZ, Sxx, Syy, Szz, Sxy, Syz, Sxz
```

---

## Configuration

Edit `src/config.py` to define:

- Paths
- Physical constants
- Hyperparameters
- Training budget
- Device configuration

Edit `src/domain.py` to select the active subset or geometry family:

```python
ACTIVE_SET = SUBSET_16   # choose from SUBSET_4 / 8 / 16 / 32 / 64
```

---

## Training

Run training from the `src` folder:

```bash
cd src
python train.py
```

> If `geopino_checkpoint.pth` already exists, training resumes automatically from the latest saved state.

---

## Outputs

Training generates model checkpoints, logs, and diagnostic figures in the configured output path:

| File                     | Description                           |
|--------------------------|---------------------------------------|
| `geopino_checkpoint.pth` | Resume checkpoint                     |
| `geopino_model.pth`      | Final trained model                   |
| `training_log.csv`       | Per-epoch loss history                |
| `scan_log.csv`           | Per-geometry scan metrics             |
| `training_curves.pdf`    | Global loss evolution                 |
| `scatter_geo*.pdf`       | FEM vs prediction scatter plots       |
| `sidebyside_*.pdf`       | FEM / prediction / error comparisons  |
| `vonmises_iso_*.pdf`     | Von Mises iso-view comparisons        |

Additional experiment outputs are organized under `results/`:

```text
results/
├── ablation study/
├── final model/
│   ├── lug 3d/
│   └── plate with a hole/
└── scaling study/
```

---

## Data Generation

The `data_generation/` folder contains scripts and CSV files used to generate the training and validation datasets for both geometry families.

### Sampling Strategy

- A **64-sample master dataset** is generated using **Latin Hypercube Sampling (LHS)** with centered-discrepancy optimization.
- Nested subsets of size 4, 8, 16, and 32 are extracted using **Farthest Point Sampling (FPS)**, so each smaller subset is contained in the larger ones.
- This supports reproducible scaling studies.

### Validation Split

For each geometry family, the validation set contains **16 samples**:

- 12 inside the convex hull of the training set
- 4 outside the training domain for mild extrapolation assessment

### Regeneration

```bash
cd data_generation/Plate_with_a_hole
python plate_hole_lhs_generation.py

cd ../Lug_3D
python lug_lhs_generation.py

cd ..
python scaling_study.py
```

---

## CAD Automation

The `cad_automation/` folder contains the automation workflow used to generate CAD exports for batch FEM preprocessing. It includes:

- Autodesk Inventor iLogic rules
- Macro-based export automation
- Python scripts for STL and STEP collection and renaming

This pipeline reads parametric values from CSV files, updates the CAD model, and exports:

- Body STL
- Load STL
- Fixed STL
- STEP geometry

All outputs are stored into indexed case folders for later ANSYS simulation.

---

## Notebook

The `notebooks/` folder contains the Colab-oriented training and evaluation workflow for GeoPINONet. It is intended for:

- Rapid experimentation
- Model debugging
- Visualization
- GPU-based Colab execution

> The notebook was designed for **Google Colab** and is best used with a GPU runtime.

---

## Results

The `results/` folder stores the main outputs used throughout the study:

| Folder            | Contents                                             |
|-------------------|------------------------------------------------------|
| `ablation study/` | Sensitivity analyses and component-wise evaluations  |
| `scaling study/`  | Performance trends across nested training subsets    |
| `final model/`    | Final trained-model outputs for each geometry family |

---

## Requirements

Main dependencies are listed in `requirements.txt`. Typical packages used in this repository include:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `trimesh`
- `rtree`
- `openpyxl`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

This repository includes:

- Dataset generation scripts
- Pre-generated training and validation CSV files
- CAD automation scripts
- Notebook-based experiments
- Training source code
- Structured result folders for scaling, ablation, and final models

The goal is to make the full GeoPINONet pipeline reproducible, from parametric geometry generation to surrogate model training and evaluation.

---

## Citation

If you use GeoPINONet in your research, please cite:

```bibtex
@article{urrutia2026geopinonet,
  author  = {Urrutia, Vicente and others},
  title   = {GeoPINONet: A Physics-Informed Neural Operator for 3D Structural Mechanics},
  journal = {Under review},
  year    = {2026},
  doi     = {DOI pending}
}
```

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License. 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

You are free to:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution**: You must give appropriate credit (citing the paper and this repository), provide a link to the license, and indicate if changes were made.

See the [LICENSE](LICENSE) file for the full legal text.
