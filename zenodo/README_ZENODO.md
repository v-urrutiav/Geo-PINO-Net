# GeoPINONet Zenodo archive

This Zenodo record contains the complete datasets, trained model weights,
archived source-code snapshot, and result files associated with the manuscript:

**GeoPINONet: A Geometry-Conditioned Physics-Informed Neural Operator for 3D Elasticity with Real-Time Inference**

DOI: https://doi.org/10.5281/zenodo.19816527

---

## Contents

This archive is organized into separate compressed files:

- `GeoPINONet_Lug_3D_dataset_v1.zip`  
  Complete 3D lug finite-element dataset, including training and validation geometries.

- `GeoPINONet_Plate_with_a_hole_dataset_v1.zip`  
  Complete plate-with-hole finite-element dataset, including training and validation geometries.

- `GeoPINONet_trained_models_v1.zip`  
  Final trained GeoPINONet model checkpoints for both geometry families.

- `GeoPINONet_results_v1.zip`  
  Final-model metrics, validation metrics, scaling-study results, ablation-study outputs,
  reproduced tables, and training logs.

- `Geo-PINO-Net_source_v1.zip`  
  Snapshot of the public source-code repository at the time of manuscript submission.

---

## Dataset structure

The complete dataset archives follow the same naming convention used by the
source-code repository.

Each geometry case contains:

```text
{i}_body.stl
{i}_load.stl
{i}_fixed.stl
{i}_compression.csv
{i}_lateral_bending.csv
```

where `{i}` is the geometry index.

The STL files define:

- `body`: complete geometry.
- `load`: loaded boundary surface.
- `fixed`: fixed boundary surface.

The CSV files contain the ANSYS finite-element solutions for the two canonical
load cases.

---

## FEM CSV columns

Each FEM CSV file contains nodal coordinates, displacement components, and
Cauchy stress components:

```text
x, y, z, ux, uy, uz, sxx, syy, szz, sxy, syz, sxz
```

The expected stress-component order is:

```text
sxx, syy, szz, sxy, syz, sxz
```

---

## Load cases

Two canonical load modes are included:

- `compression`: axial compression load case.
- `lateral_bending`: lateral bending load case.

In the linear-elastic regime, combined load responses can be obtained by
linear superposition of the two canonical model predictions.

---

## Trained model checkpoints

The trained model archive should be extracted into the repository as:

```text
trained_models/
├── Lug_3D/
│   └── 64_lug_final.pth
└── Plate_with_a_hole/
    └── 64_hole_final.pth
```

These checkpoints are used by the inference demo scripts in the GitHub
repository.

---

## Result files

The result archive includes:

```text
results/
├── ablation_study/
├── final_model/
├── reproduced_tables/
└── scaling_study/
```

These files support reproduction of the main tables reported in the manuscript.
The table-generation script in the source repository reads the raw metric files
and writes LaTeX tables to `results/reproduced_tables/`.

---

## Source code

The development repository is available at:

https://github.com/v-urrutiav/GeoPINONet

The archived source-code snapshot included in this Zenodo record corresponds to
the repository state used for manuscript submission.

---

## Basic usage

After cloning the GitHub repository and installing the dependencies, the
lightweight checks can be run with:

```bash
python -m compileall src scripts
python scripts/smoke_imports.py
python scripts/run_example_lug.py
python scripts/run_example_plate.py
```

After downloading and placing the trained model checkpoints, a real inference
demo can be run with:

```bash
python scripts/demo_inference_lug.py --f-comp 1000 --f-lat 0
```

Paper tables can be regenerated with:

```bash
python scripts/reproduce_tables.py
```

---

## Citation

If you use this archive, please cite this Zenodo record and the associated
manuscript.

```bibtex
@misc{urrutia2026geopinonet_archive,
  author       = {Urrutia Vald{\'e}s, Vicente and La F{\'e}-Perdomo, Iv{\'a}n},
  title        = {GeoPINONet datasets, trained models, and source code for geometry-conditioned 3D elasticity},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19816527},
  url          = {https://doi.org/10.5281/zenodo.19816527}
}
```

---

## Authors

- Vicente Urrutia Valdés  
  ORCID: https://orcid.org/0009-0001-6349-5747

- Iván La Fé-Perdomo
  ORCID: https://orcid.org/0000-0002-4042-1534


Pontificia Universidad Católica de Valparaíso, Chile.

---

## License

The dataset, trained model weights, source-code snapshot, and result files are
provided under the license specified in the Zenodo record.
