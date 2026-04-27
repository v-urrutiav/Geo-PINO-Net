# Trained models

This folder is reserved for the trained GeoPINONet model checkpoints.

The checkpoint files are not stored directly in the GitHub repository due to file
size. They are archived on Zenodo:

**https://doi.org/10.5281/zenodo.19816527**

---

## Expected structure

After downloading the trained model archive, this folder should have the
following structure:

```text
trained_models/
├── download_models.py
├── README.md
├── Lug_3D/
│   └── 64_lug_final.pth
└── Plate_with_a_hole/
    └── 64_hole_final.pth
```

---

## Downloading the checkpoints

From the repository root, run:

```bash
python trained_models/download_models.py
```

This downloads the trained model archive from Zenodo and installs the checkpoint
files into the expected folders.

To force a new download and overwrite existing checkpoint files:

```bash
python trained_models/download_models.py --force
```

To remove the downloaded ZIP archive after extraction:

```bash
python trained_models/download_models.py --clean
```

---

## Checkpoint descriptions

### `Lug_3D/64_lug_final.pth`

Final GeoPINONet checkpoint trained on the 3D lug family using 64 training
geometries.

### `Plate_with_a_hole/64_hole_final.pth`

Final GeoPINONet checkpoint trained on the plate-with-hole family using 64
training geometries.

---

## Usage

The checkpoints are used by the inference and example scripts, for example:

```bash
python scripts/demo_inference_lug.py --f-comp 1000 --f-lat 0
```

The scripts expect the checkpoint files to be located in the structure shown
above.

---

## Notes

- Do not commit `.pth`, `.pt`, or `.ckpt` files to GitHub.
- Keep trained model weights in the Zenodo archive.
- The GitHub repository contains only this README and the download utility.

---

## Citation

If you use these trained models, please cite the Zenodo archive and the
associated manuscript:

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
