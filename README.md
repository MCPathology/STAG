# STAG: Biologically Guided Spatial Transcriptomics Prediction via Hypergraph Learning

> Under review. Dataset download links will be made publicly available upon
> acceptance of the manuscript.

STAG predicts spatial gene expression from histopathology images. This release
contains two training pipelines:

- `2D/`: spot-level prediction from single-section H&E image patches.
- `3D/`: pseudo-3D prediction from preprocessed serial-section graph data.

Detailed instructions are provided in [`2D/README.md`](2D/README.md) and
[`3D/README.md`](3D/README.md). Full documentation is available in
[`DOCS.md`](DOCS.md).

## Repository Layout

```text
STAG/
|-- 2D/                 # 2D STAG training, datasets, models, and ablations
|-- 3D/                 # pseudo-3D STAG training and serial-section datasets
|-- DOCS.md             # extended documentation
`-- README.md
```

## Installation

Install dependencies inside the subfolder you want to run:

```bash
# 2D experiments
cd 2D
pip install -r requirements.txt

# 3D experiments
cd ../3D
pip install -r requirements.txt
```

The self-supervised ResNet18 backbone weight is expected at
`weights/tenpercent_resnet18.ckpt`. If it is missing, the code downloads it on
the first run.

## 2D Data

Place the 2D datasets under `2D/data/`:

```text
2D/data/
|-- GSE144240/                                               # cSCC
|-- HER2/                                                    # HER2ST
|-- Human_breast_cancer_in_situ_capturing_transcriptomics/  # HBC
`-- Hest1k_datasets/                                         # HEST-1k subsets
```

Expected files:

```text
GSE144240/
|-- *.jpg
|-- *_stdata.tsv
`-- *_spot_data-selection-P*.tsv

HER2/
|-- images/HE/*.jpg
|-- count-matrices/*.tsv
`-- spot-selection/*_selection.tsv

Human_breast_cancer_in_situ_capturing_transcriptomics/
|-- *.jpg
|-- *_stdata.tsv
`-- spots_*.csv

Hest1k_datasets/<subset>/
|-- st/*.h5ad
`-- wsis/*.tif
```

Gene panels and gene-text embeddings are stored in `2D/select_genes/` and are
loaded automatically by the training scripts.

## 2D Splits

All 2D splits are sample-level splits. Spots from the same WSI/sample are never
split across train and validation folds.

| Dataset family | Split unit | Split implementation | Default seed |
|---|---|---|---:|
| cSCC (`GSE144240`) | `.jpg` WSI files under `GSE144240/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HER2ST (`HER2`) | `.jpg` WSI files under `HER2/images/HE/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HBC | `.jpg` WSI files under the HBC folder | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HEST-1k subsets | sample IDs from `Hest1k_datasets/<subset>/st/*.h5ad` | `KFold(..., shuffle=True)` over sorted sample IDs | 1553 |

The generated split JSON records the exact train/validation files and is saved
under the run output directory. Reusing the same `--k_folds` and `--seed` reloads
the same split file.

## 2D Training

Run commands from the `2D/` directory:

```bash
cd 2D

# cSCC, 4-fold CV, 50 epochs
python train_STAG.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 8

# HER2ST, 6-fold CV, 50 epochs
python train_STAG.py --data_name HER2 --k_folds 6 --epochs 50 --batch_size 8

# HBC, 9-fold CV, 50 epochs
python train_STAG.py --data_name HBC --k_folds 9 --epochs 50 --batch_size 8

# HEST-PRAD, 6-fold CV, 50 epochs
python train_STAG.py --data_name HEST_PRAD --k_folds 6 --epochs 50 --batch_size 8
```

Recommended release settings:

| Dataset | `--data_name` | Folds | Epochs | Batch size |
|---|---:|---:|---:|---:|
| cSCC | `cSCC` | 4 | 50 | 8 |
| HER2ST | `HER2` | 6 | 50 | 8 |
| HBC | `HBC` | 9 | 50 | 8 |
| HEST-PRAD | `HEST_PRAD` | 6 | 50 | 8 |
| HEST-kidney | `HEST_kidney` | 6 | 50 | 8 |
| HEST-mouse-brain | `HEST_mouse_brain` | 5 | 50 | 8 |

For the no-text variant:

```bash
python train_STAG_notext.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 16
```

For HVG experiments:

```bash
python train_STAG_hvg.py --data_name HER2 --k_folds 6 --epochs 80 --batch_size 8 --gene_mode hvg
```

## 3D Data

The 3D pipeline uses preprocessed serial-section data, not raw WSI folders. Each
config file in `3D/config/` points to one preprocessed dataset folder.

Expected structure:

```text
3D/<dataset_dir>/
|-- cropped_imgs/
|-- <slice_name>_all_layer_data.npy
`-- <dataset>_top_250_genes.csv
```

The `*_all_layer_data.npy` files are Python dictionaries containing serial-layer
neighbors and cropped patch names. See [`3D/README.md`](3D/README.md) for the
exact dictionary format.

## 3D Splits

All 3D experiments use slice-level cross-validation. A held-out fold contains
one slice name, and all spots/layer entries from that slice are used for testing.
The remaining slices are used for training.

| Config | Slice names | Folds |
|---|---|---:|
| `stnet` | `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W` | 23 |
| `her2st` | `A, B, C, D, E, F, G, H` | 8 |
| `skin` | `A, B, C, D` | 4 |
| `pcw` | `A, B, C, D, E, F` | 6 |
| `mouse` | `A, B, C, D` | 4 |

## 3D Training

Run commands from the `3D/` directory. `--select_fold` chooses the held-out slice
fold.

```bash
cd 3D

# STNet, 23 folds, 50 epochs
python main.py --config_name stnet --mode cv --select_fold 0 --gpu 0

# HER2ST, 8 folds, 60 epochs
python main.py --config_name her2st --mode cv --select_fold 0 --gpu 0

# Skin, 4 folds, 20 epochs
python main.py --config_name skin --mode cv --select_fold 0 --gpu 0

# PCW, 6 folds, 20 epochs
python main.py --config_name pcw --mode cv --select_fold 0 --gpu 0

# Mouse, 4 folds, 40 epochs
python main.py --config_name mouse --mode cv --select_fold 0 --gpu 0
```

To run all STNet folds:

```bash
for f in $(seq 0 22); do
  python main.py --config_name stnet --mode cv --select_fold $f --gpu 0
done
```

## Outputs

The public release saves metrics only by default.

- 2D: split JSON files and `kfold_summary*.csv` metrics.
- 3D: CSV logs under `3D/logs/<date>/<run_name>/`.
- Not saved by default: model checkpoints, TensorBoard events, and full stdout
  logs.

Optional output flags:

```bash
--save_checkpoints
--save_tensorboard
--save_logs          # 2D only
```

Large raw datasets and preprocessed serial-section folders should be distributed
as external archives rather than committed directly to GitHub.
