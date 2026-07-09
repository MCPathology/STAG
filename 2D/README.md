# STAG 2D Training

This folder contains the 2D STAG model for spot-level spatial gene-expression
prediction from histopathology image patches. Run all commands from this `2D/`
directory so that the relative paths in the dataset loaders resolve correctly.

## Environment

```bash
cd 2D
pip install -r requirements.txt
```

The model uses a self-supervised ResNet18 backbone. If
`weights/tenpercent_resnet18.ckpt` is not present, the code downloads it on the
first run.

## Data Layout

Place the 2D data under `2D/data/`:

```text
2D/data/
├── GSE144240/                                               # cSCC
├── HER2/                                                    # HER2ST
├── Human_breast_cancer_in_situ_capturing_transcriptomics/  # HBC
└── Hest1k_datasets/                                         # HEST-1k subsets
```

Expected files:

```text
GSE144240/
├── *.jpg
├── *_stdata.tsv
└── *_spot_data-selection-P*.tsv

HER2/
├── images/HE/*.jpg
├── count-matrices/*.tsv
└── spot-selection/*_selection.tsv

Human_breast_cancer_in_situ_capturing_transcriptomics/
├── *.jpg
├── *_stdata.tsv
└── spots_*.csv

Hest1k_datasets/<subset>/
├── st/*.h5ad
└── wsis/*.tif
```

Gene panels and gene-text embeddings are stored in `2D/select_genes/` and are
loaded automatically by the training scripts.

## Split Protocol

All 2D splits are sample-level splits. Spots from the same WSI/sample are never
split across train and validation folds.

| Dataset family | Split unit | Split implementation | Default seed |
|---|---|---|---:|
| cSCC (`GSE144240`) | `.jpg` WSI files under `GSE144240/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HER2ST (`HER2`) | `.jpg` WSI files under `HER2/images/HE/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HBC | `.jpg` WSI files under `Human_breast_cancer_in_situ_capturing_transcriptomics/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HEST-1k subsets | sample IDs from `Hest1k_datasets/<subset>/st/*.h5ad` | `KFold(..., shuffle=True)` over sorted sample IDs | 1553 |

The generated split JSON records the exact train/validation files and is saved
under the run output directory. Reusing the same `--k_folds` and `--seed` reloads
the same split file.

## Default Outputs

The public training scripts are configured to keep the release lightweight:

- Saved by default: fold split JSON files and `kfold_summary*.csv` metrics.
- Not saved by default: model checkpoints, TensorBoard event files, full stdout logs.

Optional output flags:

```bash
--save_checkpoints   # save best-fold .pth files
--save_tensorboard   # save TensorBoard event files
--save_logs          # save training_log.txt
```

## Main STAG Training

Use `train_STAG.py` for the main text-guided STAG model.

Recommended settings used for the 2D release:

| Dataset | `--data_name` | Folds | Epochs | Batch size |
|---|---:|---:|---:|---:|
| cSCC | `cSCC` | 4 | 50 | 8 |
| HER2ST | `HER2` | 6 | 50 | 8 |
| HBC | `HBC` | 9 | 50 | 8 |
| HEST-PRAD | `HEST_PRAD` | 6 | 50 | 8 |
| HEST-kidney | `HEST_kidney` | 6 | 50 | 8 |
| HEST-mouse-brain | `HEST_mouse_brain` | 5 | 50 | 8 |

Examples:

```bash
# cSCC, 4-fold CV
python train_STAG.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 8

# HER2ST, 6-fold CV
python train_STAG.py --data_name HER2 --k_folds 6 --epochs 50 --batch_size 8

# HBC, 9-fold CV
python train_STAG.py --data_name HBC --k_folds 9 --epochs 50 --batch_size 8

# HEST-PRAD, 6-fold CV
python train_STAG.py --data_name HEST_PRAD --k_folds 6 --epochs 50 --batch_size 8
```

The script evaluates every epoch and reports the best validation metrics for each
fold in the final CSV.

## No-Text Variant

Use `train_STAG_notext.py` for the STAG variant without gene-text embeddings:

```bash
python train_STAG_notext.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 16
python train_STAG_notext.py --data_name HER2 --k_folds 6 --epochs 50 --batch_size 16
python train_STAG_notext.py --data_name HBC --k_folds 9 --epochs 50 --batch_size 16
```

## HVG Experiments

Use `train_STAG_hvg.py` for HVG gene-panel experiments. This entry currently
supports `cSCC`, `HER2`, and `HBC`.

Recommended HVG settings:

| Dataset | `--data_name` | Folds | Epochs | Batch size |
|---|---:|---:|---:|---:|
| cSCC | `cSCC` | 4 | 80 | 8 |
| HER2ST | `HER2` | 6 | 80 | 8 |
| HBC | `HBC` | 9 | 80 | 8 |

Examples:

```bash
python train_STAG_hvg.py --data_name cSCC --k_folds 4 --epochs 80 --batch_size 8 --gene_mode hvg
python train_STAG_hvg.py --data_name HER2 --k_folds 6 --epochs 80 --batch_size 8 --gene_mode hvg
python train_STAG_hvg.py --data_name HBC --k_folds 9 --epochs 80 --batch_size 8 --gene_mode hvg
```

To run a single fold for debugging:

```bash
python train_STAG_hvg.py --data_name HER2 --k_folds 6 --select_fold 0 --epochs 1 --batch_size 2
```

## Ablations

```bash
# Query/attention/contrastive ablation
python train_ab.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 8

# Geneformer text-encoder ablation
python train_geneformer_ablation.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 8

# Encoder and GNN ablations write per-fold and summary CSV files.
python ablation_encoder_cSCC.py --img_encoder resnet18 --exp_encoder mlp
python ablation_gnn_cSCC.py --gnn_type hgnn
```

## Notes

- HEST support depends on matching gene-list and text-embedding files in
  `select_genes/`. If you add a new HEST subset, add both files and update the
  mapping in `train_STAG.py`.
- Large raw datasets should not be committed directly to GitHub. Keep full data in
  an external release and place only a small toy example in the repository.
