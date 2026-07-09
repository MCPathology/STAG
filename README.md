# STAG: Biologically Guided Spatial Transcriptomics Prediction via Hypergraph Learning

> Under review. Dataset download links will be made publicly available upon
> acceptance of the manuscript.

STAG predicts spatial gene expression from histopathology images. This
repository contains two training pipelines:

- `2D/`: single-section spot-level prediction from H&E image patches.
- `3D/`: pseudo-3D prediction from preprocessed serial-section graph data.

For implementation details, see [`2D/README.md`](2D/README.md),
[`3D/README.md`](3D/README.md), and [`DOCS.md`](DOCS.md).

![STAG framework overview](assets/MEDIA_MainArch.jpg)

## Framework Overview

STAG consists of two key components: a **Query branch** and a **Neighbor
branch**. In the 2D setting, the Query branch processes the target pathology
patch and its paired spatial transcriptomics spot, while the Neighbor branch
takes neighboring regions as input. Intra-slice relationships are modeled with
hypergraphs. The two branches further use cross-attention and contrastive
learning losses (`L_s` and `L_g`) to refine and align cross-modal features
between histology and gene expression.

During inference, only the pathology encoder is used, so STAG predicts gene
expression directly from the histology image.

## Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Installation](#installation)
3. [2D Pipeline](#2d-pipeline)
4. [3D Pipeline](#3d-pipeline)
5. [Outputs](#outputs)
6. [Data Release Note](#data-release-note)
7. [Citation](#citation)

## Pipeline Overview

| Pipeline | What it does | Main script |
|---|---|---|
| 2D STAG | Predicts gene expression from one spatial transcriptomics section | `2D/train_STAG.py` |
| 2D no-text variant | Runs STAG without gene-text embeddings | `2D/train_STAG_notext.py` |
| 2D HVG variant | Runs STAG on HVG gene panels | `2D/train_STAG_hvg.py` |
| 3D STAG | Uses serial-section pseudo-3D context | `3D/main.py` |

By default, training saves metrics and fold splits only. Checkpoints,
TensorBoard files, and full stdout logs are optional flags.

## Installation

Install dependencies inside the pipeline folder you want to run.

```bash
# 2D experiments
cd 2D
pip install -r requirements.txt

# 3D experiments
cd ../3D
pip install -r requirements.txt
```

The image encoder uses a self-supervised ResNet18 checkpoint at
`weights/tenpercent_resnet18.ckpt`. If the file is missing, the code attempts to
download it on the first run.

## 2D Pipeline

### 2D Data Structure

Place all 2D datasets under `2D/data/`.

```text
2D/data/
|-- GSE144240/                                               # cSCC
|-- HER2/                                                    # HER2ST
|-- Human_breast_cancer_in_situ_capturing_transcriptomics/  # HBC
`-- Hest1k_datasets/                                         # HEST-1k subsets
```

Expected files for each dataset:

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

Gene panels and gene-text embeddings are loaded from `2D/select_genes/`.

### 2D Dataset Table

The following datasets are configured for the text-guided `train_STAG.py`
entry.

| Dataset | `--data_name` | Data folder | Required files | Gene/text files |
|---|---|---|---|---|
| cSCC | `cSCC` | `2D/data/GSE144240/` | `*.jpg`, `*_stdata.tsv`, `*_spot_data-selection-P*.tsv` | `cSCC_Selected_Genes.npy`, `cSCC_bert_text_encode.npy` |
| HER2ST | `HER2` | `2D/data/HER2/` | `images/HE/*.jpg`, `count-matrices/*.tsv`, `spot-selection/*_selection.tsv` | `HER2_Selected_Genes.npy`, `HER2_loki_text_encode.npy` |
| HBC | `HBC` | `2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics/` | `*.jpg`, `*_stdata.tsv`, `spots_*.csv` | `HBC_Selected_Genes.npy`, `STNet_loki_text_encode.npy` |
| HEST-LUAD | `HEST_LUAD` | `2D/data/Hest1k_datasets/hest_data_LUAD/` | `st/*.h5ad`, `wsis/*.tif` | `HEST_LUNG_gene.npy`, `HEST_LUNG_loki_text_encode.npy` |
| HEST-kidney | `HEST_kidney` | `2D/data/Hest1k_datasets/kidney/` | `st/*.h5ad`, `wsis/*.tif` | `HEST_KIDNEY_gene.npy`, `HEST_KIDNEY_loki_text_encode.npy` |
| HEST-mouse-brain | `HEST_mouse_brain` | `2D/data/Hest1k_datasets/mouse_brain/` | `st/*.h5ad`, `wsis/*.tif` | `HEST_MOUSE_BRAIN_gene.npy`, `HEST_MOUSE_BRAIN_loki_text_encode.npy` |
| HEST-PRAD | `HEST_PRAD` | `2D/data/Hest1k_datasets/PRAD/` | `st/*.h5ad`, `wsis/*.tif` | `HEST_PRAD_gene.npy`, `HEST_PRAD_loki_text_encode.npy` |

The parser also contains names such as `HEST_IDC`, `HEST_PAAD`, `HEST_SKCM`,
`HEST_her2st`, `HEST_Liver`, and `HEST_Lung`. Before running the text-guided
main model on those datasets, add the corresponding entries to `GENE_FILES` and
`TEXT_FILES` in `2D/train_STAG.py`.

### 2D Splits

All 2D splits are sample-level splits. Spots from the same WSI/sample are never
split across train and validation folds.

| Dataset family | Split unit | Split implementation | Default seed |
|---|---|---|---:|
| cSCC | `.jpg` WSI files under `GSE144240/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HER2ST | `.jpg` WSI files under `HER2/images/HE/` | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HBC | `.jpg` WSI files under the HBC folder | `KFold(..., shuffle=True)` over sorted image files | 1553 |
| HEST-1k subsets | sample IDs from `Hest1k_datasets/<subset>/st/*.h5ad` | `KFold(..., shuffle=True)` over sorted sample IDs | 1553 |

The generated split JSON records the exact train/validation files. Reusing the
same `--k_folds` and `--seed` reloads the same split.

### 2D Training Commands

Run commands from the `2D/` directory.

```bash
cd 2D
```

Main text-guided STAG:

```bash
# cSCC
python train_STAG.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 8

# HER2ST
python train_STAG.py --data_name HER2 --k_folds 6 --epochs 50 --batch_size 8

# HBC
python train_STAG.py --data_name HBC --k_folds 9 --epochs 50 --batch_size 8

# HEST-LUAD
python train_STAG.py --data_name HEST_LUAD --k_folds 6 --epochs 50 --batch_size 8

# HEST-kidney
python train_STAG.py --data_name HEST_kidney --k_folds 6 --epochs 50 --batch_size 8

# HEST-mouse-brain
python train_STAG.py --data_name HEST_mouse_brain --k_folds 5 --epochs 50 --batch_size 8

# HEST-PRAD
python train_STAG.py --data_name HEST_PRAD --k_folds 6 --epochs 50 --batch_size 8
```

Recommended settings:

| Dataset | `--data_name` | Folds | Epochs | Batch size |
|---|---:|---:|---:|---:|
| cSCC | `cSCC` | 4 | 50 | 8 |
| HER2ST | `HER2` | 6 | 50 | 8 |
| HBC | `HBC` | 9 | 50 | 8 |
| HEST-LUAD | `HEST_LUAD` | 6 | 50 | 8 |
| HEST-kidney | `HEST_kidney` | 6 | 50 | 8 |
| HEST-mouse-brain | `HEST_mouse_brain` | 5 | 50 | 8 |
| HEST-PRAD | `HEST_PRAD` | 6 | 50 | 8 |

Quick smoke test:

```bash
python train_STAG.py --data_name cSCC --k_folds 4 --epochs 1 --batch_size 2
```

Additional 2D variants:

```bash
# No gene-text embeddings
python train_STAG_notext.py --data_name cSCC --k_folds 4 --epochs 50 --batch_size 16
python train_STAG_notext.py --data_name HER2 --k_folds 6 --epochs 50 --batch_size 16
python train_STAG_notext.py --data_name HBC --k_folds 9 --epochs 50 --batch_size 16

# HVG gene panel
python train_STAG_hvg.py --data_name cSCC --k_folds 4 --epochs 80 --batch_size 8 --gene_mode hvg
python train_STAG_hvg.py --data_name HER2 --k_folds 6 --epochs 80 --batch_size 8 --gene_mode hvg
python train_STAG_hvg.py --data_name HBC --k_folds 9 --epochs 80 --batch_size 8 --gene_mode hvg
```

## 3D Pipeline

### 3D Data Structure

The 3D pipeline uses preprocessed serial-section data, not raw WSI folders. Each
YAML file in `3D/config/` points to one preprocessed dataset folder through
`DATASET.data_dir`.

Expected folder format:

```text
3D/<dataset_dir>/
|-- cropped_imgs/
|-- <slice_name>_all_layer_data.npy
`-- <dataset>_top_250_genes.csv
```

Each `*_all_layer_data.npy` file is a Python dictionary saved with
`np.save(..., allow_pickle=True)`. It stores serial-section neighbor entries:

```python
{
    row_key: {
        layer_key: {
            "gene_expressions": [...],
            "cropped_image_names": [...]
        }
    }
}
```

### 3D Dataset Table

| Dataset/config | Expected folder | Required files | Folds | Epochs | Batch size |
|---|---|---|---:|---:|---:|
| STNet serial sections | `3D/stnet_dataset_normal_smooth/` | `cropped_imgs/`, `A_all_layer_data.npy` through `W_all_layer_data.npy`, `stnet_top_250_genes.csv` | 23 | 50 | 16 |
| HER2ST serial sections | `3D/her2st_heg250_dataset/` | `cropped_imgs/`, `A_all_layer_data.npy` through `H_all_layer_data.npy`, `her2st_top_250_genes.csv` | 8 | 60 | 1 |
| Skin | `3D/skin_dataset_normal_smooth/` | `cropped_imgs/`, `A_all_layer_data.npy` through `D_all_layer_data.npy`, `skin_top_250_genes.csv` | 4 | 20 | 4 |
| PCW | `3D/pcw_dataset_normal_smooth/` | `cropped_imgs/`, `A_all_layer_data.npy` through `F_all_layer_data.npy`, `pcw_top_250_genes.csv` | 6 | 20 | 2 |
| Mouse | `3D/mouse_dataset_normal_smooth/` | `cropped_imgs/`, `A_all_layer_data.npy` through `D_all_layer_data.npy`, `mouse_top_250_genes.csv` | 4 | 40 | 2 |

### 3D Splits

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

### 3D Training Commands

Run commands from the `3D/` directory. Each run trains one held-out slice fold.
`--select_fold` chooses that fold.

```bash
cd 3D
```

Single-fold commands:

```bash
# STNet serial sections, fold 0 of 23
python main.py --config_name stnet --mode cv --select_fold 0 --gpu 0

# HER2ST serial sections, fold 0 of 8
python main.py --config_name her2st --mode cv --select_fold 0 --gpu 0

# Skin, fold 0 of 4
python main.py --config_name skin --mode cv --select_fold 0 --gpu 0

# PCW, fold 0 of 6
python main.py --config_name pcw --mode cv --select_fold 0 --gpu 0

# Mouse, fold 0 of 4
python main.py --config_name mouse --mode cv --select_fold 0 --gpu 0
```

Full cross-validation commands:

```bash
# STNet: folds 0-22
for f in $(seq 0 22); do python main.py --config_name stnet --mode cv --select_fold $f --gpu 0; done

# HER2ST: folds 0-7
for f in $(seq 0 7); do python main.py --config_name her2st --mode cv --select_fold $f --gpu 0; done

# Skin: folds 0-3
for f in $(seq 0 3); do python main.py --config_name skin --mode cv --select_fold $f --gpu 0; done

# PCW: folds 0-5
for f in $(seq 0 5); do python main.py --config_name pcw --mode cv --select_fold $f --gpu 0; done

# Mouse: folds 0-3
for f in $(seq 0 3); do python main.py --config_name mouse --mode cv --select_fold $f --gpu 0; done
```

## Outputs

Default outputs are intentionally lightweight:

- 2D: fold split JSON files and `kfold_summary*.csv` metrics.
- 3D: CSV logs under `3D/logs/<date>/<run_name>/`.
- Not saved by default: model checkpoints, TensorBoard events, and full stdout
  logs.

Optional flags:

```bash
--save_checkpoints   # save model checkpoints
--save_tensorboard   # save TensorBoard event files
--save_logs          # 2D only; save training_log.txt
```

If checkpoints are enabled for 3D, test a checkpoint with:

```bash
cd 3D
python main.py --config_name stnet --mode test --model_path logs/<date>/<run_name>/<checkpoint>.ckpt --gpu 0
```

## Data Release Note

Large raw datasets and preprocessed serial-section folders should be distributed
as external archives rather than committed directly to GitHub. For code-only
releases, keep the folder structure above and place downloaded data in the
corresponding paths before training.

The full data archive is stored externally as:

```text
Aliyun Drive: /data/zyc-MEDIA-Re.zip
```

See [`DATA.md`](DATA.md) for the detailed data restoration guide. On the server,
the archive can be restored with:

```bash
bash scripts/prepare_media_data_from_aliyunpan.sh \
  /mnt/pfs-gv8sxa/tts/dhg/yg/zyc/aliyunpan-v0.4.0-linux-amd64/aliyunpan \
  /data/zyc-MEDIA-Re.zip \
  .
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{QU2026104206,
  title = {STAG: Biologically guided spatial transcriptomics prediction via hypergraph learning},
  journal = {Medical Image Analysis},
  pages = {104206},
  year = {2026},
  issn = {1361-8415},
  doi = {https://doi.org/10.1016/j.media.2026.104206},
  url = {https://www.sciencedirect.com/science/article/pii/S1361841526002756},
  author = {Mingcheng Qu and Yuchuan Zhao and Guang Yang and Donglin Di and Xiu Su and Hongyan Xu and Yang Song and Lei Fan},
  keywords = {Spatial transcriptomics, Whole-slide images, Cross-modal alignment, Contrastive learning, Hypergraph learning}
}
```
