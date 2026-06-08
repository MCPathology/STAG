# STAG — Documentation

Official implementation of **STAG**, a dual-branch hypergraph framework that predicts
spatial transcriptomics (ST) gene expression from histopathology images, guided by
cross-modal contrastive learning and gene semantic priors.

This repository contains two settings:

| Folder | Setting | Description |
|--------|---------|-------------|
| [`2D/`](2D/) | 2D ST prediction | The main STAG model: per-slice gene-expression prediction on cSCC, HER2, HBC, and HEST datasets. |
| [`3D/`](3D/) | pseudo-3D ST prediction | The pseudo-3D extension (`STAG3D`) that models inter-slice dependencies across serial tissue sections (STNet, her2st, skin, pcw, mouse). |

Each subfolder is self-contained and has its own `README.md` and `requirements.txt`.

---

## Method Overview

STAG predicts the expression of a target gene panel (250 highly expressed genes) for each
spatial spot from its histology patch. It has two branches:

- **Query branch** — encodes the target spot's image patch and aligns it with its gene
  expression via cross-modal contrastive learning.
- **Neighbor branch** — builds a **hypergraph** over neighboring spots (feature-similarity
  hyperedges, distinct from spatial neighbors) and aggregates context with hypergraph
  convolutions for both the image and expression modalities.

Both modalities are fused by a multi-head cross-attention encoder, and a decoder regresses
the gene expression vector. Optionally, a frozen **gene semantic encoder** (OmiCLIP/Loki
text embeddings) injects biological priors over the target gene panel.

Training objective:

```
L = L_recon (MSE)  +  λ1 · L_contrast(query)  +  λ2 · L_contrast(neighbor)
```

The pseudo-3D variant (`3D/`) additionally builds multi-group mini-hypergraphs and an
inter-slice hypergraph to capture dependencies across serial sections, where each slice
serves as the reference in turn.

---

## Repository Structure

```
STAG-public/
├── README.md
├── DOCS.md
├── .gitignore
│
├── 2D/                                   # main 2D STAG (PyTorch)
│   ├── README.md
│   ├── requirements.txt
│   ├── train_STAG.py                     # ★ main: STAG + gene text encoder
│   ├── train_STAG_notext.py              # STAG without text encoder
│   ├── train_STAG_hvg.py                 # HVG gene-panel experiment
│   ├── train_ab.py                       # query-branch architecture ablation
│   ├── train_geneformer_ablation.py      # Geneformer text-encoder ablation
│   ├── ablation_encoder_cSCC.py          # image/expression encoder ablation
│   ├── ablation_gnn_cSCC.py              # HGNN vs GCN vs GAT ablation
│   ├── models/
│   │   └── models/
│   │       ├── model_Hypergraph_Text.py  # STAG with OmiCLIP text  → class STAG
│   │       ├── model_Hypergraph.py       # STAG without text       → class STAG
│   │       └── module.py                 # HGNN, EXPNN, cross-attention, decoder
│   ├── dataset/
│   │   ├── cSCCDataset.py  HER2Dataset.py  HBCDataset.py  HESTdataset.py      # base loaders
│   │   ├── TextcSCCDataset.py  TextHER2Dataset.py  TextHBCDataset.py  TextHESTDataset.py  # + gene text
│   │   └── NewcSCCDataset.py  NewHBCDataset.py  NewHER2Dataset.py  NewHESTDataset.py      # no-text variant
│   ├── preprocess/
│   │   ├── select_hvg.py  encode_hvg_loki.py
│   │   ├── 2-get_common_genes.py  3-get_other_hest_subsets.py
│   │   ├── extract_geneformer_cell_embeddings.py
│   │   └── 1-get_hest_genes.ipynb  4-genes_classification.ipynb
│   │       5-1-Bert_preprocess.ipynb  5-omiclip_preprocess.ipynb
│   ├── scripts/
│   │   └── run_STAG.sh  run_ablation_encoder.sh  run_geneformer_ablation.sh  run_hvg_ours.sh
│   └── select_genes/                     # bundled gene panels + text encodings (see Data section)
│
└── 3D/                                   # pseudo-3D STAG (PyTorch Lightning)
    ├── README.md
    ├── requirements.txt
    ├── main.py                           # train / test entry (cv | test | external_test | inference)
    ├── utils.py                          # config + logger helpers
    ├── get_res.py                        # parse logs → best metrics
    ├── model/
    │   ├── model.py                      # STAG3D LightningModule
    │   └── modules.py                    # hypergraph convs, cross-attention fusion, decoder
    ├── datasets/
    │   └── st_data.py                    # serial-section dataset + collate_fn
    ├── config/
    │   └── stnet.yaml  her2st.yaml  skin.yaml  pcw.yaml  mouse.yaml
    └── Scripts/
        ├── 1-Get_data.ipynb ... 7-crop-images.ipynb   # preprocessing pipeline
        └── registration.py                            # serial-section registration
```

---

## Data Directory Structure

Raw datasets are **not** included in this repository (too large). Download them from
their original sources and arrange them under `2D/data/` (or the 3D dataset folders) as
shown below. Run all scripts from the setting's root folder (`2D/` or `3D/`) so the
relative paths resolve.

> # 📌 Data Availability
>
> ## The dataset download links will be made **publicly available upon acceptance** of the manuscript.

### 2D datasets

```text
2D/data/
├── GSE144240/                                   # cSCC  (flat directory)
│   ├── <sample>.jpg                             #   WSI image      e.g. GSM4284316_P2_ST_rep1.jpg
│   ├── <sample>_stdata.tsv                      #   count matrix   (spots × genes)
│   └── <patient>_spot_data-selection-P<n>.tsv   #   spot pixel coordinates
│
├── HER2/                                        # her2st  (sub-directories)
│   ├── images/HE/<sample>.jpg                   #   WSI image
│   ├── count-matrices/<sample>.tsv             #   count matrix
│   └── spot-selection/<sample>_selection.tsv   #   spot pixel coordinates
│
├── Human_breast_cancer_in_situ_capturing_transcriptomics/   # HBC  (flat directory)
│   ├── HE_<sample>.jpg                          #   WSI image
│   ├── <sample>_stdata.tsv                      #   count matrix
│   └── spots_<sample>.csv                       #   spot pixel coordinates
│
└── Hest1k_datasets/                             # HEST-1k  (one set of files per subset)
    ├── st/<subset>.h5ad                         #   AnnData: expression + coordinates
    ├── patches/<subset>.h5                      #   per-spot patch barcodes
    └── wsis/<subset>.tif                        #   whole-slide image
        # <subset> ∈ {IDC, LUAD, SKCM, PRAD, KIDNEY, LIVER, LUNG, MOUSE_BRAIN, ...}
```

### 3D datasets

Each dataset is preprocessed by `3D/Scripts/` into a folder named like
`stnet_dataset_normal_smooth/` (also `her2st_heg250_dataset/`, `skin_dataset_normal_smooth/`,
`pcw_dataset_normal_smooth/`, `mouse_dataset_normal_smooth/`):

```text
3D/<dataset>_dataset_normal_smooth/
├── cropped_imgs/
│   └── <spot>.png                  # per-spot cropped patches
├── <name>_all_layer_data.npy       # serial-section data, one file per slice  (<name> = A, B, C, ...)
└── <dataset>_top_250_genes.csv     # target gene panel  (e.g. stnet_top_250_genes.csv)
```

Each config's `DATASET.data_dir` (in `3D/config/<dataset>.yaml`) points to this folder.

### Bundled gene panels & text encodings (`2D/select_genes/`)

These small files are **included** so STAG runs without regenerating them:

```text
2D/select_genes/
├── <dataset>_Selected_Genes.npy / <dataset>_gene.npy / <dataset>_select_genes.npy   # 250-gene panels
├── <dataset>_loki_text_encode.npy                                                   # OmiCLIP gene text embeddings
├── cSCC_bert_text_encode.npy                                                        # BERT gene text embeddings (cSCC)
└── <dataset>_mapping.csv                                                            # gene symbol → Ensembl ID mapping
    # <dataset> ∈ {cSCC, HER2, HBC, STNet, HEST_KIDNEY, HEST_LIVER, HEST_LUNG, HEST_PRAD, HEST_MOUSE_BRAIN, ...}
```

---

## Getting Started

- **2D experiments:** see [`2D/README.md`](2D/README.md)
- **3D experiments:** see [`3D/README.md`](3D/README.md)

> Only STAG's own model code is included here. Baseline implementations
> (HisToGene, Hist2ST, BLEEP, TRIPLEX, M2OST, STNet, ResNet-ABMIL, etc.) are **not**
> part of this release.

---

## Datasets

STAG is evaluated on publicly available ST datasets:

- **cSCC** — GEO accession GSE144240
- **HER2+ breast cancer** (her2st)
- **Human breast cancer (HBC)**
- **HEST-1k** subsets (LUAD, IDC, PAAD, SKCM, kidney, liver, lung, mouse brain, PRAD, ...)

Raw images/counts must be downloaded from their original sources and placed under
`2D/data/` (or `3D/.../`) following the path conventions documented in each subfolder.
The gene panels and precomputed text encodings needed to run STAG are bundled in
`2D/select_genes/`.

---

## License

Released for academic research use. See `LICENSE` (add your preferred license, e.g. MIT).
