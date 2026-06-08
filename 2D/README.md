# STAG — 2D Spatial Transcriptomics Prediction

The main STAG model for per-slice gene-expression prediction from histopathology patches.

## Directory Structure

```
2D/
├── models/models/
│   ├── model_Hypergraph_Text.py   # STAG with gene semantic (OmiCLIP) encoder  ← main model
│   ├── model_Hypergraph.py        # STAG without text encoder (variant)
│   └── module.py                  # building blocks (hypergraph conv, cross-attention, decoder)
├── dataset/
│   ├── Text{cSCC,HER2,HBC,HEST}Dataset.py   # datasets with gene text embeddings
│   ├── New{cSCC,HBC,HER2,HEST}Dataset.py    # datasets for the no-text variant
│   └── {cSCC,HER2,HBC,HEST}dataset.py       # base loaders
├── preprocess/                    # gene-panel selection + text-encoding pipeline
├── select_genes/                  # bundled gene panels + precomputed text encodings
├── scripts/                       # example run scripts
├── train_STAG.py        # ★ main training entry (STAG + text)
├── train_STAG_notext.py                 # training entry (STAG, no text)
├── train_STAG_hvg.py              # HVG gene-panel experiment
├── train_ab.py                    # architecture ablation (query branch components)
├── ablation_encoder_cSCC.py       # encoder ablation (ResNet/UNI/CONCH × MLP/SNN/Geneformer)
├── ablation_gnn_cSCC.py           # GNN ablation (HGNN vs GCN vs GAT)
└── train_geneformer_ablation.py   # Geneformer text-encoder ablation
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `einops`, `scikit-learn`, `scipy`,
`pandas`, `tqdm`, `tensorboard`. The Geneformer ablation additionally needs
`transformers` and `mygene`; the OmiCLIP/Loki text-encoding preprocessing needs the
external [Loki/OmiCLIP](https://github.com/GuangyuWangLab2021/Loki) package.

## Data Preparation

1. **Gene panels & text encodings** — already bundled in [`select_genes/`](select_genes/)
   (gene lists `*_Selected_Genes.npy`, OmiCLIP encodings `*_loki_text_encode.npy`,
   BERT encodings `*_bert_text_encode.npy`). No action needed.

2. **Raw ST datasets** — download from the original sources and place under `./data/`:

   ```
   2D/data/
   ├── GSE144240/                                                 # cSCC
   ├── HER2/                                                      # her2st
   ├── Human_breast_cancer_in_situ_capturing_transcriptomics/    # HBC
   └── Hest1k_datasets/                                           # HEST-1k subsets
   ```

   All paths use the `./data/...` convention. **Run all commands from the `2D/`
   directory** so the relative paths resolve. If your data lives elsewhere, edit the
   `DATA_PATHS` dict at the top of each training script (or pass the dataset path).

## Training

Run from the `2D/` directory.

### Main STAG (with gene semantic encoder)

```bash
python train_STAG.py --data_name cSCC \
    --emb_dim 512 --depth 2 --heads 8 \
    --k_folds 5 --epochs 50 --lr 1e-4 --batch_size 8 \
    --loss_ratio1 0.4 --loss_ratio2 0.2 --temp1 0.05 --temp2 0.05
```

`--data_name` ∈ `{cSCC, HER2, HBC, HEST_LUAD, HEST_IDC, HEST_PAAD, HEST_SKCM, HEST_kidney, HEST_mouse_brain, HEST_PRAD, ...}`.

### STAG without text encoder

```bash
python train_STAG_notext.py --data_name cSCC --k_folds 5 --epochs 50
```

## Ablations (paper)

```bash
# Hypergraph vs GCN vs GAT
python ablation_gnn_cSCC.py --gnn_type hgnn      # or gcn / gat

# Expression / image encoder ablation (Table 8)
python ablation_encoder_cSCC.py --img_encoder resnet18 --exp_encoder mlp   # baseline
python ablation_encoder_cSCC.py --img_encoder resnet18 --exp_encoder snn
```

### Geneformer expression-encoder ablation

```bash
# 1) precompute frozen Geneformer per-spot embeddings (full transcriptome)
python preprocess/extract_geneformer_cell_embeddings.py \
    --data_path ./data/GSE144240 \
    --output_dir ./data/GSE144240/geneformer_emb \
    --model_dir ./Geneformer --model_variant v1-10m --batch_size 64

# 2) train with frozen Geneformer + trainable projection MLP
python ablation_encoder_cSCC.py --img_encoder resnet18 --exp_encoder geneformer \
    --geneformer_emb_dir ./data/GSE144240/geneformer_emb --geneformer_dim 256
```

The Geneformer model weights/dictionaries are obtained by cloning the official
[Geneformer](https://huggingface.co/ctheodoris/Geneformer) repo into `./Geneformer/`.

## Notes

- Comments and docstrings have been stripped from the source for the public release.
- HEST subset directory casing may differ (`Hest1k_datasets` vs `hest1k_datasets`);
  adjust to match your local layout.
