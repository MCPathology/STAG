# STAG — pseudo-3D Spatial Transcriptomics Prediction

The pseudo-3D extension of STAG (`STAG3D`), which models inter-slice dependencies across
serial tissue sections. Built on PyTorch Lightning.

## Directory Structure

```
3D/
├── main.py            # training / testing entry
├── model/
│   ├── model.py       # STAG3D LightningModule
│   └── modules.py     # hypergraph convs, cross-attention fusion, decoder
├── datasets/
│   └── st_data.py     # serial-section dataset + collate_fn
├── config/            # per-dataset YAML configs (stnet, her2st, skin, pcw, mouse)
├── Scripts/           # data preprocessing & slice-registration pipeline
├── get_res.py         # parse training logs into best metrics
└── utils.py           # config / logger helpers
```

## Installation

```bash
pip install -r requirements.txt
```

Uses `pytorch-lightning==1.9.0`, `torch`, `torch-geometric`, `einops`, `scanpy`,
`opencv-python`, `wget`, etc. (full pinned list in `requirements.txt`).

## Data Preparation

The serial-section datasets are built from raw ST data via the notebooks in
[`Scripts/`](Scripts/) (run in order):

1. `1-Get_data.ipynb` — fetch raw data
2. `2-Preprocess.ipynb` — QC / normalization
3. `3-Preprocess-imgs.ipynb` — image preprocessing
4. `4-Preprocess-gene.ipynb` — gene-panel selection (top-250)
5. `5-Make-Dataset.ipynb` — assemble per-slice `*_all_layer_data.npy`
6. `7-crop-images.ipynb` — crop spot patches
7. `registration.py` — serial-section image registration (pseudo-3D alignment)

Each config's `DATASET.data_dir` points to the resulting dataset folder, e.g.
`stnet_dataset_normal_smooth`, `her2st_heg250_dataset`. Update these paths to your local
layout. The self-supervised ResNet18 backbone weight
(`weights/tenpercent_resnet18.ckpt`) is downloaded automatically on first run.

## Training

```bash
# k-fold cross-validation on a chosen fold
python main.py --config_name stnet --gpu 0 --mode cv --select_fold 0
```

- `--config_name` ∈ `{stnet, her2st, skin, pcw, mouse}` (loads `config/<name>.yaml`)
- `--mode` ∈ `{cv, test, external_test, inference}`
- `--select_fold` selects which fold to run (folds = `TRAINING.num_k` in the config)

Testing from a checkpoint:

```bash
python main.py --config_name stnet --mode test --model_path results/<ckpt>.ckpt
```

## Results

```bash
python get_res.py    # scans *.txt logs and prints best val_mse / val_mae / PCC top-k
```

## Notes

- Hyperparameters (epochs, batch size, loss ratios, temperatures) are set per dataset in
  `config/*.yaml`.
- Comments and docstrings have been stripped from the source for the public release.
