# STAG 3D Training

This folder contains the pseudo-3D STAG model for serial-section spatial
transcriptomics. The 3D pipeline expects preprocessed per-slice graph data rather
than raw whole-slide images. Run all commands from this `3D/` directory.

## Environment

```bash
cd 3D
pip install -r requirements.txt
```

The self-supervised ResNet18 backbone weight is stored at
`weights/tenpercent_resnet18.ckpt`. If it is missing, the code downloads it on the
first run.

## Preprocessed Data Layout

Each dataset is selected by `--config_name`, which loads `config/<name>.yaml`.
The YAML field `DATASET.data_dir` points to the preprocessed dataset folder.

Expected structure:

```text
3D/<dataset_dir>/
├── cropped_imgs/
│   ├── <spot_patch>.png
│   └── ...
├── <slice_name>_all_layer_data.npy
└── <dataset>_top_250_genes.csv
```

The `*_all_layer_data.npy` files are Python dictionaries saved with
`np.save(..., allow_pickle=True)`. Each entry stores serial-section neighbors with
two keys used by the loader:

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

The patch names in `cropped_image_names` are resolved under `cropped_imgs/`. The
loader converts `.jpg` suffixes to `.png`, so the release data should provide PNG
patches.

## Dataset Settings

| Config | Data folder | Slices/Folds | Epochs | Batch size |
|---|---|---:|---:|---:|
| `stnet` | `stnet_dataset_normal_smooth` | 15 | 50 | 16 |
| `her2st` | `her2st_heg250_dataset` | 8 | 60 | 1 |
| `skin` | `skin_dataset_normal_smooth` | 4 | 20 | 4 |
| `pcw` | `pcw_dataset_normal_smooth` | 6 | 20 | 2 |
| `mouse` | `mouse_dataset_normal_smooth` | 4 | 40 | 2 |

The fold count is determined by the serial-section slice names used in
`datasets/st_data.py`. The YAML files should match these counts.

## Split Protocol

All 3D experiments use slice-level cross-validation over the preprocessed serial
sections. A held-out fold contains one slice name, and all spots/layer entries
from that slice are used for testing. The remaining slices are used for training.
No spot-level random splitting is used.

| Config | Slice names | Fold rule |
|---|---|---|
| `her2st` | `A, B, C, D, E, F, G, H` | 8 folds; fold `i` tests slice `A` through `H` respectively |
| `stnet` | `E, F, I, J, L, M, N, O, P, R, S, T, U, V, W` | 15 folds; fold `i` tests the corresponding slice in this list |
| `skin` | `A, B, C, D` | 4 folds; fold `i` tests one slice |
| `pcw` | `A, B, C, D, E, F` | 6 folds; fold `i` tests one slice |
| `mouse` | `A, B, C, D` | 4 folds; fold `i` tests one slice |

The command-line argument `--select_fold` chooses which held-out slice fold to
run. To reproduce a complete cross-validation result, run every fold listed for
that dataset and average the saved CSV metrics.

Required files by config:

```text
stnet_dataset_normal_smooth/
├── cropped_imgs/
├── E_all_layer_data.npy ... W_all_layer_data.npy
└── stnet_top_250_genes.csv

her2st_heg250_dataset/
├── cropped_imgs/
├── A_all_layer_data.npy ... H_all_layer_data.npy
└── her2st_top_250_genes.csv

skin_dataset_normal_smooth/
├── cropped_imgs/
├── A_all_layer_data.npy ... D_all_layer_data.npy
└── skin_top_250_genes.csv

pcw_dataset_normal_smooth/
├── cropped_imgs/
├── A_all_layer_data.npy ... F_all_layer_data.npy
└── pcw_top_250_genes.csv

mouse_dataset_normal_smooth/
├── cropped_imgs/
├── A_all_layer_data.npy ... D_all_layer_data.npy
└── mouse_top_250_genes.csv
```

## Training

Run one selected fold:

```bash
python main.py --config_name stnet --mode cv --select_fold 0 --gpu 0
```

Run all folds by launching the selected folds separately:

```bash
for f in $(seq 0 14); do
  python main.py --config_name stnet --mode cv --select_fold $f --gpu 0
done
```

Dataset-specific commands:

```bash
# STNet, 15 folds, 50 epochs
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

## Outputs

The release configuration saves metrics only by default:

- Saved by default: CSV logs under `logs/<date>/<run_name>/`.
- Not saved by default: Lightning checkpoints and TensorBoard event files.

Optional output flags:

```bash
--save_checkpoints   # enable Lightning checkpoint saving
--save_tensorboard   # enable TensorBoard event files
```

## Testing from a Checkpoint

Checkpoint saving is disabled by default. If you trained with
`--save_checkpoints`, test a checkpoint with:

```bash
python main.py --config_name stnet --mode test --model_path logs/<date>/<run_name>/<checkpoint>.ckpt --gpu 0
```

## Notes

- The preprocessed folders are large and should be distributed as external data
  archives instead of being committed directly to GitHub.
- The notebooks in `Scripts/` document the preprocessing pipeline used to generate
  the `*_all_layer_data.npy` files and cropped spot patches.
