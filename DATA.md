# Data Preparation

The training data are distributed as preprocessed STAG archives. They are not
stored in normal Git history because the image and spatial transcriptomics files
are large.

Download the data packages from the public share:

```text
Data share link: <to be added>
```

## Package List

```text
SHA256SUMS.txt
STAG-2D-GSE144240.tar.zst
STAG-2D-HER2.tar.zst
STAG-2D-HBC.tar.zst
STAG-2D-HEST-kidney.tar.zst
STAG-2D-HEST-mouse_brain.tar.zst
STAG-2D-HEST-PRAD.tar.zst
STAG-3D-HBC-stnet.tar.zst
STAG-weights-resnet18.tar.zst
```

`HEST_LUAD` is not included in the released processed package.

## Restore Data

Place the downloaded archives in the repository root and extract them:

```bash
tar --use-compress-program=unzstd -xf STAG-2D-GSE144240.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HER2.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HBC.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HEST-kidney.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HEST-mouse_brain.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HEST-PRAD.tar.zst
tar --use-compress-program=unzstd -xf STAG-3D-HBC-stnet.tar.zst
tar --use-compress-program=unzstd -xf STAG-weights-resnet18.tar.zst

sha256sum -c SHA256SUMS.txt
```

The expected layout after extraction is:

```text
2D/data/
|-- GSE144240/
|-- HER2/
|-- Human_breast_cancer_in_situ_capturing_transcriptomics/
`-- Hest1k_datasets/
    |-- kidney/
    |-- mouse_brain/
    `-- PRAD/

2D/weights/tenpercent_resnet18.ckpt

3D/
|-- stnet_dataset_normal_smooth/
`-- weights/tenpercent_resnet18.ckpt
```

## Package-to-Experiment Mapping

| Package | Restored folder | Main command |
|---|---|---|
| `STAG-2D-GSE144240.tar.zst` | `2D/data/GSE144240/` | `python train_STAG.py --data_name cSCC ...` |
| `STAG-2D-HER2.tar.zst` | `2D/data/HER2/` | `python train_STAG.py --data_name HER2 ...` |
| `STAG-2D-HBC.tar.zst` | `2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics/` | `python train_STAG.py --data_name HBC ...` |
| `STAG-2D-HEST-kidney.tar.zst` | `2D/data/Hest1k_datasets/kidney/` | `python train_STAG.py --data_name HEST_kidney ...` |
| `STAG-2D-HEST-mouse_brain.tar.zst` | `2D/data/Hest1k_datasets/mouse_brain/` | `python train_STAG.py --data_name HEST_mouse_brain ...` |
| `STAG-2D-HEST-PRAD.tar.zst` | `2D/data/Hest1k_datasets/PRAD/` | `python train_STAG.py --data_name HEST_PRAD ...` |
| `STAG-3D-HBC-stnet.tar.zst` | `3D/stnet_dataset_normal_smooth/` | `python main.py --config_name stnet ...` |
| `STAG-weights-resnet18.tar.zst` | `2D/weights/`, `3D/weights/` | encoder checkpoint |

## Smoke Tests

After extraction, run a short check:

```bash
cd 2D
python train_STAG.py --data_name cSCC --k_folds 4 --select_fold 0 --epochs 1 --batch_size 2
python train_STAG.py --data_name HER2 --k_folds 6 --select_fold 0 --epochs 1 --batch_size 2
python train_STAG.py --data_name HBC --k_folds 9 --select_fold 0 --epochs 1 --batch_size 2
python train_STAG.py --data_name HEST_kidney --k_folds 3 --select_fold 0 --epochs 1 --batch_size 1
python train_STAG.py --data_name HEST_mouse_brain --k_folds 3 --select_fold 0 --epochs 1 --batch_size 1
python train_STAG.py --data_name HEST_PRAD --k_folds 3 --select_fold 0 --epochs 1 --batch_size 1
```

For 3D:

```bash
cd 3D
python main.py --config_name stnet --mode cv --select_fold 0 --gpu 0
```
