# Data Preparation

The full raw and preprocessed data are distributed outside this GitHub
repository because the archive is too large for a normal Git repository.

Current archive location:

```text
Aliyun Drive: /data/zyc-MEDIA-Re.zip
Archive size: about 200.95 GB
```

GitHub has a hard 100 MB single-file limit and is not suitable for hosting the
full raw WSI/ST archive. This repository therefore tracks:

- source code;
- gene panels and gene-text embeddings under `2D/select_genes/`;
- data layout documentation;
- restoration scripts.

The following data are intentionally not committed:

- `2D/data/`;
- preprocessed 3D dataset folders such as `3D/stnet_dataset_normal_smooth/`;
- raw WSI files such as `.tif`, `.tiff`, `.svs`;
- downloaded `.zip`/`.tar` archives.

## Restore from Aliyun Drive on the Server

From the repository root, run:

```bash
bash scripts/prepare_media_data_from_aliyunpan.sh \
  /mnt/pfs-gv8sxa/tts/dhg/yg/zyc/aliyunpan-v0.4.0-linux-amd64/aliyunpan \
  /data/zyc-MEDIA-Re.zip \
  .
```

The script downloads the archive, extracts it, and copies recognized folders
into the expected STAG layout:

```text
2D/data/
|-- GSE144240/
|-- HER2/
|-- Human_breast_cancer_in_situ_capturing_transcriptomics/
`-- Hest1k_datasets/

3D/
|-- stnet_dataset_normal_smooth/
|-- her2st_heg250_dataset/
|-- skin_dataset_normal_smooth/
|-- pcw_dataset_normal_smooth/
`-- mouse_dataset_normal_smooth/
```

If the archive has a different top-level folder name, the script still searches
inside the extracted tree for the recognized dataset folders.

## Verify Data Placement

After extraction, check:

```bash
ls 2D/data
ls 3D/*dataset*
```

Then run a short smoke test:

```bash
cd 2D
python train_STAG.py --data_name cSCC --k_folds 4 --epochs 1 --batch_size 2
```

For 3D:

```bash
cd 3D
python main.py --config_name stnet --mode cv --select_fold 0 --gpu 0
```
