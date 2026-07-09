# Data Preparation

The full raw and preprocessed data are distributed outside this GitHub
repository because the archive is too large for a normal Git repository.

Current archive location:

```text
Aliyun Drive: /data/zyc-MEDIA-Re.zip
Archive size: about 200.95 GB
```

GitHub has a hard 100 MB single-file limit for normal Git objects and is not
suitable for committing the full raw WSI/ST archive directly to the repository.
The ready-to-run non-HEST 2D datasets are also several GB in total, so they
should be released as GitHub Release assets, Git LFS objects with sufficient
quota, or a HuggingFace Dataset. This repository therefore tracks:

- source code;
- gene panels and gene-text embeddings under `2D/select_genes/`;
- data layout documentation;
- restoration scripts.

The following data are intentionally not committed to normal Git history:

- `2D/data/`;
- preprocessed 3D dataset folders such as `3D/stnet_dataset_normal_smooth/`;
- raw WSI files such as `.tif`, `.tiff`, `.svs`;
- downloaded `.zip`/`.tar` archives.

## Ready Data Status

The following folders have been verified with the public training code:

```text
2D/data/GSE144240/
2D/data/HER2/
2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics/
3D/stnet_dataset_normal_smooth/
```

`2D/data/Hest1k_datasets/` is restored separately. `HEST_LUAD` and optional 3D
settings other than the HBC/STNet serial-section setting are marked as
supplementary.

## Download the Full Data Archive

The primary data source is the full Aliyun Drive archive:

```text
Aliyun Drive path: /data/zyc-MEDIA-Re.zip
```

From the repository root, run the restoration script. It downloads the archive
with the Aliyun Drive CLI and restores the recognized ready folders:

```bash
bash scripts/prepare_media_data_from_aliyunpan.sh \
  /path/to/aliyunpan \
  /data/zyc-MEDIA-Re.zip \
  .
```

Manual download, if needed:

```bash
./aliyunpan download --saveto /path/to/download_dir /data/zyc-MEDIA-Re.zip
```

This creates the main ready-to-run layout:

```text
2D/data/GSE144240/
2D/data/HER2/
2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics/
2D/weights/tenpercent_resnet18.ckpt
3D/stnet_dataset_normal_smooth/
3D/weights/tenpercent_resnet18.ckpt
```

## Optional Ready-Asset Mirror

The same ready-to-run folders can also be mirrored as smaller GitHub Release or
HuggingFace assets. For the GitHub Release mirror, download all assets from:

```text
https://github.com/MCPathology/STAG/releases/tag/data-v20260709
```

Then restore from the repository root:

```bash
tar --use-compress-program=unzstd -xf STAG-2D-GSE144240.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HER2.tar.zst
tar --use-compress-program=unzstd -xf STAG-2D-HBC.tar.zst

cat STAG-3D-HBC-stnet.tar.zst.part-* > STAG-3D-HBC-stnet.tar.zst
tar --use-compress-program=unzstd -xf STAG-3D-HBC-stnet.tar.zst

tar --use-compress-program=unzstd -xf STAG-weights-resnet18.tar.zst
sha256sum -c SHA256SUMS.txt
```

## Release Upload Helper

To publish the prepared assets to a GitHub release, use:

```bash
GITHUB_TOKEN=<token-with-contents-write> \
  bash scripts/upload_release_assets.sh /path/to/STAG_release_assets_20260709
```

For organization-owned repositories, make sure the token satisfies the
organization policy and has access to create releases for this repository.

## Restore from Aliyun Drive on the Project Server

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
`-- Human_breast_cancer_in_situ_capturing_transcriptomics/

3D/
`-- stnet_dataset_normal_smooth/
```

If the archive has a different top-level folder name, the script still searches
inside the extracted tree for the recognized dataset folders.

## Restore HEST-1k Subsets

HEST-1k data are restored from the full archive into:

```text
2D/data/Hest1k_datasets/PRAD/
2D/data/Hest1k_datasets/kidney/
2D/data/Hest1k_datasets/mouse_brain/
```

Each subset should keep the HEST loader layout:

```text
<subset>/st/*.h5ad
<subset>/wsis/*.tif
```

Selective restore example:

```bash
unzip /path/to/zyc-MEDIA-Re.zip \
  'zyc-MEDIA-Re/Data/Hest1k_datasets/PRAD/*' \
  'zyc-MEDIA-Re/Data/Hest1k_datasets/kidney/*' \
  'zyc-MEDIA-Re/Data/Hest1k_datasets/mouse_brain/*' \
  -d /tmp/stag_hest_restore

mkdir -p 2D/data/Hest1k_datasets
rsync -a /tmp/stag_hest_restore/zyc-MEDIA-Re/Data/Hest1k_datasets/PRAD 2D/data/Hest1k_datasets/
rsync -a /tmp/stag_hest_restore/zyc-MEDIA-Re/Data/Hest1k_datasets/kidney 2D/data/Hest1k_datasets/
rsync -a /tmp/stag_hest_restore/zyc-MEDIA-Re/Data/Hest1k_datasets/mouse_brain 2D/data/Hest1k_datasets/
```

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
