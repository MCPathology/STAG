#!/usr/bin/env bash
set -euo pipefail

ALIYUNPAN_BIN="${1:-aliyunpan}"
CLOUD_ARCHIVE="${2:-/data/zyc-MEDIA-Re.zip}"
REPO_ROOT="${3:-$(pwd)}"

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
WORK_DIR="${REPO_ROOT}/.data_restore"
ARCHIVE_DIR="${WORK_DIR}/archives"
EXTRACT_DIR="${WORK_DIR}/extracted"
ARCHIVE_NAME="$(basename "$CLOUD_ARCHIVE")"
LOCAL_ARCHIVE="${ARCHIVE_DIR}/${ARCHIVE_NAME}"

mkdir -p "$ARCHIVE_DIR" "$EXTRACT_DIR" "${REPO_ROOT}/2D/data"

echo "[STAG data] Repository root: ${REPO_ROOT}"
echo "[STAG data] Cloud archive: ${CLOUD_ARCHIVE}"
echo "[STAG data] Local archive: ${LOCAL_ARCHIVE}"

if [[ ! -x "$ALIYUNPAN_BIN" ]]; then
  echo "[ERROR] aliyunpan executable not found or not executable: ${ALIYUNPAN_BIN}" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_ARCHIVE" ]]; then
  echo "[STAG data] Downloading archive from Aliyun Drive..."
  "$ALIYUNPAN_BIN" download --saveto "$ARCHIVE_DIR" --ow "$CLOUD_ARCHIVE"

  DOWNLOADED="$(find "$ARCHIVE_DIR" -type f -name "$ARCHIVE_NAME" | head -n 1 || true)"
  if [[ -z "$DOWNLOADED" ]]; then
    echo "[ERROR] Download finished but ${ARCHIVE_NAME} was not found under ${ARCHIVE_DIR}" >&2
    exit 1
  fi
  if [[ "$DOWNLOADED" != "$LOCAL_ARCHIVE" ]]; then
    mv "$DOWNLOADED" "$LOCAL_ARCHIVE"
  fi
else
  echo "[STAG data] Archive already exists; skip download."
fi

echo "[STAG data] Extracting archive..."
if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$LOCAL_ARCHIVE" -d "$EXTRACT_DIR"
else
  python - <<PY
import zipfile
from pathlib import Path
archive = Path(r"$LOCAL_ARCHIVE")
out = Path(r"$EXTRACT_DIR")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
PY
fi

copy_first_match() {
  local name="$1"
  local dest="$2"
  local src
  src="$(find "$EXTRACT_DIR" -type d -name "$name" | head -n 1 || true)"
  if [[ -z "$src" ]]; then
    echo "[WARN] Not found in archive: ${name}"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[STAG data] Copy ${src} -> ${dest}"
  rm -rf "$dest"
  cp -a "$src" "$dest"
}

copy_first_match "GSE144240" "${REPO_ROOT}/2D/data/GSE144240"
copy_first_match "HER2" "${REPO_ROOT}/2D/data/HER2"
copy_first_match "Human_breast_cancer_in_situ_capturing_transcriptomics" "${REPO_ROOT}/2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics"
copy_first_match "Hest1k_datasets" "${REPO_ROOT}/2D/data/Hest1k_datasets"

copy_first_match "stnet_dataset_normal_smooth" "${REPO_ROOT}/3D/stnet_dataset_normal_smooth"
copy_first_match "her2st_heg250_dataset" "${REPO_ROOT}/3D/her2st_heg250_dataset"
copy_first_match "skin_dataset_normal_smooth" "${REPO_ROOT}/3D/skin_dataset_normal_smooth"
copy_first_match "pcw_dataset_normal_smooth" "${REPO_ROOT}/3D/pcw_dataset_normal_smooth"
copy_first_match "mouse_dataset_normal_smooth" "${REPO_ROOT}/3D/mouse_dataset_normal_smooth"

echo "[STAG data] Done. Check restored folders with:"
echo "  ls ${REPO_ROOT}/2D/data"
echo "  ls ${REPO_ROOT}/3D/*dataset*"
