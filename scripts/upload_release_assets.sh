#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: set GITHUB_TOKEN before running this script." >&2
  exit 1
fi

ASSET_DIR="${1:-}"
TAG="${2:-data-v20260709}"
REPO="${3:-MCPathology/STAG}"

if [[ -z "$ASSET_DIR" || ! -d "$ASSET_DIR" ]]; then
  echo "Usage: GITHUB_TOKEN=... bash scripts/upload_release_assets.sh <asset_dir> [tag] [repo]" >&2
  exit 1
fi

cd "$ASSET_DIR"

cat > RELEASE_BODY.md <<'EOF'
Ready-to-run data assets for STAG.

Contents:
- `STAG-2D-GSE144240.tar.zst`: extracts to `2D/data/GSE144240/`
- `STAG-2D-HER2.tar.zst`: extracts to `2D/data/HER2/`
- `STAG-2D-HBC.tar.zst`: extracts to `2D/data/Human_breast_cancer_in_situ_capturing_transcriptomics/`
- `STAG-3D-HBC-stnet.tar.zst.part-000` and `part-001`: concatenate, then extract to restore `3D/stnet_dataset_normal_smooth/`
- `STAG-weights-resnet18.tar.zst`: optional ResNet18 backbone weights for `2D/` and `3D/`
- `SHA256SUMS.txt`: checksums

Reconstruct the 3D HBC/STNet archive:

```bash
cat STAG-3D-HBC-stnet.tar.zst.part-* > STAG-3D-HBC-stnet.tar.zst
tar --use-compress-program=unzstd -xf STAG-3D-HBC-stnet.tar.zst
```
EOF

api="https://api.github.com/repos/${REPO}"
headers=(
  -H "Authorization: Bearer ${GITHUB_TOKEN}"
  -H "Accept: application/vnd.github+json"
  -H "X-GitHub-Api-Version: 2022-11-28"
)

release_json="$(mktemp)"
status="$(curl -sS -o "$release_json" -w "%{http_code}" "${headers[@]}" "${api}/releases/tags/${TAG}")"

if [[ "$status" == "404" ]]; then
  status="$(curl -sS -o "$release_json" -w "%{http_code}" "${headers[@]}" \
    -H "Content-Type: application/json" \
    -d "$(python3 - <<PY
import json
body=open('RELEASE_BODY.md', encoding='utf-8').read()
print(json.dumps({'tag_name':'${TAG}','name':'STAG ready data assets (2026-07-09)','body':body,'draft':False,'prerelease':False}))
PY
)" \
    "${api}/releases")"
fi

if [[ "$status" != "200" && "$status" != "201" ]]; then
  echo "ERROR: failed to create/read release (HTTP $status)" >&2
  cat "$release_json" >&2
  exit 1
fi

release_id="$(python3 - <<PY
import json
print(json.load(open('$release_json'))['id'])
PY
)"
upload_url="https://uploads.github.com/repos/${REPO}/releases/${release_id}/assets"

assets=(
  SHA256SUMS.txt
  STAG-2D-GSE144240.tar.zst
  STAG-2D-HER2.tar.zst
  STAG-2D-HBC.tar.zst
  STAG-3D-HBC-stnet.tar.zst.part-000
  STAG-3D-HBC-stnet.tar.zst.part-001
  STAG-weights-resnet18.tar.zst
)

for asset in "${assets[@]}"; do
  [[ -f "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 1; }
  echo "Uploading $asset"
  status="$(curl -sS -o /tmp/stag_upload_response.json -w "%{http_code}" \
    "${headers[@]}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$asset" \
    "${upload_url}?name=$(python3 - <<PY
from urllib.parse import quote
print(quote('$asset'))
PY
)")"
  if [[ "$status" != "201" ]]; then
    echo "ERROR: upload failed for $asset (HTTP $status)" >&2
    cat /tmp/stag_upload_response.json >&2
    exit 1
  fi
done

echo "Uploaded assets to https://github.com/${REPO}/releases/tag/${TAG}"
