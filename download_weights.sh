#!/usr/bin/env bash
# Resumable parallel download of Z-Image-Turbo weights.
# Re-running this script resumes from where it left off (curl -C -).
set -u

REPO="Tongyi-MAI/Z-Image-Turbo"
RAW="/home/peanut/Projects/strata-diffusion/weights"
BASE="https://huggingface.co/$REPO/resolve/main"

mkdir -p "$RAW"

FILES=(
  "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
  "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
  "text_encoder/model-00001-of-00003.safetensors"
  "text_encoder/model-00002-of-00003.safetensors"
)

download_one() {
  local f="$1"
  local out="$RAW/$(basename "$f")"
  if [ -f "$out.done" ]; then
    echo "SKIP (done) $f"
    return 0
  fi
  echo "START $f"
  curl -L --fail --retry 5 --retry-delay 3 -C - \
       -o "$out" "$BASE/$f" 2>&1 | tail -1
  if [ -f "$out" ]; then
    touch "$out.done"
    echo "FINISHED $f ($(stat -c%s "$out") bytes)"
  else
    echo "FAILED $f"
  fi
}

# Download the three transformer shards in parallel (biggest wins first),
# then the text encoder shards.
download_one "${FILES[0]}" &
download_one "${FILES[1]}" &
download_one "${FILES[2]}" &
wait
download_one "${FILES[3]}" &
download_one "${FILES[4]}" &
wait

echo "ALL_DOWNLOADS_ATTEMPTED"
ls -lh "$RAW"
