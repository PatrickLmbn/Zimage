#!/usr/bin/env bash
# Build a local diffusers-layout model dir for Z-Image-Turbo:
#   - INT4 (sdnq) transformer + text_encoder + vae from Disty0/...-SDNQ-uint4-svd-r32
#   - tokenizer / scheduler / configs from that repo
#   - tokenizer files fall back to the official repo if missing
set -u

INT4="Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"
OFFICIAL="Tongyi-MAI/Z-Image-Turbo"
DEST="/home/peanut/Projects/strata-diffusion/models/z-image-turbo-int4"

mkdir -p "$DEST"/{transformer,text_encoder,vae,tokenizer,scheduler}

fetch() {  # repo path dest
  local repo="$1" path="$2" dest="$3"
  if [ -s "$dest" ]; then
    echo "SKIP  $path"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  curl -L --fail --retry 5 --retry-delay 2 -C - \
       -o "$dest" "https://huggingface.co/$repo/resolve/main/$path" >/dev/null 2>&1 \
    && echo "OK    $path" || echo "FAIL  $path"
}

echo "--- small files (INT4 repo) ---"
for f in model_index.json \
         scheduler/scheduler_config.json \
         text_encoder/config.json \
         text_encoder/generation_config.json \
         text_encoder/quantization_config.json \
         tokenizer/tokenizer.json \
         tokenizer/tokenizer_config.json \
         tokenizer/merges.txt \
         tokenizer/vocab.json \
         transformer/config.json \
         transformer/quantization_config.json \
         vae/config.json; do
  fetch "$INT4" "$f" "$DEST/$f"
done

echo "--- big files (INT4 repo) ---"
fetch "$INT4" "transformer/diffusion_pytorch_model.safetensors" "$DEST/transformer/diffusion_pytorch_model.safetensors" &
fetch "$INT4" "text_encoder/model.safetensors"             "$DEST/text_encoder/model.safetensors" &
fetch "$INT4" "vae/diffusion_pytorch_model.safetensors"    "$DEST/vae/diffusion_pytorch_model.safetensors" &
wait

echo "--- tokenizer fallback (official repo) ---"
for f in tokenizer/tokenizer.json tokenizer/tokenizer_config.json tokenizer/merges.txt tokenizer/vocab.txt tokenizer/vocab.json; do
  if [ ! -s "$DEST/$f" ]; then
    fetch "$OFFICIAL" "$f" "$DEST/$f"
  fi
done

echo "--- result ---"
du -sh "$DEST"
find "$DEST" -type f -printf "%10s  %P\n" | sort -k2
