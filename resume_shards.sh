#!/usr/bin/env bash
# Resume the two large INT4 shards in parallel. Safe to re-run.
set -u
D="/home/peanut/Projects/strata-diffusion/models/z-image-turbo-int4"
R="Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"
B="https://huggingface.co/$R/resolve/main"

get() {
  local path="$1" out="$2" want="$3"
  while true; do
    local have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    if [ "$have" -ge "$want" ]; then
      echo "COMPLETE $path ($have bytes)"
      return 0
    fi
    curl -L --fail --retry 3 --retry-delay 2 -C - -o "$out" "$B/$path" >/dev/null 2>&1
    sleep 1
  done
}

get "transformer/diffusion_pytorch_model.safetensors" "$D/transformer/diffusion_pytorch_model.safetensors" 3474875882 &
get "text_encoder/model.safetensors" "$D/text_encoder/model.safetensors" 2839159572 &
wait
echo "ALL_SHARDS_COMPLETE"
