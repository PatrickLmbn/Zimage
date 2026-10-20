"""Resumable download of Tongyi-MAI/Z-Image-Turbo weights."""
import os
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

t0 = time.time()
path = snapshot_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    repo_type="model",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
    max_workers=4,
)
print("DOWNLOAD_DONE", path, f"{time.time() - t0:.1f}s", flush=True)
