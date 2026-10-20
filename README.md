# Zimage

A local web UI for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image
generation, quantized to INT4 so it runs on a 4 GB GPU (RTX 2050).

![stack](https://img.shields.io/badge/backend-FastAPI-informational) ![stack](https://img.shields.io/badge/frontend-React_19%20%2B%20Tailwind_v4-informational) ![model](https://img.shields.io/badge/model-sdnq_uint4-9cf3ef)

## What it is

A FastAPI backend that loads the INT4 Z-Image-Turbo pipeline and a React frontend to
drive it: type a prompt, pick a resolution and step count, watch the job run with live
VRAM/CPU/RAM telemetry, and browse the results in a gallery.

## How it works

The model is loaded once at startup and kept resident, so requests skip the ~25 s load
time. The text encoder stays on **CPU** (never touches VRAM); only the transformer uses
group offload, with `num_blocks_per_group` reconfigured per request so just a few
transformer blocks are resident on the GPU at a time. That keeps peak VRAM at ~0.77 GB
instead of the ~4 GB the full model wants. One generation runs at a time behind a lock,
since the GPU is a single shared resource.

## Setup

```bash
./fetch_int4_model.sh     # download the INT4 checkpoint into models/
cd frontend && npm install && cd ..
```

Weights and the model directory are gitignored — re-fetch them with the scripts above.

## Run

```bash
./run.sh                  # backend :8000 + frontend :5173
./run.sh backend          # backend only
./run.sh frontend         # frontend only
```

Open **http://127.0.0.1:5173**.

## Performance on an RTX 2050 (16 GB RAM)

| Metric | Value |
|---|---|
| 8 steps @ 512px, 1 block | ~34 s (≈4.2 s/step) |
| Peak VRAM | 0.77 GB |
| Backend RSS | ~1.5 GB idle, ~5.9 GB under load |
| Prompt encode | ~18 s (CPU) |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model, VRAM free, RSS |
| GET | `/api/settings` | allowed sizes, steps, block counts |
| GET | `/api/system` | realtime CPU/RAM/swap/disk/GPU telemetry |
| GET | `/api/headroom` | predicted VRAM cost for a (size, blocks) choice |
| POST | `/api/generate` | queue a job → `{job_id}` |
| GET | `/api/jobs` | all jobs, newest first |
| GET | `/api/jobs/{id}` | job state/progress/metrics |
| DELETE | `/api/jobs/{id}` | cancel a job |
| GET | `/api/image/{id}` | generated PNG |
| GET | `/api/gallery` | all generated images with prompt + metrics |

See [README-webui.md](README-webui.md) for the full UI behaviour and memory-strategy notes.

## License

Model weights are subject to the upstream [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
license. The application code has no license file yet — add one before treating it as
open source.
