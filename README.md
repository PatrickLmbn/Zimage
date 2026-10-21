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
time. Two things make it fit on a 4 GB card:

- **The text encoder stays in RAM but encodes on the GPU.** It is 2.65 GB of packed
  INT4, which does not fit on the card alongside everything else, so it is staged in
  and out around each request (`vae → CPU` → `encoder → GPU` → encode → back). On
  the card the encode takes ~1 s instead of ~16 s on the CPU.
- **The transformer streams one block at a time.** Only one of its 30 blocks is
  resident in VRAM; the next is prefetched over PCIe on a side stream while the
  current one computes. That keeps peak VRAM at ~0.77 GB instead of the ~3.2 GB the
  full model wants.

One generation runs at a time behind a lock, since the GPU is a single shared
resource.

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

Measured through the HTTP API, 8 steps @ 512 px:

| Metric | Value |
|---|---|
| Full request | ~28 s |
| Prompt encode (GPU) | ~2 s |
| Generation | ~25 s (≈3.2 s/step) |
| VAE decode | ~0.6 s |
| Peak VRAM (generation) | 0.77 GB |
| Backend RSS | ~1.6 GB idle, ~8.2 GB under load |

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
