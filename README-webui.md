# Z-Image-Turbo INT4 — Local Web UI

A local browser interface for the Z-Image-Turbo INT4 (sdnq uint4) model running on
an RTX 2050 4 GB. React + shadcn/ui + Lucide icons, strictly monochrome light/dark.

```
backend/     FastAPI server — loads the model, runs generation jobs
             sysmon.py    CPU/RAM/swap/disk/GPU telemetry for the UI
frontend/    Vite + React 19 + Tailwind v4 + shadcn/ui (base/nova)
outputs/     generated PNGs (web_<jobid>.png) + metadata sidecars
             (web_<jobid>.json)
baseline.py  standalone CLI generation, same memory strategy
```

## Run

```bash
./run.sh              # both: backend :8000, frontend :5173
./run.sh backend      # backend only
./run.sh frontend     # frontend only
```

Then open **http://127.0.0.1:5173**.

Manual equivalent:

```bash
cd backend && ../.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model, VRAM free, RSS |
| GET | `/api/settings` | allowed sizes, steps, block counts |
| GET | `/api/system` | realtime CPU/per-core/RAM/swap/disk/GPU telemetry |
| GET | `/api/headroom` | predicted VRAM cost for a (size, blocks) choice, or the full grid |
| POST | `/api/generate` | queue a job → `{job_id}` |
| GET | `/api/jobs` | all jobs, newest first (state restore) |
| GET | `/api/jobs/{id}` | job state/progress/metrics |
| DELETE | `/api/jobs/{id}` | cancel a job (queued: immediate; running: next step) |
| GET | `/api/image/{id}` | generated PNG |
| GET | `/api/gallery` | all generated images with prompt + metrics, newest first |

## UI behaviour

- **Refresh-safe.** The active job id is kept in `localStorage`
  (`zit.activeJob`). On load the UI reattaches to a running job and keeps
  showing its stage, progress and the Stop button.
- **Stop button.** Replaces Generate while a job is active. Queued jobs are
  cancelled immediately; running jobs abort at the next step boundary
  (diffusers has no hard interrupt, so cancellation is cooperative).
- **System monitor.** Right column. Polls `/api/system` at 1 Hz: GPU load and
  RAM history charts, per-core CPU load, CPU, RAM, swap (zram), VRAM, temp,
  power, disk usage, and the pipeline's own state (offload mode, resident VRAM
  blocks, backend RSS).
- **VRAM headroom preview.** Under the Resolution / VRAM-blocks controls, a
  panel shows what the current choice will cost: the predicted device-level
  VRAM peak, the free space left afterwards, a verdict (comfortable / ok /
  tight / won't fit), and which phase sets the peak. Backed by
  `backend/headroom.py`, a model calibrated on this machine by measuring the
  real transformer forward and the real VAE decode for every (size, blocks)
  combination. It reproduces the observed production `peak_vram_gb` values
  exactly (worst error 2 MiB).
- **GPU memory figures.** `vram_free_gb` / `vram_used_gb` come from NVML
  `nvmlDeviceGetMemoryInfo` — device-wide physical VRAM, and they match
  `nvidia-smi` to within a few MiB. The **VRAM bar** fills with
  `memory_percent` = `used / total` (true occupancy), *not* NVML's
  `utilization.memory`, which measures memory-controller busy time and is
  reported separately as `memory_controller_percent`. `/api/health` reports
  free VRAM via `torch.cuda.mem_get_info` (same physical figure, one decimal
  place different).
- **Info icons.** Every control (prompt, resolution, VRAM blocks, steps, seed)
  has a circled ⓘ that explains what it does and how it affects speed/VRAM.
- **Layout.** Three columns — Prompt | Gallery | System — as equal-height
  cards that share the same top and bottom edges. The page fills the viewport
  with no page-level scrollbar; the gallery scrolls inside its own card, and
  the prompt controls scroll inside theirs so the Generate/Stop button stays
  pinned and reachable.
- **Durable gallery metadata.** Each finished job writes a
  `outputs/web_<jobid>.json` sidecar next to its PNG. `/api/gallery` reads
  the in-memory job when the backend is still running it and falls back to the
  sidecar otherwise, so prompts and metrics survive a backend restart.

## Memory strategy (why it fits in 4 GB VRAM / 16 GB RAM)

- The text encoder (2.65 GB INT4) stays **resident on CPU** — loaded once at
  startup, so each request skips an ~18 s load. It never touches VRAM.
- Only the **transformer** uses group offload. `num_blocks_per_group` is
  reconfigured per request: the `group_offloading` hooks are removed and
  re-registered with the new group size (verified: with 30 blocks the group
  size tracks the request exactly — 1→1, 2→2, 4→2/4, 8→6/8).
  Peak VRAM is **0.77 GB** at 1 block.
- `use_stream=False` — stream mode keeps a pinned CPU copy of every weight
  (+3.25 GB), which swaps on this box.
- `pipe.transformer.enable_group_offload(...)`, **not** `pipe.enable_group_offload(...)`.
  The pipeline-level call sweeps every submodule and drags the text encoder onto
  the GPU, which OOMs the card. Likewise `exclude_modules` means "keep this ON
  the accelerator", not "skip this one".
- Prompt encoding uses **dynamic padding** (`padding=False`). The stock
  pipeline pads to 512 tokens, making a 16-token prompt take 64 s instead of
  17 s for identical output.
- One generation at a time, behind a lock — the GPU is a single shared resource.
- Disk offload (`offload_to_disk_path`) does **not** work with sdnq: the uint4
  packed weights are non-contiguous, so diffusers' disk-backed offload fails to
  restore them.

## Measured on this box (RTX 2050, 16 GB RAM)

| Metric | Value |
|---|---|
| 8 steps @ 512px, 1 block | ~34 s (≈4.2 s/step) |
| Peak VRAM | 0.77 GB of ~2.5 GB free |
| Backend RSS | ~1.5 GB idle, ~5.9 GB under load |
| Prompt encode | ~18 s (CPU) |
