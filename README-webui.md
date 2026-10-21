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
- **VRAM headroom preview.** Under the Resolution control, a panel shows what the
  current choice will cost: the predicted device-level VRAM peak, the free space
  left afterwards, a verdict (comfortable / ok / tight / won't fit), and which
  phase sets the peak. Backed by `backend/headroom.py`, a model calibrated on
  this machine by measuring the real transformer forward and the real VAE
  decode for every (size, blocks) combination. It reproduces the observed
  production `peak_vram_gb` values exactly (worst error 2 MiB).
- **VRAM blocks control.** Removed. The streamed offload requires exactly one
  block per group, and measurements showed larger groups are slower, so the
  choice was always a no-op. The API still accepts `blocks` for compatibility
  and ignores it.
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

- The text encoder (2.65 GB INT4) stays **resident on CPU** between requests, so
  each request skips an ~18 s load. It is moved **onto the GPU for the encode
  itself** and moved straight back: sdnq has CUDA kernels for the quantized
  Qwen3, so the encode takes ~0.8 s on the card instead of ~16 s on the CPU.
  Because the encoder and the VAE decode peak do not fit together in 4 GB, the
  handoff is staged around every request:
  `vae → CPU` → `encoder → GPU` → encode → `encoder → CPU` → `vae → GPU`.
- Only the **transformer** uses group offload, with **one** block resident in
  VRAM at a time and the next prefetched over PCIe on a side stream.
- `use_stream=True` **is** enabled, together with `low_cpu_mem_usage=True`. The
  stream is what lets diffusers overlap the transfer with the compute. The
  `low_cpu_mem_usage` flag is what makes it affordable here: without it
  diffusers keeps a second *pinned* CPU copy of every weight (+3.25 GB), which
  does swap on this box. With it the pinned copy is made on the fly, so peak RSS
  is actually **lower** than the no-stream path (measured 4.2 GB vs 6.3 GB, swap
  delta ~0).
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

## Where the time goes

The bottleneck is PCIe, not compute. Instrumenting diffusers' `onload_`/`offload_`
hooks to sum the logical bytes of every tensor crossing the bus shows **6.5 GB
of PCIe traffic per denoising step** — 52 GB for an 8-step 512 px image — while
GPU compute is only ~7% of wall time. Every layer crosses the bus twice per step
because the group-offload hook onloads in `pre_forward` and offloads back in
`post_forward`.

Two things follow from that:

- **More resident blocks is slower, not faster.** Measured: 1 block 3.94–4.17
  s/step, 3 blocks 4.95–5.46, 4 blocks 4.74–4.95. The block count is therefore
  fixed at 1.
- **The prompt encode was the single biggest cost.** At ~16 s it was more than
  a third of a request. Moving it to the GPU is where most of the speedup came
  from.

## Measured on this box (RTX 2050, 16 GB RAM)

Full request, 8 steps @ 512 px, measured through the HTTP API (2 runs each):

| | old | new |
|---|---|---|
| Wall clock | 51.5 s / 48.6 s | **28–29 s** |
| Prompt encode | ~16 s (CPU) | **1.8–2.7 s (GPU)** |
| Generation | 30–36 s | 25–26 s |
| Per step | 3.79–4.44 s | **3.17–3.20 s** |
| Decode | (inside pipeline) | 0.5–0.8 s |
| Peak VRAM | 0.766 GB | 0.772 GB |
| Peak RSS | ~10.3 GB | ~8.2 GB |

### Why the encode is "try GPU, fall back to CPU"

The encoder is 2.7 GB of packed INT4, and on the GPU it also needs sdnq's ~48 MB
per-layer dequantisation buffer. The card only has ~3.7 GB usable after the CUDA
context, so the fit depends on what else is on it. Two things make it reliable:

- The VAE is kept **off** the card between requests (staged in only for the
  decode), which buys the encode ~170 MB.
- Do **not** call `empty_cache()` before staging the encoder. Releasing the
  cached segment forces a fresh 2.7 GB driver allocation that fragments; reusing
  the existing block is what makes it fit.

Even so, on a cold allocator the first encode could not find a contiguous block,
so the backend performs one throwaway encode at startup. With the warmup, three
consecutive requests all took the ~2 s GPU path with no fallback. If the card is
genuinely too full (another app holding VRAM), the encode falls back to the CPU
(~17 s) rather than failing the request.

| Other metrics | Value |
|---|---|
| 8 steps @ 512px | ~26–28 s generation (≈3.3 s/step) |
| Peak VRAM (generation) | 0.77 GB of ~2.8 GB free |
| Peak VRAM (during encode) | 2.80 GB — the encoder briefly owns the card |
| Backend RSS | ~1.6 GB idle, ~8.2 GB under load |

### A note on reproducibility

The GPU encode does not produce bit-identical embeddings to the CPU encode
(cosine similarity 1.00001, ~45% of pixels differ). It is bf16 kernel
nondeterminism, not a bug, but it means seeds from before this change will not
reproduce exactly.
