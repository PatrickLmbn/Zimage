"""
FastAPI backend for the Z-Image-Turbo INT4 (sdnq) web UI.

Serves the React frontend and exposes generation over HTTP.

Design constraints (RTX 2050 4 GB VRAM / 16 GB RAM, ~7 GB RAM actually free):

  * The INT4 transformer is 3.25 GB packed and the text encoder is 2.65 GB packed.
    They cannot both live in VRAM, and together they strain RAM.

  * The fix is TWO SEPARATE resident sets that never overlap in VRAM:
      - the transformer is block-offloaded to VRAM (1 block at a time, ~0.17 GB)
      - the text encoder lives on CPU and is loaded ONCE at startup, staying
        resident so each request skips an ~18 s load
    The text encoder is CPU-only, so it never competes for VRAM.

  * IMPORTANT: diffusers' `enable_group_offload(exclude_modules=[...])` means
    "do NOT offload these, keep them on the accelerator" -- the opposite of what
    the name suggests. Passing the text encoder there moves 2.65 GB to the GPU
    and OOMs the 4 GB card. The text encoder is therefore offloaded separately
    with its own CPU placement, and only the transformer uses group offload.

  * PROMPT ENCODING RUNS ON THE GPU. The text encoder is 2.65 GB of packed INT4
    and sdnq has CUDA kernels for it, so it encodes in ~0.8 s on the card versus
    ~16-18 s on the CPU. It cannot stay there: the VAE decode needs its own
    activation peak and the two together do not fit in 4 GB. So the encoder is
    staged in and out around each request:
        1. vae   -> CPU      (frees 160 MB)
        2. encoder -> GPU    (needs ~2.7 GB free)
        3. encode
        4. encoder -> CPU    (releases 2.7 GB)
        5. vae   -> GPU
    Measured: a full 8-step 512px request drops from ~70 s to ~30 s.

  * GROUP OFFLOAD USES A STREAM. `use_stream=True` lets diffusers prefetch the
    next block group over PCIe while the current one computes, which cuts
    generation from ~4.5 s/step to ~3.5 s/step. `low_cpu_mem_usage=True` is
    required alongside it: without it diffusers keeps a second PINNED CPU copy
    of every weight (+3.25 GB), which does swap on this box. With it, the
    pinned copy is made on the fly and peak RSS is LOWER than the no-stream
    path (measured 4.2 GB vs 6.3 GB, swap delta ~0).

  * Prompt encoding uses DYNAMIC padding. The stock pipeline pads to 512 tokens;
    a typical prompt is only ~16-30 tokens, and padding to 512 makes the CPU
    encode take 64 s instead of 17 s for identical output.

  * Generation runs in one worker thread behind a lock: the GPU is a single
    shared resource and the model is not thread-safe.

Endpoints:
    GET  /api/health        -> status, model info, VRAM/RAM
    GET  /api/settings      -> resolution / steps / block-size options
    POST /api/generate      -> queue a generation job, returns job id
    GET  /api/jobs/{id}     -> job status (queued/running/done/error)
    GET  /api/image/{id}    -> the generated PNG
    GET  /api/gallery       -> previously generated images
    DELETE /api/jobs/{id}   -> cancel a queued job
"""

import ctypes
import gc
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

# Set BEFORE importing transformers/diffusers, otherwise the SDNQ quantization
# config on the text encoder is ignored and it loads as bf16 (2x RAM).
os.environ.setdefault("DIFFUSERS_SDNQ_TRANSFORMERS", "1")
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import headroom  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.environ.get("ZIMAGE_MODEL_DIR", PROJECT_ROOT / "models" / "z-image-turbo-int4")
)
OUT_DIR = Path(os.environ.get("ZIMAGE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

JobState = Literal["queued", "running", "done", "error"]

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # pragma: no cover
    _LIBC = None


def malloc_trim() -> None:
    """Return freed glibc arenas to the OS so RSS reflects real usage."""
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


# --------------------------------------------------------------------------- models
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=8, ge=1, le=50)
    size: int = Field(default=512)
    seed: int = Field(default=-1, ge=-1, le=2**31 - 1)
    guidance: float = Field(default=0.0, ge=0.0, le=20.0)
    # Accepted for API compatibility but ignored: the streamed group offload
    # requires exactly one block per group (diffusers overrides anything
    # larger), and measurements showed larger groups are SLOWER anyway
    # (1 block 3.94-4.17 s/step, 3 blocks 4.95-5.46, 4 blocks 4.74-4.95).
    blocks: int = Field(default=1, ge=1, le=8)


class Job(BaseModel):
    id: str
    state: JobState = "queued"
    prompt: str = ""
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    progress: float = 0.0
    stage: str = "queued"
    image_url: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None


# --------------------------------------------------------------------------- state
JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()

# Cooperative cancellation: diffusers has no interrupt API, so a cancel request
# sets this flag and the step callback raises to abort the run.
CANCEL_FLAGS: dict[str, threading.Event] = {}

PIPE: Any = None
TOKENIZER: Any = None
TEXT_ENCODER: Any = None
LOAD_ERROR: str | None = None
ACTIVE_BLOCKS: int = 1

# Sizes must be a multiple of vae_scale_factor * 2 == 16.
ALLOWED_SIZES = [256, 384, 512, 640, 768]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


# --------------------------------------------------------------------------- metadata
def meta_path(job_id: str) -> Path:
    """Sidecar JSON file holding a job's prompt and metrics."""
    return OUT_DIR / f"web_{job_id}.json"


def save_job_meta(job: Job) -> None:
    """Write a job's metadata to disk next to its PNG.

    Without this the gallery loses every prompt and metric as soon as the
    backend restarts, because the in-memory JOBS dict is the only other
    place the data lives.
    """
    try:
        meta_path(job.id).write_text(
            json.dumps(
                {
                    "job_id": job.id,
                    "prompt": job.prompt,
                    "created_at": job.created_at,
                    "metrics": job.metrics,
                },
                indent=2,
            )
        )
    except OSError as exc:
        log(f"job {job.id}: could not save metadata: {exc}")


def load_job_meta(job_id: str) -> dict[str, Any] | None:
    """Read a job's persisted metadata, if any."""
    path = meta_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- model
def set_block_offload(num_blocks: int) -> None:
    """(Re)configure group offload so `num_blocks` transformer blocks sit in
    VRAM at a time.

    diffusers registers one `group_offloading` hook per block group, so changing
    the group size means removing every existing hook and re-applying the
    offload with the new `num_blocks_per_group`.

    The stream is ON: it lets diffusers prefetch the next group over PCIe while
    the current one computes, which is the difference between ~4.5 s/step and
    ~3.5 s/step on this card. `low_cpu_mem_usage=True` is what makes that safe
    here -- without it diffusers keeps a second pinned CPU copy of every weight
    (+3.25 GB) and this box swaps. With it, the pinned copy is made on the fly,
    so peak RSS is actually LOWER than the no-stream path.

    Note: diffusers forces `num_blocks_per_group=1` when a stream is used, so
    the block count is effectively always 1 on this configuration.

    Must only be called while the GPU lock is held, or between jobs.
    """
    global ACTIVE_BLOCKS
    if PIPE is None:
        raise RuntimeError("pipeline not loaded")

    # Drop the root registry's hook and every child's own registry instance.
    root_reg = getattr(PIPE.transformer, "_diffusers_hook", None)
    if root_reg is not None:
        root_reg.remove_hook("group_offloading", recurse=True)
    for _name, mod in list(PIPE.transformer.named_modules()):
        reg = getattr(mod, "_diffusers_hook", None)
        if reg is None:
            continue
        if reg.get_hook("group_offloading") is not None:
            reg.remove_hook("group_offloading", recurse=False)

    PIPE.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=num_blocks,
        use_stream=True,
        low_cpu_mem_usage=True,
    )
    ACTIVE_BLOCKS = num_blocks
    gc.collect()
    malloc_trim()
    torch.cuda.empty_cache()
    log(f"  group offload reconfigured -> {num_blocks} block(s) per group (streamed)")


def load_pipeline() -> None:
    """Load the pipeline and the text encoder. Both stay resident on CPU."""
    global PIPE, TOKENIZER, TEXT_ENCODER, LOAD_ERROR

    from diffusers import ZImagePipeline
    import sdnq  # noqa: F401  -- registers sdnq into transformers.quantizers.auto
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    log(f"loading pipeline from {MODEL_DIR} ...")
    t0 = time.time()
    PIPE = ZImagePipeline.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    log(f"  pipeline loaded in {time.time() - t0:.1f}s  RSS={rss_mb():.0f} MB")

    # The text encoder stays resident in RAM for the whole session: it is the
    # same cost whether we load it now or per-request, and keeping it resident
    # removes an ~18 s load from every request. It is moved to the GPU only for
    # the duration of the encode (see encode_prompt) and moved straight back.
    log("  loading text encoder (CPU, resident) ...")
    t0 = time.time()
    TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    TEXT_ENCODER = Qwen3ForCausalLM.from_pretrained(
        str(MODEL_DIR / "text_encoder"), dtype=torch.bfloat16, device_map="cpu"
    )
    PIPE.text_encoder = TEXT_ENCODER
    log(f"  text encoder loaded in {time.time() - t0:.1f}s  RSS={rss_mb():.0f} MB")

    # Block-offload the transformer: only N blocks live in VRAM at a time.
    # Apply this to the TRANSFORMER ONLY. `pipe.enable_group_offload(...)` would
    # sweep every component and drag the 2.65 GB text encoder onto the GPU and
    # OOM the 4 GB card. And `exclude_modules` does not help: it means "keep
    # this on the accelerator", not "skip this one".
    set_block_offload(1)
    # The VAE is small (160 MB) but its decode is the activation peak of the
    # whole pipeline, so it is kept OFF the card between requests and staged in
    # only for the decode. Leaving it resident would eat into the prompt
    # encode's headroom, which is the tightest moment in the request.
    PIPE.vae.to("cpu")
    log("  transformer group-offloaded (streamed), vae on cpu, text encoder on cpu")

    gc.collect()
    malloc_trim()
    torch.cuda.empty_cache()
    log(
        f"  ready. RSS={rss_mb():.0f} MB  "
        f"VRAM free={torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB"
    )

    # Warm the prompt-encode path. On a cold allocator the encoder's 2.7 GB
    # cannot find a contiguous block and the first real request falls back to
    # the ~17 s CPU encode. Doing one throwaway encode here makes the block
    # resident so every request gets the ~1 s GPU path.
    #
    # NOTE: on this card the GPU encode usually does NOT fit at all -- the
    # desktop holds ~1.2 GB and the encoder wants the rest. In that case the
    # warmup falls back to the CPU too, and every request pays ~17 s. The
    # log line below says which path is actually live, because "warmup done
    # (GPU path resident)" used to be printed even when it had fallen back.
    try:
        t0 = time.time()
        _ = encode_prompt("warmup")
        del _
        gc.collect()
        torch.cuda.empty_cache()
        encode_free = torch.cuda.mem_get_info(0)[0] / 1024**3
        if encode_free >= 2.85:
            log(f"  prompt-encode warmup done, GPU path resident ({encode_free:.2f} GB free)")
        else:
            log(
                f"  prompt-encode warmup done, but the GPU path does not fit "
                f"({encode_free:.2f} GB free, needs 2.85 GB) -- requests will use "
                f"the CPU encode (~17 s each)"
            )
    except Exception as exc:  # noqa: BLE001
        log(f"  warmup skipped: {type(exc).__name__}: {exc}")


def _encode_text(text: str, device: str) -> "torch.Tensor":
    """Run the encode on `device`. Caller manages where the encoder lives."""
    templated = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = TOKENIZER(
        [templated],
        padding=False,  # dynamic: a 16-token prompt must not cost a 512-token pass
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    mask = inputs.attention_mask.bool()
    with torch.no_grad():
        out = TEXT_ENCODER(
            input_ids=inputs.input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
    return out.hidden_states[-2][0][mask[0]].clone()


def encode_prompt(text: str) -> "torch.Tensor":
    """Encode a prompt, preferring the GPU but falling back to the CPU.

    The text encoder is 2.7 GB of packed INT4. On the CPU it takes 16-18 s per
    request; sdnq has CUDA kernels for it, so on the card it takes ~1 s.

    On the GPU it is a very close fit: 2.7 GB of weights plus sdnq's ~48 MB
    per-layer dequantisation buffer, on a card that only has ~3.7 GB usable
    after the CUDA context and any other process. Whether that fits depends on
    what else is on the card at the time, so the encode is attempted on the GPU
    and falls back to the CPU rather than failing the request. The fallback
    costs ~15 s, but it is correct, and it keeps the fast path whenever the
    card has room.

    The VAE is kept on the CPU between requests: it is staged in for the decode
    instead, so the encoder gets the whole card for this phase.

    The caller must hold the GPU lock.
    """
    if TEXT_ENCODER is None:
        raise RuntimeError("text encoder not loaded")

    # Make sure nothing else of ours is holding VRAM. NOTE: no empty_cache()
    # here. Releasing the cached segment forces the encoder's 2.7 GB to come
    # from a fresh driver allocation, which fragments on this card and makes
    # the encode OOM; reusing the existing cached block is what lets it fit.
    PIPE.vae.to("cpu")
    torch.cuda.synchronize()

    try:
        TEXT_ENCODER.to("cuda")
        torch.cuda.synchronize()
        try:
            embeds = _encode_text(text, "cuda")
        finally:
            TEXT_ENCODER.to("cpu")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return embeds
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        # RuntimeError covers the device-mismatch that sdnq raises when a
        # partial move leaves some layers behind after a failed .to("cuda").
        if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(exc).lower():
            raise
        log(f"  GPU prompt encode OOM ({type(exc).__name__}), falling back to CPU")
        gc.collect()
        malloc_trim()
        torch.cuda.empty_cache()
        TEXT_ENCODER.to("cpu")
        torch.cuda.synchronize()
        return _encode_text(text, "cpu")


def decode_latents(latents: "torch.Tensor") -> Any:
    """Decode latents to a PIL image, staging the VAE onto the card.

    The VAE is 160 MB and its decode is the activation peak of the whole
    pipeline, so it is kept off the card between requests and staged in only
    for the decode. It shares the card with nothing else at this point: the
    text encoder has already gone back to the CPU and the transformer's blocks
    are offloaded one at a time.

    `torch.no_grad()` is essential here. The pipeline's own `__call__` is
    decorated with it, so its decode runs without an autograd graph. Decoding
    by hand without it builds a graph that holds EVERY VAE activation alive
    (2.76 GB instead of 174 MB) and the decode OOMs on a 4 GB card.
    """
    PIPE.vae.to("cuda")
    torch.cuda.synchronize()
    try:
        with torch.no_grad():
            latents = latents.to(PIPE.vae.dtype)
            latents = (latents / PIPE.vae.config.scaling_factor) + PIPE.vae.config.shift_factor
            image = PIPE.vae.decode(latents, return_dict=False)[0]
            # postprocess returns a LIST of images (one per batch element).
            out = PIPE.image_processor.postprocess(image, output_type="pil")
            return out[0] if isinstance(out, list) else out
    finally:
        PIPE.vae.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class _JobCancelled(Exception):
    """Raised inside the step callback to abort a running generation."""


def run_generation(job: Job, req: GenerateRequest) -> None:
    """Execute one generation. Runs under GPU_LOCK."""
    cancel_evt = CANCEL_FLAGS.setdefault(job.id, threading.Event())
    try:
        with GPU_LOCK:
            job.state = "running"
            job.started_at = time.time()
            job.stage = "encoding prompt"
            job.progress = 0.05
            log(f"job {job.id}: encoding prompt on GPU ({len(req.prompt)} chars)")

            t_enc = time.perf_counter()
            embeds = encode_prompt(req.prompt)
            encode_s = time.perf_counter() - t_enc
            log(f"job {job.id}: prompt encoded in {encode_s:.2f}s")
            job.stage = "generating"
            job.progress = 0.2

            # The block count is fixed at 1: the streamed group offload requires
            # it, and larger groups measured slower anyway. Nothing to
            # reconfigure per request.

            seed = (
                req.seed
                if req.seed >= 0
                else int(torch.randint(0, 2**31 - 1, (1,)).item())
            )
            log(f"job {job.id}: generating {req.steps} steps @ {req.size}px seed={seed}")

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            def on_step(pipe, step_i, timestep, kwargs):
                # Cooperative cancel: the step boundary is the only safe place
                # to abort a CUDA run, so check the flag between steps.
                if cancel_evt.is_set():
                    raise _JobCancelled()
                job.progress = 0.2 + 0.7 * ((step_i + 1) / max(req.steps, 1))
                job.stage = f"step {step_i + 1}/{req.steps}"
                return kwargs

            result = PIPE(
                prompt=None,
                prompt_embeds=[embeds],
                height=req.size,
                width=req.size,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
                # Return the latents and decode ourselves, so the VAE can be
                # staged onto the card only for the decode. Letting the
                # pipeline decode would require the VAE to be resident the
                # whole time, which is what squeezes the prompt encode.
                output_type="latent",
            )
            gen_s = time.time() - t0

            # The decode is a separate phase with its own activation peak.
            t_dec = time.perf_counter()
            image = decode_latents(result.images)
            decode_s = time.perf_counter() - t_dec

            out_path = OUT_DIR / f"web_{job.id}.png"
            image.save(out_path)

            peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
            job.image_url = f"/api/image/{job.id}"
            job.metrics = {
                "generation_seconds": round(gen_s, 2),
                "seconds_per_step": round(gen_s / max(req.steps, 1), 2),
                "encode_seconds": round(encode_s, 2),
                "decode_seconds": round(decode_s, 2),
                "total_seconds": round(encode_s + gen_s + decode_s, 2),
                "peak_vram_gb": round(peak_vram, 3),
                "rss_mb": round(rss_mb()),
                "seed": seed,
                "size": req.size,
                "steps": req.steps,
                "blocks": req.blocks,
            }
            # Persist the metadata next to the PNG. The in-memory JOBS dict is
            # lost on restart, which would strip the prompt and metrics from
            # every card in the gallery.
            save_job_meta(job)
            job.progress = 1.0
            job.state = "done"
            job.stage = "done"
            log(
                f"job {job.id}: done in {gen_s:.1f}s "
                f"(encode {encode_s:.1f}s, decode {decode_s:.1f}s)  "
                f"peakVRAM={peak_vram:.2f} GB"
            )

    except _JobCancelled:
        job.state = "error"
        job.stage = "cancelled"
        job.error = "Cancelled by user"
        log(f"job {job.id}: cancelled by user")
    except Exception as exc:  # noqa: BLE001
        job.state = "error"
        job.stage = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        log(f"job {job.id}: ERROR {job.error}")
    finally:
        job.finished_at = time.time()
        CANCEL_FLAGS.pop(job.id, None)
        gc.collect()
        malloc_trim()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- app
app = FastAPI(title="Z-Image-Turbo INT4 (RTX 2050)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global LOAD_ERROR
    try:
        load_pipeline()
    except Exception as exc:  # noqa: BLE001
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        log(f"FAILED to load pipeline: {LOAD_ERROR}")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "error" if LOAD_ERROR else ("ready" if PIPE is not None else "loading"),
        "model": "Z-Image-Turbo INT4 (sdnq uint4)",
        "device": "RTX 2050 4GB",
        "error": LOAD_ERROR,
        "rss_mb": round(rss_mb()),
        "vram_free_gb": (
            round(torch.cuda.mem_get_info(0)[0] / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        ),
    }


@app.get("/api/settings")
def settings() -> dict[str, Any]:
    return {
        "sizes": ALLOWED_SIZES,
        "default_size": 512,
        "default_steps": 8,
        "max_steps": 50,
        "blocks": [1],
        "active_blocks": ACTIVE_BLOCKS,
        "notes": {
            "encode": (
                "The prompt encoder wants the card to itself (~2.85 GB free). "
                "If it does not fit it falls back to the CPU: the request still "
                "works, but the encode takes ~17 s instead of ~1 s."
            ),
            "vram": "One transformer block is resident in VRAM at a time (streamed prefetch).",
            "size": "Sizes must be a multiple of 16.",
        },
    }


@app.get("/api/headroom")
def headroom_endpoint(
    size: int | None = None, blocks: int | None = None
) -> dict[str, Any]:
    """Predict the VRAM cost of a generation for the current card state.

    With no parameters it returns the full grid of (size, blocks) options so
    the UI can show a preview for every combination. With `size` and `blocks`
    it returns the prediction for that single choice.

    The numbers come from `headroom.py`, which was calibrated by measuring the
    real transformer forward and the real VAE decode for every combination on
    this machine (worst error 1 MiB against the measured device peaks).
    """
    import sysmon

    stats = sysmon.snapshot(OUT_DIR)
    gpu = stats.get("gpu", {})
    total = float(gpu.get("vram_total_gb") or 0.0)
    used = float(gpu.get("vram_used_gb") or 0.0)
    free = float(gpu.get("vram_free_gb") or 0.0)

    if size is not None and blocks is not None:
        pred = headroom.predict(
            size, blocks, vram_total_gb=total, vram_used_gb=used, vram_free_gb=free
        )
        pred["explain"] = headroom.explain(size, blocks)
        pred["explain_encode"] = headroom.explain_encode(
            bool(pred.get("encode_uses_gpu")), free * 1024
        )
        return pred

    grid = headroom.predict_all(
        vram_total_gb=total, vram_used_gb=used, vram_free_gb=free
    )
    return {
        "vram_total_gb": total,
        "vram_used_gb": used,
        "vram_free_gb": free,
        "grid": grid,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    if LOAD_ERROR:
        raise HTTPException(status_code=503, detail=LOAD_ERROR)
    if PIPE is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if req.size % 16 != 0:
        raise HTTPException(status_code=400, detail="size must be a multiple of 16")

    job = Job(id=uuid.uuid4().hex[:12], prompt=req.prompt, created_at=time.time())
    with JOBS_LOCK:
        JOBS[job.id] = job

    threading.Thread(target=run_generation, args=(job, req), daemon=True).start()
    return {"job_id": job.id, "state": job.state}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.model_dump()


@app.get("/api/image/{job_id}")
def image(job_id: str) -> FileResponse:
    path = OUT_DIR / f"web_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/gallery")
def gallery() -> list[dict[str, Any]]:
    """Previously generated images, newest first.

    Metadata comes from the in-memory job when the backend is still running
    that job, otherwise from the sidecar JSON written when the image was
    generated. Without the sidecar fallback, a backend restart would strip
    every prompt and metric from the gallery.
    """
    items = []
    for path in sorted(
        OUT_DIR.glob("web_*.png"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        job_id = path.stem.removeprefix("web_")
        with JOBS_LOCK:
            job = JOBS.get(job_id)

        prompt = job.prompt if job else ""
        metrics = job.metrics if job else None
        created_at = path.stat().st_mtime

        if not prompt and not metrics:
            meta = load_job_meta(job_id)
            if meta:
                prompt = meta.get("prompt") or ""
                metrics = meta.get("metrics")
                created_at = meta.get("created_at") or created_at

        items.append(
            {
                "job_id": job_id,
                "url": f"/api/image/{job_id}",
                "prompt": prompt,
                "created_at": created_at,
                "metrics": metrics,
            }
        )
    return items


@app.delete("/api/jobs/{job_id}")
def cancel(job_id: str) -> dict[str, Any]:
    """Cancel a job.

    A queued job is marked cancelled immediately. A running job gets a
    cooperative cancel: the step callback checks the flag at the next step
    boundary and aborts the run (diffusers has no hard interrupt).
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.state in ("done", "error"):
            raise HTTPException(status_code=409, detail="job already finished")
        state = job.state

    if state == "queued":
        with JOBS_LOCK:
            job.state = "error"
            job.stage = "cancelled"
            job.error = "Cancelled by user"
        CANCEL_FLAGS.pop(job_id, None)
        log(f"job {job_id}: cancelled while queued")
        return {"ok": True, "state": job.state, "immediate": True}

    # Running: set the flag; the step callback will abort within one step.
    evt = CANCEL_FLAGS.get(job_id)
    if evt is None:
        # The worker already cleaned up (race with completion).
        raise HTTPException(status_code=409, detail="job is not running")
    evt.set()
    with JOBS_LOCK:
        job.stage = "cancelling"
    log(f"job {job_id}: cancel requested")
    return {"ok": True, "state": "running", "immediate": False}


@app.get("/api/system")
def system() -> dict[str, Any]:
    """Realtime system telemetry: CPU, RAM, swap, disk, GPU."""
    import sysmon

    return sysmon.snapshot(OUT_DIR)


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """All known jobs, newest first. Lets the UI restore state after a refresh."""
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    return [j.model_dump() for j in jobs]
