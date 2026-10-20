"""
Phase 1 baseline: generate an image with Z-Image-Turbo (INT4 / sdnq) on the RTX 2050.

Machine constraints this script is built around (Ryzen 7 7435HS 8C/16T, 16 GB RAM,
RTX 2050 4 GB VRAM, ~2.8 GB VRAM actually free, ~6.8 GB RAM free):

  * The INT4 transformer is 3.25 GB packed -> it does NOT fit in VRAM, and it does
    not fit in RAM alongside the text encoder (3.25 + 2.65 = 5.9 GB).
  * So components are staged, never all resident at once:
        1. text encoder  -> CPU, encode the prompt, then FREE it
        2. transformer   -> CPU (packed INT4), block-offloaded to VRAM a few blocks
                            at a time
        3. vae           -> GPU (160 MB, fp32 upcast to 320 MB), decode the latents
  * Group offload runs with use_stream=False ON PURPOSE. With use_stream=True,
    diffusers keeps a second PINNED CPU copy of every weight (+3.25 GB RAM) and this
    box has only ~6.8 GB free. Without a stream there is exactly ONE copy of each
    weight, living in either RAM or VRAM.
  * Default resolution is 512x512, not the model's 1024x1024, because a 1024x1024
    latent is 64x64 -> 4096 tokens and the unified attention sequence (tokens + 512
    caption tokens) pushes activation memory past what is left of the 4 GB card.
  * SDNQ_USE_TORCH_COMPILE=0: inductor compilation costs a large init peak plus
    minutes of compile time for no benefit on a correctness baseline.

Usage:
    python baseline.py --prompt "a cute cat" --steps 8 --size 512
"""

import argparse
import gc
import json
import os
import resource
import time

# Must be set before transformers/diffusers read the text encoder config. Without it
# the SDNQ quantization config on the text encoder is silently ignored and it loads
# as bf16 (2x the RAM, and the weights do not fit next to the transformer).
os.environ.setdefault("DIFFUSERS_SDNQ_TRANSFORMERS", "1")
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get(
    "ZIMAGE_MODEL_DIR", str(PROJECT_ROOT / "models" / "z-image-turbo-int4")
)
OUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cute cat sitting on a windowsill")
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--blocks", type=int, default=1, help="transformer blocks kept in VRAM at once")
    ap.add_argument("--out", default="baseline")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 70)
    log("Z-Image-Turbo INT4 baseline | RTX 2050 4GB / 16GB RAM")
    log("=" * 70)
    log(f"prompt   : {args.prompt}")
    log(f"steps    : {args.steps}  size: {args.size}x{args.size}  seed: {args.seed}")
    log(f"guidance : {args.guidance}  blocks/group: {args.blocks}")
    log(f"RAM peak : {rss_mb():.0f} MB at start")
    log("-" * 70)

    from diffusers import ZImagePipeline
    import sdnq  # noqa: F401  -- registers sdnq into transformers.quantizers.auto

    # ---------------------------------------------------------------- load pipeline
    log("loading pipeline (all components on CPU)")
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    log(f"  assembled in {time.time() - t0:.1f}s  RAM peak={rss_mb():.0f} MB")

    te = pipe.text_encoder
    q = te.model.layers[0].mlp.gate_proj
    log(f"  text_encoder quantized: {hasattr(q, 'sdnq_dequantizer')} ({type(q).__name__})")
    log(f"  transformer layers    : {len(pipe.transformer.layers)}")
    log(f"  vae dtype             : {pipe.vae.dtype}")

    # ---------------------------------------------------------------- encode prompt
    # The text encoder stays on CPU (2.65 GB packed; the GPU does not need it).
    # Encode the prompt, then drop the encoder before the transformer starts
    # streaming blocks to VRAM.
    log("encoding prompt (text encoder on CPU)")
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative if args.negative else None,
        do_classifier_free_guidance=args.guidance > 0,
        device=torch.device("cpu"),
        max_sequence_length=512,
    )
    log(f"  encoded in {time.time() - t0:.1f}s  embeds={[tuple(e.shape) for e in prompt_embeds]}")
    log(f"  RAM peak after encode : {rss_mb():.0f} MB")

    # Free the text encoder -- 2.65 GB of RAM we cannot keep resident.
    pipe.text_encoder = None
    gc.collect()
    log(f"  text encoder freed. RAM peak={rss_mb():.0f} MB")

    # ---------------------------------------------------------------- offload setup
    log("enabling group offload (use_stream=False: no pinned duplicate copy)")
    # Apply to the TRANSFORMER ONLY: pipe.enable_group_offload() would sweep every
    # component and drag the text encoder onto the GPU (OOM on a 4 GB card).
    # `exclude_modules` does not help -- it means "keep this on the accelerator".
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=args.blocks,
        use_stream=False,
    )
    log(f"  transformer: block_level, {args.blocks} block(s)/group")

    # The VAE is small (160 MB) but its decode is the activation peak, so park it
    # on the GPU permanently instead of offloading it.
    pipe.vae.to("cuda")
    log("  vae: resident on cuda")

    free_cuda()
    vram_free = torch.cuda.mem_get_info(0)[0] / 1024**3
    log(f"  VRAM free before generate: {vram_free:.2f} GB   RAM peak={rss_mb():.0f} MB")
    log("-" * 70)

    # ---------------------------------------------------------------- generate
    gen_kwargs = dict(
        prompt=None,
        prompt_embeds=prompt_embeds,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )
    if args.guidance > 0 and negative_prompt_embeds:
        gen_kwargs["negative_prompt_embeds"] = negative_prompt_embeds

    torch.cuda.reset_peak_memory_stats()
    log(f"generating ({args.steps} steps)...")
    t1 = time.time()
    try:
        result = pipe(**gen_kwargs)
    except torch.cuda.OutOfMemoryError as e:
        log(f"CUDA OOM: {e}")
        log(f"peak VRAM at OOM: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        raise
    gen_s = time.time() - t1

    image = result.images[0]
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
    out_path = os.path.join(OUT_DIR, f"{args.out}.png")
    image.save(out_path)

    log("-" * 70)
    log(f"DONE  generation {gen_s:.1f}s ({gen_s / max(args.steps, 1):.1f}s/step)")
    log(f"      peak VRAM      : {peak_vram:.2f} GB")
    log(f"      peak RAM       : {rss_mb():.0f} MB")
    log(f"      image          : {out_path}  {image.size}")

    metrics = {
        "prompt": args.prompt,
        "steps": args.steps,
        "size": args.size,
        "seed": args.seed,
        "guidance": args.guidance,
        "blocks_per_group": args.blocks,
        "generation_seconds": round(gen_s, 2),
        "seconds_per_step": round(gen_s / max(args.steps, 1), 2),
        "peak_vram_gb": round(peak_vram, 3),
        "peak_rss_mb": round(rss_mb()),
        "model": "Z-Image-Turbo INT4-sdnq",
    }
    mpath = os.path.join(OUT_DIR, f"{args.out}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"      metrics        : {mpath}")


if __name__ == "__main__":
    main()
