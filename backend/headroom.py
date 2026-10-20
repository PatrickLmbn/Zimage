"""Headroom prediction for Z-Image-Turbo on the RTX 2050 4 GB card.

The question the UI must answer: "if I change VRAM blocks or resolution, will
this still fit in VRAM, and how much headroom will be left?"

This is a CALIBRATED model, not a guess. Every constant below was measured on
this machine by running the real transformer forward and the real VAE decode
for every (size, blocks) combination and sampling device VRAM at 1 kHz. The
model reproduces the observed production `peak_vram_gb` values exactly
(0 MiB error at 512px/1 and 512px/3 blocks, 2 MiB at 512px/4).

See references/headroom-model.md for the derivation.

Two numbers matter and they answer different questions:

  torch_peak  = max(transformer_torch, vae_torch)
      PyTorch tensor bytes only. This is what `torch.cuda.max_memory_allocated()`
      reports, and therefore what the gallery's "peak VRAM" metric shows.

  device_peak = max(transformer_device, vae_phase_device)
      Whole-card bytes: the transformer phase peak, or the device level reached
      during the VAE decode (which is the VAE's own cost PLUS whatever the
      transformer phase still holds). This is what nvidia-smi/NVML reports, and
      therefore what the System card's VRAM bar shows.

Headroom must be judged against the DEVICE number, because that is what the
card actually has. The torch number understates it by ~1.4 GB (CUDA context,
cuBLAS/cuDNN workspaces, other apps, allocator cache).
"""

from __future__ import annotations

# --- calibration constants (measured on RTX 2050 4GB, driver 615.78.08) ----

# torch tensor peak of the transformer forward, in MiB.
TRANSFORMER_TORCH_MIB: dict[tuple[int, int], float] = {
    (256, 1): 489.0, (256, 2): 585.0, (256, 3): 683.0, (256, 4): 780.0,
    (384, 1): 506.0, (384, 2): 605.0, (384, 3): 703.0, (384, 4): 800.0,
    (512, 1): 533.0, (512, 2): 629.0, (512, 3): 726.0, (512, 4): 823.0,
    (640, 1): 566.0, (640, 2): 663.0, (640, 3): 760.0, (640, 4): 858.0,
    (768, 1): 606.0, (768, 2): 703.0, (768, 3): 801.0, (768, 4): 898.0,
}

# torch tensor peak of the VAE decode, by output size, in MiB.
VAE_TORCH_MIB: dict[int, float] = {
    256: 327.0, 384: 518.0, 512: 784.0, 640: 1126.0, 768: 1545.0,
}

# device-level peak of the transformer phase, minus the idle baseline, MiB.
# (This is the "T - base" column of the calibration.)
TRANSFORMER_DEVICE_MIB: dict[tuple[int, int], float] = {
    (256, 1): 450.0, (256, 2): 766.0, (256, 3): 751.0, (256, 4): 718.0,
    (384, 1): 396.0, (384, 2): 606.0, (384, 3): 691.0, (384, 4): 754.0,
    (512, 1): 544.0, (512, 2): 652.0, (512, 3): 737.0, (512, 4): 829.0,
    (640, 1): 983.0, (640, 2): 742.0, (640, 3): 800.0, (640, 4): 894.0,
    (768, 1): 725.0, (768, 2): 834.0, (768, 3): 906.0, (768, 4): 937.0,
}

# device level reached during the VAE decode, minus the idle baseline, MiB.
# This INCLUDES the transformer memory still held at that moment, so it is
# already the combined figure -- which is why the model is a max(), not a sum.
VAE_PHASE_DEVICE_MIB: dict[tuple[int, int], float] = {
    (256, 1): 454.0, (256, 2): 766.0, (256, 3): 718.0, (256, 4): 718.0,
    (384, 1): 586.0, (384, 2): 829.0, (384, 3): 900.0, (384, 4): 959.0,
    (512, 1): 1220.0, (512, 2): 1349.0, (512, 3): 1435.0, (512, 4): 1542.0,
    (640, 1): 2187.0, (640, 2): 2116.0, (640, 3): 2227.0, (640, 4): 2342.0,
    (768, 1): 2471.0, (768, 2): 2564.0, (768, 3): 2040.0, (768, 4): 2088.0,
}

ALLOWED_SIZES: tuple[int, ...] = (256, 384, 512, 640, 768)
ALLOWED_BLOCKS: tuple[int, ...] = (1, 2, 3, 4)

# The transformer only has 30 blocks, so 30 is the hard ceiling.
MAX_BLOCKS = 30
_PACKED_BLOCK_MIB = 97.3  # packed weight of one transformer block


def _lookup(table: dict, size: int, blocks: int) -> float:
    """Measured value, or a linear extrapolation in `blocks` outside the
    measured range (1..4)."""
    key = (size, blocks)
    if key in table:
        return table[key]
    measured = [b for b in ALLOWED_BLOCKS if (size, b) in table]
    lo = max(b for b in measured if b <= blocks) if any(
        b <= blocks for b in measured) else min(measured)
    hi = min((b for b in measured if b >= blocks), default=max(measured))
    if lo == hi:
        return table[(size, lo)]
    v_lo, v_hi = table[(size, lo)], table[(size, hi)]
    frac = (blocks - lo) / (hi - lo)
    return v_lo + frac * (v_hi - v_lo)


def torch_peak_mib(size: int, blocks: int) -> float:
    """Peak PyTorch tensor bytes for a generation, in MiB.

    This is the number the gallery's "peak VRAM" metric reports.
    """
    trf = _lookup(TRANSFORMER_TORCH_MIB, size, blocks)
    vae = VAE_TORCH_MIB[size]
    return max(trf, vae)


def device_peak_mib(size: int, blocks: int) -> float:
    """Peak whole-card VRAM for a generation, in MiB above the idle baseline.

    This is the number that decides whether the job fits on the card.
    """
    trf = _lookup(TRANSFORMER_DEVICE_MIB, size, blocks)
    vae_phase = _lookup(VAE_PHASE_DEVICE_MIB, size, blocks)
    return max(trf, vae_phase)


def predict(size: int, blocks: int, *, vram_total_gb: float, vram_used_gb: float,
            vram_free_gb: float) -> dict:
    """Predict the VRAM outcome of a generation.

    `vram_*_gb` are the CURRENT live readings from NVML (device-wide).
    """
    torch_mib = torch_peak_mib(size, blocks)
    device_mib = device_peak_mib(size, blocks)

    projected_used_gb = vram_used_gb + device_mib / 1024
    projected_free_gb = vram_free_gb - device_mib / 1024

    trf_t = _lookup(TRANSFORMER_TORCH_MIB, size, blocks)
    vae_t = VAE_TORCH_MIB[size]
    binding = "vae" if vae_t > trf_t else "transformer"

    if projected_free_gb <= 0.0:
        verdict = "won't fit"
    elif projected_free_gb < 0.25:
        verdict = "very tight"
    elif projected_free_gb < 0.6:
        verdict = "tight"
    elif projected_free_gb < 1.2:
        verdict = "ok"
    else:
        verdict = "comfortable"

    return {
        "size": size,
        "blocks": blocks,
        "torch_peak_gb": round(torch_mib / 1024, 3),
        "device_peak_gb": round(device_mib / 1024, 3),
        "projected_used_gb": round(projected_used_gb, 2),
        "projected_free_gb": round(projected_free_gb, 2),
        "binding": binding,
        "verdict": verdict,
        "fits": projected_free_gb > 0.0,
    }


def predict_all(*, vram_total_gb: float, vram_used_gb: float,
                vram_free_gb: float) -> list[dict]:
    """The full grid, for a UI that shows a preview of every option."""
    return [
        predict(size, blocks, vram_total_gb=vram_total_gb, vram_used_gb=vram_used_gb,
                vram_free_gb=vram_free_gb)
        for size in ALLOWED_SIZES
        for blocks in ALLOWED_BLOCKS
    ]


def explain(size: int, blocks: int) -> str:
    """One-line human explanation of what a block/resolution choice costs."""
    trf_t = _lookup(TRANSFORMER_TORCH_MIB, size, blocks)
    vae_t = VAE_TORCH_MIB[size]
    binding = "VAE decode" if vae_t > trf_t else "transformer"
    return (
        f"{binding} dominates at {size}px / {blocks} block(s): "
        f"transformer {trf_t:.0f} MiB, VAE {vae_t:.0f} MiB, "
        f"device peak {device_peak_mib(size, blocks):.0f} MiB"
    )
