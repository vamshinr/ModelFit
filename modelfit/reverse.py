"""Reverse mode: given a model + target tokens/sec, compute the minimum
hardware that would deliver it, and surface real GPUs that clear the bar.
"""
from __future__ import annotations

from typing import Optional

from modelfit.models import ModelSpec, QUANT_BYTES_PER_PARAM
from modelfit.quantization import (
    weights_bytes, kv_bytes_for, GPU_OVERHEAD_BYTES,
    SAFETY_HEADROOM, ACTIVATION_FRACTION,
)
from modelfit.hardware import GPU_BANDWIDTH
from modelfit.scoring import BANDWIDTH_UTILIZATION, MOE_SPEEDUP

GB = 1024**3
MB = 1024**2


def _required_bandwidth_gbps(model: ModelSpec, quant: str, context: int,
                              target_tps: float) -> float:
    """Solve: tps = bandwidth * util / (active_weights + kv_partial)
    Returns peak GB/s the GPU/RAM needs to advertise to hit target_tps.
    """
    active_ratio = (model.active_params_b / model.params_b) if model.params_b else 1.0
    w = weights_bytes(model, quant)
    kv = kv_bytes_for(model, context)
    active_weight_bytes = w * active_ratio
    bytes_per_step = active_weight_bytes + (kv * 0.1)
    # If MoE, the speed model gives back ~MOE_SPEEDUP. Invert it.
    moe_factor = MOE_SPEEDUP if active_ratio < 0.99 else 1.0
    needed_bandwidth_bytes_per_sec = target_tps * bytes_per_step / moe_factor / BANDWIDTH_UTILIZATION
    return needed_bandwidth_bytes_per_sec / (1024**3)


def _required_vram_gb(model: ModelSpec, quant: str, context: int) -> tuple[float, float, float]:
    w = weights_bytes(model, quant)
    kv = kv_bytes_for(model, context)
    activations = int((w + kv) * ACTIVATION_FRACTION)
    total = w + kv + activations + GPU_OVERHEAD_BYTES
    # Account for the safety headroom we use during fit (need a bit more raw VRAM).
    total = int(total / SAFETY_HEADROOM)
    return w / GB, kv / GB, total / GB


def recommend_hardware(model: ModelSpec,
                       target_tps: float = 30.0,
                       context: Optional[int] = None,
                       quant: str = "Q4_K_M") -> dict:
    if quant not in QUANT_BYTES_PER_PARAM:
        raise ValueError(f"Unknown quant {quant}")
    ctx = context or model.context_max
    weights_gb, kv_gb, vram_gb = _required_vram_gb(model, quant, ctx)
    bw_gbps = _required_bandwidth_gbps(model, quant, ctx, target_tps)

    # Filter known GPUs that meet both VRAM and bandwidth bars.
    candidates = []
    # We don't have VRAM in the lookup table — apply rough chip-family rules.
    GPU_VRAM = {
        "rtx 5090": 32, "rtx 5080": 16, "rtx 4090": 24, "rtx 4080 super": 16, "rtx 4080": 16,
        "rtx 4070 ti super": 16, "rtx 4070 ti": 12, "rtx 4070 super": 12, "rtx 4070": 12,
        "rtx 4060 ti": 16, "rtx 4060": 8,
        "rtx 3090 ti": 24, "rtx 3090": 24, "rtx 3080 ti": 12, "rtx 3080": 10,
        "rtx 3070 ti": 8, "rtx 3070": 8, "rtx 3060": 12,
        "h200": 141, "h100": 80, "a100": 80, "l40s": 48, "a6000": 48, "a40": 48,
        "rx 7900 xtx": 24, "rx 7900 xt": 20, "mi300x": 192, "mi250x": 128,
        "m1 max": 32, "m1 ultra": 64, "m2 max": 64, "m2 ultra": 128,
        "m3 max": 96, "m4 max": 128,
    }
    for name, bw in GPU_BANDWIDTH.items():
        vram = GPU_VRAM.get(name, 0)
        if vram == 0:
            continue
        meets = vram >= vram_gb and bw >= bw_gbps
        marginal = vram >= vram_gb * 0.95 and bw >= bw_gbps * 0.85
        if meets or (marginal and len(candidates) < 8):
            candidates.append({
                "name": name.upper(),
                "vram_gb": vram,
                "bandwidth_gbps": bw,
                "meets": meets,
            })
    # Sort: meets-first, then by VRAM ascending (cheapest sufficient option first).
    candidates.sort(key=lambda c: (not c["meets"], c["vram_gb"], -c["bandwidth_gbps"]))

    return {
        "model": model.id,
        "name": model.name,
        "quant": quant,
        "context": ctx,
        "target_tps": target_tps,
        "weights_gb": round(weights_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "min_vram_gb": round(vram_gb, 2),
        "min_bandwidth_gbps": round(bw_gbps, 1),
        "candidate_gpus": candidates[:12],
    }
