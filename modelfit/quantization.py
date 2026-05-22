"""Memory-fit engine.

Given a ModelSpec and a HardwareProfile, walk Q8_0 → Q2_K, walk context
from max down to a configurable floor (default 50% of max), and return the
best (quant, context) that actually fits in the chosen memory tier.

Memory accounting:
    weights      = params_b * 1e9 * bytes_per_param[quant]
    kv_cache     = kv_bytes_per_token * context_tokens
    activations  = ~5% of (weights + kv) — runtime workspace
    overhead     = framework + drivers (we use 800 MB GPU, 400 MB CPU)

The "tier" is GPU VRAM if a GPU is present (Apple unified counts as VRAM
since the GPU can read the whole RAM); otherwise system RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from modelfit.models import ModelSpec, QUANT_BYTES_PER_PARAM, QUANT_ORDER, QUANT_QUALITY
from modelfit.hardware import HardwareProfile

GB = 1024**3
MB = 1024**2

GPU_OVERHEAD_BYTES = 800 * MB        # driver + CUDA context + runtime
CPU_OVERHEAD_BYTES = 400 * MB
ACTIVATION_FRACTION = 0.05           # rough runtime activation workspace
SAFETY_HEADROOM = 0.92               # leave 8% of the tier free
# Walk context down by these fractions when full doesn't fit.
CONTEXT_FALLBACKS = [1.0, 0.5, 0.25, 0.125]


@dataclass
class FitResult:
    fits: bool
    quant: Optional[str] = None
    context: Optional[int] = None
    weights_bytes: int = 0
    kv_bytes: int = 0
    overhead_bytes: int = 0
    total_bytes: int = 0
    tier: str = "cpu"             # cpu | gpu | unified
    budget_bytes: int = 0
    quality_retention: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fits": self.fits,
            "quant": self.quant,
            "context": self.context,
            "weights_gb": round(self.weights_bytes / GB, 3),
            "kv_cache_gb": round(self.kv_bytes / GB, 3),
            "overhead_gb": round(self.overhead_bytes / GB, 3),
            "total_gb": round(self.total_bytes / GB, 3),
            "budget_gb": round(self.budget_bytes / GB, 3),
            "tier": self.tier,
            "quality_retention": round(self.quality_retention, 3),
            "reason": self.reason,
        }


def weights_bytes(model: ModelSpec, quant: str) -> int:
    """Bytes the weights occupy at a given GGUF quantization level."""
    bpp = QUANT_BYTES_PER_PARAM[quant]
    return int(model.params_b * 1_000_000_000 * bpp)


def kv_bytes_for(model: ModelSpec, context_tokens: int) -> int:
    return int(model.kv_bytes_per_token * context_tokens)


def memory_required(model: ModelSpec, quant: str, context: int) -> int:
    w = weights_bytes(model, quant)
    kv = kv_bytes_for(model, context)
    activations = int((w + kv) * ACTIVATION_FRACTION)
    return w + kv + activations


def _budget_for(hw: HardwareProfile) -> tuple[int, str]:
    """Pick the tier we'll actually load into."""
    if hw.unified_memory:
        # Apple Silicon: the GPU shares the same DRAM. Use ~75% of total.
        return int(hw.ram_bytes * 0.75) - GPU_OVERHEAD_BYTES, "unified"
    vram = sum(g.vram_bytes for g in hw.gpus)
    if vram > 0:
        return vram - GPU_OVERHEAD_BYTES, "gpu"
    return hw.ram_bytes - CPU_OVERHEAD_BYTES, "cpu"


def fit_model(model: ModelSpec, hw: HardwareProfile,
              min_context: int = 2048,
              max_quant: str = "Q8_0",
              min_quant: str = "Q2_K") -> FitResult:
    """Walk quant from best → worst and context from max → min, returning
    the *highest quality* configuration that fits.

    Caller can pin a higher floor (min_context) if they truly need long
    context, even at the cost of saying "doesn't fit".
    """
    budget_raw, tier = _budget_for(hw)
    overhead = GPU_OVERHEAD_BYTES if tier in ("gpu", "unified") else CPU_OVERHEAD_BYTES
    budget = int(budget_raw * SAFETY_HEADROOM)

    # Slice the quant walk to caller-allowed range.
    quants = QUANT_ORDER
    if max_quant in quants:
        quants = quants[quants.index(max_quant):]
    if min_quant in quants:
        quants = quants[: quants.index(min_quant) + 1]

    floor = max(min_context, 512)
    contexts: List[int] = []
    for frac in CONTEXT_FALLBACKS:
        c = int(model.context_max * frac)
        if c >= floor and c not in contexts:
            contexts.append(c)
    if floor not in contexts and floor < model.context_max:
        contexts.append(floor)

    best_failure: Optional[FitResult] = None

    for quant in quants:
        for ctx in contexts:
            total = memory_required(model, quant, ctx) + overhead
            if total <= budget:
                return FitResult(
                    fits=True, quant=quant, context=ctx,
                    weights_bytes=weights_bytes(model, quant),
                    kv_bytes=kv_bytes_for(model, ctx),
                    overhead_bytes=overhead,
                    total_bytes=total,
                    tier=tier,
                    budget_bytes=budget,
                    quality_retention=QUANT_QUALITY[quant],
                    reason=(
                        f"fits at {quant} with {ctx:,} ctx "
                        f"({total / GB:.1f} GB ≤ {budget / GB:.1f} GB {tier})"
                    ),
                )
            best_failure = FitResult(
                fits=False, quant=quant, context=ctx,
                weights_bytes=weights_bytes(model, quant),
                kv_bytes=kv_bytes_for(model, ctx),
                overhead_bytes=overhead,
                total_bytes=total,
                tier=tier,
                budget_bytes=budget,
                quality_retention=QUANT_QUALITY[quant],
                reason=(f"closest miss: {total / GB:.1f} GB > "
                        f"{budget / GB:.1f} GB at {quant}/{ctx:,}"),
            )

    if best_failure is None:
        return FitResult(fits=False, tier=tier, budget_bytes=budget,
                         reason="no quantization configurations considered")
    best_failure.reason = (
        f"won't fit even at Q2_K/{min(contexts):,} ctx — "
        f"needs {best_failure.total_bytes / GB:.1f} GB, have {budget / GB:.1f} GB"
    )
    return best_failure


def all_quant_options(model: ModelSpec, hw: HardwareProfile,
                      context: Optional[int] = None) -> List[FitResult]:
    """For inspection: every quantization at a fixed (or max) context."""
    ctx = context or model.context_max
    out = []
    for quant in QUANT_ORDER:
        budget_raw, tier = _budget_for(hw)
        overhead = GPU_OVERHEAD_BYTES if tier in ("gpu", "unified") else CPU_OVERHEAD_BYTES
        budget = int(budget_raw * SAFETY_HEADROOM)
        total = memory_required(model, quant, ctx) + overhead
        out.append(FitResult(
            fits=total <= budget,
            quant=quant, context=ctx,
            weights_bytes=weights_bytes(model, quant),
            kv_bytes=kv_bytes_for(model, ctx),
            overhead_bytes=overhead,
            total_bytes=total,
            tier=tier, budget_bytes=budget,
            quality_retention=QUANT_QUALITY[quant],
        ))
    return out
