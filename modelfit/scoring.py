"""Composite scoring on Quality, Speed, Context, Capability.

The final score is a weighted sum of four 0-100 sub-scores. Weights shift
based on the user's task profile (chat vs reasoning vs code vs research).

Anti-patterns we deliberately avoid:
    * Scoring models that don't fit at all (returned with score=0).
    * Letting a 7B beat a 70B on raw "quality" alone — the speed sub-score
      penalizes slow generation, so a 1 tok/s 70B can still lose to a
      40 tok/s 14B for interactive chat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from modelfit.models import ModelSpec
from modelfit.hardware import HardwareProfile
from modelfit.quantization import FitResult, fit_model

GB = 1024**3


# Use-case → (quality, speed, context, capability) weight tuple.
USE_CASE_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "chat":        (0.30, 0.35, 0.15, 0.20),
    "reasoning":   (0.50, 0.15, 0.20, 0.15),
    "code":        (0.35, 0.25, 0.20, 0.20),
    "math":        (0.50, 0.15, 0.15, 0.20),
    "long-context":(0.25, 0.15, 0.45, 0.15),
    "agent":       (0.35, 0.25, 0.25, 0.15),
    "research":    (0.40, 0.20, 0.25, 0.15),
    "balanced":    (0.30, 0.30, 0.20, 0.20),
}


# ---------------------------------------------------------------------------
# Speed estimate: tokens/sec ≈ effective_bandwidth / model_in_memory_bytes
# (Memory-bandwidth bound is the right first-order model for autoregressive
# decode. We discount bandwidth by a utilization factor.)
# ---------------------------------------------------------------------------
BANDWIDTH_UTILIZATION = 0.65   # real-world fraction of peak bandwidth
MOE_SPEEDUP = 1.7              # MoE only reads active experts per token
MIN_TPS, MAX_TPS = 0.3, 250.0  # clamp for log scaling


def estimate_tokens_per_sec(model: ModelSpec, fit: FitResult,
                            hw: HardwareProfile) -> float:
    if not fit.fits or fit.quant is None:
        return 0.0
    # Active weight bytes — for MoE the read set per token is smaller.
    active_ratio = (model.active_params_b / model.params_b) if model.params_b else 1.0
    active_weight_bytes = fit.weights_bytes * active_ratio
    # KV cache reads happen each step, so include them.
    bytes_per_step = active_weight_bytes + (fit.kv_bytes * 0.1)  # KV read is partial
    bandwidth = hw.effective_bandwidth_gbps * (1024**3) * BANDWIDTH_UTILIZATION
    if bytes_per_step <= 0:
        return 0.0
    tps = bandwidth / bytes_per_step
    if active_ratio < 0.99:
        tps *= MOE_SPEEDUP
    return max(MIN_TPS, min(MAX_TPS, tps))


# ---------------------------------------------------------------------------
# Sub-scores (each 0-100)
# ---------------------------------------------------------------------------
def quality_subscore(model: ModelSpec, fit: FitResult) -> float:
    base = model.quality
    return max(0.0, min(100.0, base * fit.quality_retention))


def speed_subscore(tps: float) -> float:
    # Log-scaled so 10 tps ≈ 50, 50 tps ≈ 75, 200 tps ≈ 100.
    if tps <= 0:
        return 0.0
    # log10(0.5) ≈ -0.3, log10(200) ≈ 2.3 → map to 0-100.
    norm = (math.log10(tps) - math.log10(MIN_TPS)) / (math.log10(MAX_TPS) - math.log10(MIN_TPS))
    return max(0.0, min(100.0, norm * 100))


def context_subscore(fit_context: int, model_max: int) -> float:
    if fit_context <= 0:
        return 0.0
    # Reward both absolute context and how close we got to model's trained max.
    abs_score = min(100.0, math.log2(max(2048, fit_context) / 2048) * 18.0)
    fraction = min(1.0, fit_context / model_max) if model_max else 1.0
    return abs_score * (0.6 + 0.4 * fraction)


def capability_subscore(model: ModelSpec, use_case: str) -> float:
    """How well the model type matches what the user is trying to do."""
    t = model.type
    tags = set(model.tags)
    uc = use_case.lower()

    # Base by direct type match
    base = {
        "chat":        {"chat": 95, "instruct": 88, "base": 50, "code": 60,
                         "reasoning": 80, "math": 65, "vision": 75},
        "reasoning":   {"reasoning": 100, "instruct": 75, "chat": 70, "math": 80,
                         "code": 65, "base": 40, "vision": 60},
        "code":        {"code": 100, "instruct": 78, "chat": 65, "base": 50,
                         "reasoning": 80, "math": 65, "vision": 55},
        "math":        {"math": 100, "reasoning": 90, "instruct": 70, "chat": 65,
                         "code": 70, "base": 40, "vision": 50},
        "long-context":{"instruct": 85, "chat": 82, "code": 80, "reasoning": 80,
                         "math": 60, "base": 55, "vision": 70},
        "agent":       {"instruct": 95, "chat": 88, "code": 80, "reasoning": 85,
                         "math": 60, "base": 50, "vision": 65},
        "research":    {"reasoning": 90, "instruct": 85, "chat": 80, "math": 80,
                         "code": 70, "base": 60, "vision": 65},
        "balanced":    {"instruct": 88, "chat": 85, "code": 75, "reasoning": 80,
                         "math": 70, "base": 55, "vision": 70},
    }.get(uc, {})
    score = base.get(t, 60)

    # Tag boosts (e.g. "code" tag amplifies code use case, multilingual for chat)
    if uc == "code" and "code" in tags:
        score = min(100, score + 5)
    if uc == "reasoning" and "reasoning" in tags:
        score = min(100, score + 5)
    if uc == "math" and "math" in tags:
        score = min(100, score + 5)
    return float(score)


@dataclass
class ScoredModel:
    model: ModelSpec
    fit: FitResult
    tokens_per_sec: float
    quality: float
    speed: float
    context: float
    capability: float
    composite: float
    use_case: str
    weights: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "id": self.model.id,
            "name": self.model.name,
            "family": self.model.family,
            "params_b": self.model.params_b,
            "active_params_b": self.model.active_params_b,
            "type": self.model.type,
            "tags": list(self.model.tags),
            "fits": self.fit.fits,
            "quant": self.fit.quant,
            "context": self.fit.context,
            "context_max": self.model.context_max,
            "memory_gb": round(self.fit.total_bytes / GB, 2),
            "budget_gb": round(self.fit.budget_bytes / GB, 2),
            "tier": self.fit.tier,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "subscores": {
                "quality": round(self.quality, 1),
                "speed": round(self.speed, 1),
                "context": round(self.context, 1),
                "capability": round(self.capability, 1),
            },
            "composite": round(self.composite, 1),
            "use_case": self.use_case,
            "reason": self.fit.reason,
        }


def score_model(model: ModelSpec, hw: HardwareProfile,
                use_case: str = "balanced",
                min_context: int = 2048) -> ScoredModel:
    weights = USE_CASE_WEIGHTS.get(use_case, USE_CASE_WEIGHTS["balanced"])
    fit = fit_model(model, hw, min_context=min_context)
    if not fit.fits:
        return ScoredModel(
            model=model, fit=fit, tokens_per_sec=0.0,
            quality=0, speed=0, context=0, capability=0,
            composite=0, use_case=use_case, weights=weights,
        )
    tps = estimate_tokens_per_sec(model, fit, hw)
    q = quality_subscore(model, fit)
    s = speed_subscore(tps)
    c = context_subscore(fit.context or 0, model.context_max)
    cap = capability_subscore(model, use_case)
    wq, ws, wc, wcap = weights
    composite = wq * q + ws * s + wc * c + wcap * cap
    return ScoredModel(
        model=model, fit=fit, tokens_per_sec=tps,
        quality=q, speed=s, context=c, capability=cap,
        composite=composite, use_case=use_case, weights=weights,
    )


def rank_all(hw: HardwareProfile, use_case: str = "balanced",
             min_context: int = 2048,
             include_unfit: bool = False,
             family: Optional[str] = None,
             type_filter: Optional[str] = None,
             min_quality: float = 0.0) -> List[ScoredModel]:
    from modelfit.models import get_catalog
    out: List[ScoredModel] = []
    for m in get_catalog():
        if family and m.family.lower() != family.lower():
            continue
        if type_filter and m.type != type_filter:
            continue
        if m.quality < min_quality:
            continue
        s = score_model(m, hw, use_case=use_case, min_context=min_context)
        if not include_unfit and not s.fit.fits:
            continue
        out.append(s)
    out.sort(key=lambda x: x.composite, reverse=True)
    return out
