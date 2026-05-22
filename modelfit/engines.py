"""Inference-engine catalog + throughput prediction.

The base speed model in `scoring.py` is memory-bandwidth bound and assumes a
*single-request* decode (the realistic case for a local chatbot). Real
production deployments use very different engines whose characteristics shift
tokens/sec by a lot:

  * vLLM     — Continuous batching + PagedAttention. Single-request throughput
                ~ same as llama.cpp, but at high concurrency can serve 5-20×
                more tokens/sec aggregate.
  * TensorRT-LLM — Fused CUDA kernels, FP8, in-flight batching. On H100/H200
                the gold standard. Single-request is ~1.6× a hand-tuned
                llama.cpp; aggregate is much higher.
  * llama.cpp — GGUF + AVX/Metal kernels. Best for CPU and Apple Silicon;
                most popular for local use.
  * Ollama   — Thin wrapper around llama.cpp. ~Same throughput.
  * MLX      — Apple's framework. ~0.9× llama.cpp on Mac (but bigger models
                fit cleanly in unified memory).
  * ExLlamaV2 — Quantized inference on consumer NVIDIA GPUs. ~1.25× llama.cpp
                on RTX cards at Q4.
  * Hugging Face Transformers — Reference implementation. SLOW (~0.4×
                llama.cpp). Use a real engine in production.
  * TGI (HF Text Generation Inference) — Production HF server. Continuous
                batching, FlashAttention. ~1.2× llama.cpp single-request.

Numbers below are *single-request* multipliers vs. the bandwidth-bound base
estimate. Aggregate (concurrent) throughput scales with `concurrency_factor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from modelfit.models import ModelSpec
from modelfit.hardware import HardwareProfile
from modelfit.quantization import FitResult, fit_model
from modelfit.scoring import estimate_tokens_per_sec


@dataclass(frozen=True)
class Engine:
    id: str
    name: str
    # Speed multiplier vs the bandwidth-bound base estimate, single-request.
    single_req_multiplier: float
    # How much aggregate throughput scales at high concurrency (continuous batching).
    concurrency_factor: float
    # Which platforms this engine actually runs on.
    platforms: Set[str]
    # Which quantization families it supports.
    quants: Set[str]
    # Headline strengths, weaknesses (1-line each) for the beginner.
    pros: List[str]
    cons: List[str]
    install_hint: str
    notes: str = ""

    @property
    def is_gpu_only(self) -> bool:
        return "cpu" not in self.platforms

    def supports_quant(self, quant: str) -> bool:
        if quant in self.quants:
            return True
        # Family matching ("any GGUF Q*" etc.)
        if quant.startswith("Q") and "GGUF" in self.quants:
            return True
        return False

    def supports_hw(self, hw: HardwareProfile) -> bool:
        if hw.unified_memory:
            return "apple" in self.platforms
        if hw.gpus:
            vendors = {g.vendor for g in hw.gpus}
            if "nvidia" in vendors and "nvidia" in self.platforms: return True
            if "amd" in vendors and "amd" in self.platforms: return True
            if "intel" in vendors and "intel" in self.platforms: return True
        return "cpu" in self.platforms

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "single_req_multiplier": self.single_req_multiplier,
            "concurrency_factor": self.concurrency_factor,
            "platforms": sorted(self.platforms),
            "quants": sorted(self.quants),
            "pros": list(self.pros),
            "cons": list(self.cons),
            "install_hint": self.install_hint,
            "notes": self.notes,
        }


ENGINES: List[Engine] = [
    Engine(
        id="llama.cpp", name="llama.cpp",
        single_req_multiplier=1.00, concurrency_factor=1.2,
        platforms={"cpu", "nvidia", "amd", "apple", "intel"},
        quants={"GGUF", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                 "Q3_K_M", "Q3_K_S", "Q2_K", "F16"},
        pros=["Runs everywhere — CPU, NVIDIA, AMD, Apple Metal, even phones",
              "Smallest dependency footprint",
              "Best support for GGUF quantization formats"],
        cons=["Single-request focused — limited continuous batching",
              "C++ build can intimidate beginners"],
        install_hint="git clone https://github.com/ggerganov/llama.cpp && make",
        notes="The reference engine for local LLMs. Powers Ollama, LM Studio, Jan, GPT4All.",
    ),
    Engine(
        id="ollama", name="Ollama",
        single_req_multiplier=0.98, concurrency_factor=1.3,
        platforms={"cpu", "nvidia", "amd", "apple"},
        quants={"GGUF", "Q4_K_M", "Q5_K_M", "Q8_0"},
        pros=["One-command install on Mac/Linux/Windows",
              "Built-in model library — `ollama run llama3.2:3b` and you're chatting",
              "REST API and OpenAI-compatible endpoint included"],
        cons=["Less tuneable than raw llama.cpp",
              "Limited control over which quant gets pulled"],
        install_hint="curl -fsSL https://ollama.com/install.sh | sh",
        notes="Thin wrapper around llama.cpp. The easiest path to running models locally.",
    ),
    Engine(
        id="vllm", name="vLLM",
        single_req_multiplier=1.05, concurrency_factor=8.0,
        platforms={"nvidia", "amd"},
        quants={"AWQ", "GPTQ", "FP8", "F16", "BF16"},
        pros=["PagedAttention — efficient KV cache management",
              "Continuous batching — 5-20× throughput at high concurrency",
              "OpenAI-compatible API server built in"],
        cons=["NVIDIA / AMD GPU only — won't help on Mac or CPU",
              "Heavyweight: full PyTorch + CUDA toolchain"],
        install_hint="pip install vllm",
        notes="The standard for production self-hosted inference on NVIDIA. "
              "Single-request ~ same as llama.cpp, but aggregate throughput is huge.",
    ),
    Engine(
        id="tensorrt-llm", name="TensorRT-LLM",
        single_req_multiplier=1.6, concurrency_factor=10.0,
        platforms={"nvidia"},
        quants={"FP8", "INT8", "F16", "BF16", "AWQ"},
        pros=["Fastest single-request and aggregate throughput on NVIDIA",
              "FP8 on Hopper/Blackwell — ~2× the throughput of F16",
              "In-flight batching, fused kernels"],
        cons=["NVIDIA-only", "Setup is non-trivial — build engines per model",
              "Best benefits on H100 / H200 / B200"],
        install_hint="pip install tensorrt_llm  (and follow the engine-build guide)",
        notes="Production deployment for NVIDIA. Best when you can pre-compile model engines.",
    ),
    Engine(
        id="tgi", name="Hugging Face TGI",
        single_req_multiplier=1.20, concurrency_factor=6.0,
        platforms={"nvidia", "amd"},
        quants={"AWQ", "GPTQ", "F16", "BF16"},
        pros=["Production-ready HTTP server",
              "FlashAttention v2, paged attention, continuous batching",
              "Best HF Hub integration"],
        cons=["GPU only", "Docker-first deploy is opinionated"],
        install_hint="docker run --gpus all ghcr.io/huggingface/text-generation-inference",
        notes="HuggingFace's production serving stack — what they run on the Hub.",
    ),
    Engine(
        id="exllamav2", name="ExLlamaV2",
        single_req_multiplier=1.25, concurrency_factor=1.5,
        platforms={"nvidia"},
        quants={"EXL2", "GPTQ"},
        pros=["Very fast single-request on consumer RTX GPUs",
              "EXL2 quant format allows non-uniform precision per layer"],
        cons=["NVIDIA only, no native CPU/Mac",
              "Smaller ecosystem than llama.cpp"],
        install_hint="pip install exllamav2",
        notes="Strong single-user choice on RTX 3090 / 4090. Faster than llama.cpp at Q4.",
    ),
    Engine(
        id="mlx", name="MLX (Apple)",
        single_req_multiplier=0.92, concurrency_factor=1.4,
        platforms={"apple"},
        quants={"MLX-4bit", "MLX-8bit", "F16", "BF16"},
        pros=["First-class Apple Silicon support",
              "Memory-efficient — supports larger models in unified memory",
              "Lazy evaluation + automatic memory management"],
        cons=["Mac only", "Smaller community than llama.cpp"],
        install_hint="pip install mlx-lm",
        notes="Apple's research framework. Slightly slower than llama.cpp's Metal backend "
              "today, but improving rapidly.",
    ),
    Engine(
        id="transformers", name="HF Transformers (vanilla)",
        single_req_multiplier=0.40, concurrency_factor=1.0,
        platforms={"cpu", "nvidia", "amd", "apple", "intel"},
        quants={"F16", "BF16", "INT8", "INT4-bnb", "GPTQ", "AWQ"},
        pros=["Reference implementation — every new model lands here first",
              "Easiest path for research / fine-tuning"],
        cons=["SLOW for inference — no fused kernels, no paged attention",
              "Use only for prototyping; pick a real engine for serving"],
        install_hint="pip install transformers",
        notes="Great for development, bad for serving. Don't use this in production.",
    ),
    Engine(
        id="sglang", name="SGLang",
        single_req_multiplier=1.10, concurrency_factor=9.0,
        platforms={"nvidia", "amd"},
        quants={"FP8", "AWQ", "GPTQ", "F16", "BF16"},
        pros=["RadixAttention — fast prefix sharing across requests",
              "Strong structured-output / constrained-decoding support",
              "Comparable or better than vLLM at high concurrency"],
        cons=["Newer / smaller community than vLLM",
              "GPU only"],
        install_hint="pip install sglang",
        notes="Rising star for production inference. Excellent for agent workflows "
              "with shared prompt prefixes.",
    ),
    Engine(
        id="lmstudio", name="LM Studio",
        single_req_multiplier=0.98, concurrency_factor=1.2,
        platforms={"cpu", "nvidia", "apple"},
        quants={"GGUF", "Q4_K_M", "Q5_K_M", "Q8_0"},
        pros=["Friendly desktop app — model browser + chat UI in one",
              "No terminal required",
              "OpenAI-compatible local server"],
        cons=["Closed source",
              "Less performant than raw llama.cpp for power users"],
        install_hint="Download from https://lmstudio.ai",
        notes="Same engine as llama.cpp underneath, wrapped in a GUI. "
              "Great for the discover → download → chat loop.",
    ),
]

ENGINE_BY_ID = {e.id: e for e in ENGINES}


@dataclass
class ThroughputPrediction:
    engine: Engine
    fits: bool
    single_user_tps: float
    aggregate_tps: float           # at high concurrency
    explanation: str

    def to_dict(self) -> dict:
        return {
            "engine_id": self.engine.id,
            "engine_name": self.engine.name,
            "fits": self.fits,
            "single_user_tps": round(self.single_user_tps, 1),
            "aggregate_tps_high_concurrency": round(self.aggregate_tps, 1),
            "explanation": self.explanation,
        }


def predict_throughput(model: ModelSpec, hw: HardwareProfile,
                       min_context: int = 4096) -> List[ThroughputPrediction]:
    """For each engine that can serve this (model, hw) pair, predict
    single-user and aggregate (continuous-batching) tokens/sec.

    The estimate is intentionally rough — a coarse but honest sketch is more
    helpful than a precise-looking lie.
    """
    fit = fit_model(model, hw, min_context=min_context)
    if not fit.fits:
        # No engine can run what doesn't fit.
        return [
            ThroughputPrediction(
                engine=e, fits=False,
                single_user_tps=0.0, aggregate_tps=0.0,
                explanation=(
                    "Model doesn't fit on this hardware even at Q2_K. "
                    "Run `modelfit reverse <model>` to see what would fit."),
            ) for e in ENGINES if e.supports_hw(hw)
        ]
    base_tps = estimate_tokens_per_sec(model, fit, hw)
    out: List[ThroughputPrediction] = []
    for e in ENGINES:
        if not e.supports_hw(hw):
            continue
        if not e.supports_quant(fit.quant):
            out.append(ThroughputPrediction(
                engine=e, fits=False,
                single_user_tps=0.0, aggregate_tps=0.0,
                explanation=(f"{e.name} doesn't support the {fit.quant} format. "
                              f"Convert to one of: {', '.join(sorted(e.quants)[:5])}…"),
            ))
            continue
        single = base_tps * e.single_req_multiplier
        agg = single * e.concurrency_factor
        explanation = _explain(e, model, hw, fit, single, agg)
        out.append(ThroughputPrediction(
            engine=e, fits=True,
            single_user_tps=single, aggregate_tps=agg,
            explanation=explanation,
        ))
    # Sort: fitting engines first, then by single-user throughput.
    out.sort(key=lambda p: (not p.fits, -p.single_user_tps))
    return out


def _explain(e: Engine, m: ModelSpec, hw: HardwareProfile,
             fit: FitResult, single: float, agg: float) -> str:
    parts = []
    mult = e.single_req_multiplier
    if mult >= 1.4:
        parts.append(f"Fused/optimised kernels make this engine ~{mult:.1f}× the baseline.")
    elif mult >= 1.05:
        parts.append(f"~{mult:.2f}× the baseline — a modest engine-side speedup.")
    elif mult >= 0.9:
        parts.append("About the same as the memory-bandwidth-bound baseline.")
    else:
        parts.append(f"~{mult:.2f}× the baseline — slower than tuned alternatives.")

    if e.concurrency_factor >= 5:
        parts.append(
            f"At high concurrency (batched serving), aggregate throughput "
            f"scales to roughly {agg:.0f} tok/s thanks to continuous batching / "
            f"PagedAttention."
        )
    elif e.concurrency_factor >= 1.5:
        parts.append(f"Modest batching speedup at concurrency (~{e.concurrency_factor:.1f}×).")
    else:
        parts.append("Limited concurrency benefit — best for single-user chat.")

    # Hardware-specific notes
    if e.id == "tensorrt-llm" and hw.best_gpu and "h" in hw.best_gpu.name.lower():
        parts.append("On Hopper/Blackwell GPUs FP8 unlocks another ~2× — not modelled here.")
    if e.id == "mlx" and not hw.unified_memory:
        parts.append("MLX is Apple-only — this won't apply to your hardware.")
    if e.id == "vllm" and (not hw.gpus or hw.unified_memory):
        parts.append("vLLM needs an NVIDIA/AMD GPU — won't run on this machine.")

    return " ".join(parts)


def explain_choice(predictions: List[ThroughputPrediction]) -> str:
    """A human-readable summary of which engine to actually choose."""
    fitting = [p for p in predictions if p.fits]
    if not fitting:
        return ("No engine can serve this combination. Either the model "
                "doesn't fit, or no supported engine matches your hardware.")
    # Best single-user
    best_single = max(fitting, key=lambda p: p.single_user_tps)
    # Best aggregate (production)
    best_agg = max(fitting, key=lambda p: p.aggregate_tps)
    lines = []
    lines.append(f"For a chat / single-user workload → use {best_single.engine.name} "
                 f"(~{best_single.single_user_tps:.0f} tok/s).")
    if best_agg.engine.id != best_single.engine.id:
        lines.append(f"For serving many users at once → use {best_agg.engine.name} "
                     f"(~{best_agg.aggregate_tps:.0f} tok/s aggregate).")
    else:
        lines.append(f"It's also the best at high concurrency "
                     f"(~{best_agg.aggregate_tps:.0f} tok/s aggregate).")
    return " ".join(lines)
