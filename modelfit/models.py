"""Data model + builders for the model catalog.

A ModelSpec is the immutable description of a model. The runtime fit/score
results live in scoring.ScoredModel — keep this file pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# GGUF quantization sizing: average bytes per parameter for each level.
# Numbers track real-world ggml/llama.cpp file sizes within ~3% across
# Llama, Mistral, Qwen, and Gemma checkpoints.
QUANT_BYTES_PER_PARAM = {
    "F16":    2.00,
    "Q8_0":   1.06,
    "Q6_K":   0.82,
    "Q5_K_M": 0.71,
    "Q5_K_S": 0.69,
    "Q5_0":   0.69,
    "Q4_K_M": 0.59,
    "Q4_K_S": 0.56,
    "Q4_0":   0.56,
    "Q3_K_M": 0.49,
    "Q3_K_S": 0.45,
    "Q2_K":   0.40,
}

# Walk order from highest to lowest quality. The sizer picks the first that fits.
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"]

# Relative quality retention vs F16 (rough community consensus from perplexity
# studies). The scorer applies this as a quality multiplier.
QUANT_QUALITY = {
    "F16":    1.000,
    "Q8_0":   0.998,
    "Q6_K":   0.992,
    "Q5_K_M": 0.985,
    "Q5_K_S": 0.980,
    "Q5_0":   0.978,
    "Q4_K_M": 0.965,
    "Q4_K_S": 0.955,
    "Q4_0":   0.950,
    "Q3_K_M": 0.910,
    "Q3_K_S": 0.880,
    "Q2_K":   0.820,
}

VALID_TYPES = {"base", "instruct", "chat", "code", "reasoning", "math", "vision"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    params_b: float            # Total params in billions
    active_params_b: float     # Active params/token (== params_b for dense, smaller for MoE)
    context_max: int           # Max trained context (tokens)
    type: str                  # base | instruct | chat | code | reasoning | math | vision
    quality: float             # 0–100 intrinsic quality at F16
    kv_bytes_per_token: int    # Per-token KV-cache size in bytes (fp16 KV)
    license: str = "open"
    tags: tuple = ()

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(f"Bad type {self.type} for {self.id}")

    def to_dict(self) -> dict:
        return {**asdict(self), "tags": list(self.tags)}


# ---------------------------------------------------------------------------
# KV-cache estimator
#
# KV per token (bytes) = 2 * n_layers * n_kv_heads * head_dim * 2 (fp16)
#
# We pick (layers, kv_heads, head_dim) by size bucket and architecture flags.
# GQA models share KV heads across query heads → much smaller cache.
# ---------------------------------------------------------------------------
def kv_per_token(params_b: float, gqa: bool = True,
                 layers: Optional[int] = None,
                 kv_heads: Optional[int] = None,
                 head_dim: int = 128) -> int:
    if layers is None or kv_heads is None:
        p = params_b
        if p <= 0.5:    L, KV, HD = 12, 2,  64
        elif p <= 1.5:  L, KV, HD = 16, 4,  64
        elif p <= 2.5:  L, KV, HD = 22, 4,  128
        elif p <= 4:    L, KV, HD = 28, 8,  128
        elif p <= 9:    L, KV, HD = 32, 8,  128
        elif p <= 14:   L, KV, HD = 40, 8,  128
        elif p <= 16:   L, KV, HD = 40, 8,  128
        elif p <= 25:   L, KV, HD = 48, 8,  128
        elif p <= 35:   L, KV, HD = 60, 8,  128
        elif p <= 45:   L, KV, HD = 60, 8,  128
        elif p <= 75:   L, KV, HD = 80, 8,  128
        elif p <= 110:  L, KV, HD = 80, 8,  128
        elif p <= 140:  L, KV, HD = 88, 8,  128
        else:           L, KV, HD = 126, 8, 128
        layers = L
        kv_heads = KV if gqa else max(KV, int((p ** 0.4) * 8))
        head_dim = HD
    return 2 * layers * kv_heads * head_dim * 2  # fp16 K + V


# ---------------------------------------------------------------------------
# Catalog builder — emits 206 curated specs across the open-weights ecosystem.
# ---------------------------------------------------------------------------
def _q(b: float, gqa: bool = True, **kw) -> int:
    return kv_per_token(b, gqa=gqa, **kw)


def build_catalog() -> List[ModelSpec]:
    M: List[ModelSpec] = []

    def add(*args, **kwargs):
        M.append(ModelSpec(*args, **kwargs))

    # ===== Llama family =====
    # Llama 2 (no GQA on 7B/13B, GQA on 70B)
    add("llama-2-7b", "Llama 2 7B", "Llama", 6.74, 6.74, 4096, "base", 56,
        _q(6.74, gqa=False), tags=("meta",))
    add("llama-2-7b-chat", "Llama 2 7B Chat", "Llama", 6.74, 6.74, 4096, "chat", 58,
        _q(6.74, gqa=False), tags=("meta",))
    add("llama-2-13b", "Llama 2 13B", "Llama", 13.0, 13.0, 4096, "base", 60,
        _q(13.0, gqa=False), tags=("meta",))
    add("llama-2-13b-chat", "Llama 2 13B Chat", "Llama", 13.0, 13.0, 4096, "chat", 62,
        _q(13.0, gqa=False), tags=("meta",))
    add("llama-2-70b", "Llama 2 70B", "Llama", 69.0, 69.0, 4096, "base", 71,
        _q(69.0), tags=("meta",))
    add("llama-2-70b-chat", "Llama 2 70B Chat", "Llama", 69.0, 69.0, 4096, "chat", 72,
        _q(69.0), tags=("meta",))
    # Code Llama
    add("codellama-7b", "CodeLlama 7B", "CodeLlama", 6.74, 6.74, 16384, "code", 55,
        _q(6.74, gqa=False), tags=("meta", "code"))
    add("codellama-7b-instruct", "CodeLlama 7B Instruct", "CodeLlama", 6.74, 6.74, 16384, "code", 58,
        _q(6.74, gqa=False), tags=("meta", "code"))
    add("codellama-7b-python", "CodeLlama 7B Python", "CodeLlama", 6.74, 6.74, 16384, "code", 60,
        _q(6.74, gqa=False), tags=("meta", "code", "python"))
    add("codellama-13b", "CodeLlama 13B", "CodeLlama", 13.0, 13.0, 16384, "code", 60,
        _q(13.0, gqa=False), tags=("meta", "code"))
    add("codellama-13b-instruct", "CodeLlama 13B Instruct", "CodeLlama", 13.0, 13.0, 16384, "code", 63,
        _q(13.0, gqa=False), tags=("meta", "code"))
    add("codellama-13b-python", "CodeLlama 13B Python", "CodeLlama", 13.0, 13.0, 16384, "code", 64,
        _q(13.0, gqa=False), tags=("meta", "code", "python"))
    add("codellama-34b", "CodeLlama 34B", "CodeLlama", 34.0, 34.0, 16384, "code", 67,
        _q(34.0), tags=("meta", "code"))
    add("codellama-34b-instruct", "CodeLlama 34B Instruct", "CodeLlama", 34.0, 34.0, 16384, "code", 70,
        _q(34.0), tags=("meta", "code"))
    add("codellama-34b-python", "CodeLlama 34B Python", "CodeLlama", 34.0, 34.0, 16384, "code", 72,
        _q(34.0), tags=("meta", "code", "python"))
    add("codellama-70b", "CodeLlama 70B", "CodeLlama", 69.0, 69.0, 16384, "code", 73,
        _q(69.0), tags=("meta", "code"))
    add("codellama-70b-instruct", "CodeLlama 70B Instruct", "CodeLlama", 69.0, 69.0, 16384, "code", 75,
        _q(69.0), tags=("meta", "code"))
    # Llama 3
    add("llama-3-8b", "Llama 3 8B", "Llama", 8.03, 8.03, 8192, "base", 68,
        _q(8.03), tags=("meta",))
    add("llama-3-8b-instruct", "Llama 3 8B Instruct", "Llama", 8.03, 8.03, 8192, "instruct", 72,
        _q(8.03), tags=("meta",))
    add("llama-3-70b", "Llama 3 70B", "Llama", 70.6, 70.6, 8192, "base", 79,
        _q(70.6), tags=("meta",))
    add("llama-3-70b-instruct", "Llama 3 70B Instruct", "Llama", 70.6, 70.6, 8192, "instruct", 82,
        _q(70.6), tags=("meta",))
    # Llama 3.1 (128K context, GQA)
    add("llama-3.1-8b", "Llama 3.1 8B", "Llama", 8.03, 8.03, 131072, "base", 70, _q(8.03), tags=("meta",))
    add("llama-3.1-8b-instruct", "Llama 3.1 8B Instruct", "Llama", 8.03, 8.03, 131072, "instruct", 74, _q(8.03), tags=("meta",))
    add("llama-3.1-70b", "Llama 3.1 70B", "Llama", 70.6, 70.6, 131072, "base", 81, _q(70.6), tags=("meta",))
    add("llama-3.1-70b-instruct", "Llama 3.1 70B Instruct", "Llama", 70.6, 70.6, 131072, "instruct", 84, _q(70.6), tags=("meta",))
    add("llama-3.1-405b", "Llama 3.1 405B", "Llama", 405.0, 405.0, 131072, "base", 87, _q(405.0, layers=126, kv_heads=8, head_dim=128), tags=("meta",))
    add("llama-3.1-405b-instruct", "Llama 3.1 405B Instruct", "Llama", 405.0, 405.0, 131072, "instruct", 89, _q(405.0, layers=126, kv_heads=8, head_dim=128), tags=("meta",))
    # Llama 3.2
    add("llama-3.2-1b", "Llama 3.2 1B", "Llama", 1.23, 1.23, 131072, "base", 48, _q(1.23, layers=16, kv_heads=8, head_dim=64), tags=("meta",))
    add("llama-3.2-1b-instruct", "Llama 3.2 1B Instruct", "Llama", 1.23, 1.23, 131072, "instruct", 53, _q(1.23, layers=16, kv_heads=8, head_dim=64), tags=("meta",))
    add("llama-3.2-3b", "Llama 3.2 3B", "Llama", 3.21, 3.21, 131072, "base", 62, _q(3.21, layers=28, kv_heads=8, head_dim=128), tags=("meta",))
    add("llama-3.2-3b-instruct", "Llama 3.2 3B Instruct", "Llama", 3.21, 3.21, 131072, "instruct", 66, _q(3.21, layers=28, kv_heads=8, head_dim=128), tags=("meta",))
    add("llama-3.2-11b-vision", "Llama 3.2 11B Vision", "Llama", 10.7, 10.7, 131072, "vision", 73, _q(10.7), tags=("meta", "vision"))
    add("llama-3.2-90b-vision", "Llama 3.2 90B Vision", "Llama", 88.0, 88.0, 131072, "vision", 83, _q(88.0), tags=("meta", "vision"))
    # Llama 3.3
    add("llama-3.3-70b-instruct", "Llama 3.3 70B Instruct", "Llama", 70.6, 70.6, 131072, "instruct", 85, _q(70.6), tags=("meta",))

    # ===== Mistral / Mixtral =====
    add("mistral-7b-v0.1", "Mistral 7B v0.1", "Mistral", 7.24, 7.24, 32768, "base", 64, _q(7.24), tags=("mistral",))
    add("mistral-7b-instruct-v0.1", "Mistral 7B Instruct v0.1", "Mistral", 7.24, 7.24, 32768, "instruct", 65, _q(7.24), tags=("mistral",))
    add("mistral-7b-instruct-v0.2", "Mistral 7B Instruct v0.2", "Mistral", 7.24, 7.24, 32768, "instruct", 68, _q(7.24), tags=("mistral",))
    add("mistral-7b-instruct-v0.3", "Mistral 7B Instruct v0.3", "Mistral", 7.24, 7.24, 32768, "instruct", 69, _q(7.24), tags=("mistral",))
    add("mixtral-8x7b", "Mixtral 8x7B", "Mistral", 46.7, 12.9, 32768, "base", 74, _q(46.7, layers=32, kv_heads=8, head_dim=128), tags=("mistral", "moe"))
    add("mixtral-8x7b-instruct", "Mixtral 8x7B Instruct", "Mistral", 46.7, 12.9, 32768, "instruct", 76, _q(46.7, layers=32, kv_heads=8, head_dim=128), tags=("mistral", "moe"))
    add("mixtral-8x22b", "Mixtral 8x22B", "Mistral", 141.0, 39.0, 65536, "base", 80, _q(141.0, layers=56, kv_heads=8, head_dim=128), tags=("mistral", "moe"))
    add("mixtral-8x22b-instruct", "Mixtral 8x22B Instruct", "Mistral", 141.0, 39.0, 65536, "instruct", 82, _q(141.0, layers=56, kv_heads=8, head_dim=128), tags=("mistral", "moe"))
    add("mistral-nemo-12b", "Mistral Nemo 12B", "Mistral", 12.2, 12.2, 131072, "base", 70, _q(12.2), tags=("mistral",))
    add("mistral-nemo-12b-instruct", "Mistral Nemo 12B Instruct", "Mistral", 12.2, 12.2, 131072, "instruct", 73, _q(12.2), tags=("mistral",))
    add("mistral-small-22b", "Mistral Small 22B", "Mistral", 22.2, 22.2, 32768, "instruct", 76, _q(22.2), tags=("mistral",))
    add("mistral-small-3-24b", "Mistral Small 3 24B", "Mistral", 24.0, 24.0, 32768, "instruct", 78, _q(24.0), tags=("mistral",))
    add("mistral-large-123b", "Mistral Large 123B", "Mistral", 123.0, 123.0, 131072, "instruct", 84, _q(123.0), tags=("mistral",))
    add("mathstral-7b", "Mathstral 7B", "Mistral", 7.24, 7.24, 32768, "math", 70, _q(7.24), tags=("mistral", "math"))
    add("codestral-22b", "Codestral 22B", "Mistral", 22.2, 22.2, 32768, "code", 77, _q(22.2), tags=("mistral", "code"))
    add("codestral-mamba-7b", "Codestral Mamba 7B", "Mistral", 7.3, 7.3, 256000, "code", 71, _q(7.3), tags=("mistral", "code", "mamba"))

    # ===== Qwen family =====
    # Qwen 1.5
    for p, q, l, kv, hd in [(0.5, 44, 24, 16, 64), (1.8, 54, 24, 16, 128),
                              (4.0, 62, 40, 20, 128), (7.0, 67, 32, 32, 128),
                              (14.0, 73, 40, 40, 128), (32.0, 78, 64, 8, 128),
                              (72.0, 81, 80, 8, 128), (110.0, 83, 80, 8, 128)]:
        add(f"qwen1.5-{p:g}b-chat", f"Qwen 1.5 {p:g}B Chat", "Qwen", float(p), float(p),
            32768, "chat", q, _q(p, layers=l, kv_heads=kv, head_dim=hd), tags=("alibaba",))
    # Qwen2
    for p, q, l, kv in [(0.5, 50, 24, 2), (1.5, 60, 28, 2), (7.0, 73, 28, 4),
                          (72.0, 82, 80, 8)]:
        add(f"qwen2-{p:g}b-instruct", f"Qwen2 {p:g}B Instruct", "Qwen", float(p), float(p),
            131072, "instruct", q, _q(p, layers=l, kv_heads=kv), tags=("alibaba",))
    add("qwen2-57b-a14b", "Qwen2 57B-A14B MoE", "Qwen", 57.0, 14.0, 65536, "instruct", 76,
        _q(57.0, layers=28, kv_heads=4), tags=("alibaba", "moe"))
    # Qwen2.5
    for p, q, l, kv in [(0.5, 52, 24, 2), (1.5, 63, 28, 2), (3.0, 68, 36, 2),
                          (7.0, 75, 28, 4), (14.0, 78, 48, 8),
                          (32.0, 81, 64, 8), (72.0, 84, 80, 8)]:
        add(f"qwen2.5-{p:g}b-instruct", f"Qwen2.5 {p:g}B Instruct", "Qwen", float(p), float(p),
            131072, "instruct", q, _q(p, layers=l, kv_heads=kv), tags=("alibaba",))
    # Qwen2.5 Coder
    for p, q in [(0.5, 50), (1.5, 60), (3.0, 67), (7.0, 76), (14.0, 80), (32.0, 84)]:
        add(f"qwen2.5-coder-{p:g}b-instruct", f"Qwen2.5-Coder {p:g}B Instruct",
            "Qwen", float(p), float(p), 131072, "code", q,
            _q(p), tags=("alibaba", "code"))
    # Qwen2.5 Math
    for p, q in [(1.5, 65), (7.0, 78), (72.0, 85)]:
        add(f"qwen2.5-math-{p:g}b-instruct", f"Qwen2.5-Math {p:g}B Instruct",
            "Qwen", float(p), float(p), 4096, "math", q,
            _q(p), tags=("alibaba", "math"))
    # Qwen3
    for p, q, l, kv in [(0.6, 55, 28, 2), (1.7, 66, 28, 2),
                          (4.0, 72, 36, 4), (8.0, 78, 36, 8),
                          (14.0, 81, 48, 8), (32.0, 84, 64, 8)]:
        add(f"qwen3-{p:g}b", f"Qwen3 {p:g}B", "Qwen", float(p), float(p),
            131072, "instruct", q, _q(p, layers=l, kv_heads=kv), tags=("alibaba",))
    add("qwen3-30b-a3b", "Qwen3 30B-A3B MoE", "Qwen", 30.5, 3.3, 131072, "instruct", 80,
        _q(30.5, layers=48, kv_heads=4), tags=("alibaba", "moe"))
    add("qwen3-235b-a22b", "Qwen3 235B-A22B MoE", "Qwen", 235.0, 22.0, 131072, "instruct", 88,
        _q(235.0, layers=94, kv_heads=8), tags=("alibaba", "moe"))
    add("qwq-32b-preview", "QwQ 32B Preview", "Qwen", 32.5, 32.5, 32768, "reasoning", 85,
        _q(32.5, layers=64, kv_heads=8), tags=("alibaba", "reasoning"))
    add("qwq-32b", "QwQ 32B", "Qwen", 32.5, 32.5, 131072, "reasoning", 87,
        _q(32.5, layers=64, kv_heads=8), tags=("alibaba", "reasoning"))

    # ===== Gemma =====
    add("gemma-2b", "Gemma 2B", "Gemma", 2.5, 2.5, 8192, "base", 50, _q(2.5), tags=("google",))
    add("gemma-2b-it", "Gemma 2B Instruct", "Gemma", 2.5, 2.5, 8192, "instruct", 53, _q(2.5), tags=("google",))
    add("gemma-7b", "Gemma 7B", "Gemma", 8.5, 8.5, 8192, "base", 66, _q(8.5), tags=("google",))
    add("gemma-7b-it", "Gemma 7B Instruct", "Gemma", 8.5, 8.5, 8192, "instruct", 69, _q(8.5), tags=("google",))
    add("gemma-2-2b", "Gemma 2 2B", "Gemma", 2.6, 2.6, 8192, "base", 58, _q(2.6), tags=("google",))
    add("gemma-2-2b-it", "Gemma 2 2B Instruct", "Gemma", 2.6, 2.6, 8192, "instruct", 62, _q(2.6), tags=("google",))
    add("gemma-2-9b", "Gemma 2 9B", "Gemma", 9.2, 9.2, 8192, "base", 72, _q(9.2), tags=("google",))
    add("gemma-2-9b-it", "Gemma 2 9B Instruct", "Gemma", 9.2, 9.2, 8192, "instruct", 75, _q(9.2), tags=("google",))
    add("gemma-2-27b", "Gemma 2 27B", "Gemma", 27.2, 27.2, 8192, "base", 78, _q(27.2), tags=("google",))
    add("gemma-2-27b-it", "Gemma 2 27B Instruct", "Gemma", 27.2, 27.2, 8192, "instruct", 80, _q(27.2), tags=("google",))
    add("gemma-3-1b", "Gemma 3 1B", "Gemma", 1.0, 1.0, 32768, "instruct", 56, _q(1.0), tags=("google",))
    add("gemma-3-4b", "Gemma 3 4B", "Gemma", 4.3, 4.3, 131072, "instruct", 68, _q(4.3), tags=("google", "vision"))
    add("gemma-3-12b", "Gemma 3 12B", "Gemma", 12.0, 12.0, 131072, "instruct", 76, _q(12.0), tags=("google", "vision"))
    add("gemma-3-27b", "Gemma 3 27B", "Gemma", 27.0, 27.0, 131072, "instruct", 82, _q(27.0), tags=("google", "vision"))
    add("codegemma-2b", "CodeGemma 2B", "Gemma", 2.5, 2.5, 8192, "code", 55, _q(2.5), tags=("google", "code"))
    add("codegemma-7b", "CodeGemma 7B", "Gemma", 8.5, 8.5, 8192, "code", 67, _q(8.5), tags=("google", "code"))
    add("codegemma-7b-instruct", "CodeGemma 7B Instruct", "Gemma", 8.5, 8.5, 8192, "code", 70, _q(8.5), tags=("google", "code"))

    # ===== Phi (Microsoft) =====
    add("phi-2", "Phi-2 2.7B", "Phi", 2.7, 2.7, 2048, "base", 55, _q(2.7), tags=("microsoft",))
    add("phi-3-mini-4k", "Phi-3 Mini 3.8B (4K)", "Phi", 3.8, 3.8, 4096, "instruct", 68, _q(3.8), tags=("microsoft",))
    add("phi-3-mini-128k", "Phi-3 Mini 3.8B (128K)", "Phi", 3.8, 3.8, 131072, "instruct", 68, _q(3.8), tags=("microsoft",))
    add("phi-3-small-7b-128k", "Phi-3 Small 7B (128K)", "Phi", 7.4, 7.4, 131072, "instruct", 73, _q(7.4), tags=("microsoft",))
    add("phi-3-medium-14b-128k", "Phi-3 Medium 14B (128K)", "Phi", 14.0, 14.0, 131072, "instruct", 77, _q(14.0), tags=("microsoft",))
    add("phi-3.5-mini-instruct", "Phi-3.5 Mini Instruct", "Phi", 3.8, 3.8, 131072, "instruct", 71, _q(3.8), tags=("microsoft",))
    add("phi-3.5-moe-instruct", "Phi-3.5 MoE 41.9B-A6.6B", "Phi", 41.9, 6.6, 131072, "instruct", 78,
        _q(41.9, layers=32, kv_heads=8), tags=("microsoft", "moe"))
    add("phi-4", "Phi-4 14B", "Phi", 14.7, 14.7, 16384, "instruct", 80, _q(14.7), tags=("microsoft",))
    add("phi-4-mini", "Phi-4 Mini 3.8B", "Phi", 3.8, 3.8, 131072, "instruct", 73, _q(3.8), tags=("microsoft",))

    # ===== DeepSeek =====
    add("deepseek-coder-1.3b", "DeepSeek Coder 1.3B", "DeepSeek", 1.3, 1.3, 16384, "code", 55, _q(1.3), tags=("deepseek", "code"))
    add("deepseek-coder-6.7b", "DeepSeek Coder 6.7B", "DeepSeek", 6.7, 6.7, 16384, "code", 68, _q(6.7), tags=("deepseek", "code"))
    add("deepseek-coder-33b", "DeepSeek Coder 33B", "DeepSeek", 33.0, 33.0, 16384, "code", 76, _q(33.0), tags=("deepseek", "code"))
    add("deepseek-coder-v2-lite-16b", "DeepSeek Coder V2 Lite 16B-A2.4B", "DeepSeek",
        15.7, 2.4, 131072, "code", 77, _q(15.7, layers=27, kv_heads=8), tags=("deepseek", "code", "moe"))
    add("deepseek-coder-v2-236b", "DeepSeek Coder V2 236B-A21B", "DeepSeek",
        236.0, 21.0, 131072, "code", 85, _q(236.0, layers=60, kv_heads=8), tags=("deepseek", "code", "moe"))
    add("deepseek-v2-lite-16b", "DeepSeek V2 Lite 16B-A2.4B", "DeepSeek",
        15.7, 2.4, 32768, "instruct", 71, _q(15.7, layers=27, kv_heads=8), tags=("deepseek", "moe"))
    add("deepseek-v2-236b", "DeepSeek V2 236B-A21B", "DeepSeek",
        236.0, 21.0, 131072, "instruct", 80, _q(236.0, layers=60, kv_heads=8), tags=("deepseek", "moe"))
    add("deepseek-v3-671b", "DeepSeek V3 671B-A37B", "DeepSeek",
        671.0, 37.0, 131072, "instruct", 88, _q(671.0, layers=61, kv_heads=8), tags=("deepseek", "moe"))
    add("deepseek-r1", "DeepSeek R1 671B-A37B", "DeepSeek",
        671.0, 37.0, 131072, "reasoning", 90, _q(671.0, layers=61, kv_heads=8), tags=("deepseek", "moe", "reasoning"))
    add("deepseek-r1-zero", "DeepSeek R1 Zero 671B-A37B", "DeepSeek",
        671.0, 37.0, 131072, "reasoning", 87, _q(671.0, layers=61, kv_heads=8), tags=("deepseek", "moe", "reasoning"))
    add("deepseek-r1-distill-qwen-1.5b", "DeepSeek R1 Distill Qwen 1.5B", "DeepSeek",
        1.5, 1.5, 131072, "reasoning", 60, _q(1.5), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-r1-distill-qwen-7b", "DeepSeek R1 Distill Qwen 7B", "DeepSeek",
        7.0, 7.0, 131072, "reasoning", 74, _q(7.0), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-r1-distill-llama-8b", "DeepSeek R1 Distill Llama 8B", "DeepSeek",
        8.0, 8.0, 131072, "reasoning", 75, _q(8.0), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-r1-distill-qwen-14b", "DeepSeek R1 Distill Qwen 14B", "DeepSeek",
        14.0, 14.0, 131072, "reasoning", 80, _q(14.0), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-r1-distill-qwen-32b", "DeepSeek R1 Distill Qwen 32B", "DeepSeek",
        32.0, 32.0, 131072, "reasoning", 84, _q(32.0), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill Llama 70B", "DeepSeek",
        70.0, 70.0, 131072, "reasoning", 86, _q(70.0), tags=("deepseek", "reasoning", "distill"))
    add("deepseek-math-7b", "DeepSeek Math 7B", "DeepSeek", 7.0, 7.0, 4096, "math", 70,
        _q(7.0), tags=("deepseek", "math"))

    # ===== Yi =====
    add("yi-6b", "Yi 6B", "Yi", 6.0, 6.0, 4096, "base", 60, _q(6.0), tags=("01ai",))
    add("yi-9b", "Yi 9B", "Yi", 8.8, 8.8, 4096, "base", 65, _q(8.8), tags=("01ai",))
    add("yi-34b", "Yi 34B", "Yi", 34.0, 34.0, 200000, "base", 72, _q(34.0), tags=("01ai",))
    add("yi-1.5-6b-chat", "Yi 1.5 6B Chat", "Yi", 6.0, 6.0, 4096, "chat", 64, _q(6.0), tags=("01ai",))
    add("yi-1.5-9b-chat", "Yi 1.5 9B Chat", "Yi", 8.8, 8.8, 16384, "chat", 70, _q(8.8), tags=("01ai",))
    add("yi-1.5-34b-chat", "Yi 1.5 34B Chat", "Yi", 34.0, 34.0, 16384, "chat", 76, _q(34.0), tags=("01ai",))
    add("yi-coder-1.5b", "Yi Coder 1.5B", "Yi", 1.5, 1.5, 131072, "code", 56, _q(1.5), tags=("01ai", "code"))
    add("yi-coder-9b", "Yi Coder 9B", "Yi", 8.8, 8.8, 131072, "code", 72, _q(8.8), tags=("01ai", "code"))

    # ===== Cohere Command R =====
    add("command-r-35b", "Command R 35B", "Cohere", 35.0, 35.0, 131072, "instruct", 76, _q(35.0), tags=("cohere",))
    add("command-r-plus-104b", "Command R+ 104B", "Cohere", 104.0, 104.0, 131072, "instruct", 81, _q(104.0, layers=64), tags=("cohere",))
    add("command-r7b", "Command R7B", "Cohere", 7.0, 7.0, 131072, "instruct", 70, _q(7.0), tags=("cohere",))

    # ===== StarCoder =====
    add("starcoder2-3b", "StarCoder2 3B", "StarCoder", 3.0, 3.0, 16384, "code", 60, _q(3.0), tags=("bigcode", "code"))
    add("starcoder2-7b", "StarCoder2 7B", "StarCoder", 7.0, 7.0, 16384, "code", 67, _q(7.0), tags=("bigcode", "code"))
    add("starcoder2-15b", "StarCoder2 15B", "StarCoder", 15.0, 15.0, 16384, "code", 71, _q(15.0), tags=("bigcode", "code"))
    add("starcoder2-15b-instruct", "StarCoder2 15B Instruct", "StarCoder", 15.0, 15.0, 16384, "code", 74, _q(15.0), tags=("bigcode", "code"))

    # ===== Falcon =====
    add("falcon-7b", "Falcon 7B", "Falcon", 7.0, 7.0, 2048, "base", 55, _q(7.0), tags=("tii",))
    add("falcon-40b", "Falcon 40B", "Falcon", 40.0, 40.0, 2048, "base", 64, _q(40.0), tags=("tii",))
    add("falcon-180b", "Falcon 180B", "Falcon", 180.0, 180.0, 2048, "base", 73, _q(180.0, layers=80), tags=("tii",))
    add("falcon2-11b", "Falcon 2 11B", "Falcon", 11.0, 11.0, 8192, "base", 65, _q(11.0), tags=("tii",))
    add("falcon3-1b", "Falcon 3 1B Instruct", "Falcon", 1.7, 1.7, 8192, "instruct", 56, _q(1.7), tags=("tii",))
    add("falcon3-3b", "Falcon 3 3B Instruct", "Falcon", 3.2, 3.2, 32768, "instruct", 64, _q(3.2), tags=("tii",))
    add("falcon3-7b", "Falcon 3 7B Instruct", "Falcon", 7.5, 7.5, 32768, "instruct", 71, _q(7.5), tags=("tii",))
    add("falcon3-10b", "Falcon 3 10B Instruct", "Falcon", 10.3, 10.3, 32768, "instruct", 74, _q(10.3), tags=("tii",))

    # ===== SOLAR =====
    add("solar-10.7b-instruct", "SOLAR 10.7B Instruct", "SOLAR", 10.7, 10.7, 4096, "instruct", 70, _q(10.7), tags=("upstage",))

    # ===== WizardLM =====
    add("wizardlm-2-7b", "WizardLM 2 7B", "WizardLM", 7.0, 7.0, 32768, "chat", 70, _q(7.0), tags=("microsoft",))
    add("wizardlm-2-8x22b", "WizardLM 2 8x22B", "WizardLM", 141.0, 39.0, 65536, "chat", 82, _q(141.0, layers=56), tags=("microsoft", "moe"))
    add("wizardcoder-7b", "WizardCoder 7B", "WizardLM", 7.0, 7.0, 16384, "code", 64, _q(7.0), tags=("microsoft", "code"))
    add("wizardcoder-15b", "WizardCoder 15B", "WizardLM", 15.0, 15.0, 8192, "code", 68, _q(15.0), tags=("microsoft", "code"))
    add("wizardcoder-33b", "WizardCoder 33B", "WizardLM", 33.0, 33.0, 16384, "code", 74, _q(33.0), tags=("microsoft", "code"))
    add("wizardmath-7b", "WizardMath 7B", "WizardLM", 7.0, 7.0, 4096, "math", 67, _q(7.0), tags=("microsoft", "math"))
    add("wizardmath-70b", "WizardMath 70B", "WizardLM", 70.0, 70.0, 4096, "math", 78, _q(70.0), tags=("microsoft", "math"))

    # ===== Small / specialized chat =====
    add("stablelm-zephyr-3b", "StableLM Zephyr 3B", "StableLM", 2.8, 2.8, 4096, "chat", 56, _q(2.8), tags=("stability",))
    add("tinyllama-1.1b", "TinyLlama 1.1B Chat", "TinyLlama", 1.1, 1.1, 2048, "chat", 42, _q(1.1, layers=22, kv_heads=4, head_dim=64), tags=())
    add("openchat-3.6-8b", "OpenChat 3.6 8B", "OpenChat", 8.0, 8.0, 8192, "chat", 73, _q(8.0), tags=())
    add("hermes-2-pro-mistral-7b", "Hermes 2 Pro Mistral 7B", "NousResearch", 7.24, 7.24, 32768, "chat", 70, _q(7.24), tags=("nous",))
    add("hermes-2-theta-llama-8b", "Hermes 2 Theta Llama 8B", "NousResearch", 8.03, 8.03, 8192, "chat", 71, _q(8.03), tags=("nous",))
    add("hermes-3-llama-8b", "Hermes 3 Llama 8B", "NousResearch", 8.03, 8.03, 131072, "chat", 73, _q(8.03), tags=("nous",))
    add("hermes-3-llama-70b", "Hermes 3 Llama 70B", "NousResearch", 70.6, 70.6, 131072, "chat", 82, _q(70.6), tags=("nous",))
    add("hermes-3-llama-405b", "Hermes 3 Llama 405B", "NousResearch", 405.0, 405.0, 131072, "chat", 86, _q(405.0, layers=126), tags=("nous",))
    add("dolphin-2.5-mixtral-8x7b", "Dolphin 2.5 Mixtral 8x7B", "Dolphin", 46.7, 12.9, 32768, "chat", 74, _q(46.7, layers=32), tags=("dolphin", "moe"))
    add("dolphin-mistral-7b", "Dolphin Mistral 7B", "Dolphin", 7.24, 7.24, 32768, "chat", 68, _q(7.24), tags=("dolphin",))
    add("dolphin-2.9-llama3-8b", "Dolphin 2.9 Llama3 8B", "Dolphin", 8.03, 8.03, 8192, "chat", 71, _q(8.03), tags=("dolphin",))
    add("dolphin-2.9-llama3-70b", "Dolphin 2.9 Llama3 70B", "Dolphin", 70.6, 70.6, 8192, "chat", 80, _q(70.6), tags=("dolphin",))
    add("nous-hermes-2-mixtral-8x7b", "Nous Hermes 2 Mixtral 8x7B", "NousResearch", 46.7, 12.9, 32768, "chat", 76, _q(46.7, layers=32), tags=("nous", "moe"))

    # ===== Microsoft Orca =====
    add("orca-2-7b", "Orca 2 7B", "Orca", 6.74, 6.74, 4096, "chat", 62, _q(6.74, gqa=False), tags=("microsoft",))
    add("orca-2-13b", "Orca 2 13B", "Orca", 13.0, 13.0, 4096, "chat", 66, _q(13.0, gqa=False), tags=("microsoft",))

    # ===== Vicuna / Zephyr =====
    add("vicuna-7b-v1.5", "Vicuna 7B v1.5", "Vicuna", 6.74, 6.74, 4096, "chat", 58, _q(6.74, gqa=False), tags=("lmsys",))
    add("vicuna-13b-v1.5", "Vicuna 13B v1.5", "Vicuna", 13.0, 13.0, 4096, "chat", 62, _q(13.0, gqa=False), tags=("lmsys",))
    add("zephyr-7b-beta", "Zephyr 7B Beta", "Zephyr", 7.24, 7.24, 32768, "chat", 68, _q(7.24), tags=("huggingface",))

    # ===== Bagel (Jon Durbin) =====
    add("bagel-7b", "Bagel 7B v0.4", "Bagel", 7.24, 7.24, 32768, "chat", 67, _q(7.24), tags=())
    add("bagel-34b", "Bagel 34B v0.4", "Bagel", 34.0, 34.0, 200000, "chat", 75, _q(34.0), tags=())

    # ===== NVIDIA Nemotron =====
    add("nemotron-70b-instruct", "Llama 3.1 Nemotron 70B Instruct", "Nemotron", 70.6, 70.6, 131072, "instruct", 86, _q(70.6), tags=("nvidia",))
    add("nemotron-mini-4b", "Nemotron Mini 4B Instruct", "Nemotron", 4.2, 4.2, 4096, "instruct", 62, _q(4.2), tags=("nvidia",))
    add("nemotron-nano-9b", "Nemotron Nano 9B", "Nemotron", 9.0, 9.0, 131072, "instruct", 74, _q(9.0), tags=("nvidia",))

    # ===== IBM Granite =====
    add("granite-3b-code", "Granite 3B Code", "Granite", 3.0, 3.0, 128000, "code", 62, _q(3.0), tags=("ibm", "code"))
    add("granite-3b-code-instruct", "Granite 3B Code Instruct", "Granite", 3.0, 3.0, 128000, "code", 66, _q(3.0), tags=("ibm", "code"))
    add("granite-8b-code", "Granite 8B Code", "Granite", 8.0, 8.0, 128000, "code", 70, _q(8.0), tags=("ibm", "code"))
    add("granite-8b-code-instruct", "Granite 8B Code Instruct", "Granite", 8.0, 8.0, 128000, "code", 73, _q(8.0), tags=("ibm", "code"))
    add("granite-3.1-2b-instruct", "Granite 3.1 2B Instruct", "Granite", 2.0, 2.0, 131072, "instruct", 60, _q(2.0), tags=("ibm",))
    add("granite-3.1-8b-instruct", "Granite 3.1 8B Instruct", "Granite", 8.0, 8.0, 131072, "instruct", 72, _q(8.0), tags=("ibm",))

    # ===== HF SmolLM2 =====
    add("smollm2-135m-instruct", "SmolLM2 135M Instruct", "SmolLM", 0.135, 0.135, 8192, "instruct", 30, _q(0.135, layers=30, kv_heads=3, head_dim=64), tags=("huggingface",))
    add("smollm2-360m-instruct", "SmolLM2 360M Instruct", "SmolLM", 0.36, 0.36, 8192, "instruct", 38, _q(0.36, layers=32, kv_heads=5, head_dim=64), tags=("huggingface",))
    add("smollm2-1.7b-instruct", "SmolLM2 1.7B Instruct", "SmolLM", 1.7, 1.7, 8192, "instruct", 56, _q(1.7), tags=("huggingface",))

    # ===== OLMo =====
    add("olmo-7b", "OLMo 7B", "OLMo", 7.0, 7.0, 4096, "base", 56, _q(7.0, gqa=False), tags=("ai2",))

    # ===== EXAONE =====
    add("exaone-3.5-7.8b-instruct", "EXAONE 3.5 7.8B Instruct", "EXAONE", 7.8, 7.8, 32768, "instruct", 73, _q(7.8), tags=("lg",))

    # ===== Cohere Aya =====
    add("aya-23-8b", "Aya 23 8B", "Aya", 8.0, 8.0, 8192, "chat", 66, _q(8.0), tags=("cohere", "multilingual"))
    add("aya-23-35b", "Aya 23 35B", "Aya", 35.0, 35.0, 8192, "chat", 74, _q(35.0), tags=("cohere", "multilingual"))
    add("aya-expanse-8b", "Aya Expanse 8B", "Aya", 8.0, 8.0, 8192, "chat", 70, _q(8.0), tags=("cohere", "multilingual"))
    add("aya-expanse-32b", "Aya Expanse 32B", "Aya", 32.0, 32.0, 8192, "chat", 76, _q(32.0), tags=("cohere", "multilingual"))

    # ===== Misc =====
    add("smaug-72b", "Smaug 72B", "Smaug", 72.0, 72.0, 8192, "chat", 80, _q(72.0), tags=("abacus",))
    add("glm-4-9b-chat", "GLM-4 9B Chat", "ChatGLM", 9.4, 9.4, 131072, "chat", 73, _q(9.4), tags=("zhipu",))
    add("internlm2.5-7b-chat", "InternLM 2.5 7B Chat", "InternLM", 7.7, 7.7, 32768, "chat", 71, _q(7.7), tags=("shanghai-ai",))
    add("internlm2.5-20b-chat", "InternLM 2.5 20B Chat", "InternLM", 20.0, 20.0, 32768, "chat", 76, _q(20.0), tags=("shanghai-ai",))
    add("internlm3-8b-instruct", "InternLM 3 8B Instruct", "InternLM", 8.8, 8.8, 32768, "instruct", 75, _q(8.8), tags=("shanghai-ai",))

    return M


# Lazily instantiated; cheap, but no need to repeat on every CLI call.
_CATALOG: Optional[List[ModelSpec]] = None


def get_catalog(include_external: bool = True) -> List[ModelSpec]:
    """Return the active catalog.

    By default, this is the curated 206-entry baseline plus any externally
    synced specs (e.g. HuggingFace via `modelfit catalog sync`). Curated
    entries always win on duplicate ids — manual quality scores are kept.
    """
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = build_catalog()
    if not include_external:
        return _CATALOG
    # Lazy import to avoid circular: catalog_sync imports from models.
    try:
        from modelfit.catalog_sync import cached_specs
        ext = cached_specs()
    except Exception:
        ext = []
    if not ext:
        return _CATALOG
    by_id = {m.id: m for m in ext}
    for m in _CATALOG:
        by_id[m.id] = m  # curated wins
    return list(by_id.values())


def find_model(model_id_or_name: str) -> Optional[ModelSpec]:
    target = model_id_or_name.lower()
    for m in get_catalog():
        if m.id.lower() == target or m.name.lower() == target:
            return m
    # Substring fallback.
    matches = [m for m in get_catalog()
               if target in m.id.lower() or target in m.name.lower()]
    return matches[0] if len(matches) == 1 else None
