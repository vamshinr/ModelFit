"""Plain-language layer for absolute beginners.

This is THE translation layer between the geeky internals (Q4_K_M, KV cache,
tokens/sec) and what a brand-new user actually needs to know:

  "Will this run on my computer? Will it feel fast? How good are the answers?"

Every function here returns user-facing strings. No domain knowledge required
from the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modelfit.models import ModelSpec, QUANT_BYTES_PER_PARAM
from modelfit.scoring import ScoredModel
from modelfit.quantization import FitResult, GB


# ---------------------------------------------------------------------------
# Quantization → plain English
# ---------------------------------------------------------------------------
QUANT_LABEL = {
    "F16":    ("Full precision",      "★★★★★", "No quality loss, but huge files. Only worth it on big GPUs."),
    "Q8_0":   ("Excellent quality",   "★★★★★", "Practically identical to the full model. Best when you have the memory."),
    "Q6_K":   ("Great quality",       "★★★★★", "Almost indistinguishable from the full model."),
    "Q5_K_M": ("Great quality",       "★★★★½", "Very small quality drop. A common sweet spot."),
    "Q5_K_S": ("Good quality",        "★★★★½", "Slightly smaller than Q5_K_M, same idea."),
    "Q5_0":   ("Good quality",        "★★★★½", "Older 5-bit format. Use Q5_K_M instead if available."),
    "Q4_K_M": ("Good quality",        "★★★★",  "The most popular quant. Strong quality, half the memory of Q8."),
    "Q4_K_S": ("Good quality",        "★★★★",  "Slightly smaller than Q4_K_M with a tiny quality drop."),
    "Q4_0":   ("Decent quality",      "★★★½",  "Older 4-bit format. Prefer Q4_K_M."),
    "Q3_K_M": ("Compressed",          "★★★",   "Noticeable quality drop. Use when memory is tight."),
    "Q3_K_S": ("Heavily compressed",  "★★½",   "Saves more memory but answers get worse."),
    "Q2_K":   ("Heavily compressed",  "★★",    "Last resort. Quality drops a lot but it'll run."),
}


def quant_label(quant: Optional[str]) -> tuple[str, str, str]:
    """(label, stars, one-line explainer) for the given quant code."""
    if not quant:
        return ("—", "", "")
    return QUANT_LABEL.get(quant, (quant, "?", ""))


# ---------------------------------------------------------------------------
# Speed (tokens/sec) → plain English
# ---------------------------------------------------------------------------
def speed_label(tps: float) -> tuple[str, str]:
    """(headline, what-it-feels-like)"""
    if tps <= 0:
        return ("Won't run", "Doesn't fit on this machine.")
    if tps >= 40:
        return ("⚡ Instant",  "Replies stream faster than you can read.")
    if tps >= 20:
        return ("🚀 Snappy",  "Feels like a fast chatbot — barely any wait.")
    if tps >= 8:
        return ("✅ Usable",  "OK for chat. Roughly 1 word every few seconds.")
    if tps >= 3:
        return ("🐢 Slow",    "Workable for short questions; tiresome for long ones.")
    return ("🦥 Very slow",   "A few words per second. Probably too slow to use day-to-day.")


# ---------------------------------------------------------------------------
# Context (tokens) → plain English
# ---------------------------------------------------------------------------
def context_label(tokens: int) -> tuple[str, str]:
    """(headline, rough size comparison)"""
    if tokens <= 0:
        return ("—", "")
    # ~0.75 words per token; ~250 words per page.
    pages = tokens * 0.75 / 250
    if tokens >= 131072:
        return ("📚 Whole book",       f"Holds ~{int(pages):,} pages of text at once.")
    if tokens >= 32768:
        return ("📖 Long document",    f"Holds ~{int(pages):,} pages of text at once.")
    if tokens >= 16384:
        return ("📑 Medium document",  f"Holds ~{int(pages):,} pages of text at once.")
    if tokens >= 8192:
        return ("📄 Short document",   f"Holds ~{int(pages):,} pages of text at once.")
    if tokens >= 4096:
        return ("📝 Long email",       f"About {int(tokens*0.75):,} words at once.")
    return    ("✉️ Short prompt",     f"About {int(tokens*0.75):,} words at once.")


# ---------------------------------------------------------------------------
# Quality (intrinsic 0-100) → plain English
# ---------------------------------------------------------------------------
def quality_label(quality_score: float) -> tuple[str, str]:
    if quality_score >= 85:
        return ("🏆 Excellent",  "On par with the best open models available.")
    if quality_score >= 75:
        return ("⭐ Great",      "Strong answers — competitive with paid services for most tasks.")
    if quality_score >= 65:
        return ("👍 Good",       "Solid for everyday use; may stumble on hard reasoning.")
    if quality_score >= 55:
        return ("🆗 Decent",     "Fine for simple chat and writing; weaker on reasoning.")
    return ("🪵 Basic",          "Best for small / experimental use.")


# ---------------------------------------------------------------------------
# Type → plain English
# ---------------------------------------------------------------------------
TYPE_LABEL = {
    "chat":      "💬 Chat — built for back-and-forth conversation.",
    "instruct":  "🎯 Instruct — follows task instructions reliably.",
    "code":      "💻 Code — fine-tuned to write & explain programs.",
    "reasoning": "🧠 Reasoning — thinks step-by-step before answering.",
    "math":      "🧮 Math — specialised for arithmetic and word problems.",
    "base":      "🧱 Base — raw model, no instruction tuning. Beginners want one of the others.",
    "vision":    "🖼️ Vision — understands images alongside text.",
}


def type_label(t: str) -> str:
    return TYPE_LABEL.get(t, t)


# ---------------------------------------------------------------------------
# Overall verdict for a fit
# ---------------------------------------------------------------------------
def verdict(scored: ScoredModel) -> tuple[str, str]:
    """(headline, one-line summary of what running this means)"""
    if not scored.fit.fits:
        return ("❌ Won't fit", scored.fit.reason)
    composite = scored.composite
    headroom = scored.fit.budget_bytes - scored.fit.total_bytes
    headroom_gb = headroom / GB
    if composite >= 75 and headroom_gb >= 1.0:
        h = "🌟 Excellent pick"
    elif composite >= 65:
        h = "✅ Good pick"
    elif composite >= 50:
        h = "👍 Works"
    else:
        h = "⚠️ Tight"

    sp_head, _ = speed_label(scored.tokens_per_sec)
    ctx_head, _ = context_label(scored.fit.context or 0)
    qlabel, _stars, _desc = quant_label(scored.fit.quant)
    summary = f"{sp_head} · {ctx_head} · {qlabel}"
    return (h, summary)


# ---------------------------------------------------------------------------
# Download-size estimate
# ---------------------------------------------------------------------------
def download_size_gb(model: ModelSpec, quant: str) -> float:
    """Approximate GGUF file size on disk (similar to runtime weights bytes)."""
    return model.params_b * 1e9 * QUANT_BYTES_PER_PARAM.get(quant, 0.6) / GB


def download_label(gb: float) -> str:
    if gb < 1: return f"{gb*1024:.0f} MB"
    if gb < 100: return f"{gb:.1f} GB"
    return f"{gb:.0f} GB"


# ---------------------------------------------------------------------------
# "What does this model do well?" — a one-line pitch per type
# ---------------------------------------------------------------------------
def best_for(model: ModelSpec) -> str:
    tag_hints = {
        "code": "writing & explaining code",
        "math": "math problems and step-by-step proofs",
        "reasoning": "logic puzzles and multi-step reasoning",
        "multilingual": "non-English languages",
        "vision": "questions about images",
        "moe": "high quality with relatively fast generation",
    }
    type_hints = {
        "chat":      "casual conversation, writing help, summarising",
        "instruct":  "following instructions, drafting emails / docs, classification",
        "code":      "writing & debugging code in many languages",
        "reasoning": "math, logic, and multi-step problem solving",
        "math":      "math problems",
        "base":      "fine-tuning experiments (not chat — pick an instruct model instead)",
        "vision":    "describing images and answering questions about them",
    }
    pitches = [type_hints.get(model.type, "general use")]
    for t in model.tags:
        if t in tag_hints:
            pitches.append(tag_hints[t])
    # De-dupe while preserving order.
    seen, out = set(), []
    for p in pitches:
        if p not in seen:
            seen.add(p); out.append(p)
    return "; ".join(out)


# ---------------------------------------------------------------------------
# Memory tier explainer
# ---------------------------------------------------------------------------
def tier_explanation(tier: str) -> str:
    if tier == "gpu":
        return "Runs on the GPU (fast — what you want)."
    if tier == "unified":
        return "Runs on Apple Silicon unified memory — GPU and CPU share RAM."
    return "Runs on CPU only (slower; consider a GPU for serious use)."


# ---------------------------------------------------------------------------
# Why didn't a model fit?  Beginner-friendly diagnosis
# ---------------------------------------------------------------------------
def why_doesnt_fit(model: ModelSpec, fit: FitResult, budget_gb: float) -> str:
    need = fit.total_bytes / GB if fit.total_bytes else 0
    short_by = max(0, need - budget_gb)
    return (
        f"This model needs about {need:.0f} GB to run even at the smallest "
        f"compression. You have ~{budget_gb:.0f} GB available — short by "
        f"{short_by:.0f} GB. Try a smaller model in the same family, or look "
        f"at the 'reverse' command to see what hardware would fit it."
    )


# ---------------------------------------------------------------------------
# Use-case → plain-English description (for `wizard`, etc.)
# ---------------------------------------------------------------------------
USE_CASE_DESCRIPTION = {
    "chat":         "Casual conversation, writing help, summarising — speed matters.",
    "reasoning":    "Hard problems, logic, multi-step thinking — answer quality matters most.",
    "code":         "Writing or debugging code — balance of quality and speed.",
    "math":         "Math problems — quality over speed.",
    "long-context": "Reading long documents or whole books — context length matters most.",
    "agent":        "Tool use, RAG, agentic workflows — needs to follow instructions well.",
    "research":     "Research, analysis, deep reading — quality + context.",
    "balanced":     "I'm not sure / general everyday use.",
}
