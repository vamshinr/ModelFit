"""Single-pick recommendation logic.

Given a use case + the user's hardware, pick ONE model that's a great
starting point — not the abstract highest-scoring entry, but a balanced,
beginner-friendly choice.

We bias toward:
  * Instruct/chat models (not 'base')
  * Models with a verified Ollama tag (so we can give a one-line install)
  * Some headroom in the memory budget (so the system doesn't OOM on long chats)
  * Reasonable size — avoid the absolute smallest unless that's all that fits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from modelfit.hardware import HardwareProfile
from modelfit.scoring import ScoredModel, rank_all
from modelfit.runners import OLLAMA_TAGS
from modelfit.quantization import GB


@dataclass
class Recommendation:
    primary: ScoredModel
    runner_ups: List[ScoredModel]
    why: str
    use_case: str

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.to_dict(),
            "runner_ups": [s.to_dict() for s in self.runner_ups],
            "why": self.why,
            "use_case": self.use_case,
        }


def _beginner_friendliness_bonus(s: ScoredModel) -> float:
    """How much to nudge a score toward beginner-friendliness."""
    bonus = 0.0
    # Has a one-command Ollama install — huge win for beginners.
    if s.model.id in OLLAMA_TAGS:
        bonus += 4
    # Penalise base models for beginners (they don't follow instructions).
    if s.model.type == "base":
        bonus -= 25
    # Bonus for instruct/chat over reasoning/code unless the use case asks for them.
    if s.model.type in ("instruct", "chat"):
        bonus += 1.5
    # Prefer comfortable fit (>= 20% headroom in budget).
    if s.fit.budget_bytes:
        headroom = (s.fit.budget_bytes - s.fit.total_bytes) / s.fit.budget_bytes
        if headroom >= 0.25:
            bonus += 2
        elif headroom < 0.05:
            bonus -= 3
    # A *very* tiny model (sub-1B) is usually worse for an enjoyable chat —
    # only recommend if literally nothing else fits.
    if s.model.params_b < 0.8:
        bonus -= 4
    return bonus


def recommend(hw: HardwareProfile, use_case: str = "balanced",
              min_context: int = 4096) -> Optional[Recommendation]:
    ranked = rank_all(hw, use_case=use_case, min_context=min_context)
    if not ranked:
        # Try again with a smaller context floor; the user just needs *something*.
        ranked = rank_all(hw, use_case=use_case, min_context=2048)
    if not ranked:
        return None

    # Re-rank by beginner-friendliness adjustment.
    ranked_with_bonus = sorted(
        ranked,
        key=lambda s: s.composite + _beginner_friendliness_bonus(s),
        reverse=True,
    )
    primary = ranked_with_bonus[0]
    runner_ups = [s for s in ranked_with_bonus[1:4] if s.model.id != primary.model.id]

    why = _explain_pick(primary, hw, use_case)
    return Recommendation(primary=primary, runner_ups=runner_ups,
                          why=why, use_case=use_case)


def _explain_pick(s: ScoredModel, hw: HardwareProfile, use_case: str) -> str:
    """Plain-English 'why this model'."""
    parts = []
    m = s.model
    parts.append(f"For '{use_case}' on your machine, {m.name} is a balanced pick.")
    if m.id in OLLAMA_TAGS:
        parts.append("It's installable in one command via Ollama, so you can start chatting in a couple of minutes.")
    parts.append(
        f"It fits at {s.fit.quant} compression with {s.fit.context:,} tokens of context "
        f"— that's about {s.fit.total_bytes / GB:.1f} GB of memory."
    )
    if s.tokens_per_sec >= 20:
        parts.append(f"You should see ~{s.tokens_per_sec:.0f} tokens/sec, which feels snappy in a chat window.")
    elif s.tokens_per_sec >= 8:
        parts.append(f"Expect ~{s.tokens_per_sec:.0f} tokens/sec — usable for chat, a bit slow for long answers.")
    else:
        parts.append(
            f"Expect only ~{s.tokens_per_sec:.0f} tokens/sec on this hardware. "
            f"If that feels too slow, look at the runner-ups (smaller models)."
        )
    if m.params_b >= 30:
        parts.append("This is one of the bigger models that still fits — quality should be strong.")
    elif m.params_b < 3:
        parts.append(
            "It's a small model — fast and friendly, but it may stumble on hard reasoning. "
            "If you have more memory headroom, consider a 7B+ runner-up."
        )
    return " ".join(parts)
