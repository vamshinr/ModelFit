"""HuggingFace catalog sync.

Pulls a curated slice of HuggingFace's `text-generation` model list into a
local cache, and merges it with the built-in 206-model curated catalog at
runtime (via `models.get_catalog`, which calls `apply_external_catalog`).

Design choices:

* The curated catalog is *always* the baseline — manually tuned quality
  scores there are authoritative and never overwritten.
* HF gives us name, downloads, tags, and (sometimes) safetensors param
  counts, but not architectural details. We derive context length and KV
  cache from heuristics + per-architecture rules, with explicit notes.
* We hit only the public `/api/models` endpoint — no auth needed for
  public models. A token can be supplied via $HF_TOKEN to extend rate
  limits / pull gated models.
* The cache lives at ~/.modelfit/catalog_hf.json so re-running `sync`
  is incremental and works offline.

This file deliberately uses ONLY stdlib so it works in any Python install.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modelfit.models import (
    ModelSpec, kv_per_token, VALID_TYPES, build_catalog,
)


CACHE_DIR = Path.home() / ".modelfit"
CACHE_FILE = CACHE_DIR / "catalog_hf.json"
SETTINGS_FILE = CACHE_DIR / "settings.json"

HF_API_BASE = "https://huggingface.co/api"
USER_AGENT = "modelfit/0.1 (https://github.com/modelfit)"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def cache_path() -> Path:
    return CACHE_FILE


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> Dict[str, Any]:
    if not CACHE_FILE.exists():
        return {"models": [], "synced_at": None, "source": "huggingface"}
    try:
        with CACHE_FILE.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"models": [], "synced_at": None, "source": "huggingface"}


def save_cache(data: Dict[str, Any]) -> None:
    _ensure_dir()
    with CACHE_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def clear_cache() -> None:
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def load_settings() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {"providers": {"huggingface": {"enabled": True}}}
    try:
        with SETTINGS_FILE.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"providers": {"huggingface": {"enabled": True}}}


def save_settings(s: Dict[str, Any]) -> None:
    _ensure_dir()
    with SETTINGS_FILE.open("w") as f:
        json.dump(s, f, indent=2)


def set_provider_enabled(provider: str, enabled: bool) -> None:
    s = load_settings()
    s.setdefault("providers", {}).setdefault(provider, {})["enabled"] = enabled
    save_settings(s)


def is_provider_enabled(provider: str) -> bool:
    s = load_settings()
    return bool(s.get("providers", {}).get(provider, {}).get("enabled", True))


# ---------------------------------------------------------------------------
# Provenance / source listing
# ---------------------------------------------------------------------------
def list_sources() -> List[Dict[str, Any]]:
    cache = load_cache()
    return [
        {
            "name": "Built-in curated catalog",
            "enabled": True,
            "last_sync": "embedded",
            "count": len(build_catalog()),
        },
        {
            "name": "HuggingFace Hub",
            "enabled": is_provider_enabled("huggingface"),
            "last_sync": cache.get("synced_at"),
            "count": len(cache.get("models", [])),
        },
    ]


# ---------------------------------------------------------------------------
# HF API client
# ---------------------------------------------------------------------------
def _http_get(url: str, *, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def fetch_model_list(*, limit: int = 100, sort: str = "downloads",
                     timeout: int = 15) -> List[Dict[str, Any]]:
    """Call HF /api/models with the text-generation pipeline filter."""
    # Trending isn't a real sort param on /api/models — emulate by sorting
    # 'likes' and trimming to recent.
    sort_param = "downloads" if sort == "trending" else sort
    qs = urllib.parse.urlencode({
        "pipeline_tag": "text-generation",
        "sort": sort_param,
        "direction": "-1",
        "limit": limit,
        # Pull tags so we can classify type (instruct, chat, code, …).
        "full": "true",
    })
    url = f"{HF_API_BASE}/models?{qs}"
    raw = _http_get(url, timeout=timeout)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected HF response shape: {type(data).__name__}")
    return data


def fetch_model_config(repo_id: str, *, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Try to fetch the model's config.json — gives us architectural details."""
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        raw = _http_get(url, timeout=timeout)
        return json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError,
             json.JSONDecodeError, TimeoutError, OSError):
        return None


# ---------------------------------------------------------------------------
# HF entry → ModelSpec inference
# ---------------------------------------------------------------------------
def _slugify(s: str) -> str:
    s = s.lower().replace("/", "-")
    return re.sub(r"[^a-z0-9._-]+", "-", s).strip("-")


def _param_count_from_safetensors(entry: Dict[str, Any]) -> Optional[float]:
    """Best-effort total params from the safetensors metadata HF returns."""
    st = entry.get("safetensors")
    if not isinstance(st, dict):
        return None
    total = st.get("total") or st.get("parameters")
    if isinstance(total, (int, float)) and total > 0:
        return float(total) / 1e9
    if isinstance(total, dict):
        # Sometimes it's {"F16": 8030000000} etc.
        params = sum(v for v in total.values() if isinstance(v, (int, float)))
        if params:
            return params / 1e9
    return None


def _params_from_name(name: str) -> Optional[float]:
    """Pattern-match common '7B' / '13B' / '70B' / '405B' suffixes."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*[bB]", name)
    if m:
        # MoE pattern like "8x7B" → 8 experts × 7B ≈ active 7B, total ~56B
        experts, per = float(m.group(1)), float(m.group(2))
        return experts * per
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-z])", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?![a-z])", name)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _context_from_name(name: str) -> int:
    """Heuristic context length when config.json isn't reachable (gated repos
    without HF_TOKEN, deleted files, etc). Returns the family default."""
    n = name.lower()
    # Explicit suffixes win.
    if "1m" in n or "1048k" in n:
        return 1_048_576
    if "256k" in n:
        return 262_144
    if "200k" in n:
        return 200_000
    if "128k" in n:
        return 131_072
    if "64k" in n:
        return 65_536
    if "32k" in n:
        return 32_768
    if "16k" in n:
        return 16_384
    if "8k" in n:
        return 8_192
    if "4k" in n:
        return 4_096
    # Family defaults — based on each release's stated context window.
    if "llama-3.1" in n or "llama-3.2" in n or "llama-3.3" in n:
        return 131_072
    if "llama-3" in n:
        return 8_192
    if "llama-2" in n:
        return 4_096
    if "qwen3" in n or "qwen2.5" in n or "qwen2" in n:
        return 32_768
    if "qwen1.5" in n or "qwen-1.5" in n:
        return 32_768
    if "mistral-nemo" in n or "mistral-small-3" in n:
        return 131_072
    if "mistral" in n or "mixtral" in n:
        return 32_768
    if "deepseek-v3" in n or "deepseek-r1" in n or "deepseek-v2" in n:
        return 131_072
    if "deepseek-coder" in n:
        return 16_384
    if "phi-3" in n or "phi-4" in n or "phi-3.5" in n:
        return 131_072
    if "gemma-3" in n:
        return 131_072
    if "gemma-2" in n or "gemma" in n:
        return 8_192
    if "command-r" in n:
        return 131_072
    if "yi-1.5" in n:
        return 16_384
    if "yi-coder" in n:
        return 131_072
    if "yi-" in n:
        return 4_096
    if "gpt-oss" in n:
        return 131_072
    return 8_192  # generous modern default; was 4096


def _classify_type(tags: List[str], name: str) -> str:
    """Guess type from HF tags and name."""
    tagset = {t.lower() for t in tags}
    n = name.lower()
    if "code" in tagset or "coder" in n or "code-" in n:
        return "code"
    if "math" in tagset or "math" in n:
        return "math"
    if "reasoning" in tagset or "r1" in n or "qwq" in n or "thinking" in n:
        return "reasoning"
    if "vision" in tagset or "vl" in n or "-vision" in n or "multimodal" in tagset:
        return "vision"
    if "instruct" in n or "instruct" in tagset:
        return "instruct"
    if "chat" in n or "chat" in tagset:
        return "chat"
    return "base"


def _quality_from_downloads(downloads: int, likes: int) -> float:
    """Rough quality prior from popularity. Real benchmarks would be better.
    A model with 10M downloads is almost certainly competent; a model with
    100 is much riskier. Bounded to [40, 80] so it never beats a curated
    quality score that we set by hand.
    """
    score = 40.0
    if downloads > 0:
        # log10 scale: 1e3 → +5, 1e6 → +20, 1e9 → +30
        import math
        score += min(35.0, math.log10(max(downloads, 1)) * 4.5)
    if likes > 0:
        import math
        score += min(8.0, math.log10(max(likes, 1)) * 2.0)
    return max(40.0, min(80.0, score))


def _build_spec_from_hf(entry: Dict[str, Any],
                        config: Optional[Dict[str, Any]] = None) -> Optional[ModelSpec]:
    repo_id = entry.get("modelId") or entry.get("id")
    if not repo_id:
        return None
    name = repo_id.split("/")[-1]
    tags = entry.get("tags", []) or []
    downloads = int(entry.get("downloads") or 0)
    likes = int(entry.get("likes") or 0)

    # Params
    params_b = _param_count_from_safetensors(entry) or _params_from_name(name)
    if params_b is None or params_b <= 0 or params_b > 2000:
        return None  # Skip — we can't size memory without param count.

    # Context length — prefer config.json's max_position_embeddings,
    # otherwise fall back to a name/family heuristic so gated repos
    # (no HF_TOKEN → no config) don't get pinned to 4K.
    context_max = _context_from_name(name)
    if config:
        ctx = (config.get("max_position_embeddings")
                or config.get("rope_scaling", {}).get("max_position_embeddings"))
        if isinstance(ctx, int) and ctx >= 2048:
            context_max = min(ctx, 1_048_576)  # cap at 1M

    # Architecture details for KV cache
    layers = config.get("num_hidden_layers") if config else None
    kv_heads = (config.get("num_key_value_heads") if config else None) or layers
    head_dim = None
    if config and config.get("hidden_size") and config.get("num_attention_heads"):
        head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    kv = kv_per_token(params_b,
                       layers=layers if isinstance(layers, int) else None,
                       kv_heads=kv_heads if isinstance(kv_heads, int) else None,
                       head_dim=head_dim if isinstance(head_dim, int) else 128)

    type_ = _classify_type(tags, name)
    if type_ not in VALID_TYPES:
        type_ = "instruct"

    # MoE detection — config gives "num_experts" or "num_local_experts".
    active_params_b = params_b
    if config:
        n_experts = config.get("num_local_experts") or config.get("num_experts")
        experts_per_tok = config.get("num_experts_per_tok")
        if n_experts and experts_per_tok:
            active_params_b = params_b * (experts_per_tok / n_experts)

    quality = _quality_from_downloads(downloads, likes)
    # ID uses the repo name only (not the org prefix) so HF entries can
    # collide with — and be superseded by — curated entries in get_catalog().
    # Without this, "meta-llama/Llama-3.1-8B-Instruct" became
    # "meta-llama-llama-3.1-8b-instruct" and lived alongside the curated
    # "llama-3.1-8b-instruct" as a duplicate.
    spec = ModelSpec(
        id=_slugify(name),
        name=name,
        family=repo_id.split("/")[0],
        params_b=round(params_b, 3),
        active_params_b=round(active_params_b, 3),
        context_max=context_max,
        type=type_,
        quality=quality,
        kv_bytes_per_token=int(kv),
        license=entry.get("library_name", "open") or "open",
        tags=("hf",) + tuple(sorted({t.lower() for t in tags
                                       if t.lower() in {"moe", "code", "math",
                                                          "reasoning", "vision"}})),
    )
    return spec


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------
def sync_from_huggingface(limit: int = 100, sort: str = "downloads",
                          min_downloads: int = 1000,
                          timeout: int = 15,
                          fetch_configs: bool = True) -> Dict[str, Any]:
    """Pull a slice of HF models, derive ModelSpecs, persist to cache.

    Returns a sync report dict (fetched, merged, updated, skipped).
    """
    started = time.time()
    cache = load_cache()
    existing = {m["id"]: m for m in cache.get("models", [])}
    fetched = fetch_model_list(limit=limit, sort=sort, timeout=timeout)

    merged = 0
    updated = 0
    skipped = 0
    new_models: List[Dict[str, Any]] = []
    for entry in fetched:
        if int(entry.get("downloads") or 0) < min_downloads:
            skipped += 1
            continue
        repo_id = entry.get("modelId") or entry.get("id")
        if not repo_id:
            skipped += 1
            continue
        config = fetch_model_config(repo_id, timeout=min(timeout, 8)) if fetch_configs else None
        spec = _build_spec_from_hf(entry, config)
        if spec is None:
            skipped += 1
            continue
        spec_dict = spec.to_dict()
        spec_dict["_meta"] = {
            "downloads": entry.get("downloads"),
            "likes": entry.get("likes"),
            "repo_id": repo_id,
            "synced_at": int(time.time()),
        }
        if spec.id in existing:
            updated += 1
        else:
            merged += 1
        new_models.append(spec_dict)

    # Merge new_models over existing.
    by_id = {m["id"]: m for m in cache.get("models", [])}
    for m in new_models:
        by_id[m["id"]] = m
    cache["models"] = list(by_id.values())
    cache["synced_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cache["source"] = "huggingface"
    save_cache(cache)

    return {
        "ok": True,
        "fetched": len(fetched),
        "merged": merged,
        "updated": updated,
        "skipped": skipped,
        "elapsed_sec": round(time.time() - started, 2),
        "cache_path": str(CACHE_FILE),
    }


# ---------------------------------------------------------------------------
# Catalog augmentation: feed HF cache into models.get_catalog()
# ---------------------------------------------------------------------------
def cached_specs() -> List[ModelSpec]:
    """Return ModelSpec objects from the HF cache (if any)."""
    if not is_provider_enabled("huggingface"):
        return []
    cache = load_cache()
    out: List[ModelSpec] = []
    for raw in cache.get("models", []):
        # Strip our _meta field for ModelSpec; tags must be tuple.
        d = {k: v for k, v in raw.items() if k != "_meta"}
        if isinstance(d.get("tags"), list):
            d["tags"] = tuple(d["tags"])
        try:
            out.append(ModelSpec(**d))
        except (TypeError, ValueError):
            # Skip entries from older cache formats.
            continue
    return out
