"""Map an internal model id to ready-to-paste install/run commands for the
popular local-LLM runners. This is what beginners actually need: a copy-paste
line that produces a working chatbot in 60 seconds.

We aim for *real* tags / repo names where we can. If a model isn't on a
particular runner, we say so plainly rather than guessing wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modelfit.models import ModelSpec, QUANT_BYTES_PER_PARAM
from modelfit.beginner import download_size_gb, download_label


# Curated mapping: model-id → Ollama library tag
# (kept short on purpose — we map the well-known ones and gracefully
# tell beginners to search the catalog otherwise)
OLLAMA_TAGS = {
    "llama-3.1-8b-instruct":    "llama3.1:8b",
    "llama-3.1-70b-instruct":   "llama3.1:70b",
    "llama-3.2-1b-instruct":    "llama3.2:1b",
    "llama-3.2-3b-instruct":    "llama3.2:3b",
    "llama-3.3-70b-instruct":   "llama3.3:70b",
    "llama-3-8b-instruct":      "llama3:8b",
    "llama-3-70b-instruct":     "llama3:70b",
    "llama-2-7b-chat":          "llama2:7b",
    "llama-2-13b-chat":         "llama2:13b",
    "llama-2-70b-chat":         "llama2:70b",
    "mistral-7b-instruct-v0.3": "mistral:7b",
    "mistral-7b-instruct-v0.2": "mistral:7b-instruct-v0.2-q4_K_M",
    "mistral-nemo-12b-instruct":"mistral-nemo:12b",
    "mistral-small-3-24b":      "mistral-small:24b",
    "mistral-large-123b":       "mistral-large:123b",
    "mixtral-8x7b-instruct":    "mixtral:8x7b",
    "mixtral-8x22b-instruct":   "mixtral:8x22b",
    "codestral-22b":            "codestral:22b",
    "gemma-2-2b-it":            "gemma2:2b",
    "gemma-2-9b-it":            "gemma2:9b",
    "gemma-2-27b-it":           "gemma2:27b",
    "gemma-3-1b":               "gemma3:1b",
    "gemma-3-4b":               "gemma3:4b",
    "gemma-3-12b":              "gemma3:12b",
    "gemma-3-27b":              "gemma3:27b",
    "qwen2.5-0.5b-instruct":    "qwen2.5:0.5b",
    "qwen2.5-1.5b-instruct":    "qwen2.5:1.5b",
    "qwen2.5-3b-instruct":      "qwen2.5:3b",
    "qwen2.5-7b-instruct":      "qwen2.5:7b",
    "qwen2.5-14b-instruct":     "qwen2.5:14b",
    "qwen2.5-32b-instruct":     "qwen2.5:32b",
    "qwen2.5-72b-instruct":     "qwen2.5:72b",
    "qwen2.5-coder-0.5b-instruct": "qwen2.5-coder:0.5b",
    "qwen2.5-coder-1.5b-instruct": "qwen2.5-coder:1.5b",
    "qwen2.5-coder-3b-instruct":   "qwen2.5-coder:3b",
    "qwen2.5-coder-7b-instruct":   "qwen2.5-coder:7b",
    "qwen2.5-coder-14b-instruct":  "qwen2.5-coder:14b",
    "qwen2.5-coder-32b-instruct":  "qwen2.5-coder:32b",
    "qwen3-0.6b":               "qwen3:0.6b",
    "qwen3-1.7b":               "qwen3:1.7b",
    "qwen3-4b":                 "qwen3:4b",
    "qwen3-8b":                 "qwen3:8b",
    "qwen3-14b":                "qwen3:14b",
    "qwen3-32b":                "qwen3:32b",
    "qwq-32b":                  "qwq:32b",
    "phi-3-mini-4k":            "phi3:mini",
    "phi-3-medium-14b-128k":    "phi3:medium",
    "phi-3.5-mini-instruct":    "phi3.5:3.8b",
    "phi-4":                    "phi4:14b",
    "phi-4-mini":               "phi4-mini:3.8b",
    "deepseek-coder-1.3b":      "deepseek-coder:1.3b",
    "deepseek-coder-6.7b":      "deepseek-coder:6.7b",
    "deepseek-coder-33b":       "deepseek-coder:33b",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1:1.5b",
    "deepseek-r1-distill-qwen-7b":   "deepseek-r1:7b",
    "deepseek-r1-distill-llama-8b":  "deepseek-r1:8b",
    "deepseek-r1-distill-qwen-14b":  "deepseek-r1:14b",
    "deepseek-r1-distill-qwen-32b":  "deepseek-r1:32b",
    "deepseek-r1-distill-llama-70b": "deepseek-r1:70b",
    "tinyllama-1.1b":           "tinyllama:1.1b",
    "command-r-35b":            "command-r:35b",
    "command-r-plus-104b":      "command-r-plus:104b",
    "command-r7b":              "command-r7b:7b",
    "starcoder2-3b":            "starcoder2:3b",
    "starcoder2-7b":            "starcoder2:7b",
    "starcoder2-15b":           "starcoder2:15b",
    "granite-3.1-2b-instruct":  "granite3.1-dense:2b",
    "granite-3.1-8b-instruct":  "granite3.1-dense:8b",
    "smollm2-135m-instruct":    "smollm2:135m",
    "smollm2-360m-instruct":    "smollm2:360m",
    "smollm2-1.7b-instruct":    "smollm2:1.7b",
    "nemotron-mini-4b":         "nemotron-mini:4b",
    "glm-4-9b-chat":            "glm4:9b",
}

# HuggingFace GGUF repos (community quants — naming follows the convention
# "bartowski/<model>-GGUF" or "TheBloke/<model>-GGUF" for older models).
HF_GGUF_REPOS = {
    "llama-3.1-8b-instruct":   "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "llama-3.1-70b-instruct":  "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
    "llama-3.2-3b-instruct":   "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "llama-3.3-70b-instruct":  "bartowski/Llama-3.3-70B-Instruct-GGUF",
    "mistral-7b-instruct-v0.3":"bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "mistral-nemo-12b-instruct":"bartowski/Mistral-Nemo-Instruct-2407-GGUF",
    "mixtral-8x7b-instruct":   "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    "gemma-2-9b-it":           "bartowski/gemma-2-9b-it-GGUF",
    "gemma-2-27b-it":          "bartowski/gemma-2-27b-it-GGUF",
    "qwen2.5-7b-instruct":     "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5-14b-instruct":    "bartowski/Qwen2.5-14B-Instruct-GGUF",
    "qwen2.5-32b-instruct":    "bartowski/Qwen2.5-32B-Instruct-GGUF",
    "qwen2.5-72b-instruct":    "bartowski/Qwen2.5-72B-Instruct-GGUF",
    "qwen2.5-coder-7b-instruct":  "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwen2.5-coder-32b-instruct": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
    "qwq-32b":                 "bartowski/QwQ-32B-GGUF",
    "phi-4":                   "bartowski/phi-4-GGUF",
    "deepseek-r1-distill-qwen-14b": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    "deepseek-r1-distill-qwen-32b": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
    "deepseek-r1-distill-llama-70b":"bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
    "deepseek-coder-v2-lite-16b":   "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
}


@dataclass
class RunnerCommands:
    model_id: str
    quant: str
    download_gb: float
    download_label: str
    ollama: Optional[str] = None
    lmstudio: Optional[str] = None
    llamacpp: Optional[str] = None
    huggingface_repo: Optional[str] = None
    huggingface_file: Optional[str] = None
    notes: list = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "quant": self.quant,
            "download_gb": round(self.download_gb, 2),
            "download_label": self.download_label,
            "ollama": self.ollama,
            "lmstudio": self.lmstudio,
            "llamacpp": self.llamacpp,
            "huggingface_repo": self.huggingface_repo,
            "huggingface_file": self.huggingface_file,
            "notes": list(self.notes or []),
        }


def _gguf_filename_for(model: ModelSpec, quant: str) -> str:
    """Best-effort filename inside a HF GGUF repo. Real filenames vary."""
    # bartowski convention: "<Model-Name>-<quant>.gguf" with underscores.
    family_slug = model.name.replace(" ", "-").replace("/", "-")
    return f"{family_slug}-{quant}.gguf"


def get_commands(model: ModelSpec, quant: str = "Q4_K_M") -> RunnerCommands:
    """Build install/run commands a beginner can copy-paste."""
    if quant not in QUANT_BYTES_PER_PARAM:
        quant = "Q4_K_M"
    download_gb = download_size_gb(model, quant)
    rc = RunnerCommands(
        model_id=model.id,
        quant=quant,
        download_gb=download_gb,
        download_label=download_label(download_gb),
        notes=[],
    )

    # ---- Ollama (easiest path) ----
    if model.id in OLLAMA_TAGS:
        # Ollama's default tag uses Q4_0/Q4_K_M; if user picked something else,
        # surface the alternative-tag pattern.
        base_tag = OLLAMA_TAGS[model.id]
        if quant in ("Q4_K_M", "Q4_0"):
            rc.ollama = f"ollama run {base_tag}"
        else:
            base = base_tag.split(":")[0]
            size = base_tag.split(":")[1] if ":" in base_tag else ""
            rc.ollama = f"ollama run {base}:{size}-instruct-{quant.lower().replace('_','-')}"
            rc.notes.append("If the suffixed tag isn't found, browse: https://ollama.com/library/" + base)
    else:
        rc.notes.append("Not in the Ollama library — try the HuggingFace / llama.cpp route below.")

    # ---- LM Studio (search UI; we give a search command) ----
    rc.lmstudio = f"# In LM Studio, click 'Discover' and search for:\n#    {model.name}\n# Then download the file labeled '{quant}'."

    # ---- HuggingFace GGUF (download from a known repo) ----
    if model.id in HF_GGUF_REPOS:
        rc.huggingface_repo = HF_GGUF_REPOS[model.id]
        rc.huggingface_file = _gguf_filename_for(model, quant)
        rc.llamacpp = (
            f"# 1. Download the GGUF weights:\n"
            f"huggingface-cli download {rc.huggingface_repo} {rc.huggingface_file} --local-dir ./models\n"
            f"\n# 2. Run the chat server (after building llama.cpp):\n"
            f"./llama-server -m ./models/{rc.huggingface_file} -c {min(model.context_max, 8192)}"
        )
    else:
        rc.notes.append(
            "We don't have a verified HuggingFace GGUF repo for this one. "
            "Search huggingface.co for the model name + 'GGUF'."
        )

    # Memory-warning notes for absolute beginners
    if download_gb > 60:
        rc.notes.insert(0, f"⚠️ Big download (~{download_label(download_gb)}). "
                            "Make sure you have the disk space.")
    elif download_gb > 20:
        rc.notes.insert(0, f"Download is about {download_label(download_gb)}.")

    return rc
