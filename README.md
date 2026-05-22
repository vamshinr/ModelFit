# ModelFit

**"Which LLM should I run on my computer?"** — answered, in plain English.

`modelfit` is a CLI + GUI for people who have never run a local AI model before, and a sharp tool for people who have. It looks at your computer (CPU, RAM, GPU), checks 206 open-source LLMs against your specs, predicts throughput across major inference engines, and explains every term in plain language.

It ships in two flavours:

| Build | What you get | Distribution |
|---|---|---|
| **Python** (`modelfit`) | Full CLI + REST API + interactive wizard + HuggingFace catalog sync | `pip install -e .` |
| **C++** (`modelfit-cli` + `modelfit-gui`) | Native binary CLI (Windows/macOS/Linux) and Dear ImGui desktop GUI | `cmake --build cpp/build` |

Same model catalog, same scoring math, two front-ends.

---

## Overview

**The problem.** Teams deploying open-weight LLMs on their own hardware face a recurring question: which models will actually run, and which inference engine will run them fastest? Answering that today means cross-referencing param counts, quantization tradeoffs, KV-cache math, and engine-specific kernels — every time hardware or a model release changes.

**The approach.** ModelFit detects the host system (CPU, RAM, GPU/VRAM, memory bandwidth) via platform-native APIs — `sysctlbyname` + IOKit on Apple Silicon, `/proc` and `nvidia-smi` on Linux, DXGI on Windows — then ranks a catalog of 206 curated open-weight models (plus a live HuggingFace sync, ~254 models today) across **quality, speed, memory fit, and context**. For each model it walks the quantization hierarchy (Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K), picks the highest-quality combination that fits within an 8% memory headroom, then computes a composite score weighted by use-case (`chat` prioritizes speed, `reasoning` prioritizes quality, `long-context` prioritizes window size). Throughput is estimated from memory-bandwidth-bound decoding for nine inference engines (vLLM, llama.cpp, TGI, TensorRT-LLM, SGLang, ExLlamaV2, Ollama, MLX, HF Transformers). MoE architectures (Mixtral, DeepSeek V3, Qwen3-MoE) use active rather than total parameters for the speed term.

**Where it fits.**
- **Beginners** — `modelfit wizard` asks two questions, returns one model + one install command.
- **Power users** — `modelfit rank -u code` for full rankings; `modelfit throughput <model>` for engine-by-engine tok/s; `modelfit reverse <model>` for inverse capacity planning ("what hardware do I need for Llama-3.1-70B at 30 tok/s?").
- **Engineering teams** — `modelfit serve` exposes the same scoring as a stdlib REST API for cluster schedulers and CI agents. Provider-agnostic across Ollama, LM Studio, llama.cpp, vLLM, MLX, and TGI.

**Strategic value.** Deterministic, defensible model-selection decisions for on-premise deployments. Prevents wasted downloads of models that exceed hardware constraints, shortens time-to-production for in-house AI infrastructure, and surfaces the 10× throughput delta between inference engines before a deployment commits to one. Works across heterogeneous fleets (Apple Silicon, NVIDIA, AMD, Intel). A 109-entry plain-English glossary keeps the same outputs readable for non-ML stakeholders.

---

## First time? Run this:

```bash
modelfit wizard
```

It asks you two simple questions (what do you want to do? how much text?), then recommends one model and gives you a one-line install command.

That's the whole "getting started" experience.

---

## What it tells you

For every model it considers, you get four things in plain language:

| | Example output |
|---|---|
| **Quality** | ★★★★★ Excellent quality |
| **Speed** | ⚡ Instant (or 🚀 Snappy / ✅ Usable / 🐢 Slow) |
| **Context** | 📚 Whole book (or 📖 Long document / 📝 Long email / ✉️ Short prompt) |
| **Memory** | 2.3 GB used of 4.8 GB available |
| **Download** | ~1 GB on disk |

No GB-per-parameter, no perplexity scores, no jargon you have to look up first.

---

## Install

```bash
git clone <this repo>
cd modelfit
pip install -e .

# Optional: extras
pip install -e '.[tui]'    # interactive textual UI
```

Requires Python 3.9+. Only dependencies are `rich` and `psutil`.

### Debian / Ubuntu (Python 3.11+): use a venv

On modern Debian/Ubuntu, `pip install -e .` against the system Python errors out with **`error: externally-managed-environment`** (PEP 668). The fix is to install into a project-local virtual environment:

```bash
# One-time setup
sudo apt install -y python3-venv python3-full   # if you don't already have it
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Every new shell
source /path/to/ModelFit/.venv/bin/activate
modelfit wizard
```

Prefer not to activate? Call the binary directly: `./.venv/bin/modelfit ...`.

Alternatives, in order of preference:
- **pipx** (`pipx install -e .`) — good for installing the CLI globally in an isolated env, but its editable-install support is limited; skip if you plan to hack on the code.
- **`pip install -e . --break-system-packages`** — works but writes into the OS Python's `site-packages` and can be clobbered by `apt upgrade`. Acceptable in a throwaway VM or container; avoid on anything you care about.

---

## The five commands you'll actually use

### 1. `modelfit wizard` — for total beginners

Interactive Q&A. Asks what you want, recommends one model, hands you the install command. Two minutes from "what is an LLM" to chatting locally.

### 2. `modelfit` — show all the models that fit

```bash
modelfit                 # default: balanced use case
modelfit -u chat         # leaning toward speed for chatting
modelfit -u code         # for writing code
modelfit -u reasoning    # for hard problems
```

Top of the list is what we recommend, with plain-English quality / speed / context labels.

### 3. `modelfit recommend` — give me ONE pick

```bash
modelfit recommend -u code
```

Picks the best balanced choice for your hardware, explains *why* in 3 sentences, and shows you the Ollama / LM Studio / llama.cpp install commands.

### 4. `modelfit get <model>` — how do I install this one?

```bash
modelfit get llama-3.2-3b-instruct
```

Shows the copy-paste install commands for **Ollama**, **HuggingFace + llama.cpp**, and **LM Studio**, plus the download size.

### 5. `modelfit explain <term>` — what does this word mean?

```bash
modelfit explain quantization
modelfit explain context
modelfit explain vram
modelfit explain                              # list all terms
modelfit explain --grouped                    # listing organized by category
modelfit explain --category Training          # only show training-related terms
modelfit explain --search "rl from human"     # full-text search (multi-word OK)
modelfit explain -c list                      # see all categories
```

The glossary covers **109 entries** across 10 categories:

| Category | Examples |
|---|---|
| Basics | LLM, token, context, parameter |
| Hardware | VRAM, RAM, bandwidth, HBM, Tensor Cores, AMX |
| Quantization | Q4_K_M, FP8, AWQ, GPTQ, EXL2, GGUF |
| Training | RLHF, DPO, LoRA, QLoRA, SFT, backprop, perplexity, distillation |
| Architecture | Attention, FlashAttention, PagedAttention, GQA, RoPE, RMSNorm, SwiGLU |
| Inference | Prefill, decode, TTFT, speculative decoding, continuous batching, temperature/top-p |
| Engines & runners | Ollama, llama.cpp, vLLM, TGI, TensorRT-LLM, MLX |
| Applications | Function-calling, tool-use, agents, RAG, embedding models |
| Safety & alignment | Alignment, hallucination, jailbreak |
| Theory | Scaling laws, Chinchilla |

Every entry has a one-line summary, a paragraph of detail, an analogy, and links to related entries (`See also: gqa, kv-cache`). The desktop GUI exposes the same search inline.

---

## What "fits" means

A model "fits" when, after picking a quality level (compression) and a context length (how much text it can read at once), the total memory needed is less than what your computer has.

`modelfit` automatically:

- **Tries the highest quality first**, walks down (`Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K`) only if needed.
- **Tries the full context first**, walks down (100% → 50% → 25%) only if needed.
- **Picks the highest-quality combo that fits.**
- Leaves 8% memory headroom so your machine doesn't choke on the first long prompt.

If you've never heard those terms, no problem — the default output hides them entirely. Add `--expert` to see the full math.

---

## Run a model end-to-end (worked example)

```text
$ modelfit wizard
# … wizard interaction …

🎯 Your pick: Gemma 3 1B
  ✅ Good pick
  Quality:  ★★★★★  Excellent quality
  Speed:    ⚡ Instant
  Context:  📖 Long document (98 pages at a time)
  Memory:   2.3 GB of 4.8 GB available
  Download: ~1 GB on disk

🚀 Easiest install (Ollama):
   ollama run gemma3:1b
```

That's it. Run those two commands and you're chatting with a model in about 3 minutes (mostly download time).

---

## More commands (when you outgrow the basics)

| Command | Purpose |
|---|---|
| `modelfit rank` | Ranked table (the default) |
| `modelfit info <model>` | Detailed fit/score for one model |
| `modelfit inspect` | What hardware did it detect? |
| `modelfit reverse <model>` | What hardware would I need to run X? |
| `modelfit throughput <model>` | Predicted tok/s across vLLM, llama.cpp, TGI, TRT-LLM, … |
| `modelfit engines [id]` | List the inference engines we know about (pros/cons/install) |
| `modelfit catalog sync` | Pull the latest models from HuggingFace and merge them in |
| `modelfit catalog sources` | Show enabled catalog providers + last sync time |
| `modelfit list` | The full 206-model catalog |
| `modelfit tui` | Interactive Textual UI (needs `[tui]` extra) |
| `modelfit serve` | REST API for cluster schedulers |

Every command supports `--json` for scripting.

---

## Inference-engine throughput predictions

Beginners pick **a model**. Advanced users also need to pick **the engine that serves it**, and that decision moves throughput by **10×** or more. `modelfit throughput` predicts tokens/sec for every engine that can serve the model:

```bash
modelfit throughput llama-3.1-8b-instruct
```

| Engine | Single user | At high concurrency | Why |
|---|---|---|---|
| TensorRT-LLM | ~1.6× baseline | up to 10× | Fused FP8 kernels on H100/H200/Blackwell |
| vLLM | ~1.05× | ~8× | PagedAttention + continuous batching |
| SGLang | ~1.10× | ~9× | RadixAttention — fast prefix sharing |
| TGI | ~1.20× | ~6× | FlashAttention v2 + continuous batching |
| ExLlamaV2 | ~1.25× | ~1.5× | Custom CUDA kernels, EXL2 quant; great on RTX |
| llama.cpp | 1.00× baseline | ~1.2× | Everywhere — CPU, NVIDIA, AMD, Apple, Intel |
| Ollama | ~0.98× | ~1.3× | llama.cpp under the hood, easiest install |
| MLX | ~0.92× | ~1.4× | Apple Silicon only |
| HF Transformers | ~0.40× | 1.0× | Reference impl — **don't serve from this** |

For each engine the tool also tells you *why* — "GPU only, NVIDIA/AMD, won't help on your Mac"; "FP8 doesn't unlock on your RTX 4090 — you'd need an H100".

```bash
modelfit engines vllm          # detail on one engine
modelfit engines               # list them all
```

---

## HuggingFace catalog sync

The built-in catalog has 206 hand-curated entries. To stay current with new releases, you can pull from HuggingFace:

```bash
modelfit catalog sync --limit 200 --sort downloads --min-downloads 5000
modelfit catalog sources         # see what's enabled
modelfit catalog clear           # revert to curated only
```

How it works:

- Calls `https://huggingface.co/api/models?pipeline_tag=text-generation`
- For each candidate, downloads `config.json` to get architectural details (`num_hidden_layers`, `num_key_value_heads`, `max_position_embeddings`)
- Derives params from `safetensors.total` (or pattern-matches `7B` / `8x22B` from the name)
- Auto-classifies type (chat/instruct/code/reasoning/math/vision) from tags + name
- Caches results to `~/.ModelFit/catalog_hf.json`
- Merges with the curated baseline — **curated entries always win on duplicates** so manually tuned quality scores are preserved
- Supports `$HF_TOKEN` for higher rate limits / gated models

Synced models flow through every command — `rank`, `recommend`, `throughput`, etc.

Provenance is intentional: see exactly where each entry came from with `modelfit catalog sources`.

---

## C++ build — native CLI + ImGui GUI

The C++ side has two targets:

| Target | What it is |
|---|---|
| `modelfit-cli` | Single-binary CLI (~200 KB). Zero runtime deps. Mirrors `inspect / rank / info / list`. |
| `modelfit-gui` | Dear ImGui desktop app. Tabs for **Models**, **Hardware**, **Glossary** (searchable, categorized). |

Why C++ as well as Python?

- **Distributable as a standalone executable.** No Python required to ship `modelfit-cli.exe` or `modelfit-gui` to teammates.
- **Native hardware APIs.** macOS uses `sysctlbyname` + IOKit; Linux reads `/proc` and shells `nvidia-smi`; Windows enumerates GPUs via DXGI directly.
- **Cross-checks the Python.** The C++ build links a `catalog_data.h` generated from `modelfit/models.py` — anything we add to the Python catalog flows into the C++ build automatically. The test suite includes a parity check that confirms both produce the same top pick for the same hardware + use case.

### Build it

```bash
# CLI only — zero-deps native build
cd cpp && mkdir -p build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja modelfit-cli
./modelfit-cli rank --top 5

# CLI + GUI (fetches Dear ImGui + GLFW via CMake on first config)
cmake -GNinja -DMODELFIT_GUI=ON ..
ninja
./modelfit-gui
```

The GUI looks like:

```
┌─────────────────────────────────────────────────────────────┐
│ modelfit — 206 open-weight LLMs ranked for your machine     │
│ [ Models | Hardware | Glossary ]            [Refresh hw]    │
├─────────────────────────────────────────────────────────────┤
│  Optimise for: [balanced ▾]   Min ctx: [4K ▾]   Filter:____ │
│  ┌─ Ranking ──────────┐  ┌─ Detail ─────────────────────┐   │
│  │ 1  75.4  Qwen2.5…  │  │ Qwen2.5 0.5B Instruct        │   │
│  │ 2  74.8  Qwen2…    │  │ Family: Qwen / instruct       │   │
│  │ 3  72.2  DeepSeek… │  │ Quality ▓▓▓▓▓▓▓▓▓▓▓░░░░░     │   │
│  │ 4  72.2  Qwen3 0.6B│  │ Speed   ▓▓▓▓▓▓▓▓▓░░░░░░░     │   │
│  │ …                   │  │ Context ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     │   │
│  └────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Tested compiling on macOS (Apple clang). The CMake config also handles Linux (X11/OpenGL) and Windows (DXGI/OpenGL).

### Cross-compile / ship binaries

`modelfit-cli` is a single statically-linked binary — drop it into a Docker image, a CI runner, or `scp` it to a server with no setup.

```bash
# Linux from a Linux machine
cmake -DCMAKE_BUILD_TYPE=Release .. && ninja modelfit-cli
strip modelfit-cli                 # ~200 KB

# Windows (cross-compile via MinGW, or build natively in MSVC)
cmake -G "Visual Studio 17 2022" -A x64 .. && cmake --build . --config Release
# → Release/modelfit-cli.exe
```

### Regenerate the catalog header

Whenever `modelfit/models.py` changes:

```bash
python3 cpp/scripts/gen_catalog.py > cpp/include/modelfit/catalog_data.h
python3 cpp/scripts/gen_glossary.py > cpp/src/glossary_data.cpp
ninja                              # rebuild
```

Or use the CMake target: `ninja gen_catalog`.

---

## "Will this run on my laptop?" — examples

```bash
# I have a MacBook Air (16 GB unified memory)
modelfit
# → Picks: Llama 3.2 3B Instruct, Qwen2.5 7B, Phi-3.5 Mini, …

# I have an RTX 4090 (24 GB VRAM)
# (run on that machine)
modelfit -u reasoning
# → Picks: QwQ 32B, DeepSeek R1 Distill 32B, Qwen2.5 32B, …

# I'm on a Raspberry Pi / old laptop (no GPU, 8 GB RAM)
modelfit --min-context 2048
# → Picks: SmolLM2 1.7B, Llama 3.2 1B, TinyLlama, …
```

---

## REST API

For cluster schedulers, CI agents, or building your own UI:

```bash
modelfit serve --port 8765
```

```text
GET  /healthz
GET  /hardware         (this node's specs)
GET  /models
GET  /models/{id}
POST /rank             (rank for a custom hardware spec)
POST /reverse          (compute required hardware)
```

CORS-enabled, no extra deps (uses stdlib `http.server`).

---

## Website (local)

The repo ships a small static site under `website/` — a landing page at `/` and an interactive demo at `/demo/` that calls the REST API live. To run it locally:

```bash
# Terminal 1 — the API
source .venv/bin/activate
modelfit serve --port 8765

# Terminal 2 — the static site
cd website && python3 -m http.server 8080
```

Then open <http://127.0.0.1:8080/>. The demo page reads `/hardware` and `POSTs /rank` against the API on `:8765`; CORS is open by default so any origin works.

No build step — plain HTML/CSS/JS, single shared stylesheet. To deploy as a static site (GitHub Pages, Netlify, etc.), point your host at the `website/` directory; the demo will still call whatever API endpoint the user types into the "API endpoint" field.

---

## Reverse mode: "What hardware would I need?"

```bash
modelfit reverse llama-3.1-70b-instruct --target-tps 30 --context 8192
```

→ Tells you: 48 GB of VRAM, ~1800 GB/s bandwidth, and lists the GPUs that clear the bar (A100, H100, H200, MI300X, M3 Max, …).

---

## How accurate is the speed estimate?

It's an **order-of-magnitude** estimate based on memory bandwidth and active weights — close enough to pick *between* models, not close enough to use for capacity planning. Run [explain bandwidth](#) to see the math.

---

## Run the tests

```bash
python -m unittest discover tests -v
```

---

## License

MIT.
