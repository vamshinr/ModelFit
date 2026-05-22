"""Plain-language explanations for every term a beginner runs into.

Used by the `modelfit explain` subcommand. Each entry is structured so we
can show a one-line summary, a longer paragraph, and an analogy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Entry:
    term: str
    summary: str
    body: str
    analogy: str = ""
    see_also: tuple = ()


GLOSSARY: Dict[str, Entry] = {}


def _add(term, summary, body, analogy="", see_also=()):
    # Indexed under multiple aliases so search works.
    e = Entry(term=term, summary=summary, body=body.strip(),
              analogy=analogy.strip(), see_also=see_also)
    GLOSSARY[term.lower()] = e
    return e


_add("LLM",
    "Large Language Model — the kind of AI that powers chatbots.",
    """\
A neural network trained on huge amounts of text. You type something,
it predicts the next words. With enough training, this becomes a
conversational assistant that can write, summarise, code, etc.""",
    "Like autocomplete on your phone, scaled up about a billion times.",
)

_add("token",
    "The chunks of text an LLM reads and writes. Roughly 0.75 words each.",
    """\
LLMs don't see letters or whole words — they see tokens. \"Hello, world!\"
might be 3 tokens. Useful rule of thumb: 1,000 tokens ≈ 750 English words.""",
    "Like Lego bricks of language — most words are 1–2 bricks each.",
    see_also=("context", "tok/s"),
)

_add("tok/s", "Tokens per second — how fast the model generates text.",
    """\
The headline speed metric. Above ~20 tok/s feels instant for chat.
Below ~5 tok/s feels frustrating. This depends on your hardware
(mainly memory bandwidth) and how big the model is.""",
    "Think of it as 'words per second × 0.75'.",
    see_also=("token", "bandwidth"),
)

_add("context",
    "How much text the model can 'see' at once (input + output combined).",
    """\
Measured in tokens. A 4K context fits a long email; 32K fits a chapter;
128K fits a whole novel. Bigger context = more memory needed (the
KV cache grows linearly with context).""",
    "Like the size of the model's short-term memory.",
    see_also=("token", "kv-cache"),
)

_add("vram", "Video RAM — memory built into your GPU. Where LLMs run fastest.",
    """\
GPUs have their own memory, separate from system RAM. To run an LLM
fast, the whole model needs to fit in VRAM. A 24GB RTX 4090 can fit a
~30B parameter model at Q4 quantization. Bigger models need bigger
(or multiple) GPUs.""",
    "RAM is your kitchen counter; VRAM is the chef's smaller side counter — much faster, but smaller.",
    see_also=("gpu", "quantization", "ram"),
)

_add("ram", "Your computer's main memory. Used for CPU-only inference.",
    """\
If you don't have a GPU (or the model is too big for it), the model
runs in regular RAM and uses the CPU instead. Much slower than GPU,
but works for smaller models.""",
    "Roomy but slow — like cooking with everything spread out across a big counter.",
    see_also=("vram", "bandwidth"),
)

_add("bandwidth",
    "How fast memory can be read. The #1 thing that decides LLM speed.",
    """\
Each token the model generates, it has to read all the active weights
from memory. So tokens/sec ≈ memory bandwidth ÷ model size in memory.
GPUs have 5-20x the bandwidth of system RAM, which is why they're so
much faster for LLMs.""",
    "Bandwidth is the size of the pipe between the model and the math units.",
    see_also=("vram", "tok/s"),
)

_add("quantization",
    "Compressing the model's numbers to use less memory.",
    """\
Models are stored as billions of numbers (weights). Originally each
number takes 16 or 32 bits. Quantization reduces this to 4-8 bits per
number, shrinking the file 2-8x. There's a small quality drop, but
the practical impact is usually tiny — and it lets you run much bigger
models on consumer hardware.""",
    "Like JPEG compression for the model. Q8 is high-quality JPEG; Q2 is the heavily compressed thumbnail.",
    see_also=("q4_k_m", "gguf"),
)

_add("q4_k_m",
    "The 'default' quantization level — strong quality, half the size of Q8.",
    """\
Most people running local LLMs use Q4_K_M. It uses about 4.6 bits per
weight (so a 7B model is ~4.1 GB on disk), keeps ~96.5% of the model's
quality, and runs on consumer GPUs / Apple Silicon comfortably.""",
    "The sweet spot — like 'high quality' on a JPEG slider.",
    see_also=("quantization",),
)

_add("q8_0",
    "Nearly lossless quantization. Use this when you have memory to spare.",
    """\
About 1 byte per weight (8 bits + a tiny overhead). Quality is
indistinguishable from the original for most purposes. Use it if the
model fits.""",
    see_also=("quantization", "q4_k_m"),
)

_add("q2_k",
    "Heavy compression — last resort when nothing else fits.",
    """\
About 2.5 bits per weight. Noticeable quality drop — answers may be
worse or weirder. Better than not running the model at all, but if
you can afford Q3 or Q4, do that instead.""",
    see_also=("quantization",),
)

_add("gguf",
    "The standard file format for quantized LLMs (llama.cpp / Ollama / LM Studio).",
    """\
GGUF stands for 'GGML Universal Format'. It's a single .gguf file that
holds the model weights and metadata, compressed at one of the quant
levels. Most tools that run LLMs locally use GGUF files.""",
    "Like .mp4 for video — the agreed-upon container.",
    see_also=("quantization",),
)

_add("kv-cache",
    "Memory used to remember the conversation so far. Grows with context.",
    """\
As the model reads tokens, it caches intermediate values (the 'Key' and
'Value' tensors) so it doesn't have to re-compute them. This cache
grows roughly linearly with context length. At 128K context, a 70B
model's KV cache can be 20+ GB on its own.""",
    "Like the model's working memory — refilled every new conversation, but expensive while it's there.",
    see_also=("context", "vram"),
)

_add("moe",
    "Mixture of Experts — only part of the model runs per token.",
    """\
Some big models (Mixtral, DeepSeek V3) have hundreds of billions of
parameters total, but only activate ~10-20% per token. They're as
fast as a 14B-30B dense model, but as smart as a much bigger one. The
cost: you still have to fit the *whole* thing in memory.""",
    "Like a panel of specialists — only the 2 relevant ones are asked, but they all need offices.",
    see_also=("vram",),
)

_add("parameter",
    "One of the billions of numbers that make up a model's 'brain'.",
    """\
A '7B model' has 7 billion parameters. Each parameter is one number
the model learned during training. Bigger models (more parameters)
are usually smarter but slower and need more memory.""",
    see_also=("quantization",),
)

_add("instruct",
    "Variants tuned to follow instructions, not just continue text.",
    """\
'Base' models just predict the next token. 'Instruct' (or 'chat')
variants are further trained so that when you say 'Summarise this',
they actually summarise rather than rambling. Always pick an
Instruct / Chat variant for everyday use.""",
    see_also=("chat",),
)

_add("chat", "Same idea as Instruct — tuned for back-and-forth conversation.",
    """\
Often labeled 'Chat' instead of 'Instruct'. Use whichever the model
authors picked — they mean essentially the same thing.""",
    see_also=("instruct",),
)

_add("reasoning",
    "Models that show their thinking before answering. Better on hard problems.",
    """\
Examples: DeepSeek R1, QwQ. These models generate a long internal
'thinking' trace before their final answer. They're great for math
and logic, but slower per question because they generate more tokens.""",
    see_also=("instruct",),
)

_add("ollama",
    "The easiest way to run LLMs locally. One command and you're chatting.",
    """\
Install Ollama from ollama.com. Then in a terminal:
   ollama run llama3.2:3b
…downloads and starts a chat. Use `modelfit get <model>` to find the
right Ollama tag for any model in our catalog.""",
    see_also=("gguf",),
)

_add("llama.cpp",
    "The C++ library that started the local-LLM revolution. Powers most tools.",
    """\
llama.cpp is the engine inside Ollama, LM Studio, and many others.
You can also use it directly: build it, download a .gguf file, and
run `./llama-server -m model.gguf`.""",
    see_also=("gguf", "ollama"),
)

_add("lmstudio",
    "Friendly desktop app with a built-in model browser. Good for beginners.",
    """\
Download from lmstudio.ai. It has a search UI to find GGUF models,
and a chat interface. Pairs well with `modelfit` — find a model here,
then search for it in LM Studio's 'Discover' tab.""",
    see_also=("gguf",),
)

_add("gpu", "Graphics card. The fast hardware for running LLMs.",
    """\
NVIDIA (CUDA) is best supported, AMD (ROCm) and Apple Silicon (Metal)
also work. Look for VRAM size — that's the constraint, not raw speed.
24 GB VRAM is a great target for serious local LLM use.""",
    see_also=("vram", "bandwidth"),
)

_add("apple silicon",
    "Mac M-series chips. The GPU shares system RAM ('unified memory').",
    """\
On M1/M2/M3/M4 Macs, there's no separate VRAM — the GPU can use most
of system RAM. So a 64 GB M2 Max can run models that need 48+ GB of
'VRAM'. Bandwidth is the only catch (slower than top NVIDIA cards),
which limits tok/s for big models.""",
    see_also=("vram",),
)

_add("inference",
    "Running a trained model to generate answers. (As opposed to training it.)",
    """\
'Inference' = using the model. 'Training' = building the model.
This tool only cares about inference — picking a model that runs
on your machine and generates text.""",
)

_add("hugging face", "Where the open-source AI world hosts its models.",
    """\
huggingface.co — the GitHub of AI. Almost every open-weight model
lives here. Search for the model name + 'GGUF' to find quantized
versions you can run locally.""",
    see_also=("gguf",),
)


# ===========================================================================
# Training-side terms
# ===========================================================================
_add("pretraining",
    "The first, biggest training pass — teaches the model the structure of language.",
    """\
Done on trillions of tokens of unlabeled text (the internet, books, code).
The model just learns to predict the next token. After pretraining, you
have a 'base' model — knowledgeable but not yet a chatbot.""",
    "Like a kid reading every book in the library — they know words and how sentences work, but no one's told them how to be helpful yet.",
    see_also=("finetuning", "base"),
)

_add("finetuning", "Adapting a pretrained model to a specific task or style.",
    """\
You start from a pretrained model and train it further on a smaller,
focused dataset. Often that means instruction data ('here's a task, here's
a good answer'). Cheaper than pretraining — hours/days vs. months.""",
    see_also=("pretraining", "lora", "sft", "rlhf"),
)

_add("sft",
    "Supervised Fine-Tuning — teaching a base model to follow instructions.",
    """\
You show the model thousands of (prompt → ideal response) pairs and
update its weights to imitate them. SFT is what turns a base model into
an 'instruct' / 'chat' model.""",
    see_also=("finetuning", "rlhf", "dpo"),
)

_add("rlhf",
    "Reinforcement Learning from Human Feedback — the technique behind ChatGPT.",
    """\
After SFT, humans rank multiple model responses. A separate 'reward
model' learns from those rankings, and the LLM is then trained to
produce responses the reward model prefers. Hugely impactful on
helpfulness, honesty, and safety.""",
    "Like training a chef: SFT is showing them recipes; RLHF is having diners rate dishes and tweaking based on their preferences.",
    see_also=("sft", "ppo", "dpo"),
)

_add("ppo",
    "Proximal Policy Optimization — the RL algorithm classic RLHF uses.",
    """\
A specific kind of policy-gradient method. Used inside RLHF to update
the LLM so its outputs score higher on the reward model. Stable but
complex — needs a critic, a reward model, and the policy model
running simultaneously.""",
    see_also=("rlhf", "dpo"),
)

_add("dpo",
    "Direct Preference Optimization — a simpler alternative to RLHF/PPO.",
    """\
Skips the reward model entirely. You give the trainer (chosen, rejected)
pairs and it pushes the model toward chosen and away from rejected in
one closed-form update. Easier to implement, often competitive with
PPO-based RLHF.""",
    see_also=("rlhf", "ppo"),
)

_add("lora",
    "Low-Rank Adaptation — fine-tune with ~1% of the parameters.",
    """\
Instead of updating all 8 billion weights, you add small low-rank
matrices alongside the existing weights and train only those. Result:
fine-tuned 'adapter' is a few MB, can be swapped in and out at runtime,
and you can train it on a single consumer GPU.""",
    "Like adding a lens filter rather than repainting the photo.",
    see_also=("qlora", "finetuning"),
)

_add("qlora", "LoRA on top of a quantized base model — fine-tunes 70B on a single GPU.",
    """\
Combines LoRA with 4-bit quantization of the frozen base model. The
adapter weights stay full precision. Lets you fine-tune very large
models on consumer hardware.""",
    see_also=("lora", "quantization"),
)

_add("adapter", "Small fine-tuned add-on that can be plugged into a base model.",
    """\
LoRA adapters are the most common kind. You can have dozens of adapters
trained for different tasks and load whichever you need on top of the
same base model at runtime.""",
    see_also=("lora", "qlora"),
)

_add("backpropagation",
    "How neural networks learn — propagating error gradients backwards through the layers.",
    """\
After a forward pass, the loss (how wrong the prediction was) is
computed. Backprop computes how much each weight contributed to the
loss, and the optimiser uses those gradients to nudge weights in the
right direction.""",
    see_also=("gradient-descent", "optimizer"),
)

_add("gradient-descent",
    "The fundamental algorithm that trains neural networks.",
    """\
Each step: compute the loss, compute its gradient w.r.t. the weights,
take a tiny step opposite the gradient. Repeat billions of times.
Variants like AdamW dominate LLM training.""",
    see_also=("backpropagation", "optimizer", "learning-rate"),
)

_add("optimizer",
    "The algorithm that decides how much to change each weight per training step.",
    """\
Adam and AdamW are the standard for LLMs — they adapt the learning
rate per weight using running averages of gradients. Cheaper
optimisers exist (SGD) but converge slower.""",
    see_also=("gradient-descent", "learning-rate"),
)

_add("learning-rate",
    "How big each training step is. The most important hyperparameter.",
    """\
Too high → training blows up. Too low → training takes forever or gets
stuck. Typical LLM pretraining uses ~1e-4 with a warmup-then-decay
schedule.""",
    see_also=("optimizer",),
)

_add("loss",
    "How wrong the model's prediction was — the number we minimise.",
    """\
For LLMs, the loss is cross-entropy on next-token prediction. Lower
loss = the model assigned higher probability to the right tokens.""",
    see_also=("perplexity",),
)

_add("perplexity",
    "How surprised the model is by a piece of text. Lower is better.",
    """\
Exponential of the loss. Used to compare models on a held-out test set.
A perplexity of 10 means the model was effectively choosing between 10
equally likely tokens at each step.""",
    see_also=("loss",),
)

_add("epoch", "One full pass through the training dataset.",
    """\
For LLM pretraining we don't even finish one epoch — the dataset is so
big that one pass = months of compute. For fine-tuning, 1-3 epochs is
typical.""",
)

_add("batch", "How many examples the model trains on at once.",
    """\
Bigger batches = more stable gradients but more memory. LLM
pretraining uses huge effective batch sizes (millions of tokens) via
gradient accumulation across many GPUs.""",
    see_also=("gradient-accumulation",),
)

_add("gradient-accumulation",
    "Trick to simulate a big batch on small hardware.",
    """\
Process several mini-batches, sum their gradients, then take one
optimizer step. Same effect as a bigger batch but without the memory
spike.""",
    see_also=("batch",),
)

_add("tokenizer", "Code that splits text into tokens (and back).",
    """\
Most LLMs use BPE (byte-pair encoding) or SentencePiece tokenizers.
The choice matters: tokenizers tuned on code vs. multilingual text
compress differently.""",
    see_also=("token", "bpe"),
)

_add("bpe",
    "Byte-Pair Encoding — the most common tokenization algorithm.",
    """\
Starts from bytes, repeatedly merges the most frequent adjacent pair
into a new token, until you hit a vocabulary size. Result: common
words become single tokens, rare ones split into pieces.""",
    see_also=("tokenizer",),
)

_add("vocabulary",
    "The set of tokens the model knows. Usually ~32K to 200K entries.",
    """\
A bigger vocabulary means shorter token sequences (each token covers
more text) but a bigger embedding matrix. Llama 3 uses 128K, Gemma 2
uses 256K.""",
    see_also=("token", "tokenizer"),
)

_add("dataset",
    "What the model trains on. The size and quality determine almost everything.",
    """\
LLMs are trained on trillions of tokens. Famous corpora: The Pile,
RedPajama, RefinedWeb, FineWeb, Dolma. Pretraining dataset choices
(filtering, deduplication, mix ratios) are an active research area.""",
)

_add("synthetic-data",
    "Training data generated by another model rather than scraped from humans.",
    """\
Increasingly important. Models like Phi-4 are trained mostly on
carefully filtered synthetic textbooks. Tradeoff: cleaner / more
diverse vs. risk of distillation artefacts.""",
)

_add("distillation",
    "Training a small 'student' model to imitate a big 'teacher' model.",
    """\
You generate outputs from the big model and train the small one on
them. DeepSeek-R1-Distill-* are made this way: a Qwen/Llama model
fine-tuned on R1's outputs to inherit its reasoning ability.""",
    see_also=("synthetic-data",),
)

# ===========================================================================
# Architecture
# ===========================================================================
_add("transformer", "The neural network architecture every modern LLM uses.",
    """\
Introduced in 'Attention Is All You Need' (2017). Uses self-attention
to weigh how much each token relates to every other token. The 'GPT' in
ChatGPT stands for Generative Pretrained Transformer.""",
    see_also=("attention", "self-attention"),
)

_add("attention",
    "The mechanism that lets each token look at every other token.",
    """\
At every layer, each token computes a weighted blend of all preceding
tokens. The weights ('attention scores') come from comparing each
token's query against every other token's key. Quadratic cost in
context length — that's the bottleneck behind FlashAttention etc.""",
    see_also=("self-attention", "flash-attention", "kv-cache"),
)

_add("self-attention", "Attention applied within a single sequence.",
    """\
In an LLM, each token attends to all previous tokens in the same
prompt+response. That's how the model maintains coherence.""",
    see_also=("attention",),
)

_add("flash-attention",
    "Fast & memory-efficient attention kernel. Standard in every serious engine.",
    """\
Reorders the attention computation to avoid materialising the huge
attention matrix in HBM. ~3-10× faster and uses much less memory than
the naive implementation. FlashAttention v3 is current state-of-the-art.""",
    see_also=("attention", "pagedattention"),
)

_add("pagedattention",
    "vLLM's KV-cache scheme. Treats the cache like an OS page table.",
    """\
Instead of allocating a contiguous KV-cache block per request, PagedAttention
splits the cache into fixed-size blocks and lets requests share blocks.
Dramatically reduces memory fragmentation, which is what enables vLLM's
high aggregate throughput.""",
    see_also=("kv-cache", "vllm"),
)

_add("gqa",
    "Grouped-Query Attention — fewer KV heads than query heads. Shrinks KV cache.",
    """\
Standard ('multi-head') attention has matching numbers of query, key,
and value heads. GQA gives multiple query heads a single shared
key/value pair. Llama 3, Mistral, Qwen all use GQA. Result: KV cache
is ~4-8× smaller, longer contexts become tractable.""",
    see_also=("kv-cache", "mqa"),
)

_add("mqa",
    "Multi-Query Attention — extreme version of GQA with a single shared KV head.",
    """\
Saves the most memory but loses some quality vs GQA-8. Used in PaLM-2
and a few others.""",
    see_also=("gqa",),
)

_add("embedding",
    "The vector representation of a token (or any input).",
    """\
The first thing an LLM does is look up an embedding for each token —
a dense vector (e.g., 4096 dimensions). Those vectors carry the
information forward through the layers.""",
    see_also=("token",),
)

_add("positional-encoding",
    "How the model knows which token came first, second, third…",
    """\
Attention is order-agnostic by default — you have to inject position
info. Modern LLMs use RoPE (Rotary Position Embeddings), which rotates
embeddings by an angle based on their position. ALiBi is another
common scheme.""",
    see_also=("rope",),
)

_add("rope",
    "Rotary Position Embeddings — what most modern LLMs use for position info.",
    """\
Encodes position by rotating Q and K vectors by a position-dependent
angle. Plays nicely with attention math and extends to long contexts
when scaled (YaRN, NTK-aware scaling).""",
    see_also=("positional-encoding", "yarn"),
)

_add("yarn",
    "YaRN — a technique to extend a model's usable context length without retraining.",
    """\
Scales the RoPE frequencies in a particular way. Lets you push a model
trained at 4K to 128K with minimal quality loss. Used by Qwen, Yi,
DeepSeek long-context variants.""",
    see_also=("rope",),
)

_add("normalization",
    "Keeping activations in a sensible range so training doesn't blow up.",
    """\
LLMs use LayerNorm or its newer cousin RMSNorm at each transformer
block. Without it, training is unstable.""",
    see_also=("rmsnorm",),
)

_add("rmsnorm",
    "Root-Mean-Square LayerNorm. The standard normalization for modern LLMs.",
    """\
Slightly cheaper than LayerNorm because it skips the mean subtraction.
Used by Llama, Mistral, Qwen, Gemma.""",
    see_also=("normalization",),
)

_add("activation",
    "The non-linear function that lets neural networks model complex relationships.",
    """\
Without one, stacking matrices would just be one big matrix multiply.
Modern LLMs mostly use SwiGLU or GeGLU activations in their MLP
blocks.""",
    see_also=("swiglu",),
)

_add("swiglu",
    "SwiGLU — the activation function inside most modern LLM MLPs.",
    """\
A gated variant of SiLU. Empirically a bit better than plain GeLU for
language modelling. Used by Llama, Mistral, Gemma.""",
    see_also=("activation",),
)


# ===========================================================================
# Inference internals
# ===========================================================================
_add("prefill",
    "Processing the user's prompt. Fast — done in parallel.",
    """\
When you send a prompt, the model first runs through all of it at
once ('prefill'), filling the KV cache. This part is compute-bound
and can hit teraflops. Then comes decode — token by token, much
slower.""",
    see_also=("decode", "kv-cache", "ttft"),
)

_add("decode",
    "Generating the response token by token. Memory-bandwidth bound.",
    """\
After prefill, each generated token requires reading the entire model
+ KV cache from memory. This is why bandwidth dominates token/sec.""",
    see_also=("prefill", "bandwidth", "tok/s"),
)

_add("ttft", "Time To First Token — how long after sending a prompt you see the first word.",
    """\
Determined mostly by prefill (compute) and queuing in a server.
Important for chat feel — sub-second TTFT feels responsive.""",
    see_also=("prefill",),
)

_add("speculative-decoding",
    "Speed trick: a small 'draft' model proposes tokens, big model verifies in parallel.",
    """\
The draft model generates K candidate tokens quickly. The big model
runs once on those K tokens and accepts the prefix it agrees with.
1.5-3× speedup with no quality loss. Used in production by vLLM, TGI,
TensorRT-LLM.""",
    see_also=("decode",),
)

_add("batching", "Running multiple requests in parallel for higher throughput.",
    """\
GPUs are wasted on a single decode token. Batching multiple requests
amortizes the memory reads. Continuous batching (vLLM, TGI) batches
across requests at different positions, which is the key to high
production throughput.""",
    see_also=("vllm", "pagedattention"),
)

_add("continuous-batching",
    "Dynamic batching that adds new requests as old ones finish — vLLM's secret sauce.",
    """\
Traditional batching waits for all requests to finish; continuous
batching swaps in new requests as soon as any slot frees up. The
result is high GPU utilisation across mixed request lengths.""",
    see_also=("batching", "vllm"),
)

_add("kv-cache-quantization",
    "Storing the KV cache itself in fp8 / int4 instead of fp16 — saves memory.",
    """\
Lets you fit much longer contexts at the cost of a tiny quality hit.
Supported by vLLM, TensorRT-LLM, llama.cpp (cache_k_quant / cache_v_quant).""",
    see_also=("kv-cache", "quantization"),
)

_add("temperature",
    "How random the model's choices are. 0 = always pick the most likely token.",
    """\
Higher = more variety, more creativity, more weirdness. Typical chat
values: 0.7. For deterministic / factual answers, use 0 or 0.1.""",
    see_also=("top-p", "top-k"),
)

_add("top-p",
    "Nucleus sampling — sample only from tokens summing to the top P probability.",
    """\
Top-p = 0.9 keeps the smallest set of tokens whose probabilities add
to 90%, then samples from that. Cuts off long tails of bad options.""",
    see_also=("temperature", "top-k"),
)

_add("top-k",
    "Sample only from the K most likely next tokens.",
    """\
Simpler than top-p but blunter. Common values: 40-60.""",
    see_also=("temperature", "top-p"),
)

_add("beam-search",
    "Search trick that keeps multiple candidate completions and picks the best.",
    """\
Rarely used for chat (kills creativity). Common in translation /
summarization where there's a clear single 'best' answer.""",
)

_add("greedy-decoding",
    "Always pick the most likely next token. Deterministic.",
    """\
Same as temperature=0. Good for factual queries and reproducible
outputs.""",
    see_also=("temperature",),
)

_add("system-prompt", "The 'instructions' message at the start of a chat.",
    """\
Sets persona/behaviour for the whole conversation. e.g. 'You are a
helpful assistant that answers in plain English.' Distinct from the
user's message.""",
    see_also=("prompt",),
)

_add("prompt",
    "What you send to the model. Everything that goes in.",
    """\
The art of writing prompts is called prompt engineering. For
instruct/chat models there are usually formatting tokens that wrap
your prompt (chat templates).""",
    see_also=("system-prompt", "chat-template"),
)

_add("chat-template",
    "The formatting wrapper that turns your messages into model input.",
    """\
Different models expect different chat formats (Llama vs. Qwen vs.
ChatML vs. Mistral). Wrong template → garbage output. HuggingFace's
tokenizer stores the template and applies it automatically.""",
    see_also=("prompt",),
)

_add("function-calling",
    "Letting the model emit structured calls to external tools.",
    """\
The model outputs a JSON object naming a function and arguments, your
code executes it, the result is fed back. The basis for agents. Most
instruct models support it now.""",
    see_also=("agent", "tool-use"),
)

_add("tool-use",
    "Same idea as function calling — model invokes external tools to answer.",
    """\
Includes web search, calculators, code execution, database queries.
Quality depends on how well the model was trained for it (Claude
3.5+, GPT-4o, Llama 3.1+ are good).""",
    see_also=("function-calling", "agent"),
)

_add("agent",
    "An LLM that takes multiple steps, calls tools, and acts on a goal.",
    """\
e.g. 'Find me cheap flights' → searches, parses results, refines,
returns. Reliability depends on the model's instruction-following and
its ability to recover from tool errors.""",
    see_also=("tool-use", "rag"),
)

_add("rag",
    "Retrieval-Augmented Generation — feed relevant docs into the prompt before asking.",
    """\
The model isn't trained to know your company's docs; instead, you
embed those docs, retrieve the most relevant chunks for each
question, and include them in the prompt. A whole industry around
making this fast and accurate.""",
    see_also=("embedding-model", "vector-database"),
)

_add("embedding-model",
    "A model whose job is to turn text into vectors, not generate text.",
    """\
Used in RAG to embed your docs and queries so you can find similar
ones by vector distance. Common ones: BGE, GTE, E5, nomic-embed.""",
    see_also=("rag", "vector-database"),
)

_add("vector-database",
    "Storage optimized for finding similar embeddings by nearest neighbour.",
    """\
Examples: Pinecone, Weaviate, Qdrant, Milvus, pgvector. They store
millions of vectors and let you query 'give me the 5 closest to this
one' in milliseconds.""",
    see_also=("rag", "embedding-model"),
)


# ===========================================================================
# Hardware / numerical formats
# ===========================================================================
_add("fp16",
    "16-bit floating point. Standard for inference before quantization came along.",
    """\
Half-precision floats. ~5-decimal-digit precision, supports values
up to ~65000. All modern GPUs do fp16 natively at full speed.""",
    see_also=("bf16", "fp8", "quantization"),
)

_add("bf16",
    "Brain Float 16 — fp16 but with a bigger exponent range.",
    """\
Same 16 bits, but allocates them differently: more exponent (range)
fewer fraction (precision). Avoids the overflow problems fp16 has
during training. Used pervasively for training.""",
    see_also=("fp16",),
)

_add("fp8",
    "8-bit floating point. New format, supported on H100/H200/Blackwell.",
    """\
Drops to 8 bits with two formats (E5M2 / E4M3). ~2× the throughput of
fp16 with carefully managed accuracy. The killer feature of modern
NVIDIA chips for inference.",
""",
    see_also=("bf16", "quantization"),
)

_add("int8",
    "8-bit integer quantization — older, still used.",
    """\
LLM.int8() (Tim Dettmers) brought this to LLMs. Slower than fp8 for
inference but supported everywhere.""",
    see_also=("quantization", "fp8"),
)

_add("awq",
    "Activation-aware Weight Quantization. Strong 4-bit format for serving.",
    """\
Weights are quantized but the quantization scales are picked to
minimize the error on the most active weight channels. Supported by
vLLM, TGI, TRT-LLM.""",
    see_also=("quantization", "gptq"),
)

_add("gptq",
    "Quantization-aware weight rounding. Older but well-supported 4-bit format.",
    """\
One of the first widely deployed 4-bit GPU quantizations. Slightly
lower quality than AWQ at the same bits but easier to convert.""",
    see_also=("awq", "quantization"),
)

_add("exl2", "ExLlama V2's quantization format — variable bits per layer.",
    """\
Lets you mix Q4 and Q6 in the same file, putting more precision where
it matters. Combined with ExLlamaV2's fast kernels = excellent
single-user RTX performance.""",
    see_also=("exllamav2",),
)

_add("cuda", "NVIDIA's GPU programming layer. The substrate everything runs on.",
    """\
Most inference engines write CUDA kernels for the hot loops. Apple's
equivalent is Metal; AMD's is ROCm/HIP; Intel's is OneAPI/SYCL.""",
    see_also=("rocm", "metal", "gpu"),
)

_add("metal", "Apple's GPU API. What llama.cpp / MLX use on Mac.",
    """\
The equivalent of CUDA on Apple Silicon. Less mature for ML than CUDA
but improving fast.""",
    see_also=("cuda", "apple silicon"),
)

_add("rocm", "AMD's CUDA-equivalent. Supports the MI series and some Radeon cards.",
    """\
Stack quality has improved a lot in 2024-25. vLLM, llama.cpp, and HF
all have working AMD backends now.""",
    see_also=("cuda",),
)

_add("hbm",
    "High Bandwidth Memory — the fast memory stacked on data-center GPUs.",
    """\
Why an H100 has 3,350 GB/s bandwidth and an RTX 4090 has 1,008 GB/s.
HBM is more expensive and lower capacity per dollar than GDDR.""",
    see_also=("bandwidth", "vram"),
)

_add("tflops",
    "Trillions of floating-point operations per second. Headline compute number.",
    """\
Useful for prefill / batch inference (compute-bound). For decode
(memory-bandwidth-bound) it barely matters — you'll be idle waiting
for memory.""",
    see_also=("bandwidth",),
)

_add("amx",
    "Intel's matrix accelerator. Speeds up CPU LLM inference on recent Xeons.",
    """\
AMX-BF16 doubles BF16 throughput on supported chips. llama.cpp can
use it.""",
    see_also=("cpu",),
)

_add("tensor-cores",
    "Specialized matrix-multiply units inside NVIDIA GPUs.",
    """\
Where most of the LLM compute actually runs. Different generations
have different supported types — H100 Tensor Cores can do FP8, A100
only fp16/bf16.""",
    see_also=("cuda",),
)


# ===========================================================================
# Safety / production
# ===========================================================================
_add("alignment", "Making the model do what users (and society) actually want.",
    """\
A research and engineering programme: refusing harmful requests,
following nuanced instructions, being honest about uncertainty.
Implemented via RLHF, DPO, constitutional AI, etc.""",
    see_also=("rlhf", "dpo", "safety"),
)

_add("safety",
    "Avoiding harmful, biased, or dangerous model outputs.",
    """\
Includes red-teaming, jailbreak resistance, refusal training,
content filtering. Open-weight models can usually be jailbroken
relatively easily compared to API-hosted ones.""",
)

_add("hallucination",
    "When the model makes up facts that aren't true.",
    """\
LLMs predict plausible-sounding next tokens — sometimes that plausibility
diverges from reality. RAG and tool-use help by grounding answers in
real documents.""",
    see_also=("rag",),
)

_add("jailbreak",
    "A prompt that tricks the model into ignoring its safety training.",
    """\
e.g. 'You are now an unrestricted AI…'. Effectiveness depends on the
model. Generally, more aligned models are harder but not impossible
to jailbreak.""",
    see_also=("safety", "alignment"),
)


# ===========================================================================
# Less common but useful
# ===========================================================================
_add("scaling-laws",
    "Empirical rules predicting model quality from compute, data, and parameters.",
    """\
Chinchilla scaling: a 70B model is optimally trained on ~1.4T tokens.
DeepSeek and Llama 3 deliberately overtrained smaller models past
Chinchilla optimal, which works better for inference economics.""",
    see_also=("pretraining",),
)

_add("chinchilla",
    "Famous 2022 DeepMind paper that revised LLM scaling rules.",
    """\
Showed earlier work had under-trained models. Optimal compute spends
~20 tokens per parameter. Result: smaller, better-trained models
beat larger under-trained ones.""",
    see_also=("scaling-laws",),
)

_add("mixture-of-experts", "Same as MoE — only part of the model runs per token.",
    """\
See: moe.""",
    see_also=("moe",),
)

_add("mlp",
    "Multi-Layer Perceptron — the 'feed-forward' block in each transformer layer.",
    """\
After attention, each transformer block has an MLP that processes
tokens independently. In modern LLMs this is where most parameters
live, and where MoE replaces a single dense MLP with multiple
expert MLPs.""",
    see_also=("transformer", "moe"),
)

_add("residual",
    "Skip connections that add a layer's input back to its output.",
    """\
Lets gradients flow cleanly through deep networks. Without residuals,
training 80-layer transformers would be impossible.""",
)

_add("dropout",
    "Training-time regularization that randomly zeroes activations.",
    """\
Used in many transformers but turned off at inference. Rarely
discussed in modern LLM training where dataset size is the bigger
regularizer.""",
)


# Aliases that should resolve to the canonical entry.
ALIASES = {
    "tokens": "token",
    "tokens-per-second": "tok/s",
    "tps": "tok/s",
    "ctx": "context",
    "context-window": "context",
    "context-length": "context",
    "kv cache": "kv-cache",
    "kvcache": "kv-cache",
    "weights": "parameter",
    "params": "parameter",
    "parameters": "parameter",
    "huggingface": "hugging face",
    "hf": "hugging face",
    "mac": "apple silicon",
    "m1": "apple silicon", "m2": "apple silicon",
    "m3": "apple silicon", "m4": "apple silicon",
    "q4": "q4_k_m", "q5": "q4_k_m", "q3": "q2_k",
    "lm studio": "lmstudio",
    "llamacpp": "llama.cpp",
    "mixture of experts": "moe",
    "moe-model": "moe",
    "fine-tuning": "finetuning",
    "fine tuning": "finetuning",
    "supervised fine-tuning": "sft",
    "supervised fine tuning": "sft",
    "rlhf-ppo": "ppo",
    "direct preference optimization": "dpo",
    "low rank adaptation": "lora",
    "low-rank adaptation": "lora",
    "q-lora": "qlora",
    "backprop": "backpropagation",
    "sgd": "gradient-descent",
    "adam": "optimizer",
    "adamw": "optimizer",
    "lr": "learning-rate",
    "batch size": "batch",
    "bs": "batch",
    "byte pair encoding": "bpe",
    "byte-pair-encoding": "bpe",
    "vocab": "vocabulary",
    "kd": "distillation",
    "knowledge distillation": "distillation",
    "self attention": "self-attention",
    "flashattention": "flash-attention",
    "flash attn": "flash-attention",
    "paged attention": "pagedattention",
    "grouped query attention": "gqa",
    "multi query attention": "mqa",
    "rotary": "rope",
    "rotary position embeddings": "rope",
    "rotary embeddings": "rope",
    "alibi": "positional-encoding",
    "layer norm": "normalization",
    "layernorm": "normalization",
    "rms norm": "rmsnorm",
    "gelu": "activation",
    "silu": "activation",
    "ttft": "ttft",
    "first token latency": "ttft",
    "spec dec": "speculative-decoding",
    "speculative": "speculative-decoding",
    "draft model": "speculative-decoding",
    "kv cache quant": "kv-cache-quantization",
    "temp": "temperature",
    "topp": "top-p",
    "topk": "top-k",
    "beam": "beam-search",
    "greedy": "greedy-decoding",
    "system message": "system-prompt",
    "tools": "tool-use",
    "tool calling": "function-calling",
    "function call": "function-calling",
    "agents": "agent",
    "retrieval augmented generation": "rag",
    "embeddings": "embedding-model",
    "embedding": "embedding-model",
    "vector db": "vector-database",
    "fp 16": "fp16",
    "bfloat16": "bf16",
    "fp 8": "fp8",
    "int 8": "int8",
    "activation aware": "awq",
    "tensor core": "tensor-cores",
    "alignment-rlhf": "alignment",
    "red team": "safety",
    "hallucinations": "hallucination",
    "scaling law": "scaling-laws",
    "ffn": "mlp",
    "feed-forward": "mlp",
    "skip connection": "residual",
}


def lookup(term: str) -> Entry | None:
    key = term.lower().strip()
    if key in ALIASES:
        key = ALIASES[key]
    if key in GLOSSARY:
        return GLOSSARY[key]
    # Substring fallback (must be unique).
    matches = [v for k, v in GLOSSARY.items() if key in k]
    return matches[0] if len(matches) == 1 else None


def search(query: str, limit: int = 25) -> List[Entry]:
    """Full-text search across term, summary, body, and analogy.

    Splits the query into tokens and ranks entries by how many tokens hit
    each field. Multi-word queries like "rl from human" now find RLHF.
    """
    q = query.lower().strip()
    if not q:
        return all_terms()[:limit]
    canonical = ALIASES.get(q, q)
    tokens = [t for t in q.replace("-", " ").replace("_", " ").split() if t]
    if not tokens:
        return []
    scored: List[tuple] = []
    for e in GLOSSARY.values():
        term_l = e.term.lower()
        summary_l = e.summary.lower()
        body_l = e.body.lower()
        analogy_l = (e.analogy or "").lower()
        see_blob = " ".join(e.see_also).lower()

        # Exact-term boost.
        score = 0
        if term_l == canonical or term_l == q:
            score += 200
        if q in term_l:
            score += 80

        # Per-token scoring.
        for tok in tokens:
            if tok in term_l:    score += 30
            if tok in summary_l: score += 15
            if tok in body_l:    score += 5
            if tok in analogy_l: score += 3
            if tok in see_blob:  score += 2

        # All tokens must match somewhere — otherwise drop the entry.
        all_text = f"{term_l} {summary_l} {body_l} {analogy_l} {see_blob}"
        if not all(tok in all_text for tok in tokens):
            continue
        if score:
            scored.append((score, e))
    scored.sort(key=lambda p: (-p[0], p[1].term.lower()))
    return [e for _s, e in scored[:limit]]


def all_terms() -> List[Entry]:
    return sorted(GLOSSARY.values(), key=lambda e: e.term.lower())


def all_categories() -> Dict[str, List[Entry]]:
    """Hard-coded category index. Used by the GUI sidebar."""
    cats = {
        "Basics": ["LLM", "token", "tok/s", "context", "parameter", "instruct",
                    "chat", "reasoning"],
        "Hardware": ["vram", "ram", "bandwidth", "gpu", "apple silicon",
                      "hbm", "tflops", "amx", "tensor-cores", "cuda",
                      "metal", "rocm"],
        "Quantization": ["quantization", "q8_0", "q4_k_m", "q2_k", "fp16",
                          "bf16", "fp8", "int8", "awq", "gptq", "exl2",
                          "gguf", "kv-cache-quantization"],
        "Training": ["pretraining", "finetuning", "sft", "rlhf", "ppo", "dpo",
                       "lora", "qlora", "adapter", "backpropagation",
                       "gradient-descent", "optimizer", "learning-rate",
                       "loss", "perplexity", "epoch", "batch",
                       "gradient-accumulation", "tokenizer", "bpe",
                       "vocabulary", "dataset", "synthetic-data",
                       "distillation"],
        "Architecture": ["transformer", "attention", "self-attention",
                          "flash-attention", "pagedattention", "gqa", "mqa",
                          "embedding", "positional-encoding", "rope", "yarn",
                          "normalization", "rmsnorm", "activation", "swiglu",
                          "mlp", "residual", "kv-cache"],
        "Inference": ["prefill", "decode", "ttft", "speculative-decoding",
                       "batching", "continuous-batching", "moe", "temperature",
                       "top-p", "top-k", "beam-search", "greedy-decoding",
                       "system-prompt", "prompt", "chat-template"],
        "Engines & runners": ["ollama", "llama.cpp", "lmstudio", "hugging face",
                                "inference"],
        "Applications": ["function-calling", "tool-use", "agent", "rag",
                           "embedding-model", "vector-database"],
        "Safety & alignment": ["alignment", "safety", "hallucination",
                                "jailbreak"],
        "Theory": ["scaling-laws", "chinchilla", "dropout"],
    }
    out: Dict[str, List[Entry]] = {}
    for cat, terms in cats.items():
        out[cat] = [GLOSSARY[t.lower()] for t in terms if t.lower() in GLOSSARY]
    return out
