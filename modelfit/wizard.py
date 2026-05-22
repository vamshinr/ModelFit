"""Interactive Q&A for beginners.

Walks a brand-new user through:
    1. What do you want to use it for? (use case)
    2. How important is speed vs quality?
    3. How long a conversation/document do you need to handle? (context)
…then runs the recommendation engine and prints a copy-paste install command.

Pure input(); no Textual dependency.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from modelfit.hardware import detect_hardware
from modelfit.recommend import recommend
from modelfit.runners import get_commands
from modelfit.beginner import (
    speed_label, context_label, quant_label, type_label,
    verdict, best_for, tier_explanation, download_label,
)
from modelfit.scoring import USE_CASE_WEIGHTS
from modelfit.quantization import GB

console = Console()


USE_CASE_CHOICES: List[Tuple[str, str, str]] = [
    ("1", "chat",         "Casual chat, writing help, summarising"),
    ("2", "code",         "Writing or debugging code"),
    ("3", "reasoning",    "Hard problems, math, multi-step thinking"),
    ("4", "long-context", "Reading long documents / whole books"),
    ("5", "agent",        "Tool-use, RAG, agents"),
    ("6", "balanced",     "Not sure — just give me a good general model"),
]

CONTEXT_CHOICES = [
    ("1", 2048,   "Short prompts (a paragraph or two)"),
    ("2", 8192,   "Long emails / a few pages"),
    ("3", 32768,  "A long document / chapter"),
    ("4", 131072, "A whole book / huge codebase"),
]


def _menu(prompt: str, items, console=console) -> str:
    """Render a numbered menu, return the selected key."""
    lines = [prompt, ""]
    for key, _label, desc in [(k, v, d) for k, v, d in items]:
        lines.append(f"  [bold cyan]{key}[/]  {desc}")
    console.print("\n".join(lines))
    valid = {k for k, *_ in items}
    while True:
        choice = Prompt.ask("\n[bold]Choose[/]", default="1").strip()
        if choice in valid:
            return choice
        console.print(f"[red]Pick one of {sorted(valid)}.[/]")


def run_wizard() -> int:
    hw = detect_hardware()
    console.print(Panel.fit(
        "[bold]modelfit wizard[/]\n"
        "I'll ask you a couple of questions, then recommend a model that "
        "actually runs on your computer.\n\n"
        f"[dim]Detected:[/] {hw.cpu_name}  ·  {hw.ram_bytes / GB:.0f} GB RAM"
        + (f"  ·  GPU: {hw.best_gpu.name} ({hw.best_gpu.vram_bytes / GB:.0f} GB VRAM)"
           if hw.best_gpu else "  ·  No GPU detected")
        + (f"  ·  Apple unified memory" if hw.unified_memory else ""),
        border_style="cyan",
    ))

    # 1. Use case
    uc_choice = _menu(
        "\n[bold]What do you mainly want to do?[/]",
        USE_CASE_CHOICES,
    )
    use_case = next(v for k, v, _ in USE_CASE_CHOICES if k == uc_choice)

    # 2. Context (only if relevant)
    ctx_choice = _menu(
        "\n[bold]How much text does the model need to handle at once?[/]",
        CONTEXT_CHOICES,
    )
    min_context = next(v for k, v, _ in CONTEXT_CHOICES if k == ctx_choice)

    # 3. (Optional) speed vs quality preference — only if balanced/chat
    # We keep the wizard intentionally short. For 'chat' & 'balanced' we
    # leave the weights alone, which already lean toward speed/quality nicely.

    console.print(f"\n[dim]Looking for the best fit for "
                   f"'{use_case}' with at least {min_context:,} tokens of context…[/]")

    rec = recommend(hw, use_case=use_case, min_context=min_context)
    if not rec:
        console.print("\n[red]Couldn't find any model that fits with those requirements.[/]")
        console.print("Try lowering the context, or run `modelfit reverse <model>` "
                      "to see what hardware would be needed.")
        return 1

    s = rec.primary
    v_head, v_sum = verdict(s)
    sp_head, sp_feel = speed_label(s.tokens_per_sec)
    ctx_head, ctx_size = context_label(s.fit.context or 0)
    q_label, q_stars, q_desc = quant_label(s.fit.quant)

    console.print()
    console.print(Panel(
        f"[bold]{s.model.name}[/]  [dim]({s.model.id})[/]\n"
        f"[dim]{type_label(s.model.type)}[/]\n\n"
        f"  {v_head}\n"
        f"  Quality:    {q_stars}  {q_label}  [dim]· {q_desc}[/]\n"
        f"  Speed:      {sp_head}  [dim]· {sp_feel}[/]\n"
        f"  Context:    {ctx_head}  [dim]· {ctx_size}[/]\n"
        f"  Memory:     {s.fit.total_bytes / GB:.1f} GB of "
        f"{s.fit.budget_bytes / GB:.1f} GB available  [dim]· {tier_explanation(s.fit.tier)}[/]\n\n"
        f"[bold]Why this one:[/]\n{rec.why}\n\n"
        f"[bold]Best for:[/] {best_for(s.model)}",
        title="🎯 Your pick", border_style="green",
    ))

    # Show ready-to-paste install command.
    cmds = get_commands(s.model, s.fit.quant or "Q4_K_M")
    install_lines = [
        f"[bold]📥 Download size:[/] {cmds.download_label}",
        "",
    ]
    if cmds.ollama:
        install_lines += [
            "[bold]🚀 Easiest (Ollama):[/]",
            f"   [green]{cmds.ollama}[/]",
            "[dim]   (Install Ollama first from https://ollama.com)[/]",
            "",
        ]
    if cmds.huggingface_repo:
        install_lines += [
            "[bold]📦 Manual (HuggingFace + llama.cpp):[/]",
            f"   Repo: [cyan]{cmds.huggingface_repo}[/]",
            f"   File: [cyan]{cmds.huggingface_file}[/]",
            "",
        ]
    install_lines += ["[bold]🖥️ LM Studio:[/]",
                       "   Open LM Studio → 'Discover' → search:",
                       f"   [cyan]{s.model.name}[/]",
                       f"   Download the file labelled [cyan]{s.fit.quant}[/]."]
    for n in (cmds.notes or []):
        install_lines.append(f"[dim]Note: {n}[/]")

    console.print(Panel("\n".join(install_lines), title="📋 How to install",
                          border_style="cyan"))

    if rec.runner_ups:
        console.print("\n[bold]Other good options:[/]")
        for r in rec.runner_ups:
            sp = speed_label(r.tokens_per_sec)[0]
            console.print(
                f"  [cyan]{r.model.id:35s}[/]  {r.model.params_b:5.1f}B  "
                f"{r.fit.quant or '-':6s}  {sp}  [dim](score {r.composite:.0f})[/]"
            )

    console.print(
        "\n[dim]Run [bold]modelfit info " + s.model.id + "[/] for the full breakdown, "
        "or [bold]modelfit explain quantization[/] to learn the lingo.[/]"
    )
    return 0
