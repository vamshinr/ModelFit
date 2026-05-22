"""modelfit command-line interface.

Beginner-friendly by default; expert mode behind --expert.

Subcommands:
    rank        (default) Ranked table of fitting models
    recommend   Pick ONE model + plain-English explanation + install command
    wizard      Interactive Q&A — for total beginners
    get         Show ready-to-paste install commands for a model
    explain     Glossary — plain-English definitions
    info        Detailed fit/score for one model
    inspect     Detected hardware
    reverse     What hardware would I need to run X?
    tui         Interactive Textual UI
    serve       REST API for cluster schedulers
    list        Dump the full catalog
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from modelfit import __version__
from modelfit.hardware import detect_hardware, HardwareProfile
from modelfit.models import get_catalog, find_model, QUANT_ORDER
from modelfit.quantization import fit_model, all_quant_options, GB
from modelfit.scoring import rank_all, score_model, ScoredModel, USE_CASE_WEIGHTS
from modelfit.beginner import (
    quant_label, speed_label, context_label, type_label,
    verdict, best_for, tier_explanation, download_size_gb,
    download_label, why_doesnt_fit, USE_CASE_DESCRIPTION,
)

console = Console()


# ---------------------------------------------------------------------------
# Hardware summary panel — friendly version with English explanations
# ---------------------------------------------------------------------------
def _hardware_summary(hw: HardwareProfile, *, friendly: bool = True) -> Panel:
    g = hw.best_gpu
    lines = [
        f"[bold cyan]CPU:[/]  {hw.cpu_name}  ({hw.cpu_cores} cores)",
        f"[bold cyan]RAM:[/]  {hw.ram_bytes / GB:.1f} GB  "
        f"[dim]({hw.ram_available_bytes / GB:.1f} GB free right now)[/]",
    ]
    if hw.gpus:
        for gp in hw.gpus:
            lines.append(
                f"[bold magenta]GPU:[/]  {gp.name}  "
                f"[bold]{gp.vram_bytes / GB:.0f} GB VRAM[/]"
                + (f"  [dim]({gp.vendor.upper()})[/]" if gp.vendor != "unknown" else "")
            )
    else:
        lines.append("[bold magenta]GPU:[/]  none — [yellow]CPU only (slower)[/]")
    if hw.unified_memory:
        lines.append("[dim]Apple Silicon: the GPU shares system memory.[/]")
    if friendly:
        # One-line interpretation for beginners.
        budget = hw.total_vram_bytes / GB if (hw.gpus or hw.unified_memory) else hw.ram_bytes / GB
        if budget >= 60:
            verdict_line = "[green]✨ Plenty of memory — you can run big models.[/]"
        elif budget >= 24:
            verdict_line = "[green]👍 Comfortable amount — most 30B and many 70B models work.[/]"
        elif budget >= 12:
            verdict_line = "[yellow]🆗 Mid-range — 7-14B models run well; bigger ones need compression.[/]"
        elif budget >= 6:
            verdict_line = "[yellow]🤏 Limited memory — stick with 7B or smaller.[/]"
        else:
            verdict_line = "[red]🪶 Very limited — only the tiniest models will fit.[/]"
        lines.append("")
        lines.append(verdict_line)
    return Panel("\n".join(lines), title=f"Your computer ({hw.platform} {hw.arch})",
                 border_style="cyan")


def _emit_json(payload, file=sys.stdout):
    json.dump(payload, file, indent=2, default=str)
    file.write("\n")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _render_friendly_table(ranked: List[ScoredModel], limit: int) -> Table:
    t = Table(title="🏅 Models that run well on your computer",
              box=box.SIMPLE_HEAD, header_style="bold",
              expand=False, pad_edge=False)
    t.add_column("#", style="dim", justify="right", width=3)
    t.add_column("Model", style="bold", overflow="ellipsis", min_width=24)
    t.add_column("What it's good at", overflow="fold", min_width=22)
    t.add_column("Quality", min_width=14)
    t.add_column("Speed", min_width=12)
    t.add_column("Context", min_width=14)
    t.add_column("Memory", justify="right", width=8)
    for i, s in enumerate(ranked[:limit], 1):
        q_label, q_stars, _ = quant_label(s.fit.quant)
        sp_head, _ = speed_label(s.tokens_per_sec)
        ctx_head, ctx_size = context_label(s.fit.context or 0)
        mem = f"{s.fit.total_bytes / GB:.1f} GB"
        t.add_row(
            str(i),
            s.model.name,
            best_for(s.model).split(";")[0],
            f"{q_stars} [dim]{q_label}[/]",
            sp_head,
            ctx_head,
            mem,
        )
    return t


def _render_expert_table(ranked: List[ScoredModel], limit: int) -> Table:
    t = Table(title="Best models for your system (expert view)",
              box=box.SIMPLE_HEAD, header_style="bold", expand=False, pad_edge=False)
    t.add_column("#", style="dim", justify="right", width=3)
    t.add_column("Score", justify="right", width=5)
    t.add_column("Model", style="bold", overflow="ellipsis", min_width=24)
    t.add_column("Type", style="cyan", width=8)
    t.add_column("Quant", style="magenta", width=6)
    t.add_column("Ctx", justify="right", width=6)
    t.add_column("Mem", justify="right", width=7)
    t.add_column("tok/s", justify="right", width=6)
    t.add_column("Q", justify="right", width=3)
    t.add_column("S", justify="right", width=3)
    t.add_column("C", justify="right", width=3)
    t.add_column("Cap", justify="right", width=3)
    for i, s in enumerate(ranked[:limit], 1):
        ctx_str = (f"{s.fit.context // 1000}K"
                   if s.fit.context and s.fit.context >= 1000
                   else str(s.fit.context or 0))
        tps_color = "green" if s.tokens_per_sec >= 20 else ("yellow" if s.tokens_per_sec >= 5 else "red")
        score_color = "green" if s.composite >= 70 else ("yellow" if s.composite >= 50 else "white")
        t.add_row(
            str(i),
            f"[{score_color}]{s.composite:.1f}[/]",
            s.model.name,
            s.model.type,
            s.fit.quant or "-",
            ctx_str,
            f"{s.fit.total_bytes / GB:.1f}G",
            f"[{tps_color}]{s.tokens_per_sec:.1f}[/]",
            f"{s.quality:.0f}", f"{s.speed:.0f}",
            f"{s.context:.0f}", f"{s.capability:.0f}",
        )
    return t


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------
def cmd_rank(args) -> int:
    hw = detect_hardware()
    ranked = rank_all(
        hw, use_case=args.use_case, min_context=args.min_context,
        include_unfit=args.include_unfit, family=args.family,
        type_filter=args.type, min_quality=args.min_quality,
    )
    if args.json:
        _emit_json({
            "hardware": hw.to_dict(),
            "use_case": args.use_case,
            "weights": dict(zip(("quality", "speed", "context", "capability"),
                                 USE_CASE_WEIGHTS[args.use_case])),
            "total_models": len(get_catalog()),
            "fitting_models": len(ranked),
            "models": [s.to_dict() for s in ranked[:args.top]],
        })
        return 0
    console.print(_hardware_summary(hw, friendly=not args.expert))
    uc_desc = USE_CASE_DESCRIPTION.get(args.use_case, "")
    if args.expert:
        console.print(f"\n[bold]Use case:[/] {args.use_case}   "
                      f"[bold]Weights:[/] "
                      f"Q/S/C/Cap = {':'.join(str(w) for w in USE_CASE_WEIGHTS[args.use_case])}\n")
    else:
        console.print(f"\n[bold]Optimising for:[/] {args.use_case}  [dim]({uc_desc})[/]\n")
    if not ranked:
        console.print("[bold red]No models fit on this computer at the requested context length.[/]")
        console.print("Try lowering the context: [cyan]--min-context 2048[/]")
        console.print("Or see what hardware would help: [cyan]modelfit reverse <model>[/]")
        return 1
    table = _render_expert_table(ranked, args.top) if args.expert else _render_friendly_table(ranked, args.top)
    console.print(table)
    if not args.expert:
        top = ranked[0]
        console.print(
            f"\n[bold green]→[/] Easiest start: [bold]{top.model.name}[/].  "
            f"Run [cyan]modelfit recommend[/] for an explanation + install commands, "
            f"or [cyan]modelfit get {top.model.id}[/] to see how to install it."
        )
    console.print(f"\n[dim]{len(ranked)} / {len(get_catalog())} models fit "
                  f"(min context {args.min_context:,} tokens). "
                  f"Add --expert to see scores; --include-unfit to see the rest.[/]")
    return 0


# ---------------------------------------------------------------------------
# recommend — one pick + reason + install command
# ---------------------------------------------------------------------------
def cmd_recommend(args) -> int:
    from modelfit.recommend import recommend
    from modelfit.runners import get_commands
    hw = detect_hardware()
    rec = recommend(hw, use_case=args.use_case, min_context=args.min_context)
    if not rec:
        console.print("[red]No model fits on this computer.[/] "
                      "Try [cyan]modelfit reverse llama-3.2-1b-instruct[/] to see what hardware would help.")
        return 1
    if args.json:
        cmds = get_commands(rec.primary.model, rec.primary.fit.quant or "Q4_K_M")
        _emit_json({**rec.to_dict(), "install": cmds.to_dict()})
        return 0
    s = rec.primary
    v_head, _ = verdict(s)
    sp_head, sp_feel = speed_label(s.tokens_per_sec)
    ctx_head, ctx_size = context_label(s.fit.context or 0)
    q_label, q_stars, q_desc = quant_label(s.fit.quant)
    cmds = get_commands(s.model, s.fit.quant or "Q4_K_M")

    console.print(_hardware_summary(hw))
    console.print(Panel(
        f"[bold]{s.model.name}[/]  [dim]({s.model.id})[/]\n"
        f"[dim]{type_label(s.model.type)}[/]\n\n"
        f"  {v_head}\n"
        f"  Quality:    {q_stars}  {q_label}   [dim]· {q_desc}[/]\n"
        f"  Speed:      {sp_head}  [dim]· {sp_feel}[/]\n"
        f"  Context:    {ctx_head}  [dim]· {ctx_size}[/]\n"
        f"  Memory:     {s.fit.total_bytes / GB:.1f} GB used of "
        f"{s.fit.budget_bytes / GB:.1f} GB available  [dim]· {tier_explanation(s.fit.tier)}[/]\n"
        f"  Download:   ~{cmds.download_label} on disk\n\n"
        f"[bold]Why this one:[/]\n{rec.why}\n\n"
        f"[bold]Best for:[/] {best_for(s.model)}",
        title="🎯 Recommended", border_style="green",
    ))

    # Install commands
    install_lines = []
    if cmds.ollama:
        install_lines += [
            "[bold]🚀 Easiest — Ollama:[/]",
            f"   [green]{cmds.ollama}[/]",
            "[dim]   (Install Ollama first from https://ollama.com)[/]",
            "",
        ]
    if cmds.huggingface_repo:
        install_lines += [
            "[bold]📦 HuggingFace + llama.cpp:[/]",
            f"   Repo: [cyan]{cmds.huggingface_repo}[/]",
            f"   File: [cyan]{cmds.huggingface_file}[/]",
            "",
        ]
    install_lines += [
        "[bold]🖥️  LM Studio:[/]",
        "   Open LM Studio → 'Discover' tab → search:",
        f"   [cyan]{s.model.name}[/]",
        f"   Download the file labelled [cyan]{s.fit.quant}[/].",
    ]
    for n in (cmds.notes or []):
        install_lines.append(f"[dim]Note: {n}[/]")
    console.print(Panel("\n".join(install_lines), title="📋 How to install",
                         border_style="cyan"))

    if rec.runner_ups:
        console.print("\n[bold]Other strong options:[/]")
        for r in rec.runner_ups:
            sp, _ = speed_label(r.tokens_per_sec)
            console.print(f"  • [bold]{r.model.name}[/] [dim]({r.model.id})[/] — "
                          f"{sp}, {r.fit.quant} compression")
        console.print(f"\n[dim]Run [cyan]modelfit get <id>[/] for any of those install commands.[/]")
    return 0


# ---------------------------------------------------------------------------
# wizard
# ---------------------------------------------------------------------------
def cmd_wizard(args) -> int:
    from modelfit.wizard import run_wizard
    return run_wizard()


# ---------------------------------------------------------------------------
# get <model>
# ---------------------------------------------------------------------------
def cmd_get(args) -> int:
    from modelfit.runners import get_commands
    m = find_model(args.model)
    if not m:
        console.print(f"[red]Model not found:[/] {args.model}")
        console.print("Try [cyan]modelfit list[/] to see the catalog.")
        return 2
    hw = detect_hardware()
    scored = score_model(m, hw, use_case="balanced", min_context=args.min_context)
    quant = args.quant or (scored.fit.quant if scored.fit.fits else "Q4_K_M")
    cmds = get_commands(m, quant)
    if args.json:
        _emit_json({
            "model": m.to_dict(),
            "fits": scored.fit.fits,
            "chosen_quant": quant,
            "install": cmds.to_dict(),
        })
        return 0

    console.print(Panel(
        f"[bold]{m.name}[/]  [dim]({m.id})[/]\n"
        f"{type_label(m.type)}\n"
        f"[bold]Best for:[/] {best_for(m)}\n\n"
        f"[bold]Download:[/] ~{cmds.download_label} (at {quant} compression)",
        title="Model", border_style="magenta",
    ))

    if not scored.fit.fits:
        console.print(f"[yellow]⚠️  Heads up:[/] this model probably won't run on your computer. "
                       f"Run [cyan]modelfit reverse {m.id}[/] to see what hardware would be needed.\n")

    lines = []
    if cmds.ollama:
        lines += ["[bold]🚀 Easiest — Ollama:[/]",
                  f"   [green]{cmds.ollama}[/]",
                  "[dim]   (Install Ollama from https://ollama.com)[/]",
                  ""]
    if cmds.huggingface_repo:
        lines += ["[bold]📦 HuggingFace download (for llama.cpp / others):[/]",
                  f"   [cyan]huggingface-cli download {cmds.huggingface_repo} \\\n"
                  f"     {cmds.huggingface_file} --local-dir ./models[/]",
                  ""]
    if cmds.llamacpp:
        lines += ["[bold]🦙 llama.cpp:[/]",
                  *[f"   [cyan]{ln}[/]" for ln in cmds.llamacpp.splitlines() if ln.strip()],
                  ""]
    lines += [
        "[bold]🖥️  LM Studio:[/]",
        f"   Open LM Studio → Discover → search for [cyan]{m.name}[/]",
        f"   Download the file labelled [cyan]{quant}[/].",
    ]
    for n in (cmds.notes or []):
        lines.append(f"\n[dim]Note: {n}[/]")
    console.print(Panel("\n".join(lines), title="📋 Install / Run", border_style="cyan"))
    return 0


# ---------------------------------------------------------------------------
# explain <term>
# ---------------------------------------------------------------------------
def cmd_explain(args) -> int:
    from modelfit.glossary import lookup, all_terms, search, all_categories

    # Search mode: free-text across summary/body/analogy.
    if args.search:
        hits = search(args.search, limit=args.limit)
        if args.json:
            _emit_json([{"term": e.term, "summary": e.summary} for e in hits])
            return 0
        if not hits:
            console.print(f"[yellow]No matches for[/] '{args.search}'.")
            return 1
        console.print(f"[bold]Search results for[/] '[cyan]{args.search}[/]':\n")
        for e in hits:
            console.print(f"  [bold cyan]{e.term:24s}[/]  {e.summary}")
        console.print(f"\n[dim]{len(hits)} results. Run [cyan]modelfit explain <term>[/] for details.[/]")
        return 0

    # Category browsing.
    if args.category:
        cats = all_categories()
        if args.category == "list":
            for cat in cats:
                console.print(f"  [bold]{cat}[/] [dim]({len(cats[cat])} entries)[/]")
            return 0
        entries = cats.get(args.category)
        if not entries:
            console.print(f"[red]Unknown category:[/] {args.category}")
            console.print("Try [cyan]modelfit explain --category list[/]")
            return 2
        if args.json:
            _emit_json([{"term": e.term, "summary": e.summary} for e in entries])
            return 0
        console.print(f"[bold]{args.category}[/]\n")
        for e in entries:
            console.print(f"  [bold cyan]{e.term:24s}[/]  {e.summary}")
        return 0

    # No term: list everything (or per-category if --grouped).
    if not args.term:
        if args.grouped:
            cats = all_categories()
            if args.json:
                _emit_json({cat: [{"term": e.term, "summary": e.summary} for e in es]
                             for cat, es in cats.items()})
                return 0
            for cat, es in cats.items():
                console.print(f"\n[bold underline]{cat}[/]")
                for e in es:
                    console.print(f"  [cyan]{e.term:24s}[/]  {e.summary}")
            return 0
        terms = all_terms()
        if args.json:
            _emit_json([{"term": e.term, "summary": e.summary} for e in terms])
            return 0
        console.print(f"[bold]Glossary[/] [dim]({len(terms)} entries — "
                      f"[cyan]modelfit explain <term>[/] for details, "
                      f"[cyan]--search <query>[/] to search, "
                      f"[cyan]--grouped[/] for categories)[/]\n")
        for e in terms:
            console.print(f"  [bold cyan]{e.term:24s}[/]  {e.summary}")
        return 0

    e = lookup(args.term)
    if not e:
        console.print(f"[red]No glossary entry for '{args.term}'.[/]")
        console.print(f"Try [cyan]modelfit explain --search {args.term}[/] for a full-text match.")
        return 1
    if args.json:
        _emit_json({"term": e.term, "summary": e.summary, "body": e.body,
                     "analogy": e.analogy, "see_also": list(e.see_also)})
        return 0
    body = f"{e.body}"
    if e.analogy:
        body += f"\n\n[italic dim]Analogy:[/] [italic]{e.analogy}[/]"
    if e.see_also:
        body += f"\n\n[dim]See also: {', '.join(e.see_also)}[/]"
    console.print(Panel(body, title=f"📖 {e.term} — {e.summary}",
                          border_style="cyan"))
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def cmd_info(args) -> int:
    m = find_model(args.model)
    if not m:
        console.print(f"[red]Model not found:[/] {args.model}")
        return 2
    hw = detect_hardware()
    scored = score_model(m, hw, use_case=args.use_case, min_context=args.min_context)
    options = all_quant_options(m, hw, context=min(m.context_max, args.min_context * 4))
    if args.json:
        _emit_json({
            "hardware": hw.to_dict(),
            "model": m.to_dict(),
            "score": scored.to_dict(),
            "quant_options_at_ctx": [o.to_dict() for o in options],
        })
        return 0
    console.print(_hardware_summary(hw, friendly=False))
    spec_lines = [
        f"[bold]{m.name}[/]  [dim]({m.id})[/]",
        f"{type_label(m.type)}",
        f"Family: {m.family}    Params: {m.params_b:.1f}B"
        + (f"  [dim](active {m.active_params_b:.1f}B per token — MoE)[/]"
           if m.active_params_b < m.params_b * 0.95 else ""),
        f"Max context: {m.context_max:,} tokens  ({context_label(m.context_max)[0]})",
        f"Intrinsic quality: {m.quality:.0f}/100",
        f"Best for: {best_for(m)}",
    ]
    if m.tags:
        spec_lines.append(f"[dim]Tags: {', '.join(m.tags)}[/]")
    console.print(Panel("\n".join(spec_lines), title="Model", border_style="magenta"))

    if scored.fit.fits:
        v_head, _ = verdict(scored)
        q_label, q_stars, q_desc = quant_label(scored.fit.quant)
        sp_head, sp_feel = speed_label(scored.tokens_per_sec)
        ctx_head, ctx_size = context_label(scored.fit.context or 0)
        download_gb = download_size_gb(m, scored.fit.quant)
        console.print(Panel(
            f"  {v_head}\n"
            f"  Quality:    {q_stars}  {q_label}  [dim]· {q_desc}[/]\n"
            f"  Speed:      {sp_head}  [dim]· {sp_feel}[/]\n"
            f"  Context:    {ctx_head}  [dim]· {ctx_size}[/]\n"
            f"  Memory:     {scored.fit.total_bytes/GB:.1f} GB used of {scored.fit.budget_bytes/GB:.1f} GB available\n"
            f"  Download:   ~{download_label(download_gb)} on disk  [dim]· {tier_explanation(scored.fit.tier)}[/]",
            title="✅ Will run on this computer", border_style="green",
        ))
        if args.expert:
            console.print(
                f"\n[dim]Composite score: {scored.composite:.1f}  "
                f"(Q {scored.quality:.0f} / S {scored.speed:.0f} / "
                f"C {scored.context:.0f} / Cap {scored.capability:.0f})  "
                f"@ {scored.tokens_per_sec:.1f} tok/s[/]"
            )
    else:
        budget_gb = scored.fit.budget_bytes / GB if scored.fit.budget_bytes else hw.ram_bytes / GB
        console.print(Panel(
            f"  ❌ Won't run on this computer.\n\n"
            f"  {why_doesnt_fit(m, scored.fit, budget_gb)}\n\n"
            f"  Tip: [cyan]modelfit reverse {m.id}[/] shows what hardware would.",
            title="Heads up", border_style="red",
        ))

    if args.expert:
        qt = Table(title="Memory at each compression level (at the max context that fits)",
                    box=box.SIMPLE_HEAD)
        qt.add_column("Compression"); qt.add_column("Plain English")
        qt.add_column("Weights"); qt.add_column("KV Cache")
        qt.add_column("Total"); qt.add_column("Fits?")
        for o in options:
            lab, stars, _ = quant_label(o.quant)
            qt.add_row(
                f"{o.quant}",
                f"{stars} {lab}",
                f"{o.weights_bytes/GB:.2f} GB",
                f"{o.kv_bytes/GB:.2f} GB",
                f"{o.total_bytes/GB:.2f} GB",
                "[green]✓[/]" if o.fits else "[red]✗[/]",
            )
        console.print(qt)
    else:
        console.print("\n[dim]Add [cyan]--expert[/] to see the full quant sweep & raw scores.[/]")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def cmd_inspect(args) -> int:
    hw = detect_hardware()
    if args.json:
        _emit_json(hw.to_dict())
        return 0
    console.print(_hardware_summary(hw, friendly=not args.expert))
    return 0


# ---------------------------------------------------------------------------
# reverse
# ---------------------------------------------------------------------------
def cmd_reverse(args) -> int:
    from modelfit.reverse import recommend_hardware
    m = find_model(args.model)
    if not m:
        console.print(f"[red]Model not found:[/] {args.model}")
        return 2
    rec = recommend_hardware(m, target_tps=args.target_tps,
                              context=args.context or m.context_max,
                              quant=args.quant)
    if args.json:
        _emit_json(rec)
        return 0
    q_label, q_stars, _ = quant_label(args.quant)
    console.print(Panel(
        f"[bold]{m.name}[/]\n"
        f"[dim]Target: {args.target_tps:.0f} tokens/sec @ {rec['context']:,} ctx "
        f"with {args.quant} ({q_label})[/]\n\n"
        f"[bold cyan]You'd need at least:[/]\n"
        f"  💾 [bold]{rec['min_vram_gb']:.0f} GB of VRAM[/] (or Apple unified memory)\n"
        f"     [dim](weights {rec['weights_gb']:.1f} GB + context cache {rec['kv_gb']:.1f} GB + overhead)[/]\n"
        f"  ⚡ [bold]~{rec['min_bandwidth_gbps']:.0f} GB/s memory bandwidth[/]\n"
        f"     [dim](this is what really decides speed — see [cyan]modelfit explain bandwidth[/])[/]",
        title="🔍 What hardware would you need?", border_style="cyan",
    ))
    rt = Table(title="GPUs / chips that would handle it",
                 box=box.SIMPLE_HEAD)
    rt.add_column("Chip")
    rt.add_column("Memory", justify="right")
    rt.add_column("Bandwidth", justify="right")
    rt.add_column("Verdict")
    for g in rec["candidate_gpus"]:
        rt.add_row(g["name"], f"{g['vram_gb']:.0f} GB",
                    f"{g['bandwidth_gbps']:.0f} GB/s",
                    "[green]✓ comfortably[/]" if g["meets"] else "[yellow]marginal[/]")
    console.print(rt)
    return 0


# ---------------------------------------------------------------------------
# throughput — compare inference engines for a model
# ---------------------------------------------------------------------------
def cmd_throughput(args) -> int:
    from modelfit.engines import predict_throughput, explain_choice
    m = find_model(args.model)
    if not m:
        console.print(f"[red]Model not found:[/] {args.model}")
        return 2
    hw = detect_hardware()
    preds = predict_throughput(m, hw, min_context=args.min_context)
    if args.json:
        _emit_json({
            "model": m.to_dict(),
            "hardware": hw.to_dict(),
            "summary": explain_choice(preds),
            "engines": [p.to_dict() for p in preds],
        })
        return 0
    console.print(_hardware_summary(hw, friendly=False))
    console.print(Panel(
        f"[bold]{m.name}[/]  [dim]({m.id})[/]\n"
        f"Predicted tokens/sec across major inference engines.\n"
        f"[dim](Estimates are rough — bandwidth-bound model + engine multipliers.)[/]",
        title="📊 Throughput comparison", border_style="cyan",
    ))
    t = Table(box=box.SIMPLE_HEAD, header_style="bold")
    t.add_column("Engine", style="bold", min_width=22)
    t.add_column("Single user", justify="right")
    t.add_column("At high concurrency", justify="right")
    t.add_column("Platforms", style="dim")
    t.add_column("Why", overflow="fold", min_width=30)
    for p in preds:
        plats = ", ".join(sorted(p.engine.platforms))
        if p.fits:
            su = f"[green]{p.single_user_tps:.0f} tok/s[/]"
            agg = f"[bold]{p.aggregate_tps:.0f} tok/s[/]" if p.aggregate_tps > p.single_user_tps * 2 else f"{p.aggregate_tps:.0f}"
        else:
            su = "[red]—[/]"; agg = "[red]—[/]"
        t.add_row(p.engine.name, su, agg, plats, p.explanation)
    console.print(t)
    console.print(Panel(explain_choice(preds),
                          title="👉 Which engine should you use?", border_style="green"))
    return 0


# ---------------------------------------------------------------------------
# engines — list every known engine with pros / cons / install hint
# ---------------------------------------------------------------------------
def cmd_engines(args) -> int:
    from modelfit.engines import ENGINES, ENGINE_BY_ID
    if args.engine:
        e = ENGINE_BY_ID.get(args.engine)
        if not e:
            console.print(f"[red]Unknown engine:[/] {args.engine}")
            console.print(f"Known: {', '.join(ENGINE_BY_ID.keys())}")
            return 2
        if args.json:
            _emit_json(e.to_dict())
            return 0
        pros = "\n".join(f"  ✅ {p}" for p in e.pros)
        cons = "\n".join(f"  ⚠️  {c}" for c in e.cons)
        console.print(Panel(
            f"[bold]{e.name}[/]\n"
            f"[dim]{e.notes}[/]\n\n"
            f"Platforms: {', '.join(sorted(e.platforms))}\n"
            f"Quant formats: {', '.join(sorted(e.quants))}\n"
            f"Single-request speed (vs. baseline): {e.single_req_multiplier:.2f}×\n"
            f"Concurrency speedup: {e.concurrency_factor:.1f}×\n\n"
            f"[bold]Pros[/]\n{pros}\n\n"
            f"[bold]Cons[/]\n{cons}\n\n"
            f"[bold]Install:[/] [cyan]{e.install_hint}[/]",
            title=f"Engine: {e.id}", border_style="cyan",
        ))
        return 0
    if args.json:
        _emit_json([e.to_dict() for e in ENGINES])
        return 0
    t = Table(title="Inference engines modelfit knows about",
                box=box.SIMPLE_HEAD, header_style="bold")
    t.add_column("ID", style="cyan")
    t.add_column("Name", style="bold")
    t.add_column("Platforms", style="dim")
    t.add_column("Single-req ×", justify="right")
    t.add_column("Concurrency ×", justify="right")
    t.add_column("In one line")
    for e in ENGINES:
        plats = ", ".join(sorted(e.platforms))
        t.add_row(e.id, e.name, plats, f"{e.single_req_multiplier:.2f}",
                   f"{e.concurrency_factor:.1f}", e.notes.split(".")[0])
    console.print(t)
    console.print("\n[dim]Run [cyan]modelfit engines <id>[/] for details on a single engine, "
                  "or [cyan]modelfit throughput <model>[/] to see predicted tok/s on yours.[/]")
    return 0


# ---------------------------------------------------------------------------
# catalog — sync from HuggingFace + show sources
# ---------------------------------------------------------------------------
def cmd_catalog(args) -> int:
    from modelfit.catalog_sync import (
        sync_from_huggingface, cache_path, load_cache,
        list_sources, set_provider_enabled,
    )
    if args.subcommand == "sync":
        if args.json:
            try:
                res = sync_from_huggingface(limit=args.limit, sort=args.sort,
                                             min_downloads=args.min_downloads,
                                             timeout=args.timeout)
                _emit_json(res)
                return 0
            except Exception as e:
                _emit_json({"ok": False, "error": str(e)})
                return 1
        console.print(f"Syncing from HuggingFace (limit={args.limit}, sort={args.sort})…")
        try:
            res = sync_from_huggingface(limit=args.limit, sort=args.sort,
                                         min_downloads=args.min_downloads,
                                         timeout=args.timeout)
        except Exception as e:
            console.print(f"[red]Sync failed:[/] {e}")
            return 1
        console.print(f"[green]✓[/] Fetched {res['fetched']} models  ·  "
                       f"merged {res['merged']} new entries  ·  "
                       f"updated {res['updated']} existing  ·  "
                       f"skipped {res['skipped']} (insufficient metadata)")
        console.print(f"[dim]Cache: {cache_path()}[/]")
        return 0
    if args.subcommand == "sources":
        sources = list_sources()
        if args.json:
            _emit_json(sources)
            return 0
        t = Table(title="Catalog sources", box=box.SIMPLE_HEAD)
        t.add_column("Source"); t.add_column("Enabled"); t.add_column("Last sync")
        t.add_column("Model count", justify="right")
        for s in sources:
            t.add_row(s["name"],
                       "[green]✓[/]" if s["enabled"] else "[dim]·[/]",
                       s.get("last_sync") or "—",
                       str(s.get("count", "—")))
        console.print(t)
        return 0
    if args.subcommand == "clear":
        from modelfit.catalog_sync import clear_cache
        clear_cache()
        console.print("[green]✓[/] Cleared HF cache. Catalog now uses curated 206 entries only.")
        return 0
    if args.subcommand == "enable":
        set_provider_enabled(args.provider, True)
        console.print(f"[green]✓[/] Enabled provider: {args.provider}")
        return 0
    if args.subcommand == "disable":
        set_provider_enabled(args.provider, False)
        console.print(f"[green]✓[/] Disabled provider: {args.provider}")
        return 0
    console.print("[yellow]Unknown catalog subcommand.[/] Try: sync, sources, clear, enable, disable")
    return 1


# ---------------------------------------------------------------------------
# tui / serve / list
# ---------------------------------------------------------------------------
def cmd_tui(args) -> int:
    try:
        from modelfit.tui import run_tui
    except ImportError as e:
        console.print(f"[red]TUI requires textual. Install with[/] [bold]pip install 'modelfit[tui]'[/]")
        console.print(f"[dim]{e}[/]")
        return 1
    return run_tui()


def cmd_serve(args) -> int:
    from modelfit.api import serve
    serve(host=args.host, port=args.port)
    return 0


def cmd_list(args) -> int:
    cat = get_catalog()
    if args.json:
        _emit_json([m.to_dict() for m in cat])
        return 0
    t = Table(title=f"Model catalog ({len(cat)} models)", box=box.SIMPLE_HEAD)
    t.add_column("ID", style="dim")
    t.add_column("Name", style="bold")
    t.add_column("Family", style="cyan")
    t.add_column("Type", style="magenta")
    t.add_column("Params", justify="right")
    t.add_column("Ctx", justify="right")
    t.add_column("Q", justify="right")
    for m in cat:
        active = f" ({m.active_params_b:.0f}A)" if m.active_params_b < m.params_b * 0.95 else ""
        ctx = f"{m.context_max // 1000}K" if m.context_max >= 1000 else str(m.context_max)
        t.add_row(m.id, m.name, m.family, m.type, f"{m.params_b:.1f}B{active}", ctx, f"{m.quality:.0f}")
    console.print(t)
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def _add_filter_args(p: argparse.ArgumentParser):
    p.add_argument("--use-case", "-u", default="balanced",
                   choices=list(USE_CASE_WEIGHTS.keys()),
                   help="What you'll mainly use the model for")
    p.add_argument("--min-context", type=int, default=4096,
                   help="Smallest context length you can live with (in tokens)")
    p.add_argument("--top", type=int, default=15, help="How many models to show")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    p.add_argument("--include-unfit", action="store_true",
                   help="Also show models that won't fit")
    p.add_argument("--family", help="Only this family (Llama, Qwen, Mistral, …)")
    p.add_argument("--type", help="Only this type (chat, instruct, code, reasoning, math, base, vision)")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="Drop anything below this intrinsic quality (0-100)")
    p.add_argument("--expert", action="store_true",
                   help="Show raw scores + per-quant memory math")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelfit",
        description="Auto-detect your hardware, then tell you which LLMs actually run on it — in plain English.",
        epilog="New here? Run: modelfit wizard",
    )
    p.add_argument("--version", action="version", version=f"modelfit {__version__}")
    sub = p.add_subparsers(dest="command")

    p_rank = sub.add_parser("rank", help="Rank all models that fit (default command)")
    _add_filter_args(p_rank)
    p_rank.set_defaults(func=cmd_rank)

    p_rec = sub.add_parser("recommend", help="Pick ONE model + explain why")
    p_rec.add_argument("--use-case", "-u", default="balanced", choices=list(USE_CASE_WEIGHTS.keys()))
    p_rec.add_argument("--min-context", type=int, default=4096)
    p_rec.add_argument("--json", action="store_true")
    p_rec.set_defaults(func=cmd_recommend)

    p_wiz = sub.add_parser("wizard", help="Interactive Q&A for total beginners")
    p_wiz.set_defaults(func=cmd_wizard)

    p_get = sub.add_parser("get", help="Show install commands for a specific model")
    p_get.add_argument("model")
    p_get.add_argument("--quant", choices=QUANT_ORDER + ["F16"],
                        help="Pick a specific compression level (default: best fit)")
    p_get.add_argument("--min-context", type=int, default=4096)
    p_get.add_argument("--json", action="store_true")
    p_get.set_defaults(func=cmd_get)

    p_exp = sub.add_parser("explain",
                            help="Glossary — explain LLM/ML terms in plain English. "
                                  "Covers training, inference, hardware, alignment.")
    p_exp.add_argument("term", nargs="?", help="Term to look up (omit to list all)")
    p_exp.add_argument("--search", "-s", help="Search across all entries")
    p_exp.add_argument("--category", "-c",
                        help="Browse by category (pass 'list' to see them)")
    p_exp.add_argument("--grouped", action="store_true",
                        help="List all entries grouped by category")
    p_exp.add_argument("--limit", type=int, default=25,
                        help="Max search results")
    p_exp.add_argument("--json", action="store_true")
    p_exp.set_defaults(func=cmd_explain)

    p_info = sub.add_parser("info", help="Detailed info for one model")
    p_info.add_argument("model")
    p_info.add_argument("--use-case", "-u", default="balanced", choices=list(USE_CASE_WEIGHTS.keys()))
    p_info.add_argument("--min-context", type=int, default=4096)
    p_info.add_argument("--expert", action="store_true")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    p_insp = sub.add_parser("inspect", help="Show what hardware was detected on this machine")
    p_insp.add_argument("--json", action="store_true")
    p_insp.add_argument("--expert", action="store_true")
    p_insp.set_defaults(func=cmd_inspect)

    p_rev = sub.add_parser("reverse", help="What hardware would I need to run a particular model?")
    p_rev.add_argument("model")
    p_rev.add_argument("--target-tps", type=float, default=30.0,
                        help="How fast you want it to be (tokens/second; default 30)")
    p_rev.add_argument("--context", type=int)
    p_rev.add_argument("--quant", default="Q4_K_M", choices=QUANT_ORDER + ["F16"])
    p_rev.add_argument("--json", action="store_true")
    p_rev.set_defaults(func=cmd_reverse)

    p_tui = sub.add_parser("tui", help="Interactive Textual UI")
    p_tui.set_defaults(func=cmd_tui)

    p_srv = sub.add_parser("serve", help="REST API server (for cluster schedulers)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.set_defaults(func=cmd_serve)

    p_list = sub.add_parser("list", help="Dump the full 206-model catalog")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_thr = sub.add_parser("throughput",
                            help="Compare predicted tok/s across inference engines for a model")
    p_thr.add_argument("model")
    p_thr.add_argument("--min-context", type=int, default=4096)
    p_thr.add_argument("--json", action="store_true")
    p_thr.set_defaults(func=cmd_throughput)

    p_eng = sub.add_parser("engines",
                            help="List known inference engines (or details for one)")
    p_eng.add_argument("engine", nargs="?", help="Engine id (e.g. vllm, llama.cpp)")
    p_eng.add_argument("--json", action="store_true")
    p_eng.set_defaults(func=cmd_engines)

    p_cat = sub.add_parser("catalog",
                            help="Manage catalog sources (HuggingFace sync, etc.)")
    p_cat_sub = p_cat.add_subparsers(dest="subcommand", required=True)
    p_cat_sync = p_cat_sub.add_parser("sync", help="Pull latest models from HuggingFace")
    p_cat_sync.add_argument("--limit", type=int, default=100,
                              help="How many models to fetch")
    p_cat_sync.add_argument("--sort", default="downloads",
                              choices=["downloads", "likes", "trending"],
                              help="HF sort order")
    p_cat_sync.add_argument("--min-downloads", type=int, default=1000,
                              help="Skip models with fewer downloads than this")
    p_cat_sync.add_argument("--timeout", type=int, default=15)
    p_cat_sync.add_argument("--json", action="store_true")
    p_cat_sources = p_cat_sub.add_parser("sources", help="List catalog sources")
    p_cat_sources.add_argument("--json", action="store_true")
    p_cat_sub.add_parser("clear", help="Clear the HF cache (revert to curated 206)")
    p_cat_enable = p_cat_sub.add_parser("enable", help="Enable a provider")
    p_cat_enable.add_argument("provider", choices=["huggingface"])
    p_cat_disable = p_cat_sub.add_parser("disable", help="Disable a provider")
    p_cat_disable.add_argument("provider", choices=["huggingface"])
    p_cat.set_defaults(func=cmd_catalog)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    _root_flags = {"--version", "-h", "--help"}
    if not argv or (argv[0].startswith("-") and argv[0] not in _root_flags):
        argv = ["rank"] + list(argv)
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
