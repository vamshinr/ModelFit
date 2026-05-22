"""Interactive Textual TUI.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Hardware: [CPU/RAM/GPU summary]      [Use case ▼] [Filter…] │
    ├──────────────────────────────────────────┬──────────────────┤
    │ Ranked table (focusable, ↑↓ to select)   │  Details panel   │
    │                                          │  for selection   │
    └──────────────────────────────────────────┴──────────────────┘
    Keys:  q quit · u cycle use-case · / filter · enter inspect · r refresh
"""
from __future__ import annotations

import sys
from typing import List, Optional

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Header, Footer, DataTable, Static, Input, Label
    from textual.binding import Binding
    from textual.reactive import reactive
except ImportError:
    App = None  # type: ignore

from modelfit.hardware import detect_hardware, HardwareProfile
from modelfit.scoring import rank_all, score_model, ScoredModel, USE_CASE_WEIGHTS
from modelfit.models import get_catalog, find_model
from modelfit.quantization import GB

USE_CASES = list(USE_CASE_WEIGHTS.keys())


def _hardware_line(hw: HardwareProfile) -> str:
    parts = [f"CPU: {hw.cpu_name} ({hw.cpu_cores}c/{hw.cpu_threads}t)",
             f"RAM: {hw.ram_bytes / GB:.0f}GB"]
    if hw.gpus:
        for g in hw.gpus:
            parts.append(f"GPU: {g.name} {g.vram_bytes/GB:.0f}GB")
    else:
        parts.append("GPU: none")
    return "  ·  ".join(parts)


if App is not None:

    class DetailsPanel(Static):
        def update_for(self, scored: Optional[ScoredModel], hw: HardwareProfile):
            if scored is None:
                self.update("[dim]Select a model to inspect…[/]")
                return
            m = scored.model
            f = scored.fit
            lines = [
                f"[b]{m.name}[/b]  [dim]({m.id})[/dim]",
                f"Family: {m.family}   Type: {m.type}",
                f"Params: {m.params_b:.1f}B  (active {m.active_params_b:.1f}B)",
                f"Context (trained): {m.context_max:,} tokens",
                f"Quality (F16): {m.quality:.0f}/100",
                "",
                "[b]Fit[/b]",
            ]
            if f.fits:
                lines.append(f"[green]✓[/] {f.quant} @ {f.context:,} ctx → {f.total_bytes/GB:.1f}GB on {f.tier}")
                lines.append(f"   weights {f.weights_bytes/GB:.2f}GB · kv {f.kv_bytes/GB:.2f}GB · "
                             f"budget {f.budget_bytes/GB:.1f}GB")
                lines.append(f"   ~{scored.tokens_per_sec:.1f} tokens/sec")
            else:
                lines.append(f"[red]✗[/] {f.reason}")
            lines += [
                "",
                "[b]Score breakdown[/b]",
                f"  Quality:    {scored.quality:5.1f}   (weight {scored.weights[0]})",
                f"  Speed:      {scored.speed:5.1f}   (weight {scored.weights[1]})",
                f"  Context:    {scored.context:5.1f}   (weight {scored.weights[2]})",
                f"  Capability: {scored.capability:5.1f}   (weight {scored.weights[3]})",
                "",
                f"[b]Composite: {scored.composite:.1f}[/b]",
                "",
                "[dim]Tags: " + (", ".join(m.tags) or "—") + "[/]",
            ]
            self.update("\n".join(lines))

    class ModelFitApp(App):
        CSS = """
        Screen { layout: vertical; }
        #topbar { height: 5; padding: 0 1; }
        #main   { height: 1fr; }
        #table  { width: 2fr; }
        #details{ width: 1fr; padding: 1 2; border-left: solid $accent; }
        #search { dock: bottom; display: none; }
        #search.visible { display: block; }
        DataTable > .datatable--header { background: $boost; }
        """
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("u", "cycle_use_case", "Use case"),
            Binding("slash", "open_search", "Filter"),
            Binding("escape", "close_search", show=False),
            Binding("r", "refresh", "Refresh"),
            Binding("c", "cycle_min_context", "Min ctx"),
            Binding("?", "help", "Help"),
        ]

        use_case = reactive("balanced")
        filter_text = reactive("")
        min_ctx = reactive(2048)

        def __init__(self):
            super().__init__()
            self.hw = detect_hardware()
            self.ranked: List[ScoredModel] = []
            self.selected: Optional[ScoredModel] = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Static(_hardware_line(self.hw), id="topbar")
            with Horizontal(id="main"):
                yield DataTable(id="table", cursor_type="row")
                yield DetailsPanel(id="details")
            yield Input(placeholder="Filter by name / family / tag…  (Esc to close)",
                         id="search")
            yield Footer()

        def on_mount(self):
            tbl = self.query_one(DataTable)
            tbl.add_columns("#", "Score", "Model", "Type", "Quant", "Ctx", "Mem", "tok/s")
            self._recompute()
            self._set_title()

        def _set_title(self):
            self.title = "modelfit"
            self.sub_title = f"use-case={self.use_case}  min-ctx={self.min_ctx:,}  filter={self.filter_text or '—'}"

        def _recompute(self):
            self.ranked = rank_all(self.hw, use_case=self.use_case, min_context=self.min_ctx)
            if self.filter_text:
                q = self.filter_text.lower()
                self.ranked = [s for s in self.ranked
                                if q in s.model.name.lower()
                                or q in s.model.id.lower()
                                or q in s.model.family.lower()
                                or any(q in t for t in s.model.tags)]
            tbl = self.query_one(DataTable)
            tbl.clear()
            for i, s in enumerate(self.ranked, 1):
                ctx = f"{s.fit.context//1000}K" if s.fit.context and s.fit.context >= 1000 else str(s.fit.context or 0)
                tbl.add_row(
                    str(i),
                    f"{s.composite:.1f}",
                    s.model.name,
                    s.model.type,
                    s.fit.quant or "-",
                    ctx,
                    f"{s.fit.total_bytes/GB:.1f}G",
                    f"{s.tokens_per_sec:.1f}",
                )
            self._set_title()
            self.query_one(DetailsPanel).update_for(self.ranked[0] if self.ranked else None, self.hw)

        def on_data_table_row_highlighted(self, event):
            idx = event.cursor_row
            if 0 <= idx < len(self.ranked):
                self.query_one(DetailsPanel).update_for(self.ranked[idx], self.hw)

        def action_cycle_use_case(self):
            i = USE_CASES.index(self.use_case)
            self.use_case = USE_CASES[(i + 1) % len(USE_CASES)]
            self._recompute()

        def action_cycle_min_context(self):
            options = [2048, 4096, 8192, 16384, 32768, 131072]
            i = options.index(self.min_ctx) if self.min_ctx in options else 0
            self.min_ctx = options[(i + 1) % len(options)]
            self._recompute()

        def action_open_search(self):
            inp = self.query_one(Input)
            inp.add_class("visible")
            inp.value = self.filter_text
            inp.focus()

        def action_close_search(self):
            inp = self.query_one(Input)
            inp.remove_class("visible")
            self.query_one(DataTable).focus()

        def action_refresh(self):
            self.hw = detect_hardware()
            self.query_one("#topbar", Static).update(_hardware_line(self.hw))
            self._recompute()

        def on_input_submitted(self, event: Input.Submitted):
            self.filter_text = event.value.strip()
            self.action_close_search()
            self._recompute()


def run_tui() -> int:
    if App is None:
        print("textual is not installed. Install with: pip install 'modelfit[tui]'",
              file=sys.stderr)
        return 1
    app = ModelFitApp()
    app.run()
    return 0
