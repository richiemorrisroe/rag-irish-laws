"""
Irish Statutes Eval App — Shiny Core
Run: uv run shiny run --reload irish_statutes/evals-app/app.py
"""

import json
import sys
from pathlib import Path

from shiny import App, reactive, render, ui
from shinyswatch import theme

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.db import (
    create_eval_table,
    load_all_results,
    load_unrated,
    save_eval_result,
    update_rating,
)
from indexer.eval_queries import query_llm, setup_llm
from indexer.utils import setup_logger
from indexer.vstore import get_index_from_database

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = setup_logger(__file__)

# ---------------------------------------------------------------------------
# Bootstrap (runs once at startup)
# ---------------------------------------------------------------------------
create_eval_table()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = setup_llm()
index = get_index_from_database()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Evaluate",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Unrated Queries"),
                ui.output_ui("unrated_list"),
                ui.hr(),
                ui.h5("Custom Query"),
                ui.input_text_area(
                    "custom_query",
                    label=None,
                    placeholder="Type a custom query…",
                    rows=4,
                    width="100%",
                ),
                ui.input_action_button("run_custom", "Run", class_="btn-primary w-100"),
                width=320,
            ),
            ui.card(
                ui.card_header(ui.output_text("current_query_header")),
                ui.output_ui("answer_panel"),
            ),
        ),
    ),
    ui.nav_panel(
        "Results",
        ui.card(
            ui.card_header("All Rated Results"),
            ui.output_data_frame("results_table"),
        ),
    ),
    title="Irish Statute Eval",
    theme=theme.morph,
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    # Holds the id of the currently displayed eval_results row (None = nothing selected)
    current_id = reactive.Value(None)

    # Invalidate signal — bump to force DB re-reads
    refresh_trigger = reactive.Value(0)

    def bump():
        refresh_trigger.set(refresh_trigger.get() + 1)

    # ------------------------------------------------------------------
    # Reactive data helpers
    # ------------------------------------------------------------------
    @reactive.calc
    def unrated_df():
        refresh_trigger.get()  # subscribe
        return load_unrated()

    @reactive.calc
    def all_results_df():
        refresh_trigger.get()
        return load_all_results()

    @reactive.calc
    def current_row():
        rid = current_id.get()
        if rid is None:
            return None
        df = all_results_df()
        rows = df[df["id"] == rid]
        return rows.iloc[0] if not rows.empty else None

    # ------------------------------------------------------------------
    # Sidebar: unrated list
    # ------------------------------------------------------------------
    @output
    @render.ui
    def unrated_list():
        df = unrated_df()
        if df.empty:
            return ui.p("No unrated queries.", class_="text-muted")

        items = []
        for _, row in df.iterrows():
            rid = int(row["id"])
            short = str(row["query_text"])[:80]
            is_active = rid == current_id.get()
            btn = ui.input_action_button(
                f"sel_{rid}",
                short,
                class_=f"btn btn-{'primary' if is_active else 'outline-secondary'} btn-sm text-start w-100 mb-1",
            )
            items.append(btn)

        # Wire up click handlers dynamically
        for _, row in df.iterrows():
            rid = int(row["id"])
            _make_select_handler(rid)

        return ui.div(*items)

    def _make_select_handler(rid):
        @reactive.effect
        @reactive.event(getattr(input, f"sel_{rid}"))
        def _():
            current_id.set(rid)

    # Auto-select first unrated if nothing selected
    @reactive.effect
    def _auto_select():
        if current_id.get() is None:
            df = unrated_df()
            if not df.empty:
                current_id.set(int(df.iloc[0]["id"]))

    # ------------------------------------------------------------------
    # Main panel
    # ------------------------------------------------------------------
    @output
    @render.text
    def current_query_header():
        row = current_row()
        if row is None:
            return "No query selected"
        return str(row["query_text"])

    @output
    @render.ui
    def answer_panel():
        row = current_row()
        if row is None:
            return ui.p("Select a query from the sidebar or run a custom query.", class_="text-muted")

        rid = int(row["id"])
        rating = str(row["rating"])

        # Parse source nodes JSON
        try:
            nodes = json.loads(row["source_nodes"]) if row["source_nodes"] else []
        except Exception:
            nodes = []

        try:
            scores = json.loads(row["scores"]) if row["scores"] else []
        except Exception:
            scores = []

        scores_md = "\n".join(f"- {s:.4f}" for s in scores) if scores else "_None_"
        sources_md = "\n\n---\n\n".join(
            f"**Score:** {n.get('score', 'N/A')}\n\n{n.get('text', '')}" for n in nodes
        )

        rating_badge = {
            "good": ui.span("Rated: Good", class_="badge bg-success"),
            "bad": ui.span("Rated: Bad", class_="badge bg-danger"),
        }.get(rating, ui.span("Unrated", class_="badge bg-secondary"))

        return ui.div(
            # Answer
            ui.h5("Answer"),
            ui.markdown(str(row["llm_response"] or "_No response_")),
            ui.hr(),
            # Scores
            ui.h5("Similarity Scores"),
            ui.markdown(scores_md),
            ui.hr(),
            # Sources (collapsed)
            ui.h5("Source Documents"),
            ui.tags.details(
                ui.tags.summary("Show/hide sources"),
                ui.markdown(sources_md),
            ),
            ui.hr(),
            # Rating
            ui.div(
                rating_badge,
                ui.span(" "),
                ui.input_action_button(
                    f"rate_good_{rid}", "Good answer", class_="btn btn-success btn-sm"
                ),
                ui.span(" "),
                ui.input_action_button(
                    f"rate_bad_{rid}", "Bad answer", class_="btn btn-danger btn-sm"
                ),
                class_="d-flex align-items-center gap-2",
            ),
        )

    # Wire rating buttons dynamically when the row changes
    @reactive.effect
    def _wire_rating_buttons():
        row = current_row()
        if row is None:
            return
        rid = int(row["id"])
        _make_rating_handler(rid, "good")
        _make_rating_handler(rid, "bad")

    def _make_rating_handler(rid, rating):
        btn_id = f"rate_{rating}_{rid}"

        @reactive.effect
        @reactive.event(getattr(input, btn_id))
        def _():
            update_rating(rid, rating)
            bump()
            # Advance to next unrated
            df = unrated_df()
            df2 = df[df["id"] != rid]
            if not df2.empty:
                current_id.set(int(df2.iloc[0]["id"]))
            else:
                current_id.set(None)

    # ------------------------------------------------------------------
    # Custom query
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.run_custom)
    def _run_custom():
        q = input.custom_query().strip()
        if not q:
            return
        result = query_llm(index, q, top_k=5, logger=logger)
        rid = save_eval_result(
            query=q,
            response=result.response,
            scores=result.scores,
            source_nodes=result.source_nodes,
            top_k=result.top_k,
        )
        bump()
        current_id.set(rid)

    # ------------------------------------------------------------------
    # Results tab
    # ------------------------------------------------------------------
    @output
    @render.data_frame
    def results_table():
        df = all_results_df()
        cols = ["id", "query_text", "rating", "top_k", "created_at", "rated_at"]
        return render.DataGrid(df[[c for c in cols if c in df.columns]], width="100%")


# ---------------------------------------------------------------------------
app = App(app_ui, server)
