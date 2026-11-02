"""Figures for the IEEE paper (single-column width, 300 dpi).

    python paper/make_figures.py            # reads paper/results/experiments.json
writes paper/figures/system_architecture.png and paper/figures/performance_results.png.
Requires matplotlib (not a runtime dependency of CloudRAG).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMN_IN = 3.5

# Categorical slots (validated: CVD separation and normal-vision floor pass on white). Colour follows
# the system, never its rank. Contrast vs white is low for slots 3-5, so every mark also has a
# legend entry and the numbers are in the paper's tables.
SYSTEMS = ["cloudrag", "agentic-rag", "graphrag", "advanced-rag", "vanilla-rag"]
COLOR = {"cloudrag": "#2a78d6", "agentic-rag": "#eb6834", "graphrag": "#1baf7a",
         "advanced-rag": "#eda100", "vanilla-rag": "#e87ba4"}
LABEL = {"cloudrag": "CloudRAG (proposed)", "agentic-rag": "Agentic RAG", "graphrag": "GraphRAG",
         "advanced-rag": "Advanced RAG", "vanilla-rag": "Vanilla RAG"}
QTYPES = ["single", "relational", "multihop", "multipart", "global"]
QLABEL = {"single": "Single-hop", "relational": "Relational", "multihop": "Multi-hop",
          "multipart": "Multi-part", "global": "Aggregation"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "savefig.dpi": 300,
})


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


def performance_figure(res: dict, path: str) -> None:
    S = res["systems"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN_IN, 4.3), gridspec_kw={"height_ratios": [1.05, 1]})

    # (a) accuracy by question type, grouped bars with std error bars
    n = len(SYSTEMS)
    width = 0.8 / n
    for i, s in enumerate(SYSTEMS):
        xs = [q + (i - (n - 1) / 2) * width for q in range(len(QTYPES))]
        means = [S[s]["by_type"][t]["correct"]["mean"] * 100 for t in QTYPES]
        stds = [S[s]["by_type"][t]["correct"]["std"] * 100 for t in QTYPES]
        ax1.bar(xs, means, width * 0.86, color=COLOR[s], edgecolor="white", linewidth=0.4,
                hatch="////" if s == "cloudrag" else None, label=LABEL[s], zorder=2)
        ax1.errorbar(xs, means, yerr=stds, fmt="none", ecolor=INK2, elinewidth=0.5, capsize=1, zorder=3)
    ax1.set_xticks(range(len(QTYPES)), [QLABEL[t] for t in QTYPES])
    ax1.set_ylim(0, 108)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("Answer accuracy (%)")
    ax1.set_title("(a) Accuracy by question type", loc="left", color=INK)
    _style(ax1)

    # (b) quality vs latency trade-off
    for s in SYSTEMS:
        x, y = S[s]["p50_ms"], S[s]["correct"]
        ax2.errorbar(x["mean"], y["mean"] * 100, xerr=x["std"], yerr=y["std"] * 100, fmt="o", ms=5,
                     color=COLOR[s], mec="white", mew=0.8, ecolor=COLOR[s], elinewidth=0.6, capsize=1.5, zorder=3)
        ax2.annotate(LABEL[s], (x["mean"], y["mean"] * 100), xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax2.set_xscale("log")
    lo = min(S[s]["p50_ms"]["mean"] - S[s]["p50_ms"]["std"] for s in SYSTEMS)
    hi = max(S[s]["p50_ms"]["mean"] + S[s]["p50_ms"]["std"] for s in SYSTEMS)
    ax2.set_xlim(max(0.5, lo * 0.6), hi * 1.6)
    ax2.set_ylim(40, 108)
    ax2.set_xlabel("Median latency per query, p50 (ms, log scale)")
    ax2.set_ylabel("Answer accuracy (%)")
    ax2.set_title("(b) Accuracy vs. latency (mean $\\pm$ std)", loc="left", color=INK)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if max(0.5, lo * 0.6) <= t <= hi * 1.6]
    ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    _style(ax2)
    ax2.grid(axis="x", color=GRID, linewidth=0.5)

    handles = [Patch(facecolor=COLOR[s], edgecolor="white", hatch="////" if s == "cloudrag" else None,
                     label=LABEL[s]) for s in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def architecture_figure(path: str) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 106)
    ax.axis("off")
    neutral, accent, store = "#f0efec", "#cde2fb", "#f7f6f2"

    def box(cx, cy, w, h, title, sub="", fill=neutral, edge=INK2, bold=True):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    facecolor=fill, edgecolor=edge, linewidth=0.6, zorder=2))
        ax.text(cx, cy + (2.1 if sub else 0), title, ha="center", va="center", fontsize=6.6,
                fontweight="bold" if bold else "normal", zorder=3)
        if sub:
            ax.text(cx, cy - 2.6, sub, ha="center", va="center", fontsize=5.4, color=INK2, zorder=3, linespacing=1.0)

    def arrow(x1, y1, x2, y2, dashed=False, color=INK2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, linewidth=0.7,
                                     color=color, linestyle=(0, (3, 2)) if dashed else "-", zorder=1,
                                     shrinkA=0, shrinkB=0))

    box(50, 100, 46, 7, "Tenant clients", "JWT bearer token", fill="white", bold=False)
    box(50, 85, 86, 9, "API Gateway", "authentication · per-tenant rate limit · injection filter · audit log")
    box(16, 65, 27, 12, "Ingestion", "PII redaction · signing\npoisoning scan")
    box(50, 65, 27, 12, "Retrieval", "complexity router · RRF\nintegrity + ACL check", fill=accent, edge="#2a78d6")
    box(84, 65, 25, 12, "Generation", "grounded answer\ncircuit breaker")
    box(14, 44, 22, 8, "Event stream", "Redis Streams", fill=store)
    box(39.5, 44, 17, 8, "Embedding")
    box(61, 44, 17, 8, "Reranker")
    box(85, 44, 23, 8, "LLM (Claude)", "extractive fallback", fill=store)
    box(39, 22, 72, 13, "Indexing shards 0 … N (tenant-hashed)",
        "per tenant: HNSW · BM25 · knowledge graph · ACLs", fill=store)
    box(50, 5, 96, 6, "Observability: Prometheus metrics · online quality · security events · traces",
        fill="white", bold=False)

    arrow(50, 96.4, 50, 89.6)                     # client -> gateway
    arrow(30, 80.4, 22, 71.2, dashed=True)        # gateway -> ingestion
    arrow(50, 80.4, 50, 71.2)                     # gateway -> retrieval
    arrow(70, 80.4, 80, 71.2)                     # gateway -> generation
    arrow(14, 58.8, 14, 48.2, dashed=True)        # ingestion -> stream
    arrow(14, 39.8, 14, 28.7, dashed=True)        # stream -> indexing
    arrow(44, 58.8, 41, 48.2)                     # retrieval -> embedding
    arrow(56, 58.8, 60, 48.2)                     # retrieval -> reranker
    arrow(50, 58.8, 50, 28.7)                     # retrieval -> indexing
    arrow(85, 58.8, 85, 48.2)                     # generation -> llm
    arrow(33, 28.7, 37, 39.8, dashed=True)        # indexing -> embedding (embed chunks)

    ax.legend(handles=[matplotlib.lines.Line2D([], [], color=INK2, lw=0.8, label="query path"),
                       matplotlib.lines.Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="ingestion path"),
                       Patch(facecolor=accent, edgecolor="#2a78d6", lw=0.6, label="adaptive routing")],
              loc="lower right", bbox_to_anchor=(1.0, 0.24), frameon=False, fontsize=5.6, handlelength=1.8)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# -- wider charts for the repository README -------------------------------------------------------------
# Router strategies share the colour of the baseline system that implements the same pathway.
STRATEGY_COLOR = {"naive": COLOR["vanilla-rag"], "advanced": COLOR["advanced-rag"], "graph": COLOR["graphrag"],
                  "agentic": COLOR["agentic-rag"]}
README_W = 7.0


def _hbars(ax, labels, values, errors=None, color=COLOR["cloudrag"], fmt="{:.1f}"):
    """Horizontal bars, first label on top, value printed at the end of each bar."""
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, values, height=0.62, color=color, edgecolor="white", linewidth=0.5,
            xerr=errors, error_kw={"ecolor": INK2, "elinewidth": 0.6, "capsize": 1.5}, zorder=2)
    for y, v, e in zip(ys, values, errors or [0] * len(values)):
        ax.text(v + e + max(values) * 0.02, y, fmt.format(v), va="center", fontsize=6.8, color=INK)
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, max(v + e for v, e in zip(values, errors or [0] * len(values))) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def throughput_latency_figure(res: dict, path: str) -> None:
    S = res["systems"]
    order = SYSTEMS
    labels = [LABEL[s] for s in order]
    colors = [COLOR[s] for s in order]
    fig, axes = plt.subplots(1, 3, figsize=(README_W, 2.2), sharey=True)
    for ax, key, title, fmt in ((axes[0], "throughput_qps", "Throughput, 8 clients (QPS)", "{:.0f}"),
                                (axes[1], "p50_ms", "Median latency p50 (ms)", "{:.1f}"),
                                (axes[2], "cpu_ms_per_query", "CPU time per query (ms)", "{:.1f}")):
        _hbars(ax, labels, [S[s][key]["mean"] for s in order], [S[s][key]["std"] for s in order], colors, fmt)
        ax.set_title(title, loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def communication_figure(res: dict, path: str) -> None:
    comm = res["overhead"]["communication"]["query"]
    labels = [LABEL[s] for s in SYSTEMS]
    colors = [COLOR[s] for s in SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(README_W, 2.0), sharey=True)
    _hbars(axes[0], labels, [comm[s]["internal_requests_per_query"] for s in SYSTEMS], None, colors, "{:.1f}")
    axes[0].set_title("Service-to-service requests per query", loc="left", color=INK)
    _hbars(axes[1], labels, [comm[s]["internal_kb_per_query"] for s in SYSTEMS], None, colors, "{:.0f} KB")
    axes[1].set_title("Internal payload per query (KB)", loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def routing_figure(res: dict, path: str) -> None:
    routing = res["routing"]
    strategies = ["naive", "advanced", "graph", "agentic"]
    fig, ax = plt.subplots(figsize=(README_W, 2.0))
    ys = list(range(len(QTYPES)))[::-1]
    for y, q in zip(ys, QTYPES):
        counts = routing.get(q, {})
        total = sum(counts.values()) or 1
        left = 0.0
        for st in strategies:
            share = counts.get(st, 0) / total * 100
            if share:
                ax.barh(y, share, left=left, height=0.62, color=STRATEGY_COLOR[st], edgecolor="white", linewidth=1.0)
                if share >= 8:
                    ax.text(left + share / 2, y, f"{st} {share:.0f}%", ha="center", va="center", fontsize=6.6,
                            color="#0b0b0b" if st == "advanced" else "white")
                left += share
    ax.set_yticks(ys, [QLABEL[q] for q in QTYPES])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of questions routed to each pathway (%), all seeds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=STRATEGY_COLOR[s], label=s) for s in strategies], ncol=4, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def security_overhead_figure(res: dict, path: str) -> None:
    so = res["overhead"]["security"]
    rows = [("PII redaction (per doc)", "pii_redaction_us_per_doc"),
            ("Near-duplicate / hidden text (per doc)", "poisoning_doc_signals_us_per_doc"),
            ("Poisoning scan (per chunk)", "poisoning_scan_us_per_chunk"),
            ("Provenance signing (per chunk)", "provenance_sign_us_per_chunk"),
            ("Provenance verification (per chunk)", "provenance_verify_us_per_chunk"),
            ("Query injection filter (per query)", "query_guard_us_per_query")]
    fig, ax = plt.subplots(figsize=(README_W, 2.1))
    _hbars(ax, [r[0] for r in rows], [so[k]["mean"] for _, k in rows], [so[k]["std"] for _, k in rows],
           COLOR["cloudrag"], "{:.0f} µs")
    ax.set_title(f"Cost of each security control (µs). Total: {so['ingest_security_share']['mean'] * 100:.0f}% of "
                 f"ingestion time, {so['query_security_share_of_p50']['mean'] * 100:.1f}% of query latency",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "experiments.json")
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    with open(src, encoding="utf-8") as f:
        res = json.load(f)
    figures = {
        "performance_results.png": lambda p: performance_figure(res, p),
        "system_architecture.png": architecture_figure,
        "throughput_latency.png": lambda p: throughput_latency_figure(res, p),
        "communication_overhead.png": lambda p: communication_figure(res, p),
        "router_decisions.png": lambda p: routing_figure(res, p),
        "security_overhead.png": lambda p: security_overhead_figure(res, p),
    }
    for name, draw in figures.items():
        draw(os.path.join(out, name))
        print("wrote", os.path.join(out, name))


if __name__ == "__main__":
    main()
