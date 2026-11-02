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


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "experiments.json")
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    with open(src, encoding="utf-8") as f:
        res = json.load(f)
    performance_figure(res, os.path.join(out, "performance_results.png"))
    architecture_figure(os.path.join(out, "system_architecture.png"))
    print("wrote", os.path.join(out, "performance_results.png"))
    print("wrote", os.path.join(out, "system_architecture.png"))


if __name__ == "__main__":
    main()
