"""Paper results: repeated experiments over several seeds, aggregated to mean +/- std, written
as JSON, Markdown and LaTeX tables that drop into the IEEE template (booktabs style).

    cloudrag paper --seeds 7,11,23,42,2026 --out paper
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict

from ..config import Settings
from .harness import SYSTEMS, WATTS_PER_BUSY_CORE, Harness
from .overhead import communication_overhead, security_overhead

LABELS = {"vanilla-rag": "Vanilla RAG", "advanced-rag": "Advanced RAG", "graphrag": "GraphRAG",
          "agentic-rag": "Agentic RAG", "cloudrag": "CloudRAG (proposed)"}
QTYPES = ["single", "relational", "multihop", "multipart", "global"]
QTYPE_LABELS = {"single": "Single-hop", "relational": "Relational", "multihop": "Multi-hop",
                "multipart": "Multi-part", "global": "Aggregation"}


def _stat(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "values": xs}


def testbed() -> dict:
    cpu = platform.processor()
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        elif os.path.exists("/proc/cpuinfo"):
            cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    info = {"cpu": cpu, "logical_cpus": os.cpu_count(), "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version()}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    import fastapi
    import numpy
    import pydantic

    info["libraries"] = {"numpy": numpy.__version__, "fastapi": fastapi.__version__, "pydantic": pydantic.__version__}
    return info


def run_experiments(seeds: list[int], tenants: list[str], n_filler: int, top_k: int = 5,
                    settings: Settings | None = None, log=print) -> dict:
    per_seed = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        h = Harness(tenants=tenants, seed=seed, n_filler=n_filler, top_k=top_k, settings=settings)
        suites = ["quality", "throughput", "incremental", "security"] + (["load"] if i == 0 else [])
        h.run(suites=suites)
        h.report["overhead"] = {"security": security_overhead(h)}
        if i == 0:
            h.report["overhead"]["communication"] = communication_overhead(h)
        per_seed.append(h.report)
        o = h.report["quality"]["cloudrag"]["overall"]
        log(f"seed {seed}: cloudrag correctness {o['correct']:.3f}, p50 {o['p50_ms']:.1f} ms ({time.time() - t0:.0f}s)")
        h.platform.close()
    return aggregate(per_seed, seeds)


def aggregate(per_seed: list[dict], seeds: list[int]) -> dict:
    first = per_seed[0]
    metrics = ["correct", "f1", "precision_at_k", "ctx_recall", "ctx_precision", "hit", "mrr", "faithfulness",
               "answer_relevance", "abstained", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "cost_per_query_usd",
               "context_chunks", "sequential_qps", "cpu_ms_per_query", "energy_j_per_query"]
    systems = {}
    for s in SYSTEMS:
        row = {m: _stat([r["quality"][s]["overall"][m] for r in per_seed]) for m in metrics}
        row["throughput_qps"] = _stat([r["throughput"][s]["qps"] for r in per_seed])
        row["by_type"] = {t: {m: _stat([r["quality"][s]["by_type"][t][m] for r in per_seed])
                              for m in ("correct", "ctx_recall", "p50_ms")} for t in QTYPES}
        systems[s] = row

    routing: dict = defaultdict(Counter)
    for r in per_seed:
        for t, c in r["routing"].items():
            routing[t].update(c)

    sec_keys = ["rbac_leakage_rate", "rbac_authorized_recall", "cross_tenant_leakage_rate", "pii_residual_rate",
                "injection_block_rate", "benign_false_positive_rate"]
    security = {k: _stat([r["security"][k] for r in per_seed]) for k in sec_keys}
    poison_outcomes: dict = defaultdict(Counter)
    attacks: dict = defaultdict(Counter)
    probes = Counter()
    for r in per_seed:
        s = r["security"]
        probes.update(s["probes"])
        for kind, c in s["poisoning_ingest_outcomes"].items():
            poison_outcomes[kind].update(c)
        for key, v in s["poisoning_attacks"].items():
            attacks[key]["n"] += v["n"]
            attacks[key]["in_context"] += round(v["poison_in_context_rate"] * v["n"])
            attacks[key]["success"] += round(v["attack_success_rate"] * v["n"])
    security["probes"] = dict(probes)
    security["pii_values_checked"] = sum(r["security"]["pii_values_checked"] for r in per_seed)
    security["audit_chains_valid"] = all(r["security"]["audit_chain"]["ok"] for r in per_seed)
    security["audit_records"] = sum(r["security"]["audit_chain"]["records"] for r in per_seed)
    security["poisoning_ingest_outcomes"] = {k: dict(v) for k, v in poison_outcomes.items()}
    security["poisoning_attacks"] = {k: dict(v) for k, v in sorted(attacks.items())}

    sec_over = {k: _stat([r["overhead"]["security"][k] for r in per_seed]) for k in first["overhead"]["security"]}
    ing = {k: _stat([r["ingestion"][k] for r in per_seed])
           for k in ("documents", "chunks", "docs_per_s", "chunks_per_s", "index_complete_s", "index_bytes",
                     "index_bytes_per_chunk", "process_rss_mb") if k in first["ingestion"]}
    incr = {k: _stat([r["incremental"][k] for r in per_seed])
            for k in ("freshness_rate", "stale_answer_rate", "update_lag_p50_ms", "update_lag_p95_ms")}
    qcount = Counter()
    for tc_type, v in first["quality"]["cloudrag"]["by_type"].items():
        qcount[tc_type] = v["n"]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "seeds": seeds,
        "config": first["config"] | {"watts_per_busy_core": WATTS_PER_BUSY_CORE},
        "testbed": testbed(),
        "dataset": {"questions_per_seed": dict(qcount), "questions_total_per_seed": sum(qcount.values()), **ing},
        "systems": systems,
        "routing": {t: dict(c) for t, c in routing.items()},
        "security": security,
        "incremental": incr,
        "load": first.get("load", {}),
        "overhead": {"security": sec_over, "communication": first["overhead"]["communication"]},
        "per_seed": per_seed,
    }


# -- LaTeX ---------------------------------------------------------------------------------------------
def _pm(st: dict, scale: float = 1.0, digits: int = 1) -> str:
    return f"{st['mean'] * scale:.{digits}f}$\\pm${st['std'] * scale:.{digits}f}"


def _table(label: str, caption: str, cols: str, header: str, rows: list[str], small: bool = True) -> str:
    size = "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n" if small else ""
    body = "\n".join(rows)
    return (f"\\begin{{table}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n{size}"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{table}}\n")


def latex_tables(res: dict) -> dict[str, str]:
    S, cfg, n = res["systems"], res["config"], len(res["seeds"])
    best = max(S, key=lambda s: S[s]["correct"]["mean"])
    bold = lambda s, txt: f"\\textbf{{{txt}}}" if s == best else txt  # noqa: E731
    t = {}

    t["tab_performance"] = _table(
        "tab:performance", f"Performance Comparison (mean$\\pm$std over {n} seeds)", "lcccc",
        "Method & Accuracy (\\%) & Latency p50 (ms) & Throughput (QPS) & F1",
        [f"{LABELS[s]} & {bold(s, _pm(v['correct'], 100))} & {_pm(v['p50_ms'])} & {_pm(v['throughput_qps'])} & "
         f"{_pm(v['f1'], 1, 3)} \\\\" for s, v in S.items()])

    t["tab_retrieval"] = _table(
        "tab:retrieval", "Retrieval and Generation Quality (mean over seeds)", "lcccccc",
        "Method & Hit@k & MRR & Ctx-P & Ctx-R & Faith. & Rel.",
        [f"{LABELS[s]} & {v['hit']['mean']:.3f} & {v['mrr']['mean']:.3f} & {v['ctx_precision']['mean']:.3f} & "
         f"{v['ctx_recall']['mean']:.3f} & {v['faithfulness']['mean']:.3f} & {v['answer_relevance']['mean']:.3f} \\\\"
         for s, v in S.items()])

    t["tab_by_type"] = _table(
        "tab:bytype", "Answer Accuracy (\\%) by Question Type", "l" + "c" * len(QTYPES),
        "Method & " + " & ".join(QTYPE_LABELS[q] for q in QTYPES),
        [f"{LABELS[s]} & " + " & ".join(f"{v['by_type'][q]['correct']['mean'] * 100:.1f}" for q in QTYPES) + " \\\\"
         for s, v in S.items()])

    t["tab_latency"] = _table(
        "tab:latency", "Latency Distribution and Compute Cost per Query", "lcccccc",
        "Method & p50 & p95 & p99 (ms) & CPU (ms) & Energy (J) & Cost (\\$)",
        [f"{LABELS[s]} & {v['p50_ms']['mean']:.1f} & {v['p95_ms']['mean']:.1f} & {v['p99_ms']['mean']:.1f} & "
         f"{v['cpu_ms_per_query']['mean']:.1f} & {v['energy_j_per_query']['mean']:.3f} & "
         f"{v['cost_per_query_usd']['mean']:.4f} \\\\" for s, v in S.items()])

    comm = res["overhead"]["communication"]
    t["tab_communication"] = _table(
        "tab:communication", f"Communication Overhead per Query ({comm['shards']} index shards, HTTP)", "lccc",
        "Method & Internal requests & Payload (KB) & p50 over HTTP (ms)",
        [f"{LABELS[s]} & {v['internal_requests_per_query']:.1f} & {v['internal_kb_per_query']:.1f} & "
         f"{v['p50_ms_over_http']:.1f} \\\\" for s, v in comm["query"].items()])

    so = res["overhead"]["security"]
    us = lambda k: f"{so[k]['mean']:.1f}"  # noqa: E731
    t["tab_security_overhead"] = _table(
        "tab:secoverhead", "Processing Cost of Security Controls", "llr",
        "Control & Unit & Time ($\\mu$s)",
        [f"PII redaction & per document & {us('pii_redaction_us_per_doc')} \\\\",
         f"Near-duplicate / hidden-text check & per document & {us('poisoning_doc_signals_us_per_doc')} \\\\",
         f"Poisoning scan & per chunk & {us('poisoning_scan_us_per_chunk')} \\\\",
         f"Provenance signing (HMAC) & per chunk & {us('provenance_sign_us_per_chunk')} \\\\",
         f"Provenance verification & per chunk & {us('provenance_verify_us_per_chunk')} \\\\",
         f"Query injection filter & per query & {us('query_guard_us_per_query')} \\\\",
         "\\midrule",
         f"All ingestion controls & per document & {us('ingest_security_us_per_doc')} "
         f"({so['ingest_security_share']['mean'] * 100:.0f}\\%) \\\\",
         f"All query-time controls & per query & {us('query_security_us')} "
         f"({so['query_security_share_of_p50']['mean'] * 100:.1f}\\% of p50) \\\\"], small=True)

    sec = res["security"]
    pct = lambda k: f"{sec[k]['mean'] * 100:.1f}\\%"  # noqa: E731
    att = sec["poisoning_attacks"]
    outcomes = sec["poisoning_ingest_outcomes"]
    rows = [
        f"Cross-tenant leakage & {sec['probes'].get('cross_tenant_n', 0)} probes & {pct('cross_tenant_leakage_rate')} \\\\",
        f"RBAC leakage (unauthorised) & {sec['probes'].get('rbac_n', 0)} probes & {pct('rbac_leakage_rate')} \\\\",
        f"RBAC recall (authorised) & {sec['probes'].get('rbac_authorized_n', 0)} probes & {pct('rbac_authorized_recall')} \\\\",
        f"PII residue in index & {sec['pii_values_checked']} values & {pct('pii_residual_rate')} \\\\",
        f"Injection queries blocked & {5 * n} queries & {pct('injection_block_rate')} \\\\",
        f"Benign queries blocked (FP) & {5 * n} queries & {pct('benign_false_positive_rate')} \\\\",
        "\\midrule",
    ]
    for kind, label in (("injection", "Instruction injection"), ("stuffing", "Keyword stuffing"),
                        ("misinformation", "Plain misinformation")):
        o = outcomes.get(kind, {})
        handled = ", ".join(f"{v} {k}" for k, v in sorted(o.items()))
        a = att.get(f"cloudrag:{kind}", {"n": 1, "success": 0})
        rows.append(f"{label} doc. & {handled} & {a['success'] / a['n'] * 100:.1f}\\% success \\\\")
    rows.append(f"Audit chain integrity & {sec['audit_records']} records & "
                f"{'valid' if sec['audit_chains_valid'] else 'BROKEN'} \\\\")
    t["tab_security"] = _table("tab:security", "Security Evaluation", "llr", "Test & Volume & Result", rows)

    ds = res["dataset"]
    qps = ds["questions_per_seed"]
    t["tab_dataset"] = _table(
        "tab:dataset", "Evaluation Corpus (per seed)", "lr", "Property & Value",
        [f"Tenants & {len(cfg['tenants'])} \\\\",
         f"Documents & {ds['documents']['mean']:.0f} \\\\",
         f"Chunks & {ds['chunks']['mean']:.0f} \\\\",
         f"Distractor documents per tenant & {cfg['filler_docs_per_tenant']} \\\\",
         "\\midrule"]
        + [f"{QTYPE_LABELS[q]} questions & {qps.get(q, 0)} \\\\" for q in QTYPES]
        + [f"Total questions & {ds['questions_total_per_seed']} \\\\",
           f"Seeds (independent corpora) & {n} \\\\"], small=False)

    t["tab_parameters"] = _table(
        "tab:parameters", "Experimental Parameters", "lc", "Parameter & Value",
        [f"Retrieved chunks per query ($k$) & {cfg['top_k']} \\\\",
         "Chunk size & 120 words, 1-sentence overlap \\\\",
         "Embedding & feature hashing, 512 dims \\\\",
         "HNSW ($M$, $ef_{c}$, $ef_{s}$) & 16, 80, 64 \\\\",
         "BM25 ($k_1$, $b$) & 1.2, 0.75 \\\\",
         "RRF constant & 60 \\\\",
         "Candidate pool & $\\max(4k, 20)$ \\\\",
         "Graph traversal depth & 2 hops, decay 0.5 \\\\",
         "Agentic max. bridging steps & 3 \\\\",
         "Quarantine / flag threshold & 0.70 / 0.35 \\\\",
         "Query block threshold & 0.80 \\\\",
         f"Concurrent clients (throughput) & {res['per_seed'][0]['throughput']['cloudrag']['concurrency']} \\\\",
         f"Repetitions & {n} seeds ({', '.join(map(str, res['seeds']))}) \\\\",
         f"Cost model & {cfg['cost_model']} list price \\\\",
         f"Energy model & {cfg['watts_per_busy_core']} W per busy core \\\\"], small=False)

    tb = res["testbed"]
    t["testbed"] = ("\\begin{itemize}\n"
                    f"    \\item Hardware: {tb['cpu']} ({tb.get('physical_cores', '?')} cores, {tb['logical_cpus']} threads), "
                    f"{tb.get('ram_gb', '?')} GB RAM\n"
                    f"    \\item Operating System: {tb['os']}\n"
                    f"    \\item Programming Language: Python {tb['python']}\n"
                    "    \\item Simulation Platform: CloudRAG in-process deployment (all seven services, "
                    "in-memory event bus); distributed deployment on Docker Compose with Redis Streams and two "
                    "index shards\n"
                    "    \\item Software Tools: FastAPI, Uvicorn, NumPy, Pydantic, Redis 7.4, Docker 29, "
                    "Kubernetes manifests (kustomize), Prometheus\n"
                    "\\end{itemize}\n")
    return t


def markdown_summary(res: dict) -> str:
    S, n = res["systems"], len(res["seeds"])
    L = [f"# CloudRAG paper results", "",
         f"{n} seeds ({', '.join(map(str, res['seeds']))}), {res['dataset']['documents']['mean']:.0f} documents and "
         f"{res['dataset']['questions_total_per_seed']} questions per seed, top-k {res['config']['top_k']}. "
         f"Values are mean ± std across seeds. Generated {res['generated_at']}.", "",
         "## Performance comparison", "",
         "| Method | Accuracy % | p50 ms | p95 ms | Throughput QPS | F1 | CPU ms/q | Energy J/q |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in S.items():
        L.append(f"| {LABELS[s]} | {v['correct']['mean']*100:.1f} ± {v['correct']['std']*100:.1f} | "
                 f"{v['p50_ms']['mean']:.1f} ± {v['p50_ms']['std']:.1f} | {v['p95_ms']['mean']:.1f} | "
                 f"{v['throughput_qps']['mean']:.1f} ± {v['throughput_qps']['std']:.1f} | "
                 f"{v['f1']['mean']:.3f} ± {v['f1']['std']:.3f} | {v['cpu_ms_per_query']['mean']:.1f} | "
                 f"{v['energy_j_per_query']['mean']:.3f} |")
    L += ["", "## Accuracy % by question type", "", "| Method | " + " | ".join(QTYPE_LABELS[q] for q in QTYPES) + " |",
          "|---|" + "---|" * len(QTYPES)]
    for s, v in S.items():
        L.append(f"| {LABELS[s]} | " + " | ".join(
            f"{v['by_type'][q]['correct']['mean']*100:.1f} ± {v['by_type'][q]['correct']['std']*100:.1f}" for q in QTYPES) + " |")
    L += ["", "## Router decisions (all seeds)", "", "| Question type | naive | advanced | graph | agentic |", "|---|---|---|---|---|"]
    for q in QTYPES:
        c = res["routing"].get(q, {})
        L.append(f"| {QTYPE_LABELS[q]} | " + " | ".join(str(c.get(s, 0)) for s in ("naive", "advanced", "graph", "agentic")) + " |")
    comm = res["overhead"]["communication"]
    L += ["", f"## Communication overhead ({comm['shards']} shards, real HTTP serialisation)", "",
          f"Ingestion: {comm['ingest']['internal_requests_per_doc']:.1f} internal requests and "
          f"{comm['ingest']['internal_bytes_per_doc']/1024:.1f} KB per document, plus "
          f"{comm['ingest']['stream_bytes_per_doc']/1024:.2f} KB on the event stream.", "",
          "| Method | Internal requests/query | KB/query | p50 ms over HTTP |", "|---|---|---|---|"]
    for s, v in comm["query"].items():
        L.append(f"| {LABELS[s]} | {v['internal_requests_per_query']:.1f} | {v['internal_kb_per_query']:.1f} | {v['p50_ms_over_http']:.1f} |")
    so = res["overhead"]["security"]
    L += ["", "## Security control cost", "", "| Control | µs |", "|---|---|"]
    for k, v in so.items():
        if k.endswith("_share") or k.endswith("_of_p50"):
            L.append(f"| {k} | {v['mean']*100:.1f} % |")
        else:
            L.append(f"| {k} | {v['mean']:.1f} |")
    sec = res["security"]
    L += ["", "## Security results", ""]
    for k in ("cross_tenant_leakage_rate", "rbac_leakage_rate", "rbac_authorized_recall", "pii_residual_rate",
              "injection_block_rate", "benign_false_positive_rate"):
        L.append(f"- {k}: {sec[k]['mean']*100:.1f} %")
    L.append(f"- probes: {sec['probes']}; PII values checked: {sec['pii_values_checked']}; audit chains valid: "
             f"{sec['audit_chains_valid']} ({sec['audit_records']} records)")
    L.append(f"- poisoned documents at ingestion: {sec['poisoning_ingest_outcomes']}")
    L.append("- attacks (system:kind -> n / poison in context / successes): " + "; ".join(
        f"{k} {v['n']}/{v['in_context']}/{v['success']}" for k, v in sec["poisoning_attacks"].items()))
    ds, inc = res["dataset"], res["incremental"]
    L += ["", "## Ingestion, memory, freshness", "",
          f"- ingestion {ds['docs_per_s']['mean']:.0f} ± {ds['docs_per_s']['std']:.0f} docs/s end-to-end (in-process)",
          f"- index footprint {ds['index_bytes_per_chunk']['mean']/1024:.1f} KB per chunk "
          f"({ds['index_bytes']['mean']/2**20:.1f} MB for {ds['chunks']['mean']:.0f} chunks)",
          f"- process RSS {ds['process_rss_mb']['mean']:.0f} MB" if "process_rss_mb" in ds else "",
          f"- update freshness {inc['freshness_rate']['mean']*100:.0f} %, ingest-to-searchable p50 "
          f"{inc['update_lag_p50_ms']['mean']:.1f} ms",
          "", "## Testbed", "", "```", json.dumps(res["testbed"], indent=2), "```", ""]
    return "\n".join(x for x in L if x is not None)


def write_outputs(res: dict, out_dir: str) -> list[str]:
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    paths = []
    p = os.path.join(out_dir, "results", "experiments.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    paths.append(p)
    p = os.path.join(out_dir, "results", "summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(markdown_summary(res))
    paths.append(p)
    for name, tex in latex_tables(res).items():
        p = os.path.join(out_dir, "tables", f"{name}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        paths.append(p)
    return paths
