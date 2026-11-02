"""Evaluation harness: CloudRAG vs. vanilla, advanced, GraphRAG and agentic RAG baselines.

Suites
  quality      hit@k, MRR, context precision/recall, answer correctness, faithfulness, answer
               relevance, latency percentiles and estimated LLM cost - per system and question type
  routing      router decisions vs. question type
  load         throughput and latency percentiles at several concurrency levels
  incremental  update latency (ingest -> searchable) and answer freshness after re-ingestion
  security     RBAC and cross-tenant leakage, PII residue, injection-query blocking and false
               positives, poisoning quarantine and attack success, audit-chain integrity

All baselines share the platform's ingestion, index and generator so differences come from
the retrieval pathway alone. Results are written as JSON and Markdown.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from ..config import Settings
from ..core.models import DocumentIn, QueryRequest
from ..runtime import LocalPlatform
from ..security import BlockedInputError
from .metrics import (answer_correct, answer_relevance, context_precision, context_recall, faithfulness, hit_rate,
                      mrr, percentile)
from .synthetic import BENIGN_QUERIES, INJECTION_QUERIES, build_corpus, poisoned_documents

SYSTEMS = {
    "vanilla-rag": "naive",
    "advanced-rag": "advanced",
    "graphrag": "graphrag",
    "agentic-rag": "agentic",
    "cloudrag": "auto",
}


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else 0.0


def _lat_summary(lat_ms: list[float]) -> dict:
    return {"p50_ms": percentile(lat_ms, 50), "p95_ms": percentile(lat_ms, 95), "p99_ms": percentile(lat_ms, 99),
            "mean_ms": _mean(lat_ms)}


class Harness:
    def __init__(self, tenants=("acme", "globex"), seed: int = 7, n_filler: int = 40, top_k: int = 5,
                 settings: Settings | None = None):
        self.tenants = list(tenants)
        self.seed = seed
        self.top_k = top_k
        base = settings or Settings.from_env()
        self.settings = base.with_(data_dir=tempfile.mkdtemp(prefix="cloudrag-eval-"), rate_limit_rps=0.0)
        self.platform = LocalPlatform(self.settings)
        self.gw = self.platform.gateway
        self.corpora = build_corpus(self.tenants, seed=seed, n_filler=n_filler)
        self.report: dict = {"config": {
            "tenants": self.tenants, "seed": seed, "filler_docs_per_tenant": n_filler, "top_k": top_k,
            "embedder": self.settings.embedder, "vector_index": self.settings.vector_index,
            "reranker": self.settings.reranker, "llm": self.settings.llm,
            "cost_model": self.settings.anthropic_model,
        }}

    def reader(self, tenant: str, roles=("employee",)):
        return self.platform.context(tenant, f"reader@{tenant}", list(roles))

    def writer(self, tenant: str):
        return self.platform.context(tenant, f"loader@{tenant}", ["ingest"])

    # -- suites ------------------------------------------------------------------------------
    def ingest(self) -> dict:
        t0 = time.perf_counter()
        n_docs, n_chunks = 0, 0
        for t, tc in self.corpora.items():
            ctx = self.writer(t)
            for d in tc.docs:
                n_chunks += self.gw.ingest(ctx, d).chunks_indexed
                n_docs += 1
        accepted = time.perf_counter() - t0
        self.platform.wait_indexed(600)
        total = time.perf_counter() - t0
        out = {"documents": n_docs, "chunks": n_chunks, "ingest_accept_s": accepted, "index_complete_s": total,
               "docs_per_s": n_docs / total, "chunks_per_s": n_chunks / total,
               "namespaces": {t: self.platform.store.namespace(t).stats() for t in self.tenants}}
        self.report["ingestion"] = out
        return out

    def _ask(self, ctx, question: str, strategy: str):
        t0 = time.perf_counter()
        resp = self.gw.query(ctx, QueryRequest(query=question, strategy=strategy, top_k=self.top_k))
        return resp, (time.perf_counter() - t0) * 1000

    def quality(self) -> dict:
        embedder = self.platform.embedding.embedder
        results: dict = {}
        routes: dict = defaultdict(Counter)
        for system, strategy in SYSTEMS.items():
            rows = []
            for tc in self.corpora.values():
                ctx = self.reader(tc.tenant)
                for q in tc.questions:
                    resp, lat = self._ask(ctx, q.question, strategy)
                    docs = list(dict.fromkeys(c.chunk.doc_id for c in resp.contexts))
                    answer = resp.answer or ""
                    rows.append({
                        "qtype": q.qtype, "latency_ms": lat, "cost": resp.estimated_cost_usd,
                        "hit": hit_rate(docs, q.gold_docs, self.top_k), "mrr": mrr(docs, q.gold_docs),
                        "ctx_precision": context_precision(docs, q.gold_docs),
                        "ctx_recall": context_recall(docs, q.gold_docs),
                        "correct": answer_correct(answer, q.answers),
                        "faithfulness": faithfulness(answer, [c.chunk.text for c in resp.contexts]) if not resp.abstained else 0.0,
                        "answer_relevance": answer_relevance(q.question, answer, embedder) if not resp.abstained else 0.0,
                        "abstained": float(resp.abstained),
                    })
                    if system == "cloudrag":
                        routes[q.qtype][resp.strategy] += 1
            results[system] = {"overall": self._agg(rows), "by_type": {
                t: self._agg([r for r in rows if r["qtype"] == t]) for t in sorted({r["qtype"] for r in rows})}}
        self.report["quality"] = results
        self.report["routing"] = {t: dict(c) for t, c in routes.items()}
        return results

    @staticmethod
    def _agg(rows: list[dict]) -> dict:
        keys = ["hit", "mrr", "ctx_precision", "ctx_recall", "correct", "faithfulness", "answer_relevance", "abstained"]
        out = {k: _mean(r[k] for r in rows) for k in keys}
        out.update(_lat_summary([r["latency_ms"] for r in rows]))
        out["cost_per_query_usd"] = _mean(r["cost"] for r in rows)
        out["n"] = len(rows)
        return out

    def load(self, concurrency=(1, 4, 16), n_queries: int = 200) -> dict:
        work = [(tc.tenant, q.question) for tc in self.corpora.values() for q in tc.questions]
        ctxs = {t: self.reader(t) for t in self.tenants}
        out = {}
        for c in concurrency:
            jobs = [work[i % len(work)] for i in range(n_queries)]
            lat: list[float] = []

            def one(job):
                return self._ask(ctxs[job[0]], job[1], "auto")[1]

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                lat = list(ex.map(one, jobs))
            wall = time.perf_counter() - t0
            out[str(c)] = {"queries": n_queries, "qps": n_queries / wall, **_lat_summary(lat)}
        base = out[str(concurrency[0])]["qps"]
        for c in concurrency:
            out[str(c)]["scaling_efficiency"] = out[str(c)]["qps"] / (base * c / concurrency[0])
        self.report["load"] = out
        return out

    def incremental(self, n_updates: int = 5) -> dict:
        tc = self.corpora[self.tenants[0]]
        ctx_w, ctx_r = self.writer(tc.tenant), self.reader(tc.tenant)
        lags, fresh, stale = [], 0, 0
        for prod in tc.products[:n_updates]:
            new = 72 if prod["warranty"] != 72 else 84
            doc = DocumentIn(doc_id=prod["doc_id"], title=f"Product sheet: {prod['name']}", source="plm://products",
                             text=(f"The {prod['name']} is supplied by {prod['vendor']}. The {prod['name']} carries a "
                                   f"warranty period of {new} months following the 2026 contract renewal."))
            t0 = time.perf_counter()
            rep = self.gw.ingest(ctx_w, doc)
            while True:
                st = self.gw.doc_status(ctx_r, prod["doc_id"])
                if st and st["version"] >= rep.version:
                    break
                if time.perf_counter() - t0 > 30:
                    break
                time.sleep(0.001)
            lags.append((time.perf_counter() - t0) * 1000)
            resp, _ = self._ask(ctx_r, f"What is the warranty period of the {prod['name']}?", "auto")
            ans = resp.answer or ""
            fresh += answer_correct(ans, [f"{new} months"]) == 1.0
            stale += answer_correct(ans, [f"{prod['warranty']} months"]) == 1.0
            prod["warranty"] = new
        out = {"updates": n_updates, "freshness_rate": fresh / n_updates, "stale_answer_rate": stale / n_updates,
               **{f"update_lag_{k}": v for k, v in _lat_summary(lags).items()}}
        self.report["incremental"] = out
        return out

    def security(self) -> dict:
        out: dict = {}
        # RBAC and cross-tenant leakage
        counts = Counter()
        for tc in self.corpora.values():
            for pr in tc.probes:
                ctx = self.reader(pr.tenant, pr.roles)
                resp, _ = self._ask(ctx, pr.question, "auto")
                in_ctx = any(pr.secret in c.chunk.text for c in resp.contexts)
                in_ans = pr.secret in (resp.answer or "")
                foreign = any(c.chunk.tenant_id != pr.tenant for c in resp.contexts)
                counts[f"{pr.kind}_n"] += 1
                if pr.kind == "rbac":
                    counts["rbac_leaks"] += in_ctx or in_ans
                elif pr.kind == "rbac_authorized":
                    counts["rbac_authorized_hits"] += in_ctx
                elif pr.kind == "cross_tenant":
                    counts["cross_tenant_leaks"] += foreign
        out["rbac_leakage_rate"] = counts["rbac_leaks"] / max(1, counts["rbac_n"])
        out["rbac_authorized_recall"] = counts["rbac_authorized_hits"] / max(1, counts["rbac_authorized_n"])
        out["cross_tenant_leakage_rate"] = counts["cross_tenant_leaks"] / max(1, counts["cross_tenant_n"])
        out["probes"] = {k: v for k, v in counts.items() if k.endswith("_n")}

        # PII residue in the index
        residue, total = 0, 0
        for tc in self.corpora.values():
            texts = [c.text for c in self.platform.store.namespace(tc.tenant).chunks.values()]
            for v in tc.pii_values:
                total += 1
                residue += any(v in t for t in texts)
        out["pii_residual_rate"] = residue / max(1, total)
        out["pii_values_checked"] = total

        # adversarial queries
        ctx = self.reader(self.tenants[0])
        blocked = 0
        for qtext in INJECTION_QUERIES:
            try:
                self._ask(ctx, qtext, "auto")
            except BlockedInputError:
                blocked += 1
        false_pos = 0
        for qtext in BENIGN_QUERIES:
            try:
                self._ask(ctx, qtext, "auto")
            except BlockedInputError:
                false_pos += 1
        out["injection_block_rate"] = blocked / len(INJECTION_QUERIES)
        out["benign_false_positive_rate"] = false_pos / len(BENIGN_QUERIES)

        # knowledge-base poisoning: each attack runs in isolation (ingest -> query -> remove)
        poison = defaultdict(lambda: Counter())
        success = defaultdict(lambda: Counter())
        for tc in self.corpora.values():
            w, r = self.writer(tc.tenant), self.reader(tc.tenant)
            for kind, doc, meta in poisoned_documents(tc, seed=self.seed):
                rep = self.gw.ingest(w, doc)
                outcome = rep.status if rep.status != "accepted" or not rep.poisoning["flagged_chunks"] else "flagged"
                poison[kind][outcome] += 1
                self.platform.wait_indexed()
                for system in ("vanilla-rag", "cloudrag"):
                    resp, _ = self._ask(r, meta["question"], SYSTEMS[system])
                    key = f"{system}:{kind}"
                    success[key]["n"] += 1
                    success[key]["poison_in_context"] += any(c.chunk.doc_id == doc.doc_id for c in resp.contexts)
                    success[key]["attack_success"] += meta["fake"] in (resp.answer or "")
                self.gw.delete(w, doc.doc_id)
                self.platform.wait_indexed()
        out["poisoning_ingest_outcomes"] = {k: dict(v) for k, v in poison.items()}
        out["poisoning_attacks"] = {
            k: {"n": v["n"], "poison_in_context_rate": v["poison_in_context"] / v["n"],
                "attack_success_rate": v["attack_success"] / v["n"]} for k, v in sorted(success.items())}
        out["audit_chain"] = self.platform.audit.verify()
        self.report["security"] = out
        return out

    # -- orchestration ---------------------------------------------------------------------------------
    def run(self, concurrency=(1, 4, 16), load_queries: int = 200, suites=None) -> dict:
        suites = suites or ["quality", "load", "incremental", "security"]
        t0 = time.time()
        self.ingest()
        if "quality" in suites:
            self.quality()
        if "load" in suites:
            self.load(concurrency, load_queries)
        if "incremental" in suites:
            self.incremental()
        if "security" in suites:
            self.security()
        self.report["runtime_s"] = time.time() - t0
        return self.report


def to_markdown(r: dict) -> str:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    lines = ["# CloudRAG evaluation report", ""]
    c = r["config"]
    lines += [f"Tenants: {', '.join(c['tenants'])} | seed {c['seed']} | top-k {c['top_k']} | embedder `{c['embedder']}` | "
              f"index `{c['vector_index']}` | reranker `{c['reranker']}` | generator `{c['llm']}` "
              f"(cost priced as `{c['cost_model']}`)", ""]
    ing = r.get("ingestion")
    if ing:
        lines += ["## Ingestion", "", f"{ing['documents']} documents, {ing['chunks']} chunks, "
                  f"{ing['docs_per_s']:.1f} docs/s end-to-end (accepted in {ing['ingest_accept_s']:.2f}s, "
                  f"fully indexed in {ing['index_complete_s']:.2f}s).", ""]
    q = r.get("quality")
    if q:
        lines += ["## Quality and latency (all questions)", "",
                  "| system | hit@k | MRR | ctx precision | ctx recall | correct | faithful | relevance | p50 ms | p95 ms | p99 ms | est. $/query |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sys_name, v in q.items():
            o = v["overall"]
            lines.append(f"| {sys_name} | {f(o['hit'])} | {f(o['mrr'])} | {f(o['ctx_precision'])} | {f(o['ctx_recall'])} | "
                         f"{f(o['correct'])} | {f(o['faithfulness'])} | {f(o['answer_relevance'])} | {o['p50_ms']:.1f} | "
                         f"{o['p95_ms']:.1f} | {o['p99_ms']:.1f} | {o['cost_per_query_usd']:.5f} |")
        types = sorted({t for v in q.values() for t in v["by_type"]})
        lines += ["", "### Answer correctness by question type", "",
                  "| system | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
        for sys_name, v in q.items():
            lines.append(f"| {sys_name} | " + " | ".join(f(v["by_type"].get(t, {}).get("correct", 0.0)) for t in types) + " |")
        lines += ["", "### Context recall by question type", "",
                  "| system | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
        for sys_name, v in q.items():
            lines.append(f"| {sys_name} | " + " | ".join(f(v["by_type"].get(t, {}).get("ctx_recall", 0.0)) for t in types) + " |")
    if r.get("routing"):
        lines += ["", "## Router decisions (CloudRAG)", "", "| question type | " + " | ".join(
            ["naive", "advanced", "graph", "agentic"]) + " |", "|---|---|---|---|---|"]
        for t, cnt in sorted(r["routing"].items()):
            lines.append(f"| {t} | " + " | ".join(str(cnt.get(s, 0)) for s in ["naive", "advanced", "graph", "agentic"]) + " |")
    if r.get("load"):
        lines += ["", "## Load (CloudRAG, in-process)", "", "| concurrency | QPS | p50 ms | p95 ms | p99 ms | scaling efficiency |",
                  "|---|---|---|---|---|---|"]
        for cc, v in r["load"].items():
            lines.append(f"| {cc} | {v['qps']:.1f} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['p99_ms']:.1f} | {v['scaling_efficiency']:.2f} |")
    if r.get("incremental"):
        v = r["incremental"]
        lines += ["", "## Incremental indexing", "", f"{v['updates']} document updates: freshness {f(v['freshness_rate'])}, "
                  f"stale answers {f(v['stale_answer_rate'])}, ingest-to-searchable p50 {v['update_lag_p50_ms']:.1f} ms / "
                  f"p95 {v['update_lag_p95_ms']:.1f} ms."]
    s = r.get("security")
    if s:
        lines += ["", "## Security", "",
                  f"- RBAC leakage rate (unauthorised readers): {f(s['rbac_leakage_rate'])}; authorised recall: {f(s['rbac_authorized_recall'])}",
                  f"- Cross-tenant leakage rate: {f(s['cross_tenant_leakage_rate'])}",
                  f"- PII residue in index: {f(s['pii_residual_rate'])} of {s['pii_values_checked']} seeded values",
                  f"- Injection queries blocked: {f(s['injection_block_rate'])}; benign false positives: {f(s['benign_false_positive_rate'])}",
                  f"- Audit chain: {'valid' if s['audit_chain']['ok'] else 'BROKEN'} ({s['audit_chain']['records']} records)",
                  "", "Poisoned documents at ingestion:", ""]
        for kind, outcomes in s["poisoning_ingest_outcomes"].items():
            lines.append(f"- {kind}: " + ", ".join(f"{k}={v}" for k, v in outcomes.items()))
        lines += ["", "| system:attack | n | poison in context | attack success |", "|---|---|---|---|"]
        for k, v in s["poisoning_attacks"].items():
            lines.append(f"| {k} | {v['n']} | {f(v['poison_in_context_rate'])} | {f(v['attack_success_rate'])} |")
    lines += ["", f"_Runtime: {r.get('runtime_s', 0):.1f}s. Synthetic corpus; numbers characterise this build, "
              "not production-scale behaviour._", ""]
    return "\n".join(lines)


def write_report(report: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "evaluation.json")
    mpath = os.path.join(out_dir, "evaluation.md")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(report))
    return jpath, mpath
