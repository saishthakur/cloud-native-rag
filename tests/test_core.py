import numpy as np

from cloudrag.core.entities import extract_mentions, extract_relations
from cloudrag.core.text import content_terms, split_sentences, stem
from cloudrag.embedding.embedders import HashingEmbedder
from cloudrag.indexing.bm25 import BM25Index
from cloudrag.indexing.graph import KnowledgeGraph
from cloudrag.indexing.vector import FlatIndex, HNSWIndex
from cloudrag.ingestion.chunker import chunk_text
from cloudrag.retrieval.fusion import rrf
from cloudrag.retrieval.router import QueryRouter, decompose


def test_stemmer_conflates_inflections():
    assert stem("supplies") == stem("supplied") == stem("supplier") == stem("supply")
    assert stem("uses") == stem("used") == stem("use")
    assert stem("headquarters") == stem("headquartered")
    assert "zx-200" in content_terms("The ZX-200 sensor") and "200" in content_terms("ZX-200")


def test_sentences_and_entities():
    text = "Project Aurora uses the ZX-200 Sensor. The ZX-200 Sensor is supplied by Northwind Components."
    assert len(split_sentences(text)) == 2
    names = {m.name for m in extract_mentions(text)}
    assert {"Project Aurora", "ZX-200 Sensor", "Northwind Components"} <= names
    rels = {(a.name, p, b.name) for a, p, b in extract_relations(text)}
    assert ("Project Aurora", "uses", "ZX-200 Sensor") in rels
    assert ("ZX-200 Sensor", "is supplied by", "Northwind Components") in rels


def test_hnsw_recall_against_exact_search():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1500, 32)).astype(np.float32)
    hnsw, flat = HNSWIndex(32, M=12, ef_construction=80, ef_search=64), FlatIndex(32)
    for v in data:
        hnsw.add(v)
        flat.add(v)
    recall = []
    for q in rng.normal(size=(50, 32)):
        exact = {i for i, _ in flat.search(q, 10)}
        approx = {i for i, _ in hnsw.search(q, 10)}
        recall.append(len(exact & approx) / 10)
    assert np.mean(recall) >= 0.9


def test_hnsw_deletes_and_filters():
    rng = np.random.default_rng(1)
    idx = HNSWIndex(16)
    vecs = rng.normal(size=(200, 16))
    for v in vecs:
        idx.add(v)
    idx.delete(0)
    assert all(i != 0 for i, _ in idx.search(vecs[0], 5))
    even = idx.search(vecs[3], 5, allow=lambda i: i % 2 == 0)
    assert even and all(i % 2 == 0 for i, _ in even)


def test_bm25_incremental():
    bm = BM25Index()
    bm.add(1, "warranty period of the sensor is 18 months")
    bm.add(2, "travel policy for international trips")
    assert bm.search("sensor warranty", 2)[0][0] == 1
    bm.remove(1)
    assert bm.search("sensor warranty", 2) == []
    assert len(bm) == 1


def test_graph_traversal_and_removal():
    g = KnowledgeGraph()
    g.add_chunk(1, "Project Aurora uses the ZX-200 Sensor.")
    g.add_chunk(2, "The ZX-200 Sensor is supplied by Northwind Components.")
    seeds = g.link("which vendor supplies the product used by aurora project")
    assert "project aurora" in seeds
    scores, parents = g.traverse(seeds, hops=2)
    assert "northwind components" in scores and parents["northwind components"] == "zx-200 sensor"
    hits = dict(g.chunk_scores(scores, 5))
    assert set(hits) == {1, 2}
    g.remove_chunk(2)
    assert "northwind components" not in g.nodes


def test_chunker_section_acl():
    text = "Public intro sentence.\n\n[[acl roles=finance clearance=2]]Secret budget is 5 million.[[/acl]]\n\nOutro."
    chunks = chunk_text(text, max_words=50)
    restricted = [c for c in chunks if c.section_policy]
    assert len(restricted) == 1 and "budget" in restricted[0].text
    assert restricted[0].section_policy.roles == ["finance"] and restricted[0].section_policy.min_clearance == 2
    assert all("budget" not in c.text for c in chunks if not c.section_policy)


def test_embedder_similarity():
    e = HashingEmbedder(256)
    v = e.embed(["warranty period of the sensor", "sensor warranty periods", "cafeteria vendor contract"])
    assert v[0] @ v[1] > v[0] @ v[2]
    assert abs(np.linalg.norm(v[0]) - 1) < 1e-5


def test_router_decisions_and_budget():
    r = QueryRouter()
    assert r.decide("What is the warranty of the ZX-200 Sensor?", []).strategy == "naive"
    assert r.decide("Which vendor supplies the product used by Project Aurora?", ["Project Aurora"]).strategy == "graph"
    assert r.decide("Who leads Project Aurora and what is the warranty of the ZX-200 Sensor?", []).strategy == "agentic"
    assert r.decide("In which city is the vendor that supplies the product used by Project Aurora headquartered?",
                    []).strategy == "agentic"
    # priors: agentic p95 250ms, graph 80ms, advanced 40ms -> a 50ms budget lands on advanced
    d = r.decide("Who leads Project Aurora and what is the warranty of the ZX-200 Sensor?", [], latency_budget_ms=50)
    assert d.strategy == "advanced" and sum("exceeds budget" in x for x in d.reasons) == 2
    assert decompose("Compare the ZX-200 Sensor and the QK-410 Drive in terms of warranty") == [
        "the ZX-200 Sensor warranty", "the QK-410 Drive warranty"]


def test_rrf_rewards_agreement():
    from cloudrag.core.models import Chunk, ScoredChunk

    def sc(cid):
        return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id=cid, tenant_id="t", text=cid), score=1.0)

    fused = rrf({"a": [sc("x"), sc("y")], "b": [sc("y"), sc("z")]})
    assert fused[0].chunk.chunk_id == "y"
