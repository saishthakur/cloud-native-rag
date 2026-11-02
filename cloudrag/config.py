"""Runtime configuration.

Every field can be overridden with an environment variable named
``CLOUDRAG_<FIELD_NAME_UPPERCASE>`` (for example ``CLOUDRAG_LLM=anthropic``).
In Kubernetes these come from the ConfigMap / Secret in ``deploy/k8s``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace


def _cast(value: str, default):
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass(frozen=True)
class Settings:
    service_name: str = "cloudrag"
    data_dir: str = "./data"

    # --- security -----------------------------------------------------------
    secret_key: str = "dev-insecure-secret-change-me"
    provenance_key: str = "dev-insecure-provenance-key"
    token_ttl_s: int = 3600
    admin_bootstrap_key: str = "dev-admin-key"
    dev_auth: bool = True  # allow /v1/auth/token (disable when an external IdP issues tokens)
    pii_mode: str = "redact"  # redact | pseudonymize | off
    poison_quarantine_threshold: float = 0.7
    poison_flag_threshold: float = 0.35
    query_block_threshold: float = 0.8
    # provenance-based trust ceilings by source URI prefix ("prefix=trust,..."); unlisted sources get 1.0.
    # Systems of record (ERP, PLM, HRIS connectors) outrank ad-hoc uploads when facts conflict.
    source_trust: str = "upload://=0.6,external://=0.5,email://=0.7"
    max_doc_chars: int = 2_000_000
    max_query_chars: int = 2000
    rate_limit_rps: float = 50.0
    rate_limit_burst: int = 100
    audit_path: str = ""  # default: <data_dir>/audit/<service_name>.jsonl

    # --- embedding / indexing -----------------------------------------------
    embedder: str = "hashing"  # hashing | sentence-transformers:<model>
    embed_dim: int = 512
    vector_index: str = "hnsw"  # hnsw | flat
    hnsw_m: int = 16
    hnsw_ef_construction: int = 80
    hnsw_ef_search: int = 64
    chunk_tokens: int = 120
    chunk_overlap_sentences: int = 1
    snapshot_interval_s: float = 30.0
    shard_ordinal: int = -1  # -1: derive from hostname suffix (StatefulSet), else 0
    shard_count: int = 1

    # --- retrieval / reranking / generation ---------------------------------
    reranker: str = "lexical"  # lexical | cross-encoder:<model>
    rrf_k: int = 60
    agentic_max_steps: int = 3
    llm: str = "extractive"  # extractive | anthropic
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"
    anthropic_max_tokens: int = 16000

    # --- messaging ----------------------------------------------------------
    bus: str = "memory"  # memory | redis
    redis_url: str = "redis://localhost:6379/0"

    # --- service discovery (distributed mode) -------------------------------
    embedding_url: str = ""
    indexing_urls: str = ""  # comma separated, one URL per shard
    reranker_url: str = ""
    retrieval_url: str = ""
    generation_url: str = ""
    ingestion_url: str = ""
    http_timeout_s: float = 30.0

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values = {}
        for f in fields(cls):
            env = os.environ.get(f"CLOUDRAG_{f.name.upper()}")
            if env is not None:
                values[f.name] = _cast(env, f.default)
        values.update(overrides)
        return cls(**values)

    def with_(self, **overrides) -> "Settings":
        return replace(self, **overrides)

    @property
    def resolved_audit_path(self) -> str:
        if self.audit_path:
            return self.audit_path
        host = os.environ.get("HOSTNAME", "local")
        return os.path.join(self.data_dir, "audit", f"{self.service_name}-{host}.jsonl")

    @property
    def resolved_shard_ordinal(self) -> int:
        if self.shard_ordinal >= 0:
            return self.shard_ordinal
        host = os.environ.get("HOSTNAME", "")
        tail = host.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    def trust_for_source(self, source: str) -> float:
        best, best_len = 1.0, -1
        for item in self.source_trust.split(","):
            prefix, _, value = item.strip().rpartition("=")
            if prefix and source.startswith(prefix) and len(prefix) > best_len:
                best, best_len = float(value), len(prefix)
        return best

    @property
    def indexing_url_list(self) -> list[str]:
        return [u.strip() for u in self.indexing_urls.split(",") if u.strip()]
