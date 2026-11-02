"""Command line entry point.

    cloudrag serve <service>    run one microservice (or `all` for single-process mode)
    cloudrag token ...          mint a development token signed with CLOUDRAG_SECRET_KEY
    cloudrag eval ...           run the evaluation harness and write reports/
    cloudrag loadtest ...       drive a running gateway over HTTP
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .config import Settings


def _serve(args) -> None:
    import uvicorn

    from .runtime import SERVICES, build_service_app, default_port

    name = "gateway" if args.service == "all" else args.service
    if name not in SERVICES:
        raise SystemExit(f"unknown service {args.service!r}; choose one of all, {', '.join(SERVICES)}")
    settings = Settings.from_env()
    if args.service == "all":
        settings = settings.with_(bus="memory", retrieval_url="")
    app = build_service_app(name, settings)
    port = args.port or int(os.environ.get("PORT", "0")) or default_port(name)
    uvicorn.run(app, host=args.host, port=port, log_level=args.log_level)


def _token(args) -> None:
    from .core.models import Principal
    from .security.auth import TokenService

    s = Settings.from_env()
    p = Principal(user_id=args.user, tenant_id=args.tenant, roles=[r for r in args.roles.split(",") if r],
                  clearance=args.clearance)
    print(TokenService(s.secret_key, ttl_s=s.token_ttl_s).issue(p, args.ttl))


def _eval(args) -> None:
    from .evaluation.harness import Harness, write_report

    overrides = {}
    if args.llm:
        overrides["llm"] = args.llm
    if args.vector_index:
        overrides["vector_index"] = args.vector_index
    h = Harness(tenants=args.tenants.split(","), seed=args.seed, n_filler=args.filler, top_k=args.top_k,
                settings=Settings.from_env(**overrides))
    report = h.run(concurrency=tuple(int(c) for c in args.concurrency.split(",")), load_queries=args.load_queries,
                   suites=args.suites.split(",") if args.suites else None)
    jpath, mpath = write_report(report, args.out)
    print(open(mpath, encoding="utf-8").read())
    print(f"wrote {jpath} and {mpath}")


def _loadtest(args) -> None:
    import httpx

    from .evaluation.metrics import percentile

    queries = [line.strip() for line in open(args.queries, encoding="utf-8") if line.strip()]
    headers = {"authorization": f"Bearer {args.token}"}
    client = httpx.Client(base_url=args.url, timeout=60, headers=headers,
                          limits=httpx.Limits(max_connections=args.concurrency * 2))
    jobs = [queries[i % len(queries)] for i in range(args.n)]

    def one(q):
        t0 = time.perf_counter()
        r = client.post("/v1/query", json={"query": q, "top_k": args.top_k})
        return (time.perf_counter() - t0) * 1000, r.status_code

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(one, jobs))
    wall = time.perf_counter() - t0
    lat = [x for x, code in res if code == 200]
    statuses = Counter(str(code) for _, code in res)
    print(json.dumps({"requests": args.n, "concurrency": args.concurrency, "qps": args.n / wall,
                      "ok_qps": len(lat) / wall, "status_codes": dict(statuses),
                      "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95), "p99_ms": percentile(lat, 99)},
                     indent=2))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="cloudrag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="run a service")
    sp.add_argument("service")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--log-level", default="info")
    sp.set_defaults(fn=_serve)

    tp = sub.add_parser("token", help="mint a development token")
    tp.add_argument("--tenant", required=True)
    tp.add_argument("--user", required=True)
    tp.add_argument("--roles", default="")
    tp.add_argument("--clearance", type=int, default=0)
    tp.add_argument("--ttl", type=int, default=None)
    tp.set_defaults(fn=_token)

    ep = sub.add_parser("eval", help="run the evaluation harness")
    ep.add_argument("--tenants", default="acme,globex")
    ep.add_argument("--seed", type=int, default=7)
    ep.add_argument("--filler", type=int, default=40, help="distractor documents per tenant (corpus scale)")
    ep.add_argument("--top-k", type=int, default=5)
    ep.add_argument("--concurrency", default="1,4,16")
    ep.add_argument("--load-queries", type=int, default=200)
    ep.add_argument("--suites", default="", help="comma list of quality,load,incremental,security")
    ep.add_argument("--llm", default="", help="extractive | anthropic (calls the Claude API; costs money)")
    ep.add_argument("--vector-index", default="")
    ep.add_argument("--out", default="reports")
    ep.set_defaults(fn=_eval)

    lp = sub.add_parser("loadtest", help="drive a running gateway")
    lp.add_argument("--url", default="http://localhost:8000")
    lp.add_argument("--token", required=True)
    lp.add_argument("--queries", required=True, help="file with one query per line")
    lp.add_argument("-n", type=int, default=500)
    lp.add_argument("--concurrency", type=int, default=32)
    lp.add_argument("--top-k", type=int, default=5)
    lp.set_defaults(fn=_loadtest)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
