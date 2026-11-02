# Paper results

Everything the IEEE template's Experimental Setup (V) and Results (VI) sections need, generated
from the code in this repository.

| Template slot | File |
|---|---|
| Fig. `system_architecture.png` | `figures/system_architecture.png` |
| Fig. `performance_results.png` | `figures/performance_results.png` |
| V-A Dataset | `tables/tab_dataset.tex` |
| V-B Simulation/Testbed | `tables/testbed.tex` |
| V-C Parameter Settings (`tab:parameters`) | `tables/tab_parameters.tex` |
| VI-B Comparison (`tab:performance`: accuracy, latency, throughput, F1) | `tables/tab_performance.tex` |
| VI-A/B detail | `tables/tab_retrieval.tex`, `tables/tab_by_type.tex`, `tables/tab_latency.tex` |
| VI-C Security Analysis | `tables/tab_security.tex` |
| VI-D Computational/Communication Overhead | `tables/tab_latency.tex`, `tables/tab_communication.tex`, `tables/tab_security_overhead.tex` |
| All numbers, readable | `results/summary.md` |
| Raw data (every seed, every metric) | `results/experiments.json` |

## Using them in Overleaf

Upload `figures/*.png` into the template's `figures/` folder and `tables/*.tex` into a `tables/`
folder, then replace the placeholder tables in `main.tex` with, for example:

```latex
\input{tables/tab_parameters}
\input{tables/tab_performance}
```

The template already loads `booktabs` and `amsmath`, which is all the tables use. Add
`\graphicspath{{figures/}}` after `\usepackage{graphicx}` so `\includegraphics{system_architecture.png}`
finds the figures.

## How the numbers were produced

```bash
python -m cloudrag paper --seeds 7,11,23,42,2026 --tenants initech,umbrella,hooli --filler 200 --out paper
```

```bash
python paper/make_figures.py
```

Five independent synthetic corpora (one per seed), each with 3 tenants, 768 documents and 69
questions. Every value is mean ± sample standard deviation across the 5 seeds. All five systems
share the same ingestion pipeline, index and generator, so differences come from retrieval alone.
Background services were stopped during the runs.

## Metric definitions (state these in V-D)

- **Accuracy**: share of gold answer strings contained in the generated answer (partial credit for
  multi-part and aggregation questions).
- **F1**: retrieval F1 = harmonic mean of precision@k (share of retrieved documents that are gold
  evidence) and recall (share of gold evidence documents retrieved). With k = 5 and 1–3 gold
  documents per question, precision@k is capped well below 1, so F1 values are low for every system
  by construction. Compare them across systems, not against 1.
- **Latency**: wall-clock time of the full query (guard → retrieval → generation), in-process.
  "p50 over HTTP" in the communication table runs every service as a separate HTTP app.
- **Throughput**: queries per second with 8 concurrent clients, after a warm-up pass.
- **CPU time**: process CPU-seconds per query (all threads).
- **Energy**: *estimated*, not metered: CPU-seconds × 2.5 W per busy core (Intel i3-1215U: 15 W
  base power ÷ 6 cores). Say so in the paper.
- **Cost**: estimated LLM cost if the prompt were sent to Claude Opus 5.5 at list price
  ($4 / $20 per million input/output tokens); answers in these runs were generated offline.
- **Communication**: internal service-to-service HTTP requests and bytes (headers + bodies) per query.

## Caveats to state in the paper

- The corpus is synthetic and was used while developing the router and generator; results are
  optimistic for real enterprise text.
- Throughput for the fastest systems varies a lot between runs (± 85–99 QPS) because a 2–3 ms
  query on a laptop CPU is sensitive to scheduling noise.
- Scale is about 800 documents per corpus, not the 1M–10M in the original proposal; this needs a
  cluster and is future work.
- The "misinformation" poisoning attack is not detected at ingestion (45/45 accepted); it fails
  only because ad-hoc uploads carry lower source trust than systems of record.
