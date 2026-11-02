# CloudRAG final paper (IEEE journal format)

The complete paper in the IEEE Overleaf template, following the section outline in
`Project structure.odt`.

| File | Contents |
|---|---|
| `main.tex` | full paper: abstract through conclusion, Algorithm 1, equations (1)–(9) |
| `references.bib` | 17 references (the 11 from the Phase 2 proposal plus 6 method references) |
| `figures/` | `system_architecture.png`, `performance_results.png` |
| `tables/` | 10 tables generated from `paper/results/experiments.json` |

## Producing the PDF

1. In Overleaf: **New Project → Upload Project** and choose `CloudRAG_paper_overleaf.zip`
   (or upload the four items above, keeping the folder names).
2. Click **Recompile**. Overleaf's TeX Live includes `IEEEtran`, `algorithm` and `algpseudocode`.
3. **Download PDF**: this is the file to submit.

## Checks already done

The paper was compiled offline at IEEEtran journal geometry (two 21 pc columns, 10 pt Times):
0 errors, 0 overfull lines, 0 undefined references or citations, 8 pages, with every page inspected.
Two things could not be compiled offline, because those packages were not installed locally: the
`IEEEtran` class itself and the Algorithm 1 block (`algpseudocode`). Check both once on Overleaf.

## Regenerating

Tables come from `python -m cloudrag paper ...` and figures from `python paper/make_figures.py`
(see `paper/README.md`). After a new run, copy `paper/tables/*.tex` and the two figures here.
