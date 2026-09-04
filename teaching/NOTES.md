# Notes

- User prefs: pure Python, stdlib-only (no NumPy), extra depth on neighbor-selection heuristic, skip threading, real data they download (no ann-benchmarks/harness).
- Keep lessons in both `.html` (beautiful, printable) and `.md` (quick reference) — md mirrors html content.
- Prior knowledge: strong C++/Python, basic HNSW concepts; zone of proximal development starts at distance spaces + NSW, not at "what is k-NN".
- Style: Tufte-inspired, concise, every claim cited to paper or `hnswlib/hnswalg.h:line` or `ALGO_PARAMS.md:line`.

## Working notes

- Level assignment: `mult = 1/ln(M)` (`hnswlib/hnswalg.h:142`), `revSize = 1/mult`.
- L0 links: `maxM0 = 2*M` (`hnswalg.h:113`), upper layers `maxM = M`.
- ef invariant: ef >= k (`ALGO_PARAMS.md:5`).
