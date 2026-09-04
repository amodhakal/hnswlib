# Mission: Build Your Own HNSW from Scratch (Pure Python, Stdlib-Only)

## Why
You know C++/Python and basic HNSW concepts, but want to truly own the algorithm by rebuilding it without NumPy or C++ — so you can explain, modify, and debug every part and validate on real vector data you download yourself.

## Success looks like
- Pure-Python `src/hnsw.py` with no NumPy/C++ that builds, searches, persists, and handles deletions on lists of floats.
- Can trace a query top-down through layers and explain the heuristic neighbor selection on paper.
- Validated against brute-force on real data (e.g. SIFT/GloVe slice) with recall@10 >0.90 at ef=50 and QPS numbers you measured yourself.
- Code you can extend (new space, filter, M/ef tuning) without consulting hnswlib.

## Constraints
- Stdlib only in `src/` (`math`, `random`, `heapq`, `array`, `struct`, `pickle`). Data loaders may use only stdlib. No NumPy in implementation.
- Skip threading (GIL-limited). Spend extra time on neighbor-selection heuristic (split into 2 lessons).
- Real datasets downloaded by you (no ann-benchmarks harness, no custom harness required).

## Out of scope
- Thread-safe concurrent inserts, SIMD/distance micro-optimizations, alternative ANN families (IVF, PQ, LSH).
