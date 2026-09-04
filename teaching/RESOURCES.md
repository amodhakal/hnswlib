# HNSW Resources

## Knowledge

- [Paper: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs — Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320)
  Primary spec. Use for: hierarchy, level distribution, search/insert algorithms, heuristic definition (Alg. 4), parameter guidance (§4-5).

- [Paper: Approximate nearest neighbor algorithm based on navigable small world graphs — Malkov et al. (2014)](https://doi.org/10.1016/j.is.2014.03.002)
  NSW foundation. Use for: navigable small-world idea, greedy search, why NSW alone degrades in low-d/low-recall.

- [Paper: Skip lists — Pugh (1990)](https://dl.acm.org/doi/10.1145/78973.78977)
  Conceptual ancestor for probabilistic level assignment. Use for: intuition behind `level = floor(-ln(U)*mult)`.

- [Code: hnswlib — `hnswlib/hnswalg.h`, `space_l2.h`, `space_ip.h`, `visited_list_pool.h`](https://github.com/nmslib/hnswlib)
  Reference implementation in this repo. Use for: exact insertion/search order, link storage (`maxM0 = 2*M`), visited-list trick, heuristic pruning code. Cite as `hnswlib/hnswalg.h:142` etc.

- [Code: FAISS `IndexHNSW.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.cpp)
  Alternative design choice. Use for: comparing neighbor storage and ef handling.

- [Doc: `ALGO_PARAMS.md` and `TESTING_RECALL.md` in this repo](../ALGO_PARAMS.md)
  Parameter semantics (M, ef, ef_construction) and recall-vs-ef benchmarking pattern. Use for: tuning validation.

- [Dataset: SIFT1M / TEXMEX / BigANN fvecs/bvecs format](http://corpus-texmex.irisa.fr/)
  Real-data testbed. Use for: `struct`-based fvecs loader, dimension handling.

## Wisdom (Communities)

- [r/MachineLearning — ANN threads](https://www.reddit.com/r/MachineLearning/)
  High-signal discussion of ANN trade-offs. Use for: practical M/ef choices, failure modes.

- [hnswlib GitHub Issues](https://github.com/nmslib/hnswlib/issues)
  Bug patterns and edge cases. Use for: deletion/replace semantics, persistence gotchas.

## Gaps

- No single pure-Python stdlib HNSW walkthrough with matching tests — this workspace fills it.
