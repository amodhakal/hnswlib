# HNSW Glossary

Pure-Python HNSW workspace language. Every lesson adheres to these terms.

## Terms

**HNSW**:
Hierarchical Navigable Small World graph — multi-layer proximity graph where each element appears on layers 0..level, `level ~ floor(-ln(U)/ln(M))`.
_Avoid_: hierarchical NSW `without` hyphen.

**NSW**:
Single-layer navigable small-world graph; HNSW layer 0 generalizes it.
_Avoid_: flat graph.

**Level**:
Maximum layer an element belongs to, drawn from exponential decay with `mult = 1/ln(M)`. Higher level = sparser.
_Avoid_: layer id for element; use "element level".

**Layer**:
One proximity graph in the hierarchy (0 is dense base, maxlevel is sparse top). All elements of level >= layer appear here.
_Avoid_: level (ambiguous).

**Entry point**:
Top-layer node where search starts (`enterpoint_node_` in `hnswlib/hnswalg.h:45`). Updated when new maxlevel appears.

**M**:
Max outgoing links per element per upper layer; L0 allows `maxM0 = 2*M`.
_Avoid_: degree, fanout.

**maxM / maxM0**:
Caps for upper layers and layer 0 respectively.

**ef / ef_construction**:
Size of dynamic candidate list during search / construction; ef >= k (`ALGO_PARAMS.md:5`). Larger = more accurate, slower.

**Candidate set**:
Priority queue (max-heap via `heapq` with negated distances) of size `ef` holding closest seen.

**Visited list**:
Epoch-tagged `bytearray` or `VisitedListPool` entry marking nodes already evaluated in current search, avoiding `set` overhead.

**Heuristic (neighbor selection)**:
Pruning rule keeping a candidate only if it is closer to query than to any already-kept neighbor (diversity). See Alg. 4 in paper.
_Avoid_: simple closest-M (that is the `simple` baseline).

**Recall@k**:
Fraction of true k-NN (from brute force) returned by HNSW.

**Space**:
Distance functor (`L2Space`, `IPSpace`, `CosineSpace`). IP and cosine noted not metric (`README.md:54`).

**Tombstone**:
Deleted element kept in place, skipped during search; `mark_deleted` sets flag.
