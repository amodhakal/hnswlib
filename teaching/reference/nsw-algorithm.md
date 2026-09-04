# Reference: NSW — Single-Layer Graph & Greedy Search

Single layer underlying HNSW L0. Paper Malkov et al. 2014 §2; code `hnswlib/hnswalg.h:searchKnn` inner loop.

## Graph

Undirected (bidirectional) proximity graph: each node keeps `maxM` neighbors. Constructed incrementally.

## Search layer (`search_layer(q, entry, ef, layer)`)

Inputs: query vec, entry id, ef, layer idx.
Output: `ef` closest ids sorted by dist.

1. `visited = bytearray(n); visited[entry]=1` + epoch tag or fresh array.
2. `candidates = [(-dist(entry,q), entry)]` max-heap via negated distances (`heapq`).
3. `nearest = [(-dist(entry,q), entry)]` sorted best.
4. While candidates:
   `c = heappop(candidates)` (closest remaining).
   If `dist(c,q) > dist(furthest in nearest)` break.
   For each neighbor `v` of `c` on `layer` if not visited: compute `d = dist(v,q)`, push to candidates and nearest, trim nearest to `ef`.

Visited as `bytearray` avoids `set` allocation per query; pool pattern in `visited_list_pool.h:1`.

## Insertion (single layer)

`candidates = search_layer(q, entry, ef_construction, layer)`; `neighbors = select_neighbors(candidates, M, heuristic)`; link bidirectionally, pruning neighbors that exceed `M`.

## Complexity

Greedy walk `O(log n)` average on NSW; degrades on low-d/clustered — motivation for hierarchy.
