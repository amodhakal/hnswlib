# Reference: HNSW Architecture

Paper Malkov & Yashunin 2018 §3; `hnswlib/hnswalg.h:34-144`.

## Multilayer set

Elements inserted with `level = floor(-ln(unif(0,1)) * mult)` where `mult = 1/ln(M)` (`hnswalg.h:142`), `revSize = 1/mult`. `maxM = M`, `maxM0 = 2*M` (`hnswalg.h:113`). `maxlevel` tracks highest present; `enterpoint_node_` is top-layer entry.

A node appears on every layer `0..level` (nested subsets).

## Construction

For new element with `level = L`:

1. Search from `enterpoint` top-down with `ef=1` for layers `maxlevel .. L+1`, updating entry to closest at each.
2. For `L .. 0`: `candidates = search_layer(q, entry, ef_construction, layer)`; `neighbors = select_neighbors(..., M or maxM0)`; link both ways; prune.

## Query

`ef >= k` required. `cur = enterpoint`. For `maxlevel .. 1`: `cur = search_layer(q, cur, ef=1, layer)[0]` (greedy). At L0: `result = search_layer(q, cur, ef, 0)` return `k` closest.

## Memory

Per element upper layers: `maxM*sizeof(int)+overhead`; L0 double. See `hnswalg.h:120-123` contiguous layout; pure-Python uses `list[array('I')]` per layer.

## Parameters

`M` 12-48 typical; `ef_construction ~ 200` good; `M*ef_construction` roughly constant (`ALGO_PARAMS.md:22`). `ef` trades recall vs QPS.
