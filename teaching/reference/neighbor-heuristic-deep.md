# Reference: Neighbor-Selection Heuristic (Deep)

The algorithmic heart. Paper Alg. 4; `hnswlib/hnswalg.h:selectNeighbors` / `getNeighborsByHeuristic`.

## Simple vs Heuristic

*Simple*: `M` smallest `dist(q, cand)`.
*Heuristic*: picks diverse neighbors — keeps `cand` only if `dist(cand, q) < dist(cand, any_already_kept)` (or tighter). Prevents clumping, preserves long edges.

## Algorithm (heuristic)

Input: `candidates` sorted by `dist(q, c)`, `M`, `q`.
```
result = []
for c in candidates (ascending dist to q):
    if len(result) >= M: break
    keep = True
    for r in result:
        if dist(c, r) < dist(c, q):   # r occludes c
            keep = False; break
    if keep: result.append(c)
# If result < M, heuristic may return fewer; caller may fill remaining with closest.
```

hnswlib variant extends with pruning of existing neighbors after bidirectional connect.

## Why it matters

On clustered / low intrinsic dim data, simple keeps all neighbors in one cluster, graph loses navigability. Heuristic keeps one per direction, long jump edges survive.

## Edge cases

- Ties on `dist` — stable sort preserves insertion order.
- `dist(c,r)` needs same `Space` as query (not precomputed).
- When `M` small, heuristic may underfill — fallback to closest fill is correct per paper.

## Lesson split

**05a** implements simple, **05b** implements heuristic + 3 hand-crafted 2D cases where heuristic wins (equilateral, line, cluster).

## Test oracle

3 points: q=(0,0), a=(1,0), b=(2,0) collinear: simple picks {a,b}, heuristic picks {a} only (b occluded by a). Verify.

## Citations

- Paper Alg. 4
- `hnswlib/hnswalg.h:getNeighborsByHeuristic` inner `if (dist_to_query < dist_to_result)` check.
