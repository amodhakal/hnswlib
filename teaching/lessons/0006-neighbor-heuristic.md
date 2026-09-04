# Neighbor Selection — Heuristic (Diversity Pruning) — Deep Dive

*Lesson 06 · 180 min · reference/neighbor-heuristic-deep.md · heuristics.py*

*The heart of HNSW. Keep a candidate only if it is not shadowed by an already-kept neighbor.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Why extra time

This pruning is what makes HNSW robust on low-d/clustered data. Paper Alg. 4; `hnswalg.h:getNeighborsByHeuristic` inner check `if dist(c, kept) < dist(c, q)` then occluded.

## Study

- `reference/neighbor-heuristic-deep.md` full algorithm + equilateral/line/cluster drawings.
- Trace `hnswalg.h` heuristic with 3 points.

## Build

```python
def select_heuristic(candidates, M, qvec, data, space):
    candidates=sorted(candidates)  # by dist to q
    result=[]
    for d_cq, cid in candidates:
        if len(result)>=M: break
        keep=True
        for rid in result:
            if space.dist(data[cid], data[rid]) < d_cq:
                keep=False; break
        if keep:
            result.append(cid)
    # fallback fill if under M (paper): add closest remaining not occluded
    if len(result)<M:
        for d_cq,cid in candidates:
            if cid not in result:
                result.append(cid)
                if len(result)>=M: break
    return result
```

Wire: `HNSW.select_neighbors(cands, M, q, heuristic=True)`.

## Three hand tests

1. Equilateral q(0,0) a(1,0) b(0.5,0.866) c(2,0) M=2 → heuristic keeps a,b (different dirs) not a,c.
2. Line q0-a1-b2 collinear → picks only a (b occluded).
3. Cluster: 5 points in tight ball + 1 far → heuristic keeps 1 cluster rep + far point.

Code them in `tests/test_heuristic.py`.

## Verify

Compare recall@5 on synthetic clustered (3 tight blobs) simple vs heuristic — heuristic should win 5-10 pts at same ef.

## Quiz

<div class="quiz" data-answer="C">
<p><strong>Heuristic keeps c if?</strong></p>
<button data-choice="A">dist(c,q) is smallest</button>
<button data-choice="B">dist(c,any kept) is large</button>
<button data-choice="C">dist(c,q) < dist(c, r) for every kept r</button>
<p class="feedback" data-ok="Paper Alg. 4." data-no="Re-read heuristic condition."></p>
</div>

## Reflection

Write `learning-records/0001-heuristic.md` — what occlusion means geometrically.

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
