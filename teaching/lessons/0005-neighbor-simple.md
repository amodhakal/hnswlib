# Neighbor Selection — Simple (Closest M)

*Lesson 05 · 90 min · reference/neighbor-heuristic-deep.md · heuristics.py*

*Baseline: keep M closest. Easy, but clusters kill it.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/neighbor-heuristic-deep.md` Simple vs Heuristic.
- `hnswalg.h:selectNeighbors` `heuristic` flag off path.

## Build

`src/heuristics.py`:
```python
def select_simple(candidates, M):
    # candidates: list of (dist, id)
    candidates.sort()
    return [cid for _,cid in candidates[:M]]
```

Use in `hnsw.py:select_neighbors(..., heuristic=False)`.

## Exercise

Hand case: q=(0,0), a=(1,0.1), b=(1,-0.1), c=(10,0), M=2. Simple picks a,b (same direction); heuristic would keep a and c. Predict, then test.

## Verify

On clustered 2-d data (3 clusters), simple recall@5 worse than heuristic (next lesson) — set up A/B.

## Quiz

<div class="quiz" data-answer="A">
<p><strong>Simple selection needs?</strong></p>
<button data-choice="A">Sort by dist(q,cand) and take M smallest</button>
<button data-choice="B">Prune by dist(cand, kept)</button>
<button data-choice="C">Random M candidates</button>
<p class="feedback" data-ok="Baseline." data-no="See ref simple definition."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
