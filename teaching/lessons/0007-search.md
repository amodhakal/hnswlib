# Multi-Layer Search — Top-Down + ef

*Lesson 07 · 120 min · reference/hnsw-architecture.md · hnsw.py knn_query*

*One greedy walk down, one accurate L0 search. ef ≥ k is law.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/hnsw-architecture.md` Query.
- `ALGO_PARAMS.md:4-6` ef semantics, `hnswlib/hnswalg.h:searchKnn`.

## Build

```python
def knn_query(self, q, k=5, ef=10):
    assert ef>=k, "ef must be >= k"
    if self.enterpoint==-1: return []
    cur=self.enterpoint
    for lc in range(self.maxlevel,0,-1):
        cand=self.search_layer(q, cur, 1, lc)
        cur=cand[0][1]
    best=self.search_layer(q, cur, ef, 0)
    best.sort()
    return [(d,cid) for d,cid in best[:k]]
```

`search_layer` must return `list[(dist,id)]` sorted.

## Verify

- ef monotonic: recall@5 non-decreasing as ef 5→10→50→100 on 1000 random 16-d.
- Exception path: `k > n` or `ef<k` raises.

## Quiz

<div class="quiz" data-answer="B">
<p><strong>Query starts?</strong></p>
<button data-choice="A">At layer 0 entry</button>
<button data-choice="B">At top-layer enterpoint, greedy ef=1 per layer</button>
<button data-choice="C">At random node per layer</button>
<p class="feedback" data-ok="Top-down coarse then fine." data-no="See architecture query steps."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
