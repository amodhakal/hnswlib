# NSW — Single-Layer Graph & Greedy Search

*Lesson 02 · 120 min · reference/nsw-algorithm.md · graph.py*

*One layer, greedy walk, candidate heap, visited bytearray — the engine of HNSW.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Why

HNSW layer 0 *is* NSW. Master `search_layer` here and every upper layer is the same call with smaller `ef`.

## Study

- `reference/nsw-algorithm.md` algorithm steps.
- `hnswlib/hnswalg.h` search inner loop (≈ line 300-450) — candidate heap as negated max-heap.

## Build

### Task 1 — Graph store

`src/graph.py`:
```python
import heapq, array
class NSW:
    def __init__(self, maxM=16): self.M=maxM; self.adj=[]; self.data=[]
    def add_node(self, vec): self.adj.append(array.array('I')); self.data.append(vec); return len(self.data)-1
```

### Task 2 — search_layer

```python
def search_layer(self, q, entry, ef, level, space):
    n=len(self.data)
    visited=bytearray(n); visited[entry]=1
    dist0=space.dist(self.data[entry], q)
    cand=[(-dist0, entry)]; heapq.heapify(cand)  # max-heap via negation? use min-heap on -dist
    # Better: use (-dist) max-heap or (dist) min-heap + separate nearest
    best=[(dist0, entry)]
    import heapq
    # implement standard: candidates = [(dist, id)] min-heap, nearest = max-heap of size ef
```

Follow reference steps; trim `best` to `ef`. Use local `dist=space.dist`, `adj=self.adj`.

## Quiz

<div class="quiz" data-answer="C">
<p><strong>Why bytearray over set for visited?</strong></p>
<button data-choice="A">bytearray is a set</button>
<button data-choice="B">set is faster for 10k nodes</button>
<button data-choice="C">O(1) indexed, no hashing/allocation per query, poolable</button>
<p class="feedback" data-ok="See visited_list_pool.h" data-no="Check reference/nsw-algorithm.md visited note."></p>
</div>

## Verify

Build 200 random 2-d nodes, `maxM=6`, naive connect 6 nearest. Query 20 random vs brute — recall@5 should be high for this toy.

Ask teacher: show your `search_layer` trace on 5-node graph.

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
