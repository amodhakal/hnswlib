# Distance Spaces — L2, IP, Cosine without NumPy

*Lesson 01 · 90 min · reference/distance-spaces.md · spaces.py*

*You cannot search without a metric. Build three spaces by hand and learn why IP is not a metric.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Why this matters

HNSW is metric-agnostic, but the `Space` you plug in determines what "nearest" means. `README.md:51-53` defines three; your `src/spaces.py` will expose `dist(a,b)`. All later recall numbers depend on this being correct.

## Study (15 min)

- `reference/distance-spaces.md` — contracts.
- `hnswlib/space_l2.h:1-40` and `space_ip.h:1-40` — how hnswlib binds `fstdistfunc_` (`hnswalg.h:56`).

## Build

### Task 1 — Implement

In `src/spaces.py`:

```python
import math

class L2Space:
    name="l2"
    def dist(self, a, b):
        return math.fsum((x-y)*(x-y) for x,y in zip(a,b))  # squared L2

class IPSpace:
    name="ip"
    def dist(self, a, b):
        return 1.0 - math.fsum(x*y for x,y in zip(a,b))

class CosineSpace:
    name="cosine"
    def dist(self, a, b):
        dot = math.fsum(x*y for x,y in zip(a,b))
        na = math.sqrt(math.fsum(x*x for x in a))
        nb = math.sqrt(math.fsum(x*x for x in b))
        if na==0 or nb==0: return 1.0
        return 1.0 - dot/(na*nb)
```

Bind locally in callers: `d = space.dist`.

### Task 2 — Verify

- Unit: `dist([0,0],[3,4])` → 25 (squared). `IP dist([1,0],[1,0])` → 0. `Cosine` of orthogonal → 1.0.
- Brute check: 20 random 8-d vecs, compare L2 to manual loop and `math.dist` (Python 3.8+) if available.

## Retrie val — Quiz

<div class="quiz" data-answer="B">
<p><strong>Which statement is true?</strong></p>
<button data-choice="A">Cosine distance is a metric and obeys triangle inequality</button>
<button data-choice="B">IP with `1-dot` is not a metric; a point can be closer to another than itself</button>
<button data-choice="C">Squared L2 and rooted L2 give different nearest-neighbor ordering</button>
<p class="feedback" data-ok="README.md:54 notes IP non-metric." data-no="Re-read reference/distance-spaces.md pitfalls."></p>
</div>

## Exercise (in-browser mental)

Compute by hand: `a=[1,2], b=[4,6]` → L2²=?, cosine=? Then run your code to confirm.

## Next

Lesson 02 builds NSW graph on top of this `Space`. Ask your teacher about any dist mismatch.

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
