# Hierarchy — Levels, Entry Point, Mult

*Lesson 03 · 90 min · reference/hnsw-architecture.md · hnsw.py skeleton*

*Skip-list intuition: each element draws a level, appears on all layers below.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/hnsw-architecture.md` + `hnswlib/hnswalg.h:142` (`mult = 1/log(M)`, `revSize = 1/mult`).
- Pugh skip lists analogy.

## Build

`src/hnsw.py` skeleton:
```python
import math, random, array
class HNSW:
    def __init__(self, M=16, ef_construction=200, space=None, seed=100, max_elements=10000):
        self.M=M; self.maxM=M; self.maxM0=M*2; self.ef_construction=ef_construction
        self.space=space or L2Space()
        self.mult=1/math.log(M); self.revSize=1/self.mult
        random.seed(seed)
        self.max_elements=max_elements
        self.levels=array.array('b')
        self.data=[]; self.links=[]  # list per node: list of array('I') per layer
        self.enterpoint=-1; self.maxlevel=-1
    def _random_level(self):
        r=random.random()
        if r==0: r=1e-12
        return int(-math.log(r)*self.mult)
```

## Verify

Histogram: draw 5000 levels with M=16, check exponential decay (P(level>=3) ≈ 1/M^3). `enterpoint` updates when new max appears.

## Quiz

<div class="quiz" data-answer="A">
<p><strong>Level formula?</strong></p>
<button data-choice="A">floor(-ln(U) * mult) with mult=1/ln(M)</button>
<button data-choice="B">uniform 0..M</button>
<button data-choice="C">ln(U)/M</button>
<p class="feedback" data-ok="hnswalg.h:142" data-no="See reference/hnsw-architecture.md."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
