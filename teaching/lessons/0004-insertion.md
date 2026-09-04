# Insertion — Search → Select → Connect

*Lesson 04 · 150 min · reference/hnsw-architecture.md · hnsw.py add_item*

*The full add path: top-down ef=1 hunt, then ef_construction per layer.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/hnsw-architecture.md` Construction section.
- `hnswalg.h` `addPoint` flow.

## Build

```python
def add_item(self, vec, label=None):
    cur_idx=len(self.data)
    level=self._random_level()
    self.data.append(vec); self.levels.append(level)
    self.links.append([array.array('I') for _ in range(level+1)])
    if self.enterpoint==-1:
        self.enterpoint=cur_idx; self.maxlevel=level; return
    # 1) top layers af=1
    cur=self.enterpoint
    for lc in range(self.maxlevel, level, -1):
        cand=self.search_layer(vec, cur, 1, lc)
        cur=cand[0][1]
    # 2) for lc=level..0
    for lc in range(min(level,self.maxlevel), -1, -1):
        cand=self.search_layer(vec, cur, self.ef_construction, lc)
        neighbors=self.select_neighbors(cand, self.maxM0 if lc==0 else self.maxM, vec)
        # bidirectional
        self.links[cur_idx][lc].extend(neighbors)
        for nb in neighbors:
            # ensure nb has layer lc
            if lc < len(self.links[nb]):
                self.links[nb][lc].append(cur_idx)
                # prune if exceeds cap (call select_neighbors on nb's adj)
```

Implement `search_layer` reusing Lesson 02 but dispatching per `layer` adj.

## Verify

Insert 1000 random 8-d, then self-recall: query each inserted point, expect itself rank 1 at ef=10.

## Quiz

<div class="quiz" data-answer="B">
<p><strong>Why ef=1 on upper layers during insert?</strong></p>
<button data-choice="A">Upper layers need high accuracy</button>
<button data-choice="B">Greedy coarse hunt is enough; L0 does accurate search</button>
<button data-choice="C">ef must equal M</button>
<p class="feedback" data-ok="Architecture ref." data-no="Re-read construction steps."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
