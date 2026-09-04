# Memory & Speed — Pure-Python Tricks

*Lesson 08 · 90 min · hnswalg.h:120-123 layout · __slots__, array, locals*

*No NumPy, so squeeze Python: contiguous arrays, slots, local binding.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `hnswalg.h:120-123` contiguous `data_level0_memory_` layout inspiration.
- `visited_list_pool.h` pooling idea.

## Build

- Store links as `list[list[array('I')]]` not `list[list[list[int]]]`.
- Add `__slots__ = ('M','maxM','maxM0','mult','levels','data','links','enterpoint','maxlevel','space','ef_construction')` to `HNSW`.
- In hot loops: `dist=space.dist; data=self.data; links=self.links` local alias.
- Visited: reuse `bytearray` epoch tag (int array + cur_epoch) instead of fresh alloc.
- Prealloc `self.data` if `max_elements` known.

## Measure

`time.perf_counter` around 500 queries before/after — report QPS delta in `NOTES.md`. Expect 10-20% gain from locals + bytearray reuse.

## Quiz

<div class="quiz" data-answer="A">
<p><strong>Best pure-Python win?</strong></p>
<button data-choice="A">Local variable binding + array('I') + bytearray pool</button>
<button data-choice="B">Using set for visited</button>
<button data-choice="C">Recursive search_layer</button>
<p class="feedback" data-ok="Profile then fix." data-no="Check optimization ref."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
