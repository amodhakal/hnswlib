# Updates & Deletions — Tombstones & Replace

*Lesson 10 · 90 min · visited_list_pool / deleted handling · hnsw.py*

*Mark-deleted, skip in search, optionally reuse slot.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `hnswalg.h:DELETE_MARK`, `deleted_elements` set, `allow_replace_deleted_`.

## Build

- `self.deleted = bytearray(max_elements)` or `set()`.
- `mark_deleted(label_or_id)`: set flag, `num_deleted+=1`; search skips `if deleted[cid]` but still traverses graph.
- `unmark_deleted`, `replace_deleted(vec, deleted_id)` reuses slot (keep links, re-insert).
- `resize_index(new_max)` realloc `levels`, `deleted`.

Search change: filter `best` before returning top k; if insufficient non-deleted, return whatever found.

## Verify

Insert 200, delete 20 random, query — deleted never appears. Replace 10 — recall restored.

## Quiz

<div class="quiz" data-answer="A">
<p><strong>Deleted node during search?</strong></p>
<button data-choice="A">Traversed but not returned; best still needs filtering</button>
<button data-choice="B">Removed from graph immediately</button>
<button data-choice="C">Ignored completely, links deleted</button>
<p class="feedback" data-ok="Tombstone pattern." data-no="See lesson body."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
