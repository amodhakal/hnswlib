# Persistence — Save / Load (struct + array)

*Lesson 09 · 90 min · reference/persistence-format.md · persistence.py*

*One binary file, little-endian, versioned — no pickle for vectors.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/persistence-format.md` header + per-element layout.
- `hnswalg.h:loadIndex` / `saveIndex` order.

## Build

`src/persistence.py`:
```python
import struct, array
MAGIC=b'HNSW'; VERSION=1
def save(hnsw, path):
    with open(path,'wb') as f:
        f.write(struct.pack('<4sHHHIh8s', MAGIC, VERSION, hnsw.M, len(hnsw.data[0]) if hnsw.data else 0, len(hnsw.data), hnsw.maxlevel, hnsw.space.name.encode().ljust(8,b'\x00')))
        for i, vec in enumerate(hnsw.data):
            lvl=hnsw.levels[i]; a=array.array('f', vec); f.write(struct.pack('<b i b', lvl, i, 0)); a.tofile(f)
            for lc in range(lvl+1):
                arr=hnsw.links[i][lc]; f.write(struct.pack('<H', len(arr))); arr.tofile(f)
```

Load reverses, rebuilds `enterpoint`.

## Verify

Save → load → identical `knn_query` results for 20 random queries; bit-identical recall.

## Quiz

<div class="quiz" data-answer="B">
<p><strong>Persistence must?</strong></p>
<button data-choice="A">Pickle entire object is fine for vectors</button>
<button data-choice="B">Struct header + array tofile, versioned, little-endian</button>
<button data-choice="C">JSON vectors + text links</button>
<p class="feedback" data-ok="Per reference/persistence-format.md." data-no="See persistence ref."></p>
</div>

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
