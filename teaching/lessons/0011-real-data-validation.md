# Real-Data Validation — Your Downloads, Brute-Force Truth

*Lesson 11 · 90 min · reference/datasets.md · datasets.py + metrics.py*

*No harness — you fetch SIFT/GloVe, we load with struct/array and compare to brute force.*

> Primary sources: [Malkov & Yashunin 2018 §3-4](https://arxiv.org/abs/1603.09320) · `hnswlib/hnswalg.h:34-144` · `reference/` docs. Ask your teacher with follow-ups.

## Study

- `reference/datasets.md` fvecs/bvecs/Glove loaders.
- `TESTING_RECALL.md` recall-vs-ef plotting pattern.

## Build

`src/datasets.py` (stdlib only):
- `load_fvecs(path, limit)` as in reference.
- `src/metrics.py`: `recall_at_k(brute_ids, hnsw_ids, k)` and `time_qps`.

Your own script (you run):

```python
from src.hnsw import HNSW
from src.spaces import L2Space
from src.datasets import load_fvecs
from src.metrics import recall_at_k
import time

base=load_fvecs('sift_base.fvecs', limit=10000)
queries=load_fvecs('sift_query.fvecs', limit=100)
# brute
# build HNSW, time build, time queries at ef=10,50,100, table recall@10 vs brute
```

## Measure

Report table: ef | recall@1 | recall@10 | recall@100 | QPS | vs `M*ef_construction` constant check (`ALGO_PARAMS.md:22`).

Download via `wget http://corpus-texmex.irisa.fr/.../sift.tar.gz` per `tests/cpp/download_bigann.py` logic.

## Verify

On SIFT10K, target **recall@10 >0.90 at ef=50, M=16, ef_construction=200**; note QPS pure-Python will be 10-50× slower than hnswlib — acceptable.

Ask teacher after first table for tuning.

---

**Links:** [Reference docs](../reference/) · [Glossary](../reference/glossary.md) · [MISSION](../MISSION.md) · Next lesson in `lessons/`
