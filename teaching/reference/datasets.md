# Reference: Real Datasets (You Download)

No harness; you fetch and load with `src/datasets.py` (stdlib only).

## fvecs / bvecs (SIFT, GIST, BigANN)

Format per vector: `dim: i` (int32) then `dim * float32` (fvecs) or `uint8` (bvecs). Loader:

```python
import struct, array
def load_fvecs(path, limit=None):
    vecs=[]
    with open(path,'rb') as f:
        while True:
            hdr=f.read(4)
            if not hdr: break
            dim=struct.unpack('<i', hdr)[0]
            buf=f.read(dim*4)
            a=array.array('f'); a.frombytes(buf)
            vecs.append(list(a))
            if limit and len(vecs)>=limit: break
    return vecs
```

Source: http://corpus-texmex.irisa.fr/ (SIFT1M sift_base.fvecs). Download via `wget` as in `tests/cpp/download_bigann.py`.

## Text vectors (GloVe)

`glove.6B.50d.txt` lines: `word v0 v1 ...`. Parse `map(float, parts[1:])`.

## Metrics (`src/metrics.py`)

```python
def recall_at_k(brute_ids, hnsw_ids, k): ...  # len(intersection)/k per query, micro-averaged
def qps(n_queries, wall_seconds): return n_queries / wall_seconds
```

Measure with `time.perf_counter()` around `knn_query` loop; report per `ef` value.
