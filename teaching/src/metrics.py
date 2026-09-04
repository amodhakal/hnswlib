"""Metrics — pure Python. See reference/datasets.md."""


def recall_at_k(brute_ids: list[list[int]], hnsw_ids: list[list[int]], k: int) -> float:
    """Micro-averaged recall@k. Each entry: list of k ids."""
    assert len(brute_ids) == len(hnsw_ids)
    total = 0
    hit = 0
    for b, h in zip(brute_ids, hnsw_ids):
        bset = set(b[:k])
        hset = set(h[:k])
        hit += len(bset & hset)
        total += k
    return hit / total if total else 0.0


def brute_knn(data: list[list[float]], queries: list[list[float]], k: int, space) -> list[list[int]]:
    """Brute-force kNN for validation."""
    res: list[list[int]] = []
    for q in queries:
        dists = [(space.dist(vec, q), i) for i, vec in enumerate(data)]
        dists.sort()
        res.append([i for _, i in dists[:k]])
    return res
