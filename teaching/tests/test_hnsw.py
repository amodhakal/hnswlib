import random
from src.hnsw import HNSW
from src.spaces import L2Space
from src.metrics import brute_knn

def test_self_recall():
    random.seed(0)
    hnsw=HNSW(M=8, ef_construction=50, space=L2Space(), seed=1, max_elements=200)
    data=[[random.random() for _ in range(8)] for _ in range(100)]
    for v in data:
        hnsw.add_item(v)
    for i, q in enumerate(data):
        res=hnsw.knn_query(q, k=1, ef=10)
        assert res[0][1]==i, f"miss {i} got {res}"

def test_recall_vs_brute():
    random.seed(1)
    hnsw=HNSW(M=16, ef_construction=100, space=L2Space(), seed=2, max_elements=500)
    data=[[random.random() for _ in range(4)] for _ in range(200)]
    for v in data:
        hnsw.add_item(v)
    queries=data[:20]
    brute=brute_knn(data, queries, k=5, space=L2Space())
    for q, b in zip(queries, brute):
        got=[cid for _,cid in hnsw.knn_query(q,k=5,ef=50)]
        # at least 3/5 overlap expected
        assert len(set(got)&set(b))>=3, f"low recall {got} vs {b}"

if __name__=="__main__":
    test_self_recall(); test_recall_vs_brute(); print("hnsw ok")
