from src.heuristics import select_simple, select_heuristic
from src.spaces import L2Space

def test_simple():
    cands=[(5,1),(1,0),(3,2)]
    assert select_simple(cands,2)==[0,2]

def test_heuristic_line():
    # q 0 - a1 - b2 collinear: a occludes b
    s=L2Space()
    data=[[0.0],[1.0],[2.0]]  # 1-d for dist
    q=[0.0]
    cands=[(1.0,1),(4.0,2)]  # dist(q,a)=1, dist(q,b)=4
    # dist(a,b)=1 < 4 so b occluded
    res=select_heuristic(cands,2,q,data,s)
    assert res==[1] or res==[1,2]  # fallback may add b, but first must be a
    assert 1 in res

def test_heuristic_diversity():
    s=L2Space()
    data=[[0,0],[1,0],[0,1],[10,0]]
    q=[0,0]
    # candidates a(1,0) dist1, b(0,1) dist1, c(10,0) dist100
    cands=[(1,1),(1,2),(100,3)]
    res=select_heuristic(cands,2,q,data,s)
    assert len(res)==2
    assert 1 in res or 2 in res

if __name__=="__main__":
    test_simple(); test_heuristic_line(); test_heuristic_diversity(); print("heuristic ok")
