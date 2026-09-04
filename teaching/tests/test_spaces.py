import math
from src.spaces import L2Space, IPSpace, CosineSpace

def test_l2():
    s = L2Space()
    assert s.dist([0,0],[3,4]) == 25
    assert s.dist([1,2],[1,2]) == 0

def test_ip():
    s = IPSpace()
    assert abs(s.dist([1,0],[1,0]) - 0) < 1e-9
    assert abs(s.dist([1,0],[0,1]) - 1) < 1e-9

def test_cosine():
    s = CosineSpace()
    assert abs(s.dist([1,0],[0,1]) - 1.0) < 1e-9
    assert abs(s.dist([1,0],[1,0]) - 0.0) < 1e-9
    assert s.dist([0,0],[1,0]) == 1.0

if __name__ == "__main__":
    test_l2(); test_ip(); test_cosine(); print("spaces ok")
