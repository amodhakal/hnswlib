"""Distance spaces — stdlib only. No NumPy. See reference/distance-spaces.md."""
import math


class L2Space:
    name = "l2"

    def dist(self, a: list[float], b: list[float]) -> float:
        # squared L2 — ordering identical to rooted
        return math.fsum((x - y) * (x - y) for x, y in zip(a, b))


class IPSpace:
    name = "ip"

    def dist(self, a: list[float], b: list[float]) -> float:
        return 1.0 - math.fsum(x * y for x, y in zip(a, b))


class CosineSpace:
    name = "cosine"

    def dist(self, a: list[float], b: list[float]) -> float:
        dot = math.fsum(x * y for x, y in zip(a, b))
        na = math.sqrt(math.fsum(x * x for x in a))
        nb = math.sqrt(math.fsum(x * x for x in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)


SPACES = {"l2": L2Space, "ip": IPSpace, "cosine": CosineSpace}
