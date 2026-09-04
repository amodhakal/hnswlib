"""NSW single-layer building block. See reference/nsw-algorithm.md."""
import heapq
import array


class NSW:
    def __init__(self, maxM: int = 16):
        self.M = maxM
        self.data: list[list[float]] = []
        self.adj: list[array.array] = []

    def add_node(self, vec: list[float]) -> int:
        self.data.append(vec)
        self.adj.append(array.array("I"))
        return len(self.data) - 1

    def search_layer(self, q: list[float], entry: int, ef: int, space) -> list[tuple[float, int]]:
        """Return ef closest (dist, id) on this single layer."""
        n = len(self.data)
        if n == 0 or entry < 0:
            return []
        visited = bytearray(n)
        visited[entry] = 1
        d0 = space.dist(self.data[entry], q)
        # candidates: min-heap by dist; best: max-heap via neg dist
        candidates: list[tuple[float, int]] = [(d0, entry)]
        best: list[tuple[float, int]] = [(-d0, entry)]  # max-heap
        heapq.heapify(candidates)
        # best as max-heap: store (-dist, id)
        while candidates:
            d_c, c = heapq.heappop(candidates)
            # furthest in best
            furthest = -best[0][0] if best else float("inf")
            if d_c > furthest:
                break
            for nb in self.adj[c]:
                if not visited[nb]:
                    visited[nb] = 1
                    d = space.dist(self.data[nb], q)
                    if len(best) < ef or d < furthest:
                        heapq.heappush(candidates, (d, nb))
                        heapq.heappush(best, (-d, nb))
                        if len(best) > ef:
                            heapq.heappop(best)
                            furthest = -best[0][0]
        # convert best to sorted ascending
        res = [(-d, cid) for d, cid in best]
        res.sort()
        return res
