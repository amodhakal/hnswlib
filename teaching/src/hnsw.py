"""HNSW — pure Python stdlib. See reference/hnsw-architecture.md."""
import math
import random
import heapq
import array
from .spaces import L2Space
from .heuristics import select_simple, select_heuristic


class HNSW:
    __slots__ = (
        "M",
        "maxM",
        "maxM0",
        "ef_construction",
        "space",
        "mult",
        "revSize",
        "max_elements",
        "levels",
        "data",
        "links",
        "enterpoint",
        "maxlevel",
        "deleted",
        "num_deleted",
    )

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        space=None,
        seed: int = 100,
        max_elements: int = 10000,
    ):
        self.M = M
        self.maxM = M
        self.maxM0 = M * 2
        self.ef_construction = max(ef_construction, M)
        self.space = space or L2Space()
        self.mult = 1 / math.log(M)  # hnswlib/hnswalg.h:142
        self.revSize = 1 / self.mult
        random.seed(seed)
        self.max_elements = max_elements
        self.levels = array.array("b")
        self.data: list[list[float]] = []
        self.links: list[list[array.array]] = []  # per node: list per layer
        self.enterpoint = -1
        self.maxlevel = -1
        self.deleted = bytearray(max_elements)
        self.num_deleted = 0

    def _random_level(self) -> int:
        r = random.random()
        if r == 0:
            r = 1e-12
        return int(-math.log(r) * self.mult)

    # ---- search on a single layer ----
    def _search_layer(self, q: list[float], entry: int, ef: int, layer: int) -> list[tuple[float, int]]:
        n = len(self.data)
        if n == 0 or entry < 0:
            return []
        visited = bytearray(n)
        visited[entry] = 1
        d0 = self.space.dist(self.data[entry], q)
        candidates: list[tuple[float, int]] = [(d0, entry)]
        best: list[tuple[float, int]] = [(-d0, entry)]
        heapq.heapify(candidates)
        while candidates:
            d_c, c = heapq.heappop(candidates)
            furthest = -best[0][0] if best else float("inf")
            if d_c > furthest:
                break
            # neighbors on this layer
            adj = self.links[c][layer] if layer < len(self.links[c]) else []
            for nb in adj:
                if not visited[nb]:
                    visited[nb] = 1
                    d = self.space.dist(self.data[nb], q)
                    if len(best) < ef or d < furthest:
                        heapq.heappush(candidates, (d, nb))
                        heapq.heappush(best, (-d, nb))
                        if len(best) > ef:
                            heapq.heappop(best)
                            furthest = -best[0][0]
        res = [(-d, cid) for d, cid in best]
        res.sort()
        return res

    def _select_neighbors(self, candidates, M, qvec, heuristic=True):
        if heuristic:
            return select_heuristic(candidates, M, qvec, self.data, self.space)
        return select_simple(candidates, M)

    def _prune(self, node: int, layer: int, M: int):
        """Enforce max degree on node/layer via heuristic."""
        adj = self.links[node][layer]
        if len(adj) <= M:
            return
        # build candidates from current adj
        qvec = self.data[node]
        cands = [(self.space.dist(self.data[nb], qvec), nb) for nb in adj]
        keep = set(self._select_neighbors(cands, M, qvec, heuristic=True))
        # filter in place — keep order of keep set sorted by dist
        self.links[node][layer] = array.array("I", [nb for _, nb in sorted(cands) if nb in keep])

    # ---- public ----
    def add_item(self, vec: list[float], label=None):
        idx = len(self.data)
        if idx >= self.max_elements:
            raise RuntimeError("max_elements exceeded; call resize_index")
        level = self._random_level()
        self.data.append(list(vec))
        self.levels.append(level)
        self.links.append([array.array("I") for _ in range(level + 1)])
        if self.enterpoint == -1:
            self.enterpoint = idx
            self.maxlevel = level
            return idx
        cur = self.enterpoint
        # greedy top layers
        for lc in range(self.maxlevel, level, -1):
            cand = self._search_layer(vec, cur, 1, lc)
            cur = cand[0][1]
        for lc in range(min(level, self.maxlevel), -1, -1):
            cand = self._search_layer(vec, cur, self.ef_construction, lc)
            M = self.maxM0 if lc == 0 else self.maxM
            neighbors = self._select_neighbors(cand, M, vec, heuristic=True)
            # bidirectional
            self.links[idx][lc].extend(neighbors)
            for nb in neighbors:
                if lc < len(self.links[nb]):
                    self.links[nb][lc].append(idx)
                    if len(self.links[nb][lc]) > M:
                        self._prune(nb, lc, M)
            # update cur for next layer
            # closest among cand becomes cur
            cur = cand[0][1] if cand else cur
        if level > self.maxlevel:
            self.enterpoint = idx
            self.maxlevel = level
        return idx

    def knn_query(self, q: list[float], k: int = 5, ef: int = 10) -> list[tuple[float, int]]:
        assert ef >= k, "ef must be >= k (ALGO_PARAMS.md:5)"
        if self.enterpoint == -1:
            return []
        cur = self.enterpoint
        for lc in range(self.maxlevel, 0, -1):
            cand = self._search_layer(q, cur, 1, lc)
            cur = cand[0][1]
        best = self._search_layer(q, cur, ef, 0)
        # filter deleted
        filtered = [(d, cid) for d, cid in best if not self.deleted[cid]]
        filtered.sort()
        return filtered[:k]

    # ---- deletions ----
    def mark_deleted(self, idx: int):
        if self.deleted[idx]:
            raise ValueError("already deleted")
        self.deleted[idx] = 1
        self.num_deleted += 1

    def unmark_deleted(self, idx: int):
        if not self.deleted[idx]:
            raise ValueError("not deleted")
        self.deleted[idx] = 0
        self.num_deleted -= 1

    def resize_index(self, new_max: int):
        if new_max < len(self.data):
            raise ValueError("new_max < current count")
        self.max_elements = new_max
        new = bytearray(new_max)
        new[: len(self.deleted)] = self.deleted[: len(self.data)]
        self.deleted = new
