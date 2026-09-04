"""Neighbor selection — simple and heuristic. See reference/neighbor-heuristic-deep.md."""


def select_simple(candidates: list[tuple[float, int]], M: int) -> list[int]:
    """candidates: sorted (dist, id) or unsorted; return M closest ids."""
    candidates = sorted(candidates)
    return [cid for _, cid in candidates[:M]]


def select_heuristic(
    candidates: list[tuple[float, int]],
    M: int,
    qvec: list[float],
    data: list[list[float]],
    space,
) -> list[int]:
    """Diversity pruning — Paper Alg. 4. Keep c only if not occluded by any kept r.

    Occluded if dist(data[c], data[r]) < dist(c, q).
    Fallback fill ensures size M if possible.
    """
    candidates = sorted(candidates)
    result: list[int] = []
    for d_cq, cid in candidates:
        if len(result) >= M:
            break
        keep = True
        for rid in result:
            if space.dist(data[cid], data[rid]) < d_cq:
                keep = False
                break
        if keep:
            result.append(cid)
    if len(result) < M:
        seen = set(result)
        for _, cid in candidates:
            if cid not in seen:
                result.append(cid)
                seen.add(cid)
                if len(result) >= M:
                    break
    return result[:M]
