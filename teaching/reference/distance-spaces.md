# Reference: Distance Spaces

Queries: `dist(a,b) -> float` for `a,b: list[float]` same `dim`. No NumPy; use `math.fsum`, `math.sqrt`.

## L2 (squared Euclidean)

`d = sum((Ai-Bi)^2)` — matches `README.md:51` `l2`. Use `math.fsum((x-y)**2 for x,y in zip(a,b))`. No sqrt unless you need true Euclidean (recall unchanged; keep squared for speed, document it).

## Inner Product (IP)

`d = 1 - dot(A,B)` with `dot = sum(Ai*Bi)` (`README.md:52`). Not a metric: element can be closer to another than itself. Return `1 - dot`.

## Cosine

`d = 1 - dot / (||A||*||B||)` (`README.md:53`). Guard `norm==0 -> 1.0`. For stored vectors, optionally normalize on insert and fall back to IP, but spec requires raw handling.

## Implementation contract (`src/spaces.py`)

```python
class Space:
    def dist(self, a: list[float], b: list[float]) -> float: ...
    @property
    def name(self) -> str: ...
```

Bind local `dist = space.dist` in hot loops.

## Pitfalls

- Mixing squared vs rooted L2 in recall check — be consistent.
- IP expects same scale; don't L2-normalize unless Cosine.
- Type: `list[float]` not `array('f')` for dist inputs (convert once).
