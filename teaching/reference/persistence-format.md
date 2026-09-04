# Reference: Persistence Format

No pickle for vectors; use `struct` + `array('f')`/`array('I')`.

## File layout (little-endian)

Header (20 bytes): `magic b'HNSW'`, `version: H`, `M: H`, `dim: H`, `num_elements: I`, `maxlevel: h`, `space_name: 8s` (padded).
Per element:
- `level: b`, `label: i`, `deleted: b`
- `vec: dim * f` (`array('f').tofile`)
- For each layer `0..level`: `deg: H`, `neighbors: deg * I`

Use `struct.pack('<4sHHHIh8s', ...)` / `unpack`. Little-endian fixed.

## Load

Allocate `max_elements`, `enterpoint`, rebuild `linkLists` as `list[list[array('I')]]`. Validate `dim` and `M` match.

## Alternatives

JSON + raw bin sidecar works but loses atomicity. Stick to single binary.

## Deleted

Tombstone byte per element; skipped in search, kept in links until `replace_deleted` compacts (Lesson 09).
