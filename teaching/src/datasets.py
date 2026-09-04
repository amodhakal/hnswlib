"""Dataset loaders — stdlib only. See reference/datasets.md."""
import struct
import array


def load_fvecs(path: str, limit: int | None = None) -> list[list[float]]:
    """TEXMEX fvecs: int32 dim then dim*float32 per vector, little-endian."""
    vecs: list[list[float]] = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) < 4:
                break
            dim = struct.unpack("<i", hdr)[0]
            buf = f.read(dim * 4)
            if len(buf) < dim * 4:
                break
            a = array.array("f")
            a.frombytes(buf)
            # fvecs are little-endian float32
            vecs.append(list(a))
            if limit and len(vecs) >= limit:
                break
    return vecs


def load_bvecs(path: str, limit: int | None = None) -> list[list[float]]:
    """bvecs: int32 dim then dim*uint8. Returns float vectors."""
    vecs: list[list[float]] = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            dim = struct.unpack("<i", hdr)[0]
            buf = f.read(dim)
            if len(buf) < dim:
                break
            vecs.append([float(b) for b in buf])
            if limit and len(vecs) >= limit:
                break
    return vecs


def load_glove_txt(path: str, limit: int | None = None) -> list[list[float]]:
    vecs: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            vecs.append([float(x) for x in parts[1:]])
            if limit and len(vecs) >= limit:
                break
    return vecs
