"""Persistence — struct + array. See reference/persistence-format.md."""
import struct
import array
from .hnsw import HNSW
from .spaces import SPACES

MAGIC = b"HNSW"
VERSION = 1
HEADER_FMT = "<4sHHHIh8s"  # magic, version, M, dim, n, maxlevel, space(8s)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def save(hnsw: HNSW, path: str):
    dim = len(hnsw.data[0]) if hnsw.data else 0
    space_name = hnsw.space.name.encode().ljust(8, b"\x00")
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                HEADER_FMT, MAGIC, VERSION, hnsw.M, dim, len(hnsw.data), hnsw.maxlevel, space_name
            )
        )
        for i, vec in enumerate(hnsw.data):
            lvl = int(hnsw.levels[i])
            deleted = int(hnsw.deleted[i]) if i < len(hnsw.deleted) else 0
            f.write(struct.pack("<b i b", lvl, i, deleted))
            arr = array.array("f", vec)
            arr.tofile(f)
            for lc in range(lvl + 1):
                nb = hnsw.links[i][lc]
                f.write(struct.pack("<H", len(nb)))
                nb.tofile(f)


def load(path: str) -> HNSW:
    with open(path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
        magic, ver, M, dim, n, maxlevel, sname = struct.unpack(HEADER_FMT, hdr)
        assert magic == MAGIC, "bad magic"
        assert ver == VERSION, f"unsupported version {ver}"
        space_name = sname.rstrip(b"\x00").decode()
        space = SPACES.get(space_name, SPACES["l2"])()
        hnsw = HNSW(M=M, ef_construction=200, space=space, max_elements=max(n, 10000))
        hnsw.maxlevel = maxlevel
        hnsw.data = []
        hnsw.links = []
        hnsw.levels = array.array("b")
        for _ in range(n):
            lvl, idx, deleted = struct.unpack("<b i b", f.read(6))
            buf = f.read(dim * 4)
            a = array.array("f")
            a.frombytes(buf)
            hnsw.data.append(list(a))
            hnsw.levels.append(lvl)
            hnsw.deleted[idx] = deleted
            layers = []
            for _ in range(lvl + 1):
                (deg,) = struct.unpack("<H", f.read(2))
                nb = array.array("I")
                if deg:
                    nb.frombytes(f.read(deg * 4))
                layers.append(nb)
            hnsw.links.append(layers)
        # recover enterpoint = first node with maxlevel
        for i, lv in enumerate(hnsw.levels):
            if lv == maxlevel:
                hnsw.enterpoint = i
                break
        return hnsw
