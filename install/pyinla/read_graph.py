# read_graph.py
"""
Read/write INLA graph objects (ASCII or binary), and build graphs from matrices.

Graph object mirrors R's structure:
- n      : number of nodes (int)
- nnbs   : list[int] number of neighbors for each node (1-based nodes)
- nbs    : list[list[int]] neighbor lists (1-based indices)
- cc     : dict with connected component info:
           { 'id': list[int], 'n': int, 'nodes': list[list[int]], 'mean': list[int|None] }

Binary format matches the R writer (1-based indices):
  int32 magic = -1
  int32 n
  repeat i=1..n:
    int32 i
    int32 nnbs[i]
    int32 neighbors[nnbs[i]]
"""

from __future__ import annotations

import gzip
import io
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp  # optional
except Exception:
    sp = None

from .sm import inla_as_dgTMatrix  # your earlier helper


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class InlaGraph:
    n: int
    nnbs: List[int]
    nbs: List[List[int]]
    cc: Optional[dict] = None
    graph_file: Optional[str] = None

    def with_cc(self) -> "InlaGraph":
        self.cc = add_graph_cc(self)
        return self


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def inla_graph_binary_file_magic() -> int:
    return -1


def _add_graph_cc_impl(g: InlaGraph) -> dict:
    n = g.n
    s = [0] * n  # component id per node; 0 = unvisited
    comp = 0

    # Adjacency (1-based on disk; use 0-based here)
    adj = [list(map(lambda x: x - 1, nbrs)) for nbrs in g.nbs]

    def bfs(start: int, label: int):
        queue = [start]
        while queue:
            u = queue.pop(0)
            if s[u] != 0:
                continue
            s[u] = label
            for v in adj[u]:
                if s[v] == 0:
                    queue.append(v)

    for i in range(n):
        if s[i] == 0:
            comp += 1
            bfs(i, comp)

    nodes = [[] for _ in range(comp)]
    for i, cid in enumerate(s):
        nodes[cid - 1].append(i + 1)  # back to 1-based

    # 'mean' factor: NA for comps of size 1; otherwise component id
    mean = [None] * n
    for cid, idxs in enumerate(nodes, start=1):
        if len(idxs) > 1:
            for i in idxs:
                mean[i - 1] = cid

    cc = {"id": s, "n": comp, "nodes": nodes, "mean": mean}
    return cc


def add_graph_cc(graph: Union[InlaGraph, Any]) -> dict:
    """
    Add connected components info; accept InlaGraph or a graph-like input.
    """
    if isinstance(graph, InlaGraph):
        return _add_graph_cc_impl(graph)
    else:
        g = inla_read_graph(graph)
        return _add_graph_cc_impl(g)


def inla_graph_size(graph_like: Any) -> int:
    return inla_read_graph(graph_like, size_only=True)


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------

def _read_ascii_internal(filename: str, offset: int = 0, size_only: bool = False) -> Optional[InlaGraph]:
    if not os.path.exists(filename):
        return None

    with open(filename, "r", encoding="utf-8") as fp:
        lines = [l.split("#", 1)[0].strip() for l in fp]
    toks: List[int] = []
    for line in lines:
        if not line:
            continue
        toks.extend([int(t) for t in line.split()])

    if not toks:
        return None

    n = toks[0]
    if size_only:
        return n  # type: ignore[return-value]

    nnbs = [0] * n
    nbs: List[List[int]] = [[] for _ in range(n)]

    k = 1  # next token index
    while k < len(toks):
        idx = toks[k] + offset
        if idx == 0:
            # zero-based file: re-read with offset=1
            return _read_ascii_internal(filename, offset=1, size_only=size_only)
        if not (1 <= idx <= n):
            raise ValueError("Index out of bounds in ASCII graph file.")
        k += 1
        deg = toks[k]
        k += 1

        nnbs[idx - 1] = deg
        if deg > 0:
            nbrs = [t + offset for t in toks[k:k + deg]]
            k += deg
            nbs[idx - 1] = nbrs

    g = InlaGraph(n=n, nnbs=nnbs, nbs=nbs, cc=None, graph_file=filename)
    if len(g.nbs) < g.n:
        g.nbs.extend([[] for _ in range(g.n - len(g.nbs))])
    g.cc = add_graph_cc(g)
    return g


def _is_gzip_file(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(2)
    return head == b"\x1f\x8b"


def _read_binary_internal(filename: str, offset: int = 0, size_only: bool = False) -> Optional[InlaGraph]:
    if not os.path.exists(filename):
        return None

    # Open gz or plain
    fh: io.BufferedReader
    if _is_gzip_file(filename):
        fh = gzip.open(filename, "rb")  # type: ignore[assignment]
    else:
        fh = open(filename, "rb")

    with fh:
        # magic
        data = fh.read(4)
        if len(data) < 4:
            return None
        (magic,) = struct.unpack("=i", data)
        if magic != inla_graph_binary_file_magic():
            return None

        (n,) = struct.unpack("=i", fh.read(4))
        if size_only:
            return n  # type: ignore[return-value]

        nnbs = [0] * n
        nbs: List[List[int]] = [[] for _ in range(n)]

        for _ in range(n):
            (idx,) = struct.unpack("=i", fh.read(4))
            idx += offset
            if idx == 0:
                # zero-based -> rewind: re-read entire file with offset=1
                return _read_binary_internal(filename, offset=1, size_only=size_only)
            if not (1 <= idx <= n):
                raise ValueError("Index out of bounds in binary graph file.")
            (deg,) = struct.unpack("=i", fh.read(4))
            nnbs[idx - 1] = deg
            if deg > 0:
                nbrs = list(struct.unpack("=" + "i" * deg, fh.read(4 * deg)))
                nbs[idx - 1] = [v + offset for v in nbrs]

        g = InlaGraph(n=n, nnbs=nnbs, nbs=nbs, cc=None, graph_file=filename)
        if len(g.nbs) < g.n:
            g.nbs.extend([[] for _ in range(g.n - len(g.nbs))])
        g.cc = add_graph_cc(g)
        return g


def _matrix_to_graph_internal(Q: Any, size_only: bool = False) -> InlaGraph:
    if Q is None:
        raise ValueError("Q must be provided")

    if sp is not None and sp.issparse(Q):
        M = inla_as_dgTMatrix(Q)
        n = M.shape[0]
        if size_only:
            return n  # type: ignore[return-value]
        # Force diagonal to 1; then neighbors are (row,col) with x!=0 and col!=row
        diag = set(range(n))
        row = M.row.astype(int)
        col = M.col.astype(int)
        x = M.data
        # Build adjacency 1-based, unique
        nbs = [[] for _ in range(n)]
        nnbs = [0] * n
        # ensure diagonal "present"
        # (not strictly needed for neighbor extraction but matches R logic)
        for i, j, v in zip(row, col, x):
            if i == j:
                continue
            if v != 0.0:
                nbs[i].append(j + 1)
        for i in range(n):
            # unique + sorted (to mimic stable behavior)
            lst = sorted(set(nbs[i]))
            nbs[i] = lst
            nnbs[i] = len(lst)
        g = InlaGraph(n=n, nnbs=nnbs, nbs=nbs).with_cc()
        return g

    # Dense numpy or similar
    A = np.asarray(Q)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {A.shape}")
    n = A.shape[0]
    if size_only:
        return n  # type: ignore[return-value]

    # Neighbors for non-zero off-diagonals
    nbs = []
    for i in range(n):
        idxs = [j + 1 for j in range(n) if j != i and A[i, j] != 0]
        nbs.append(idxs)
    nnbs = [len(lst) for lst in nbs]
    g = InlaGraph(n=n, nnbs=nnbs, nbs=nbs).with_cc()
    return g


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def inla_read_graph(*args, size_only: bool = False) -> InlaGraph:
    """
    Read a graph from:
      - filename (str): ASCII or (gz) binary detected automatically
      - InlaGraph: returned as-is (or size)
      - matrix (numpy or scipy.sparse)
      - definition (sequence of ints/strings) -> treated as inline ASCII
    """
    if not args:
        raise ValueError("Provide a graph description (file, matrix, or inline sequence)")

    graph = args[0]

    # String: path or inline single-line
    if isinstance(graph, str) or len(args) > 1 or (isinstance(graph, (int, float)) and not hasattr(graph, "shape")):
        # Assemble tokens/lines
        if isinstance(graph, str) and os.path.exists(graph) and os.path.isfile(graph):
            # Try binary then ASCII
            g = _read_binary_internal(graph, offset=0, size_only=size_only)
            if g is None:
                g = _read_ascii_internal(graph, offset=0, size_only=size_only)
            if g is None:
                raise ValueError(f"Unrecognized graph file: {graph}")
            return g
        else:
            # Treat as inline definition: write to temp file and parse
            tfile = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
            try:
                for v in args:
                    tfile.write(str(v) + "\n")
            finally:
                tfile.close()
            try:
                g = inla_read_graph(tfile.name, size_only=size_only)
                return g
            finally:
                try:
                    os.unlink(tfile.name)
                except OSError:
                    pass

    # Existing InlaGraph
    if isinstance(graph, InlaGraph):
        return graph if not size_only else graph.n  # type: ignore[return-value]

    # Matrix-like
    return _matrix_to_graph_internal(graph, size_only=size_only)


def inla_write_graph(graph_like: Any, filename: str = "graph.dat", mode: str = "binary", **kwargs) -> str:
    """
    Write graph to ASCII or binary file. Returns filename.
    """
    g = inla_read_graph(graph_like)

    mode = str(mode).lower()
    if mode == "binary":
        return _write_graph_binary_internal(g, filename)
    elif mode == "ascii":
        return _write_graph_ascii_internal(g, filename)
    else:
        raise ValueError("mode must be 'binary' or 'ascii'")


def _write_graph_ascii_internal(g: InlaGraph, filename: str) -> str:
    with open(filename, "w", encoding="utf-8") as fd:
        fd.write(f"{g.n}\n")
        for i in range(1, g.n + 1):
            nbrs = g.nbs[i - 1]
            fd.write(f"{i} {len(nbrs)}")
            if nbrs:
                fd.write(" " + " ".join(str(v) for v in nbrs))
            fd.write("\n")
    return filename


def _write_graph_binary_internal(g: InlaGraph, filename: str) -> str:
    with open(filename, "wb") as fd:
        fd.write(struct.pack("=i", inla_graph_binary_file_magic()))
        fd.write(struct.pack("=i", g.n))
        for i in range(1, g.n + 1):
            nbrs = g.nbs[i - 1]
            fd.write(struct.pack("=i", i))
            fd.write(struct.pack("=i", len(nbrs)))
            if nbrs:
                fd.write(struct.pack("=" + "i" * len(nbrs), *nbrs))
    return filename


# Optional: simple text summary (analogous to print/summary)
def inla_graph_summary(g: InlaGraph) -> str:
    nnbs_tab = {}
    for v in g.nnbs:
        nnbs_tab[v] = nnbs_tab.get(v, 0) + 1
    lines = [
        f"\tn = {g.n}",
        f"\tncc = {g.cc['n'] if g.cc else 'NA'}",
        "\tnnbs =",
    ]
    # Name/value alignment
    names = sorted(nnbs_tab.keys())
    width = max(len(str(k)) for k in names) if names else 1
    lines.append("\t  (names) " + " ".join(f"{str(k):>{width}}" for k in names))
    lines.append("\t  (count) " + " ".join(f"{str(nnbs_tab[k]):>{width}}" for k in names))
    return "\n".join(lines)
