"""
Lightweight helpers for decoding INLA binary exports.

Historically these utilities lived in the separate ``ppyinla`` distribution.
They are bundled here so that a single ``pip install pyinla`` provides all the
pieces that the high-level ``collect`` helpers expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_F8_LE = "<f8"  # canonical storage used by INLA/GMRFLib exports


def inla_read_binary_file(path: str | Path) -> Optional[np.ndarray]:
    """
    Read an INLA ``*.dat`` style file (little-endian float64 stream).

    Parameters
    ----------
    path:
        File-system location of the binary file.

    Returns
    -------
    ndarray or None
        1-D float64 array with the raw contents, ``None`` when the file is
        missing, and an empty array when the file exists but has no payload.
    """
    p = Path(path)
    if not p.exists():
        return None
    data = p.read_bytes()
    if not data:
        return np.empty((0,), dtype=np.float64)
    # ``frombuffer`` on the bytes object keeps a reference to ``data``; take a
    # copy so the caller can mutate the result without surprises.
    return np.frombuffer(data, dtype=_F8_LE).copy()


def _split_vector_blocks(vec: Sequence[float]) -> List[Tuple[float, np.ndarray]]:
    """
    Interpret the flattened ``[idx, n, x1, y1, ...]`` representation that INLA
    uses for quantiles, modes, and marginal densities.
    """
    arr = np.asarray(vec, dtype=np.float64).ravel()
    blocks: List[Tuple[float, np.ndarray]] = []
    i = 0
    n = arr.size
    while i + 1 < n:
        idx = arr[i]
        count = int(round(arr[i + 1]))
        i += 2
        if count <= 0:
            blocks.append((idx, np.zeros((0, 2), dtype=np.float64)))
            continue
        needed = 2 * count
        chunk = arr[i: i + needed]
        if chunk.size < needed:
            # Truncated file – pad with NaNs but still return what we have.
            padding = np.full((needed - chunk.size,), np.nan, dtype=np.float64)
            chunk = np.concatenate([chunk, padding])
            i = n
        else:
            i += needed
        blocks.append((idx, chunk.reshape(count, 2)))
    return blocks


def inla_interpret_vector(vec: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """
    Decode the INLA ``vector`` format into a dense matrix.

    The returned array has shape ``(max_count, 1 + 2 * n_blocks)``:
      - column 0 holds the abscissa (probabilities, grid points, …) from the
        first block when available;
      - for each subsequent block, column ``1 + 2*k`` stores the y-values and
        column ``1 + 2*k + 1`` stores the corresponding x-values. Missing
        entries are padded with ``NaN``.
    """
    if vec is None:
        return None
    blocks = _split_vector_blocks(vec)
    if not blocks:
        return np.empty((0, 0), dtype=np.float64)

    if len(blocks) == 1:
        # Common case for fixed effects / single marginals: keep the native (N×2)
        # layout so callers can iterate over (x, y) pairs directly.
        return blocks[0][1].copy()

    max_rows = max(block[1].shape[0] for block in blocks)
    n_blocks = len(blocks)

    if max_rows == 0:
        return np.zeros((0, 1 + 2 * n_blocks), dtype=np.float64)

    out = np.full((max_rows, 1 + 2 * n_blocks), np.nan, dtype=np.float64)

    first_pairs = blocks[0][1]
    if first_pairs.size:
        out[: first_pairs.shape[0], 0] = first_pairs[:, 0]

    for k, (_, pairs) in enumerate(blocks):
        if not pairs.size:
            continue
        rows = pairs.shape[0]
        out[:rows, 1 + 2 * k] = pairs[:, 1]      # y-values
        out[:rows, 1 + 2 * k + 1] = pairs[:, 0]  # x-values for reference
    return out


def inla_interpret_vector_list(
    vec: Optional[Sequence[float]],
) -> Optional[List[np.ndarray]]:
    """
    Decode a ``vector-of-vectors`` marginal structure into a list of ``(n×2)``
    arrays. Each array contains the ``(x, y)`` pairs for one marginal.
    """
    if vec is None:
        return None
    blocks = _split_vector_blocks(vec)
    if not blocks:
        return None
    return [pairs.copy() for _, pairs in blocks]
