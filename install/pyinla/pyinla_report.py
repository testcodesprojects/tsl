"""
pyinla_report.py

Utilities to visualize and tabulate INLA results from pyINLA without affecting
the original outputs (Model.ini/data.files/results.files). Use these helpers to
create publication‑ready marginal plots and clean summary tables.

Quick usage
-----------
from pyinla_report import (
    ensure_results,
    plot_fixed_marginals,
    save_fixed_marginals_grid,
    summary_fixed_df,
    save_summary_tables,
)

res = pyinla(..., inla_call='inla', collect=True)  # run INLA (not dry_run)
info = ensure_results(res)
fig = plot_fixed_marginals(info)
save_fixed_marginals_grid(info, out_dir='reports')
save_summary_tables(info, out_dir='reports')

CLI
---
python pyinla_report.py --model <inla_dir> --out reports --which fixed hyper

This will collect results from <inla_dir>, write fixed/hyper plots and tables
under the 'reports' directory, leaving the original run folders untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore

# Optional plotting dependency
try:  # pragma: no cover - plotting not exercised by unit tests
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


# -----------------------------
# Collection helpers
# -----------------------------

def _is_result_like(obj: Any) -> bool:
    return hasattr(obj, "results") or (
        hasattr(obj, "summary_fixed") or hasattr(obj, "marginals_fixed")
    )


def ensure_results(entry: Union[str, os.PathLike, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Normalize a pyINLA result or an INLA run directory into a plain dict with
    the keys produced by collect.collect_inla_results.

    Accepts either:
      - a PyINLAResult instance (with results already collected or not), or
      - a path to an 'inla.model*' directory (or its parent directory).
    """
    if _is_result_like(entry):
        # PyINLAResult: use embedded dict if present; otherwise collect using inla_dir
        res_obj = entry
        if getattr(res_obj, "results", None) is not None:
            return dict(res_obj.results)
        inla_dir = None
        try:
            inla_dir = res_obj.args.get("inla_dir")  # type: ignore[attr-defined]
        except Exception:
            pass
        if not inla_dir:
            raise ValueError("Result object has no collected results and no inla_dir to collect from.")
        from collect import collect_inla_results
        d = collect_inla_results(inla_dir, allow_parent=False, debug=debug)
        if not isinstance(d, dict):
            raise RuntimeError("collect_inla_results returned no results.")
        return d

    # Else treat as a path
    from collect import collect_inla_results
    d = collect_inla_results(str(entry), allow_parent=True, debug=debug)
    if not isinstance(d, dict):
        raise RuntimeError("collect_inla_results returned no results.")
    return d


# -----------------------------
# Summary tables
# -----------------------------

def _fmt_df(df: "pd.DataFrame", *, digits: int = 3) -> "pd.DataFrame":  # type: ignore[name-defined]
    if pd is None:
        return df
    out = df.copy()
    with pd.option_context("display.float_format", lambda v: f"{v:.{digits}g}"):
        return out


def summary_fixed_df(results: Mapping[str, Any], *, digits: int = 3) -> Optional["pd.DataFrame"]:  # type: ignore[name-defined]
    df = results.get("summary.fixed") if isinstance(results, Mapping) else None
    if df is None or pd is None:
        return df
    return _fmt_df(df, digits=digits)


def summary_hyper_df(results: Mapping[str, Any], *, digits: int = 3) -> Optional["pd.DataFrame"]:  # type: ignore[name-defined]
    df = results.get("summary.hyperpar") if isinstance(results, Mapping) else None
    if df is None or pd is None:
        return df
    return _fmt_df(df, digits=digits)


def save_summary_tables(results: Mapping[str, Any], *, out_dir: Union[str, os.PathLike], include: Sequence[str] = ("fixed", "hyper"), digits: int = 3) -> List[Path]:
    """
    Save CSV and HTML tables for requested blocks (fixed, hyper) into out_dir.
    Returns list of written files.
    """
    os.makedirs(out_dir, exist_ok=True)
    outs: List[Path] = []
    if pd is None:
        raise RuntimeError("pandas is required to export summary tables.")

    if "fixed" in include:
        df = summary_fixed_df(results, digits=digits)
        if df is not None:
            p_csv = Path(out_dir) / "summary_fixed.csv"
            p_html = Path(out_dir) / "summary_fixed.html"
            df.to_csv(p_csv)
            df.to_html(p_html)
            outs += [p_csv, p_html]
    if "hyper" in include:
        df = summary_hyper_df(results, digits=digits)
        if df is not None:
            p_csv = Path(out_dir) / "summary_hyperpar.csv"
            p_html = Path(out_dir) / "summary_hyperpar.html"
            df.to_csv(p_csv)
            df.to_html(p_html)
            outs += [p_csv, p_html]
    return outs


# -----------------------------
# Plotting helpers
# -----------------------------

def _sanitize(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name))
    return safe or "item"


def _plot_xy(ax, xy: "pd.DataFrame", label: str) -> None:  # type: ignore[name-defined]
    x = np.asarray(xy["x"], dtype=float)
    y = np.asarray(xy["y"], dtype=float)
    ax.plot(x, y, lw=1.6, color="#0d6efd")
    ax.fill_between(x, 0, y, color="#0d6efd", alpha=0.12, linewidth=0)
    ax.set_title(label)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.8)


def plot_fixed_marginals(results: Mapping[str, Any], *, names: Optional[Sequence[str]] = None, ncols: int = 3, figsize: Optional[Tuple[float, float]] = None) -> Any:  # returns Figure
    """
    Create a grid plot of fixed‑effect marginal posteriors.
    Returns a Matplotlib Figure (requires matplotlib).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    margs = results.get("marginals.fixed") if isinstance(results, Mapping) else None
    if not isinstance(margs, Mapping) or not margs:
        raise ValueError("No marginals.fixed available to plot.")
    items = list(margs.items())
    if names is not None:
        name_set = set(names)
        items = [(k, v) for (k, v) in items if k in name_set]
    n = len(items)
    if n == 0:
        raise ValueError("Requested names not found in marginals.fixed.")
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.0 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    for idx, (name, df_xy) in enumerate(items):
        r, c = divmod(idx, ncols)
        _plot_xy(axes[r][c], df_xy, name)
    # Hide unused axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    return fig


def save_fixed_marginals_grid(results: Mapping[str, Any], *, out_dir: Union[str, os.PathLike], names: Optional[Sequence[str]] = None, ncols: int = 3, fmt: str = "png", dpi: int = 150) -> Path:
    os.makedirs(out_dir, exist_ok=True)
    fig = plot_fixed_marginals(results, names=names, ncols=ncols)
    p = Path(out_dir) / f"marginals_fixed_grid.{fmt}"
    fig.savefig(p, dpi=dpi)
    return p


def save_fixed_marginals_individual(results: Mapping[str, Any], *, out_dir: Union[str, os.PathLike], names: Optional[Sequence[str]] = None, fmt: str = "png", dpi: int = 150) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    os.makedirs(out_dir, exist_ok=True)
    margs = results.get("marginals.fixed") if isinstance(results, Mapping) else None
    if not isinstance(margs, Mapping) or not margs:
        return []
    outs: List[Path] = []
    for name, df_xy in margs.items():
        if names is not None and name not in names:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        _plot_xy(ax, df_xy, name)
        fig.tight_layout()
        p = Path(out_dir) / f"marginal_fixed_{_sanitize(name)}.{fmt}"
        fig.savefig(p, dpi=dpi)
        outs.append(p)
        plt.close(fig)
    return outs


def save_hyper_marginals_individual(results: Mapping[str, Any], *, out_dir: Union[str, os.PathLike], names: Optional[Sequence[str]] = None, fmt: str = "png", dpi: int = 150) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    os.makedirs(out_dir, exist_ok=True)
    margs = results.get("marginals.hyperpar") if isinstance(results, Mapping) else None
    if not isinstance(margs, Mapping) or not margs:
        return []
    outs: List[Path] = []
    for name, df_xy in margs.items():
        if names is not None and name not in names:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        _plot_xy(ax, df_xy, name)
        fig.tight_layout()
        p = Path(out_dir) / f"marginal_hyper_{_sanitize(name)}.{fmt}"
        fig.savefig(p, dpi=dpi)
        outs.append(p)
        plt.close(fig)
    return outs


# -----------------------------
# CLI
# -----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None):  # pragma: no cover - CLI glue
    import argparse
    ap = argparse.ArgumentParser(description="Create plots/tables for pyINLA results without altering original outputs.")
    ap.add_argument("--model", required=True, help="Path to inla.model* directory or its parent (latest will be used)")
    ap.add_argument("--out", required=True, help="Output directory for reports")
    ap.add_argument("--which", nargs="*", default=["fixed", "hyper"], choices=["fixed", "hyper"], help="Which blocks to export")
    ap.add_argument("--fmt", default="png", help="Image format for plots (default: png)")
    ap.add_argument("--dpi", type=int, default=150, help="DPI for saved images")
    ap.add_argument("--cols", type=int, default=3, help="Columns in fixed marginals grid")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI glue
    args = _parse_args(argv)
    results = ensure_results(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "fixed" in args.which:
        save_fixed_marginals_grid(results, out_dir=out_dir, ncols=args.cols, fmt=args.fmt, dpi=args.dpi)
        save_fixed_marginals_individual(results, out_dir=out_dir, fmt=args.fmt, dpi=args.dpi)
    if "hyper" in args.which:
        save_hyper_marginals_individual(results, out_dir=out_dir, fmt=args.fmt, dpi=args.dpi)

    save_summary_tables(results, out_dir=out_dir, include=args.which)
    print(f"Wrote reports to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script mode
    raise SystemExit(main())

