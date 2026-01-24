# surv.py
"""
Python equivalents of surv.R (INLA survival object helpers).

We define a lightweight 'inla.surv' analogue as a dict with keys:
  - time, lower, upper, event, truncation, cure [, subject]

and provide
  - inla_surv()            # build survival object (single-event or multi-event by subject)
  - is_inla_surv(obj)
  - as_inla_surv(obj)      # validate/normalize a dict/dataframe-like
  - plot_inla_surv(obj)    # requires matplotlib
  - print_inla_surv(obj)   # textual summary per observation
  - inla_strata(x)         # factor-like coding (levels 1..k)

The behavior mirrors the R code closely, including warnings and basic checks.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ----------------------------
# Public API
# ----------------------------
def inla_surv(
    time: Sequence[float],
    event: Optional[Sequence[int]] = None,
    time2: Optional[Sequence[float]] = None,
    truncation: Optional[Sequence[float]] = None,
    subject: Optional[Sequence[int]] = None,
    cure: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Build an INLA-style survival response object.

    - If 'subject' is None: single-event structure (right/left/interval/in-interval).
    - If 'subject' is provided: multiple-event per subject (detect / not-detect).
    """
    if subject is None:
        return _inla_surv_1(time, event, time2, truncation, cure)
    else:
        if time2 is not None:
            raise ValueError("Argument 'time2' is not allowed when 'subject' is used")
        if truncation is not None:
            raise ValueError("Argument 'truncation' is not allowed when 'subject' is used")
        return _inla_surv_2(time, event, time2, truncation, cure, subject)


def is_inla_surv(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("_class") == "inla.surv"


def as_inla_surv(obj: Any) -> Dict[str, Any]:
    """
    Accept a dict-like or a (record-oriented) dataframe-like and coerce to inla.surv.
    """
    if isinstance(obj, dict):
        for nm in obj.keys():
            if nm not in {"event", "time", "lower", "upper", "truncation", "cure", "subject"}:
                raise ValueError(f"Wrong name: {nm}")
        ret = dict(obj)
        ret["_class"] = "inla.surv"
        return ret
    try:
        # Try pandas-like conversion: DataFrame -> dict of columns
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            return as_inla_surv({k: np.asarray(v) for k, v in obj.items()})
    except Exception:
        pass
    raise TypeError("Argument must be a dict or a pandas.DataFrame")


def plot_inla_surv(x: Dict[str, Any], y: Any = None, **kwargs) -> None:
    """Dispatch to single- or multi-subject plot functions."""
    if not is_inla_surv(x):
        raise TypeError("x is not an inla.surv object")
    if x.get("subject") is None:
        _plot_inla_surv_1(x, **kwargs)
    else:
        _plot_inla_surv_2(x, **kwargs)


def print_inla_surv(x: Dict[str, Any], **kwargs) -> None:
    """Print a character representation per observation."""
    if not is_inla_surv(x):
        raise TypeError("x is not an inla.surv object")
    if x.get("subject") is None:
        lines = _as_character_inla_surv_1(x)
    else:
        lines = _as_character_inla_surv_2(x)
    for s in lines:
        print(s)


def inla_strata(obj: Sequence[Any]) -> Dict[str, Any]:
    """
    Similar to survival::strata but with levels = 1, 2, ...
    Returns {'strata': np.ndarray[int], 'coding': List[Any]}
    """
    values = np.asarray(list(obj))
    # Determine sorted unique levels (approx. base::factor default)
    # For mixed types, NumPy sort may fail; fall back to stable unique order
    try:
        levels = np.unique(values)
    except Exception:
        seen, levels = set(), []
        for v in values:
            if v not in seen:
                seen.add(v)
                levels.append(v)
    level_list = list(levels)
    # map to 1..k
    index = {lvl: i + 1 for i, lvl in enumerate(level_list)}
    strata = np.array([index[v] for v in values], dtype=int)
    return {"strata": strata, "coding": level_list}


# ----------------------------
# Internals: single-event
# ----------------------------
def _inla_surv_1(
    time: Sequence[float],
    event: Optional[Sequence[int]],
    time2: Optional[Sequence[float]],
    truncation: Optional[Sequence[float]],
    cure: Optional[Union[Sequence[float], np.ndarray]],
) -> Dict[str, Any]:
    # Checks (R mirrors)
    if time is None:
        raise ValueError("Must have a 'time' argument")
    time = _as_numeric(time, name="time")
    nn = len(time)

    # If time has NA/None, do fixups to other arrays like R
    time = np.asarray(time, dtype=float)
    if np.any(np.isnan(time)):
        idx = np.isnan(time)
        if event is not None:
            event = np.array(event, dtype=float)
            event[idx] = 1
        if time2 is not None:
            time2 = np.array(time2, dtype=float)
            time2[idx] = 0.0
        if truncation is not None:
            truncation = np.array(truncation, dtype=float)
            truncation[idx] = 0.0

    if np.any(time[~np.isnan(time)] < 0):
        raise ValueError("Negative times are not allowed")
    if time2 is not None:
        t2 = _as_numeric(time2, name="time2")
        if np.any(t2[~np.isnan(t2)] < 0):
            raise ValueError("Negative times2 are not allowed")
    if truncation is not None:
        tr = _as_numeric(truncation, name="truncation")
        if np.any(np.isnan(tr)):
            raise ValueError("Non valid values for 'truncation'")
        if np.any(tr < 0):
            raise ValueError("Negative truncation times are not allowed")

    # Handle event missingness
    if event is None:
        event = np.ones(nn, dtype=int)
        warnings.warn("'event' is missing: assuming all are observed failures")
    else:
        event = np.asarray(event, dtype=float)
        if np.any(np.isnan(event)):
            event = event.copy()
            event[np.isnan(event)] = 1
            warnings.warn("Some elements in `event` are NA: set them to observed failures.")
        event = event.astype(int)

    # Validate event values
    if not np.all(np.isin(event, np.array([0, 1, 2, 3, 4]))):
        raise ValueError("Invalid value for event")

    interval_mask = (event == 3) | (event == 4)
    if np.sum(interval_mask) > 0 and time2 is None:
        raise ValueError("'time2' has to be present for interval censored data or in-interval events")
    if np.sum(interval_mask) == 0 and time2 is not None:
        warnings.warn("'time2' is ignored for data that are not interval censored")
    if time2 is None:
        time2 = np.zeros(nn, dtype=float)

    if truncation is None:
        truncation = np.zeros(nn, dtype=float)
    else:
        truncation = np.asarray(truncation, dtype=float)
        if len(truncation) != nn:
            raise ValueError("'truncation' is of the wrong dimension")

    if cure is not None:
        cure = _as_cure_matrix(cure, nn)

    surv_time = np.zeros(nn, dtype=float)
    surv_upper = np.zeros(nn, dtype=float)
    surv_lower = np.zeros(nn, dtype=float)

    observed = event == 1
    right = event == 0
    left = event == 2
    interval = event == 3
    ininterval = event == 4

    surv_time[observed] = time[observed]
    surv_lower[right] = time[right]
    surv_upper[left] = time[left]
    surv_lower[interval] = time[interval]
    surv_upper[interval] = np.asarray(time2)[interval]
    surv_time[ininterval] = time[ininterval]
    surv_lower[ininterval] = truncation[ininterval]
    surv_upper[ininterval] = np.asarray(time2)[ininterval]
    truncation = truncation.copy()
    truncation[ininterval] = 0.0

    ss = {
        "time": surv_time,
        "lower": surv_lower,
        "upper": surv_upper,
        "event": event,
        "truncation": truncation,
        "cure": cure,
        "_class": "inla.surv",
    }
    return ss


# ----------------------------
# Internals: multiple-event (subject)
# ----------------------------
def _inla_surv_2(
    time: Sequence[float],
    event: Optional[Sequence[int]],
    time2: Optional[Sequence[float]],
    truncation: Optional[Sequence[float]],
    cure: Optional[Union[Sequence[float], np.ndarray]],
    subject: Sequence[int],
) -> Dict[str, Any]:
    if time is None:
        raise ValueError("Must have a 'time' argument")
    time = _as_numeric(time, name="time")
    nn = len(time)

    time = np.asarray(time, dtype=float)
    if np.any(np.isnan(time)):
        idx = np.isnan(time)
        if event is not None:
            event = np.array(event, dtype=float)
            event[idx] = 1
        if time2 is not None:
            t2 = np.array(time2, dtype=float)
            t2[idx] = 0.0
        if truncation is not None:
            tr = np.array(truncation, dtype=float)
            tr[idx] = 0.0

    if np.any(time[~np.isnan(time)] < 0):
        raise ValueError("Negative times are not allowed")
    if time2 is not None:
        t2 = _as_numeric(time2, name="time2")
        if np.any(t2[~np.isnan(t2)] < 0):
            raise ValueError("Negative times2 are not allowed")
    if truncation is not None:
        tr = _as_numeric(truncation, name="truncation")
        if np.any(np.isnan(tr)):
            raise ValueError("Non valid values for 'truncation'")
        if np.any(tr < 0):
            raise ValueError("Negative truncation times are not allowed")

    if event is None:
        event = np.ones(nn, dtype=int)
        warnings.warn("'event' is missing: assuming all are observed failures")
    else:
        event = np.asarray(event, dtype=float)
        if np.any(np.isnan(event)):
            event = event.copy()
            event[np.isnan(event)] = 1
            warnings.warn("Some elements in `event` are NA: set them to observed failures.")
        event = event.astype(int)

    if not np.all(np.isin(event, np.array([0, 1, 2, 3, 4]))):
        raise ValueError("Invalid value for event")

    # time2 ignored case mirrored from single-event
    interval_mask = (event == 3) | (event == 4)
    if np.sum(interval_mask) > 0 and time2 is None:
        raise ValueError("'time2' has to be present for interval censored data or in-interval events")
    if np.sum(interval_mask) == 0 and time2 is not None:
        warnings.warn("'time2' is ignored for data that are not interval censored")
    if time2 is None:
        time2 = np.zeros(nn, dtype=float)

    if truncation is None:
        truncation = np.zeros(nn, dtype=float)
    else:
        truncation = np.asarray(truncation, dtype=float)
        if len(truncation) != nn:
            raise ValueError("'truncation' is of the wrong dimension")

    if cure is not None:
        cure = _as_cure_matrix(cure, nn)

    if subject is None:
        raise ValueError("'subject' is missing")
    subject = np.asarray(subject, dtype=int)

    # Build the object
    ss = {
        "time": time.astype(float),
        "lower": np.zeros(nn, dtype=float),  # not used in this mode
        "upper": np.zeros(nn, dtype=float),  # not used in this mode
        "event": event,
        "truncation": truncation,
        "cure": cure,
        "subject": subject,
        "_class": "inla.surv",
    }
    return ss


# ----------------------------
# Plot/print helpers
# ----------------------------
def _plot_inla_surv_1(obj: Dict[str, Any], legend: bool = True, **kwargs) -> None:
    if not _HAVE_MPL:
        raise ImportError("matplotlib is required for plotting.")
    time = obj["time"]; upper = obj["upper"]; lower = obj["lower"]
    event = obj["event"]; truncation = obj["truncation"]
    nn = len(time)
    xmax = float(np.max([np.max(time), np.max(upper), np.max(lower), np.max(event)]))
    xax = (0.0, xmax + xmax / 8.0)
    yax = (0.0, float(nn))
    plt.figure()
    plt.plot(xax, yax)  # will just set limits; we will clear next
    plt.clf()
    plt.xlim(xax); plt.ylim(yax)
    plt.xlabel("time"); plt.ylabel("")
    # Emulate axes = FALSE + manual axes from R: we keep default axes
    for i in range(nn):
        if event[i] == 1:
            plt.plot([truncation[i], time[i]], [i + 1, i + 1])
            plt.text(time[i], i + 1, "*")
        elif event[i] == 0:
            plt.plot([truncation[i], lower[i]], [i + 1, i + 1])
            plt.text(lower[i], i + 1, ">")
        elif event[i] == 2:
            plt.plot([truncation[i], upper[i]], [i + 1, i + 1])
            plt.text(upper[i], i + 1, "<")
        elif event[i] == 3:
            plt.plot([lower[i], upper[i]], [i + 1, i + 1])
            plt.text(upper[i], i + 1, "|")
            plt.text(lower[i], i + 1, "|")
        elif event[i] == 4:
            plt.plot([lower[i], upper[i]], [i + 1, i + 1])
            plt.text(time[i], i + 1, "*")
    if legend:
        leg_text = ["failure"]; leg_symb = ["*"]
        if np.any(event == 0):
            leg_text.append("right cens"); leg_symb.append(">")
        if np.any(event == 2):
            leg_text.append("left cens"); leg_symb.append("<")
        if np.any(event == 3):
            leg_text.append("interval cens"); leg_symb.append("|")
        if np.any(event == 4):
            leg_text.append("in-interval failure"); leg_symb.append("*")
        # Just print a simple legend of text markers (no special pch mapping)
        plt.legend(leg_text, loc="upper right")
    plt.show()


def _plot_inla_surv_2(obj: Dict[str, Any], legend: bool = True, **kwargs) -> None:
    if not _HAVE_MPL:
        raise ImportError("matplotlib is required for plotting.")
    time = obj["time"]; event = obj["event"]; subject = obj["subject"]
    nn = int(np.max(subject))
    xmax = float(np.max(time)); xmin = float(np.min(time))
    xax = (xmin, xmax + xmax / 8.0)
    yax = (0.0, float(nn) + 0.5)
    plt.figure()
    plt.plot(xax, yax)
    plt.clf()
    plt.xlim(xax); plt.ylim(yax)
    plt.xlabel("time"); plt.ylabel("")
    for i in range(1, nn + 1):
        mask = subject == i
        ti = time[mask]
        yi = np.full(ti.size, i, dtype=float)
        # Connect times for this subject
        if ti.size > 0:
            plt.plot(ti, yi)
        # Mark detections and non-detections
        det_mask = mask & (event == 1)
        nd_mask = mask & (event == 0)
        if np.any(det_mask):
            plt.scatter(time[det_mask], np.full(np.sum(det_mask), i), marker="x")
        if np.any(nd_mask):
            plt.scatter(time[nd_mask], np.full(np.sum(nd_mask), i), marker="o")
    if legend:
        leg_text = ["detect", "not detect"]
        plt.legend(leg_text, loc="upper right")
    plt.show()


def _as_character_inla_surv_1(x: Dict[str, Any]) -> List[str]:
    nn = len(x["event"])
    out = [""] * nn
    ininterval = x["event"] == 4
    interval = x["event"] == 3
    right = x["event"] == 0
    left = x["event"] == 2
    failure = x["event"] == 1

    out_np = np.array(out, dtype=object)
    out_np[ininterval] = np.char.add(
        np.char.add(np.char.add("[", x["lower"][ininterval].astype(str)),
                    np.char.add(",", x["upper"][ininterval].astype(str))),
        np.char.add("] ", x["time"][ininterval].astype(str)),
    )
    out_np[interval] = np.char.add(
        np.char.add(np.char.add("[", x["lower"][interval].astype(str)),
                    np.char.add(",", x["upper"][interval].astype(str))),
        "]"
    )
    out_np[right] = np.char.add(x["lower"][right].astype(str), "+")
    out_np[left] = np.char.add(x["upper"][left].astype(str), "-")
    out_np[failure] = x["time"][failure].astype(str)
    return list(out_np)


def _as_character_inla_surv_2(x: Dict[str, Any]) -> List[str]:
    nn = len(x["event"])
    out = np.empty(nn, dtype=object)
    detect = x["event"] == 1
    notdetect = x["event"] == 0
    out[detect] = x["time"][detect].astype(str)
    out[notdetect] = np.char.add(x["time"][notdetect].astype(str), "+")
    return list(out)


# ----------------------------
# Utilities
# ----------------------------
def _as_numeric(x: Sequence[float], name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError
        return arr
    except Exception:
        raise TypeError(f"'{name}' variable is not numeric")


def _as_cure_matrix(cure: Union[Sequence[float], np.ndarray], nn: int) -> np.ndarray:
    arr = np.asarray(cure)
    if arr.ndim == 1:
        if arr.size != nn:
            raise ValueError("cure has incompatible length")
        arr = arr.reshape(nn, 1)
    elif arr.ndim == 2:
        if arr.shape[0] != nn:
            raise ValueError("cure has incompatible number of rows")
    else:
        raise ValueError("cure must be 1D or 2D array-like")
    # Replace NaNs with 0
    arr = arr.astype(float, copy=True)
    arr[np.isnan(arr)] = 0.0
    return arr


__all__ = [
    "inla_surv",
    "is_inla_surv",
    "as_inla_surv",
    "plot_inla_surv",
    "print_inla_surv",
    "inla_strata",
]
