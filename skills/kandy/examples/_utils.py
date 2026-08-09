"""Small shared helpers for runnable KANDy examples.

Keep domain logic in each example. This module only handles repeated output
and verification mechanics so scripts stay easy to scan and easy to check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import curve_fit


def ensure_outdir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap phase differences to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def print_check(passed: bool, message: str) -> bool:
    """Emit the standard example check line and return the boolean result."""
    ok = bool(passed)
    print(f"[CHECK] {'PASS' if ok else 'FAIL'}: {message}", flush=True)
    return ok


def save_metrics(path: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Write metrics as stable, JSON-serializable output."""
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(metrics), fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[METRICS] Saved {outpath}", flush=True)
    return outpath


def attractor_tv(true_traj: np.ndarray, pred_traj: np.ndarray,
                 bins: int = 8, drop: int = 50) -> tuple[float, float]:
    """Compare two orbits by their invariant measure, with a calibrated null.

    For a chaotic map a learned orbit decorrelates from the true one, so
    pointwise error is meaningless; what should agree is which parts of the
    attractor each orbit VISITS. This bins both into a shared 2-D histogram and
    returns the total-variation distance.

    The second return value is the null: the TV distance between two disjoint
    halves of the TRUE orbit. Two finite samples of the same measure are never
    identical, and with too many bins the null approaches 1 — comparing 200
    points over a 40x40 grid gives a null of 0.77, so any threshold below that
    would fail a perfect model. Always judge tv against null, not against an
    absolute constant.
    """
    t = np.asarray(true_traj)[drop:]
    p = np.asarray(pred_traj)[drop:]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(t) < 4 or len(p) < 2:
        return float("nan"), float("nan")

    lo = np.minimum(t.min(axis=0), p.min(axis=0))
    hi = np.maximum(t.max(axis=0), p.max(axis=0))
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    edges = [np.linspace(lo[i] - 0.02 * span[i], hi[i] + 0.02 * span[i], bins + 1)
             for i in range(2)]

    def _hist(a):
        h, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=edges)
        return h / max(h.sum(), 1)

    def _tv(a, b):
        return float(0.5 * np.abs(_hist(a) - _hist(b)).sum())

    half = len(t) // 2
    return _tv(t, p), _tv(t[:half], t[half:2 * half])


# ---------------------------------------------------------------------------
# Compatibility shims
#
# Two KANDy source lineages exist and neither is a superset of the other:
# one exports fit_sech2/fit_sech2_tanh from kandy.plotting, the other does not
# (but has the newer get_formula(lib=...) that most examples need). These two
# helpers have no kandy-internal dependencies, so re-implementing them here
# lets every example run against a single install. Same precedent as the
# bundled DelayEmbedding fallback in odes/mackey_glass_delay_example.py.
# ---------------------------------------------------------------------------

def _curve_fit_result(name: str, model, x, y, p0, bounds, label_fn) -> dict:
    p0 = [np.clip(p, lo + 1e-8, hi - 1e-8)
          for p, lo, hi in zip(p0, bounds[0], bounds[1])]
    try:
        params, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=200_000)
    except (RuntimeError, ValueError):
        params = np.array(p0)
    y_hat = model(x, *params)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"name": name, "params": params, "y_hat": y_hat, "r2": r2,
            "label": label_fn(params, r2), "_model": model}


def _fit_sech2(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a - A*sech^2((x-x0)/ell) — the even, dip-shaped shock profile."""
    def _model(x, a, A, x0, ell):
        z = (x - x0) / np.maximum(np.abs(ell), 1e-12)
        return a - A * (1.0 / np.cosh(z)) ** 2

    a0 = np.median(y)
    p0 = [a0, a0 - np.min(y), x[np.argmin(y)], np.std(x) / 5.0 or 1.0]
    bounds = ([-np.inf, -np.inf, x.min() - 1e-6, 1e-6],
              [np.inf, np.inf, x.max() + 1e-6, np.inf])
    return _curve_fit_result(
        "sech2", _model, x, y, p0, bounds,
        lambda p, r2: (f"sech²: {p[0]:.3f} − {p[1]:.3f}·sech²((x−{p[2]:.3f})"
                       f"/{p[3]:.3f})  ($R^2$={r2:.3f})"),
    )


def _fit_sech2_tanh(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = c + K*sech^2(z)*tanh(z) — the odd shock shape, z=(x-x0)/ell."""
    def _model(x, c, K, x0, ell):
        z = (x - x0) / np.maximum(np.abs(ell), 1e-12)
        return c + K * (1.0 / np.cosh(z)) ** 2 * np.tanh(z)

    c0 = np.median(y)
    y_centered = y - c0
    x0_0 = x[np.argmin(np.abs(y_centered))]
    K0 = np.sign(np.corrcoef(x - x0_0, y_centered)[0, 1] + 1e-12) * np.ptp(y)
    p0 = [c0, K0, x0_0, np.std(x) / 8.0 or 1.0]
    bounds = ([-np.inf, -np.inf, x.min() - 1e-6, 1e-6],
              [np.inf, np.inf, x.max() + 1e-6, np.inf])
    return _curve_fit_result(
        "sech2_tanh", _model, x, y, p0, bounds,
        lambda p, r2: (f"sech²·tanh: {p[0]:.3f} + {p[1]:.3f}·sech²·tanh  "
                       f"($R^2$={r2:.3f})"),
    )


try:                                        # prefer the library implementation
    from kandy.plotting import fit_sech2, fit_sech2_tanh
except ImportError:                         # fall back to the shims above
    fit_sech2 = _fit_sech2
    fit_sech2_tanh = _fit_sech2_tanh


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
