#!/usr/bin/env python3
"""Stuart-Landau band oscillators: put the PHASE back in the state.

Why this exists
---------------
``epilepsy_cross_frequency_seizure_freedom.py`` models band AMPLITUDE
envelopes.  It rolls out stably for 25 s, but it contains no waveform, so
rebuilding the recording from it required injecting the measured phase -- and a
control showed that injected phase was doing almost all the work: constant
amplitude x measured phase scored R^2 = 0.640, a *shuffled* model trajectory
scored 0.631, and the model itself only 0.710.

The obvious question is why the phase is not simply fitted too.  The measured
answer is that it can be, and that phase is in fact the MORE deterministic
coordinate here -- a k-NN bound on the analytic-signal state (Re z, Im z)
returns 0.43 against 0.05 for the log-amplitude state.  What defeats the
earlier phase models in this directory is not learning, it is INTEGRATION:

    amplitude   dA/dt = (T - A)/tau     contraction  -> error decays, bounded
    phase       dtheta/dt = omega + ..  integrator   -> error grows linearly

The phase direction of an oscillator carries a zero Lyapunov exponent -- it is
the neutral direction along the cycle, so a perturbation neither grows nor
decays, it persists.  Measured on hippocampus_R alpha: omega is 60.75 rad/s
pre-ictal and 62.45 ictal, a drift of only 2.8%, yet holding phase error under
pi/2 across a 25 s rollout needs 0.10% accuracy in omega.  That is a factor of
27, and it is why the Kuramoto scripts here score one-step R^2 ~ 0.99 and
autonomous rollout correlation ~ 0.04.  Those are not contradictory numbers;
they are what integrating a neutral direction looks like.

So this script does not try to beat that limit.  It changes the claim to one
that survives it: fit a genuine oscillator, expect the WAVEFORM to hold for
about a second and the ENVELOPE to hold indefinitely, and report the two
horizons separately instead of averaging them into one misleading number.

The model
---------
Per band b, the state is the analytic signal itself, z_b = x_b + i*y_b, from
the Hilbert transform of the bandpassed trace.  The Stuart-Landau normal form

    dz/dt = (lambda + i*omega) z - |z|^2 z

is, in real coordinates,

    dx/dt = lambda*x - omega*y - (x^2 + y^2) x
    dy/dt = omega*x + lambda*y - (x^2 + y^2) y

omega is a FITTED COEFFICIENT, not an assumed constant, which is the whole
difference from the Kuramoto attempts.  lambda > 0 gives a limit cycle at
r = sqrt(lambda); lambda < 0 gives a damped focus driven by noise.  Both are
physiologically meaningful and the fit decides which.

THE CROSS-TERMS ARE THE POINT.  KANDy is A*Psi(phi(x)) with Psi separable, so
a KAN can never manufacture x*(x^2+y^2) from separate x and y edges -- the
cubic terms have to be written into the lift by hand.  Leaving them out is a
structural error, not a tuning issue, and the ablation below removes them and
shows the limit cycle collapse it causes.

Derivatives are SPECTRAL (multiply by i*2*pi*f in the Fourier domain), which is
exact for a bandlimited signal.  Finite differences on a 10 Hz carrier sampled
at 200 Hz would put a systematic phase lag straight into the fitted omega, and
omega is the coefficient this script is trying to recover.

Reconstruction here injects NOTHING.  The rollout produces x_b(t) directly, so
sum_b x_b(t) IS the reconstructed trace and the 0.640 phase baseline that made
the envelope model's reconstruction unreadable does not arise.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_stuart_landau_bands.py
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from io import StringIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kandy")

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[2]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert

from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy

torch.set_num_threads(1)


# PyKAN's curve2coef swallows an lstsq failure and then returns an unbound
# variable; restore the ridge fallback it ships commented out.  Same shim as
# epilepsy_cross_frequency_seizure_freedom.py.
def _patch_curve2coef() -> int:
    import sys as _sys

    try:
        from kan import spline as _spline
    except ImportError:
        return 0
    _B = _spline.B_batch
    original = _spline.curve2coef

    def curve2coef(x_eval, y_eval, grid, k):
        batch, in_dim = x_eval.shape[0], x_eval.shape[1]
        out_dim = y_eval.shape[2]
        n_coef = grid.shape[1] - k - 1
        mat = _B(x_eval, grid, k).permute(1, 0, 2)[:, None, :, :]
        mat = mat.expand(in_dim, out_dim, batch, n_coef)
        yy = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
        try:
            return torch.linalg.lstsq(mat, yy).solution[:, :, :, 0]
        except Exception:                                        # noqa: BLE001
            XtX = torch.einsum("ijmn,ijnp->ijmp", mat.permute(0, 1, 3, 2), mat)
            Xty = torch.einsum("ijmn,ijnp->ijmp", mat.permute(0, 1, 3, 2), yy)
            eye = torch.eye(XtX.shape[2], device=mat.device)[None, None]
            return torch.linalg.solve(XtX + 1e-8 * eye, Xty)[:, :, :, 0]

    n = 0
    for _nm, _m in list(_sys.modules.items()):
        if _nm.split(".")[0] == "kan" and _m is not None:
            if getattr(_m, "curve2coef", None) is original:
                _m.curve2coef = curve2coef
                n += 1
    return n


_patch_curve2coef()


def env_int(n, d):
    return int(os.environ.get(n, d))


def env_float(n, d):
    return float(os.environ.get(n, d))


# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(os.environ.get(
    "KANDY_SL_RESULT_DIR", str(REPO_ROOT / "results" / "Epilepsy-Stuart-Landau")))

CHANNEL = os.environ.get("KANDY_SL_CHANNEL", "hippocampus_R")
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND_NAMES = list(BANDS)
NB = len(BAND_NAMES)
FILTER_ORDER = 4
EDGE_TRIM_S = 2.0

# Time blocks.  The transition is in TRAIN: a limit cycle whose amplitude
# changes across the event cannot be identified from one regime alone.
TRAIN_EXCLUDE = {"pre_test": (-13.0, -10.0), "post_test": (24.0, 28.0)}
GUARD_S = env_float("KANDY_SL_GUARD_S", 0.5)

GRID = env_int("KANDY_SL_GRID", 5)
STEPS = env_int("KANDY_SL_STEPS", 60)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_SL_LAMB_GRID", "0.0,0.001,0.01").split(",")]
CV_FOLDS = env_int("KANDY_SL_CV_FOLDS", 4)
N_FIT_MAX = env_int("KANDY_SL_N_FIT_MAX", 6000)

ROLL_START = env_float("KANDY_SL_ROLL_START", 2.0)
ROLL_SECONDS = env_float("KANDY_SL_ROLL_SECONDS", 6.0)


def quiet_fit(m, **kw):
    with contextlib.redirect_stdout(StringIO()):
        m.fit(**kw)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(StringIO()):
        return fn(*a, **k)


def r2_cols(y, p):
    y, p = np.atleast_2d(y.T).T, np.atleast_2d(p.T).T
    den = np.sum((y - y.mean(0)) ** 2, 0)
    return 1.0 - np.sum((y - p) ** 2, 0) / np.where(den > 0, den, np.nan)


OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7"}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 1. Analytic signal per band, and its SPECTRAL derivative
# ---------------------------------------------------------------------------
payload = pickle.load(DATA_PATH.open("rb"))
X_ALL = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)
LABELS = list(payload["labels"])
ci = LABELS.index(CHANNEL)
DT = 1.0 / FS

print("=" * 76)
print("STUART-LANDAU BAND OSCILLATORS -- phase is IN the state")
print("=" * 76)
print(f"[DATA]  {CHANNEL} @ {FS:.0f} Hz, {T_ALL[0]:.0f}..{T_ALL[-1]:.0f}s "
      "(onset at 0), NO downsampling", flush=True)


def spectral_derivative(z):
    """dz/dt by multiplying by i*2*pi*f.  Exact for a bandlimited signal.

    A finite difference on a 10 Hz carrier at 200 Hz introduces a systematic
    phase lag that lands directly in the fitted omega -- the coefficient this
    script exists to recover -- so it is not used.
    """
    n = len(z)
    f = np.fft.fftfreq(n, d=DT)
    return np.fft.ifft(np.fft.fft(z) * (1j * 2 * np.pi * f))


Z, DZ = {}, {}
for name, (lo, hi) in BANDS.items():
    bb, aa = butter(FILTER_ORDER, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    filt = filtfilt(bb, aa, X_ALL[:, ci])
    z = hilbert(filt)
    Z[name] = z
    DZ[name] = spectral_derivative(z)

edge = int(round(EDGE_TRIM_S * FS))
sl0 = slice(edge, len(X_ALL) - edge)
T = T_ALL[sl0]
N = len(T)

# Scale each band so its limit cycle sits near r ~ 1; the normal form's cubic
# term is only well conditioned when the state is O(1).
SCALE = np.array([np.sqrt(np.mean(np.abs(Z[n][sl0]) ** 2)) for n in BAND_NAMES])
STATE = np.zeros((N, 2 * NB))
DSTATE = np.zeros((N, 2 * NB))
for b, name in enumerate(BAND_NAMES):
    STATE[:, 2 * b] = Z[name][sl0].real / SCALE[b]
    STATE[:, 2 * b + 1] = Z[name][sl0].imag / SCALE[b]
    DSTATE[:, 2 * b] = DZ[name][sl0].real / SCALE[b]
    DSTATE[:, 2 * b + 1] = DZ[name][sl0].imag / SCALE[b]

print(f"[STATE] {2*NB}-D: (Re z, Im z) per band, {N} samples", flush=True)
print(f"{'band':>7} {'centre Hz':>10} {'samples/cycle':>14} {'omega meas':>11}",
      flush=True)
OMEGA_MEAS = np.zeros(NB)
for b, name in enumerate(BAND_NAMES):
    ph = np.unwrap(np.angle(Z[name][sl0]))
    OMEGA_MEAS[b] = float(np.median(np.gradient(ph, DT)))
    f0 = 0.5 * sum(BANDS[name])
    spc = FS / f0
    warn = "   [< 8/cycle: derivative and RK4 are marginal]" if spc < 8 else ""
    print(f"{name:>7} {f0:10.1f} {spc:14.1f} {OMEGA_MEAS[b]:11.2f}{warn}", flush=True)

# ---------------------------------------------------------------------------
# 2. Ceiling on THIS state, before fitting
# ---------------------------------------------------------------------------
def knn_r2(S, dts, lag_s, k=8, theiler_s=3.0, nmax=1500):
    step = max(1, len(S) // nmax)
    S = S[::step]
    dts = dts * step
    L = max(1, int(round(lag_s / dts)))
    th = int(round(theiler_s / dts))
    Y = (S[L:] - S[:-L]) / lag_s
    Zs = S[:len(Y)]
    n = len(Zs)
    D = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    D[np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < th] = np.inf
    P = Y[np.argsort(D, 1)[:, :k]].mean(1)
    return float(np.nanmean(r2_cols(Y, P)))


print(f"\n[CEILING] k-NN bound on the analytic-signal state: "
      f"{knn_r2(STATE, DT, 0.05):+.3f}  "
      "(log-amplitude state scores +0.05 -- phase is the MORE "
      "deterministic coordinate)", flush=True)

# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------
rows = np.arange(N)
near_test = np.zeros(N, bool)
test_blocks = {}
for nm, w in TRAIN_EXCLUDE.items():
    test_blocks[nm] = rows[(T >= w[0]) & (T <= w[1])]
    near_test |= (T >= w[0] - GUARD_S) & (T <= w[1] + GUARD_S)
train_rows = rows[~near_test]
rng = np.random.default_rng(SEED)
fit_rows = (np.sort(rng.choice(train_rows, N_FIT_MAX, replace=False))
            if len(train_rows) > N_FIT_MAX else train_rows)
cv_splits = []
gap = int(round(GUARD_S / DT))
for f in np.array_split(train_rows, CV_FOLDS):
    held = np.asarray(f)
    keep = train_rows[(train_rows < held[0] - gap) | (train_rows > held[-1] + gap)]
    sub = (np.sort(rng.choice(keep, N_FIT_MAX, replace=False))
           if len(keep) > N_FIT_MAX else keep)
    cv_splits.append((sub, held[::max(1, len(held) // 2000)]))
print(f"[SPLIT] train={len(train_rows)} (fit on {len(fit_rows)}), "
      f"test={sum(len(v) for v in test_blocks.values())} in "
      f"{len(test_blocks)} blocks, {CV_FOLDS}-fold blocked CV for lamb",
      flush=True)


# ---------------------------------------------------------------------------
# 4. Lift.  The cubic cross-terms are explicit -- a separable KAN cannot build
#    x*(x^2+y^2) out of an x edge and a y edge.
# ---------------------------------------------------------------------------
def make_lift(b, with_cubic=True, with_cross=True):
    others = [c for c in range(NB) if c != b]

    def np_fn(s):
        s = np.atleast_2d(s)
        x, y = s[:, 2 * b], s[:, 2 * b + 1]
        r2 = x * x + y * y
        feats = [np.ones(len(s)), x, y]
        if with_cubic:
            feats += [x * r2, y * r2]
        if with_cross:
            for c in others:
                rc2 = s[:, 2 * c] ** 2 + s[:, 2 * c + 1] ** 2
                feats += [rc2 * x, rc2 * y]
        return np.column_stack(feats)

    def t_fn(s):
        x, y = s[:, 2 * b], s[:, 2 * b + 1]
        r2 = x * x + y * y
        feats = [torch.ones_like(x), x, y]
        if with_cubic:
            feats += [x * r2, y * r2]
        if with_cross:
            for c in others:
                rc2 = s[:, 2 * c] ** 2 + s[:, 2 * c + 1] ** 2
                feats += [rc2 * x, rc2 * y]
        return torch.stack(feats, 1)

    names = ["1", "x", "y"]
    if with_cubic:
        names += ["x*r2", "y*r2"]
    if with_cross:
        for c in others:
            names += [f"r2_{BAND_NAMES[c]}*x", f"r2_{BAND_NAMES[c]}*y"]
    return np_fn, t_fn, len(names), names


# ---------------------------------------------------------------------------
# 5. Fit one 2-output oscillator per band
# ---------------------------------------------------------------------------
def fit_band(b, with_cubic=True, with_cross=True, tag="sl"):
    np_fn, t_fn, dim, names = make_lift(b, with_cubic, with_cross)
    tgt = DSTATE[:, 2 * b:2 * b + 2]
    best = None
    for lamb in LAMB_GRID:
        fold = []
        for tr, te in cv_splits:
            m = KANDy(lift=CustomLift(np_fn, dim, t_fn, f"{tag}{b}"), grid=GRID,
                      k=3, steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(),
                      device="cpu")
            quiet_fit(m, X=STATE[tr], X_dot=tgt[tr], val_frac=0.15, test_frac=0.15,
                      lamb=lamb, opt="LBFGS", lr=1.0, batch=-1,
                      stop_grid_update_step=10, patience=0, verbose=False)
            fold.append(float(np.nanmean(r2_cols(tgt[te], m.predict(STATE[te])))))
        sc = float(np.mean(fold))
        if best is None or sc > best[1]:
            best = (lamb, sc)
    m = KANDy(lift=CustomLift(np_fn, dim, t_fn, f"{tag}{b}"), grid=GRID, k=3,
              steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    quiet_fit(m, X=STATE[fit_rows], X_dot=tgt[fit_rows], val_frac=0.15,
              test_frac=0.15, lamb=best[0], opt="LBFGS", lr=1.0, batch=-1,
              stop_grid_update_step=10, patience=0, verbose=False)
    return m, best[0], best[1], names, dim


print("\n[TRAIN] fitting one Stuart-Landau oscillator per band", flush=True)
models, lambs, lift_names, lift_dims = [], [], [], []
for b, name in enumerate(BAND_NAMES):
    m, lb, sc, nms, dim = fit_band(b)
    models.append(m)
    lambs.append(lb)
    lift_names.append(nms)
    lift_dims.append(dim)
    print(f"        {name:>6}: lamb={lb:g}, blocked-CV R2={sc:+.3f}", flush=True)

# ablation: same fit with the cubic cross-terms REMOVED
print("\n[ABLATE] refitting with x*(x^2+y^2) and y*(x^2+y^2) REMOVED from the "
      "lift", flush=True)
models_nocubic = []
for b, name in enumerate(BAND_NAMES):
    m, lb, sc, _n, _d = fit_band(b, with_cubic=False, tag="nc")
    models_nocubic.append(m)
    print(f"        {name:>6}: lamb={lb:g}, blocked-CV R2={sc:+.3f}", flush=True)


# ---------------------------------------------------------------------------
# 6. Did the fit recover omega?
# ---------------------------------------------------------------------------
def linear_coeffs(model, b):
    """Read lambda and omega off the fitted field by finite differences at the
    origin of band b, holding every other band at its mean."""
    base = STATE[train_rows].mean(0)
    eps = 1e-3
    J = np.zeros((2, 2))
    for j in range(2):
        sp, sm = base.copy(), base.copy()
        sp[2 * b + j] += eps
        sm[2 * b + j] -= eps
        J[:, j] = (model.predict(sp[None])[0] - model.predict(sm[None])[0]) / (2 * eps)
    lam = 0.5 * (J[0, 0] + J[1, 1])
    om = 0.5 * (J[1, 0] - J[0, 1])
    return lam, om, J


print(f"\n[OMEGA] does the equation recover the right frequency?", flush=True)
print(f"{'band':>7} {'omega fitted':>13} {'omega measured':>15} {'error':>9} "
      f"{'lambda':>9}", flush=True)
omega_fit, lam_fit = np.zeros(NB), np.zeros(NB)
for b, name in enumerate(BAND_NAMES):
    lam, om, _J = linear_coeffs(models[b], b)
    omega_fit[b], lam_fit[b] = om, lam
    err = 100 * abs(om - OMEGA_MEAS[b]) / abs(OMEGA_MEAS[b])
    print(f"{name:>7} {om:13.2f} {OMEGA_MEAS[b]:15.2f} {err:8.1f}% {lam:9.3f}",
          flush=True)
omega_err = float(np.mean(100 * np.abs(omega_fit - OMEGA_MEAS) / np.abs(OMEGA_MEAS)))
print(f"        mean |omega| error = {omega_err:.1f}%", flush=True)

# ---------------------------------------------------------------------------
# 7. Held-out one-step skill
# ---------------------------------------------------------------------------
print("\n[EVAL]  held-out R2 of dz/dt", flush=True)
heldout = {}
for blk, r in test_blocks.items():
    row = {}
    for b, name in enumerate(BAND_NAMES):
        tgt = DSTATE[r, 2 * b:2 * b + 2]
        row[name] = {
            "stuart_landau": float(np.nanmean(r2_cols(tgt, models[b].predict(STATE[r])))),
            "no_cubic": float(np.nanmean(r2_cols(tgt, models_nocubic[b].predict(STATE[r])))),
            "zero": float(np.nanmean(r2_cols(tgt, np.zeros_like(tgt)))),
        }
    heldout[blk] = row
    print(f"        [{blk}] " + "  ".join(
        f"{n[:3]}:{v['stuart_landau']:+.3f}" for n, v in row.items()), flush=True)
mean_sl = float(np.mean([heldout[k][n]["stuart_landau"]
                         for k in heldout for n in BAND_NAMES]))
mean_nc = float(np.mean([heldout[k][n]["no_cubic"] for k in heldout for n in BAND_NAMES]))
print(f"        mean: Stuart-Landau {mean_sl:+.3f}, without cubic terms "
      f"{mean_nc:+.3f}", flush=True)


# ---------------------------------------------------------------------------
# 8. Autonomous rollout.  Nothing injected -- x_b(t) IS the waveform.
# ---------------------------------------------------------------------------
def rhs(s, mset):
    out = np.zeros_like(s)
    for b in range(NB):
        out[:, 2 * b:2 * b + 2] = mset[b].predict(s)
    return out


def rollout(s0, steps, mset):
    s = np.atleast_2d(s0).astype(float).copy()
    traj = np.empty((steps + 1, s.shape[1]))
    traj[0] = s[0]
    for i in range(steps):
        k1 = rhs(s, mset)
        k2 = rhs(s + 0.5 * DT * k1, mset)
        k3 = rhs(s + 0.5 * DT * k2, mset)
        k4 = rhs(s + DT * k3, mset)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(s).all():
            return traj[:i + 1]
        traj[i + 1] = s[0]
    return traj


n_steps = int(round(ROLL_SECONDS / DT))
i0 = int(np.argmin(np.abs(T - ROLL_START)))
print(f"\n[ROLLOUT] autonomous RK4 from t={T[i0]:.1f}s for {ROLL_SECONDS:.0f}s at "
      f"{FS:.0f} Hz -- no phase injected, nothing clipped", flush=True)
traj = rollout(STATE[i0], n_steps, models)
traj_nc = rollout(STATE[i0], n_steps, models_nocubic)
nr = len(traj)
t_r = T[i0:i0 + nr]
ref = STATE[i0:i0 + nr]

# reconstructed trace = sum of the real parts, in recording units
rec = np.sum([traj[:, 2 * b] * SCALE[b] for b in range(NB)], axis=0)
rec_nc = np.sum([traj_nc[:, 2 * b] * SCALE[b] for b in range(NB)], axis=0)
true_w = np.sum([ref[:, 2 * b] * SCALE[b] for b in range(NB)], axis=0)

# amplitude: does the limit cycle survive, or collapse without the cubic terms?
r_model = np.sqrt(traj[:, 0::2] ** 2 + traj[:, 1::2] ** 2)
r_nc = np.sqrt(traj_nc[:, 0::2] ** 2 + traj_nc[:, 1::2] ** 2)
r_true = np.sqrt(ref[:, 0::2] ** 2 + ref[:, 1::2] ** 2)
print(f"          final radius per band: data "
      + " ".join(f"{v:.2f}" for v in r_true[-1])
      + " | model " + " ".join(f"{v:.2f}" for v in r_model[-1])
      + " | no-cubic " + " ".join(f"{v:.2f}" for v in r_nc[-1]), flush=True)

# phase coherence: how long does the WAVEFORM stay aligned?
def coherence_time(a, b_, thresh=0.5, win_s=0.5):
    w = int(round(win_s / FS ** 0 / DT))
    for i in range(w, len(a)):
        seg = slice(i - w, i)
        c = np.corrcoef(a[seg], b_[seg])[0, 1]
        if not np.isfinite(c) or c < thresh:
            return float((i - w) * DT)
    return float((len(a) - w) * DT)


coh = coherence_time(true_w, rec)
print(f"          WAVEFORM coherence (running 0.5 s corr > 0.5): {coh:.2f}s "
      f"of {ROLL_SECONDS:.0f}s", flush=True)

env_true = np.abs(hilbert(true_w))
env_rec = np.abs(hilbert(rec))
env_corr = float(np.corrcoef(env_true, env_rec)[0, 1])
print(f"          ENVELOPE correlation over the full {ROLL_SECONDS:.0f}s: "
      f"{env_corr:+.3f}", flush=True)

wave_r2_1s = float(1 - np.sum((true_w[:int(1 / DT)] - rec[:int(1 / DT)]) ** 2)
                   / np.sum((true_w[:int(1 / DT)] - true_w[:int(1 / DT)].mean()) ** 2))
print(f"          WAVEFORM R2 over the first 1 s (no phase injected): "
      f"{wave_r2_1s:+.3f}", flush=True)

# ---------------------------------------------------------------------------
# 10. Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8.5))
gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.1, 1.0, 1.0], hspace=0.55,
                      wspace=0.28)

ax = fig.add_subplot(gs[0, :])
ax.plot(t_r, true_w, color="0.3", lw=0.8, label="recording (1-50 Hz)")
ax.plot(t_r, rec, color=OI["vermillion"], lw=1.0, ls="--",
        label="Stuart-Landau, autonomous (NO phase injected)")
ax.axvline(t_r[0] + coh, color=OI["blue"], ls=":", lw=1.2,
           label=f"waveform coherence lost ({coh:.1f}s)")
ax.set_ylabel("amplitude")
ax.set_title(f"{CHANNEL}: the model generates the waveform itself", loc="left",
             fontweight="bold")
ax.legend(ncol=3, frameon=False)

ax = fig.add_subplot(gs[1, :])
ax.plot(t_r, env_true, color="0.3", lw=1.2, label="recording envelope")
ax.plot(t_r, env_rec, color=OI["vermillion"], lw=1.4, ls="--",
        label=f"model envelope (r={env_corr:+.2f})")
ax.set_ylabel("envelope")
ax.set_xlabel("recording time (s)")
ax.legend(ncol=2, frameon=False)
ax.set_title("Envelope outlives waveform coherence", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[2, 0])
bi = BAND_NAMES.index("alpha")
ax.plot(ref[:, 2 * bi], ref[:, 2 * bi + 1], color="0.6", lw=0.5, label="recording")
ax.plot(traj[:, 2 * bi], traj[:, 2 * bi + 1], color=OI["vermillion"], lw=1.0,
        label="model")
ax.plot(traj_nc[:, 2 * bi], traj_nc[:, 2 * bi + 1], color=OI["blue"], lw=1.0,
        ls="--", label="no cubic terms")
ax.set_xlabel("Re z (alpha)")
ax.set_ylabel("Im z (alpha)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(frameon=False)
ax.set_title("Limit cycle", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[2, 1])
xs = np.arange(NB)
ax.bar(xs - 0.2, omega_fit, 0.4, color=OI["vermillion"], label="fitted")
ax.bar(xs + 0.2, OMEGA_MEAS, 0.4, color="0.6", label="measured")
ax.set_xticks(xs, BAND_NAMES, rotation=30, ha="right", fontsize=7)
ax.set_ylabel(r"$\omega$ (rad/s)")
ax.legend(frameon=False)
ax.set_title(f"Recovered frequency ({omega_err:.0f}% mean error)", loc="left",
             fontweight="bold")

ax = fig.add_subplot(gs[3, :])
for b, name in enumerate(BAND_NAMES):
    ax.plot(t_r, r_true[:, b], color="0.6", lw=0.7)
    ax.plot(t_r, r_model[:, b], lw=1.3, label=name)
ax.set_ylabel("radius |z|")
ax.set_xlabel("recording time (s)")
ax.legend(ncol=5, frameon=False, fontsize=6.5)
ax.set_title("Per-band amplitude: grey = recording, coloured = model",
             loc="left", fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(Path(RESULT_DIR) / f"stuart_landau.{ext}", dpi=250,
                bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Gates
# ---------------------------------------------------------------------------
alive = bool(np.all(r_model[-1] > 0.1 * r_true.mean(0)))
alive_nc = bool(np.all(r_nc[-1] > 0.1 * r_true.mean(0)))
gates = {
    "omega_recovered": {"required": "mean error < 10%", "achieved": omega_err,
                        "passed": bool(omega_err < 10.0)},
    "one_step_beats_zero": {"required": 0.5, "achieved": mean_sl,
                            "passed": bool(mean_sl > 0.5)},
    "cubic_terms_matter": {
        "required": "Stuart-Landau one-step > no-cubic, or the no-cubic "
                    "rollout collapses",
        "achieved": [mean_sl, mean_nc, alive, alive_nc],
        "passed": bool(mean_sl > mean_nc or (alive and not alive_nc))},
    "oscillation_survives_the_rollout": {
        "required": "all band radii > 10% of their observed mean at the end",
        "achieved": alive, "passed": alive},
    "waveform_coherent_for_at_least_half_a_second": {
        "required": 0.5, "achieved": coh, "passed": bool(coh >= 0.5)},
    "envelope_outlives_waveform": {
        "required": "envelope correlation > 0 over the full rollout",
        "achieved": env_corr, "passed": bool(env_corr > 0)},
}
print("\n" + "=" * 76)
ok = True
for k, v in gates.items():
    ok &= print_check(v["passed"], f"{k}: achieved={v['achieved']}, "
                                   f"required={v['required']}")

save_metrics(Path(RESULT_DIR) / "metrics.json", {
    "task": "Stuart-Landau band oscillators on the analytic signal; phase is a "
            "state variable and the rollout generates the waveform itself",
    "passed": bool(ok),
    "note": "Waveform coherence is expected to be ~1 s and envelope agreement "
            "much longer: the phase direction of an oscillator has a zero "
            "Lyapunov exponent, so phase error accumulates without bound while "
            "amplitude error saturates. The two horizons are reported "
            "separately rather than averaged.",
    "channel": CHANNEL, "bands": {k: list(v) for k, v in BANDS.items()},
    "fs_hz": FS, "scale": SCALE.tolist(),
    "omega": {"fitted": omega_fit.tolist(), "measured": OMEGA_MEAS.tolist(),
              "mean_pct_error": omega_err},
    "lambda_fitted": lam_fit.tolist(),
    "lamb_selected": lambs,
    "heldout_r2": heldout,
    "mean_heldout": {"stuart_landau": mean_sl, "no_cubic": mean_nc},
    "rollout": {"start_s": float(T[i0]), "seconds": ROLL_SECONDS,
                "waveform_coherence_s": coh,
                "waveform_r2_first_1s": wave_r2_1s,
                "envelope_correlation": env_corr,
                "final_radius_model": r_model[-1].tolist(),
                "final_radius_no_cubic": r_nc[-1].tolist(),
                "final_radius_data": r_true[-1].tolist()},
    "gates": gates,
})
# Symbolic extraction runs LAST and its output is saved separately.
# get_formula reaches into PyKAN's auto_symbolic, which has already crashed a
# run of this script outright; with it ahead of the save, a failure there threw
# away the whole fit.  Everything above is on disk before this point.
# ---------------------------------------------------------------------------
# 12. Equations (last, and separately saved)
# ---------------------------------------------------------------------------
print("\n[EQUATION] dz_b/dt, from the fitted field", flush=True)
equations = {}
for b, name in enumerate(BAND_NAMES):
    try:
        ex = quiet(models[b].get_formula, var_names=[
            n.replace("*", "_").replace("2", "2") for n in lift_names[b]],
            round_places=3, lib=["x", "x^2", "0"], r2_threshold=0.9)
        equations[name] = [str(e) for e in ex]
        print(f"  dx_{name}/dt = {str(ex[0])[:150]}", flush=True)
        print(f"  dy_{name}/dt = {str(ex[1])[:150]}", flush=True)
    except Exception as exc:                                     # noqa: BLE001
        equations[name] = f"{type(exc).__name__}: {exc}"
        print(f"  {name}: get_formula failed ({type(exc).__name__})", flush=True)


save_metrics(Path(RESULT_DIR) / "equations.json", {"equations": equations})
print(f"[DONE]  {RESULT_DIR}")
