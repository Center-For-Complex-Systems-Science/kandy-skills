#!/usr/bin/env python3
"""Hero figure for the plasma/ examples — the physics, on one page.

Four panels, one per example in this directory, each showing the system the
example learns rather than the fit itself:

  a  Orszag-Tang vortex        current sheets, with magnetic field lines
  b  2D MHD turbulence         filamented current, with magnetic field lines
  c  Brio-Wu shock tube        the MHD Riemann fan, compound wave marked
  d  Kinetic heat-flux closure temperature relaxation, flux limiter inset

This script only draws; it fits nothing and writes no metrics.  It re-simulates
each system at figure resolution so it stays self-contained, like
geometry/trefoil_knot_hero.py.

In panels a and b the magnetic field lines are CONTOURS OF THE VECTOR
POTENTIAL A: because B = (A_y, -A_x), every contour of A is a field line, which
is the same fact orszag_tang_example.py exploits to make div(B) = 0 structural.
Current concentrates where those contours pinch together — that is what a
current sheet IS, and it is why the Orszag-Tang problem is the standard test.

Colours: diverging RdBu_r for signed fields (current), perceptually uniform
viridis for the positive temperature field, Okabe-Ito for the line plots — all
colourblind-safe, and all readable in greyscale because each panel also carries
a shape cue (contours, line style, marker).

Run:  python plasma/plasma_hero.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir
from kandy.plotting import use_pub_style

SEED = 42

# Okabe-Ito, colourblind-safe
INK = "#000000"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"

# ---------------------------------------------------------------------------
# Shared 2D spectral MHD machinery (vorticity / vector-potential form)
# ---------------------------------------------------------------------------
def spectral_setup(n):
    kx = np.fft.fftfreq(n, d=1.0 / n).reshape(-1, 1)
    ky = np.fft.fftfreq(n, d=1.0 / n).reshape(1, -1)
    k2 = kx ** 2 + ky ** 2
    return kx, ky, k2, np.where(k2 == 0, 1.0, k2), (np.abs(kx) < n / 3) & (np.abs(ky) < n / 3)


def mhd2d_run(n, w_hat, A_hat, nu, eta, dt, n_steps):
    """Integrate incompressible 2D MHD; return the final (w, A, j) fields."""
    KX, KY, K2, K2I, DEAL = spectral_setup(n)
    ifft = lambda z: np.real(np.fft.ifft2(z))

    def rhs(w_h, A_h):
        psi_h = np.where(K2 == 0, 0.0, w_h / K2I)
        u, v = ifft(1j * KY * psi_h), ifft(-1j * KX * psi_h)
        Bx, By = ifft(1j * KY * A_h), ifft(-1j * KX * A_h)
        w_x, w_y = ifft(1j * KX * w_h), ifft(1j * KY * w_h)
        A_x, A_y = ifft(1j * KX * A_h), ifft(1j * KY * A_h)
        j_x, j_y = ifft(1j * KX * K2 * A_h), ifft(1j * KY * K2 * A_h)
        dw = -(u * w_x + v * w_y) + (Bx * j_x + By * j_y)
        dA = -(u * A_x + v * A_y)
        return (np.fft.fft2(dw) * DEAL - nu * K2 * w_h,
                np.fft.fft2(dA) * DEAL - eta * K2 * A_h)

    for _ in range(n_steps):
        a1, b1 = rhs(w_hat, A_hat)
        a2, b2 = rhs(w_hat + 0.5 * dt * a1, A_hat + 0.5 * dt * b1)
        a3, b3 = rhs(w_hat + 0.5 * dt * a2, A_hat + 0.5 * dt * b2)
        a4, b4 = rhs(w_hat + dt * a3, A_hat + dt * b3)
        w_hat = w_hat + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        A_hat = A_hat + (dt / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4)
    return ifft(w_hat), ifft(A_hat), ifft(K2 * A_hat)


def orszag_tang_fields(n=128, nu=2e-3, eta=5e-3, dt=2e-3, n_steps=500):
    x = np.arange(n) * 2 * np.pi / n
    XX, YY = np.meshgrid(x, x, indexing="ij")
    w0 = np.cos(XX) + np.cos(YY)                    # u = (-sin y,  sin x)
    A0 = np.cos(YY) + 0.5 * np.cos(2 * XX)          # B = (-sin y,  sin 2x)
    return mhd2d_run(n, np.fft.fft2(w0), np.fft.fft2(A0), nu, eta, dt, n_steps)


def turbulence_fields(n=192, nu=4e-3, eta=4e-3, dt=2e-3, n_steps=700):
    KX, KY, K2, _, DEAL = spectral_setup(n)
    rng = np.random.default_rng(SEED)
    ifft = lambda z: np.real(np.fft.ifft2(z))

    def low_k(amp, k_peak):
        f_h = np.fft.fft2(rng.standard_normal((n, n)))
        f_h *= np.exp(-K2 / (2.0 * k_peak ** 2)) * DEAL
        f = ifft(f_h)
        return np.fft.fft2(f * amp / np.abs(f).max())

    return mhd2d_run(n, low_k(6.0, 8.0), low_k(0.6, 6.0), nu, eta, dt, n_steps)


# ---------------------------------------------------------------------------
# Brio-Wu shock tube (1D ideal MHD, MUSCL + HLL)
# ---------------------------------------------------------------------------
GAMMA, BX = 2.0, 0.75


def _cons(W):
    rho, u, v, p, By = (W[..., i] for i in range(5))
    E = p / (GAMMA - 1) + 0.5 * rho * (u ** 2 + v ** 2) + 0.5 * (BX ** 2 + By ** 2)
    return np.stack([rho, rho * u, rho * v, By, E], axis=-1)


def _prim(U):
    rho, mx, my, By, E = (U[..., i] for i in range(5))
    rho = np.maximum(rho, 1e-8)
    u, v = mx / rho, my / rho
    p = (GAMMA - 1) * (E - 0.5 * rho * (u ** 2 + v ** 2) - 0.5 * (BX ** 2 + By ** 2))
    return np.stack([rho, u, v, np.maximum(p, 1e-8), By], axis=-1)


def _flux(W):
    rho, u, v, p, By = (W[..., i] for i in range(5))
    b2 = BX ** 2 + By ** 2
    E = p / (GAMMA - 1) + 0.5 * rho * (u ** 2 + v ** 2) + 0.5 * b2
    return np.stack([
        rho * u,
        rho * u ** 2 + p + 0.5 * By ** 2 - 0.5 * BX ** 2,
        rho * u * v - BX * By,
        u * By - v * BX,
        (E + p + 0.5 * b2) * u - BX * (u * BX + v * By),
    ], axis=-1)


def _cf(W):
    rho, _, _, p, By = (W[..., i] for i in range(5))
    a2, ca2, cax2 = GAMMA * p / rho, (BX ** 2 + By ** 2) / rho, BX ** 2 / rho
    disc = np.maximum((a2 + ca2) ** 2 - 4 * a2 * cax2, 0.0)
    return np.sqrt(np.maximum(0.5 * (a2 + ca2 + np.sqrt(disc)), 1e-12))


def brio_wu(n=800, t_end=0.10, cfl=0.4):
    dx = 1.0 / n
    xc = (np.arange(n) + 0.5) * dx
    W0 = np.where((xc < 0.5)[:, None],
                  np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
                  np.array([0.125, 0.0, 0.0, 0.1, -1.0]))
    U, t = _cons(W0), 0.0
    mm = lambda a, b: np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

    def div(U):
        W = _prim(U)
        Wg = np.concatenate([W[:1], W[:1], W, W[-1:], W[-1:]], axis=0)
        s = mm(Wg[1:-1] - Wg[:-2], Wg[2:] - Wg[1:-1])
        WL, WR = (Wg[1:-1] + 0.5 * s)[:-1], (Wg[1:-1] - 0.5 * s)[1:]
        UL, UR, FL, FR = _cons(WL), _cons(WR), _flux(WL), _flux(WR)
        cL, cR = _cf(WL), _cf(WR)
        SL = np.minimum(np.minimum(WL[..., 1] - cL, WR[..., 1] - cR), 0.0)[:, None]
        SR = np.maximum(np.maximum(WL[..., 1] + cL, WR[..., 1] + cR), 0.0)[:, None]
        F = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
        return -np.diff(F, axis=0) / dx

    while t < t_end - 1e-12:
        W = _prim(U)
        dt = min(cfl * dx / np.max(np.abs(W[:, 1]) + _cf(W)), t_end - t)
        U1 = U + dt * div(U)
        U = 0.5 * (U + U1 + dt * div(U1))
        t += dt
    return xc, _prim(U)


# ---------------------------------------------------------------------------
# Kinetic heat-flux relaxation (1D, the closure example's forward problem)
# ---------------------------------------------------------------------------
# t_end is short on purpose: the flux limiter erases the steep gradients within
# t ~ 0.05, so a window out to t = 1 (what the closure example integrates) is a
# flat field with a thin coloured strip at the bottom.  That fast collapse is
# also why saturated samples are rare enough to need stratifying.
def heat_flux_run(nx=128, lam=0.50, amp=0.42, t_end=0.10, dt=2e-4, n_save=200):
    T_EQ, NU_R, KAP, A, Z0, B, TMIN = 1.0, 0.02, 0.02, 0.75, 0.06, 0.35, 0.20
    K = 2 * np.pi * np.fft.fftfreq(nx, d=2 * np.pi / nx)
    deal = np.abs(K) <= (2 / 3) * np.abs(K).max()
    d = lambda f: np.real(np.fft.ifft(1j * K * np.fft.fft(f)))
    hil = lambda f: np.real(np.fft.ifft(-1j * np.sign(K) * np.fft.fft(f)))

    def rhs(T):
        Ts = np.clip(T, TMIN, None)
        z = lam * d(np.log(Ts))
        h = hil((Ts - Ts.mean()) / Ts.mean())
        q = Ts ** 1.5 * (-A * np.tanh(z / Z0) + B * h) - KAP * d(Ts)
        return -d(np.real(np.fft.ifft(np.fft.fft(q) * deal))) - NU_R * (Ts - T_EQ)

    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    rng = np.random.default_rng(SEED)
    p = np.zeros_like(x)
    for m in range(1, 17):
        p += (rng.uniform(0.55, 1.15) / m ** 0.2) * np.cos(m * x + rng.uniform(0, 2 * np.pi))
    T = T_EQ + amp * p / np.abs(p).max()

    every = max(1, int(t_end / dt) // n_save)
    frames, times = [T.copy()], [0.0]
    for i in range(1, int(t_end / dt) + 1):
        k1 = rhs(T); k2 = rhs(T + 0.5 * dt * k1)
        k3 = rhs(T + 0.5 * dt * k2); k4 = rhs(T + dt * k3)
        T = T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % every == 0:
            frames.append(T.copy()); times.append(i * dt)
    return x, np.array(times), np.array(frames), (A, Z0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def field_panel(ax, field, A, title, subtitle):
    """Signed field as an image, with magnetic field lines (contours of A)."""
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    s = np.percentile(np.abs(field), 99.5)
    im = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-s, vmax=s, interpolation="bilinear", rasterized=True)
    ax.contour(np.linspace(0, 2 * np.pi, A.shape[0]),
               np.linspace(0, 2 * np.pi, A.shape[1]), A.T,
               levels=14, colors="k", linewidths=0.45, alpha=0.38)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=3)
    ax.text(0.5, -0.055, subtitle, transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color="0.35")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("current $j$", fontsize=7.5)
    return im


def main() -> None:
    print("[HERO] simulating Orszag-Tang ...", flush=True)
    _, A_ot, j_ot = orszag_tang_fields()
    print("[HERO] simulating 2D MHD turbulence ...", flush=True)
    _, A_tb, j_tb = turbulence_fields()
    print("[HERO] simulating Brio-Wu shock tube ...", flush=True)
    x_bw, W_bw = brio_wu()
    print("[HERO] simulating heat-flux relaxation ...", flush=True)
    x_hf, t_hf, T_hf, (A_lim, z0) = heat_flux_run()

    use_pub_style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a. Orszag-Tang current sheets
    field_panel(ax_a, j_ot, A_ot, "Orszag–Tang vortex",
                "current sheets form where field lines pinch")

    # b. Decaying 2D MHD turbulence
    field_panel(ax_b, j_tb, A_tb, "2D MHD turbulence",
                "current filaments between magnetic islands")

    # c. Brio-Wu shock tube
    rho, u, p, By = W_bw[:, 0], W_bw[:, 1], W_bw[:, 3], W_bw[:, 4]
    ax_c.plot(x_bw, rho, color=BLUE, lw=1.6, label=r"$\rho$")
    ax_c.plot(x_bw, By, color=VERMILLION, lw=1.6, ls="--", label=r"$B_y$")
    ax_c.plot(x_bw, p, color=GREEN, lw=1.1, ls=":", label=r"$p$")
    # Locate the compound wave: the sharp B_y reversal left of centre.
    left = (x_bw > 0.30) & (x_bw < 0.58)
    x_comp = float(x_bw[left][np.argmax(np.abs(np.gradient(By[left], x_bw[left])))])
    ax_c.annotate("compound wave", xy=(x_comp, By[np.argmin(np.abs(x_bw - x_comp))]),
                  xytext=(x_comp - 0.26, -0.85), fontsize=7.5, color="0.25",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color="0.45"))
    ax_c.set_xlabel("$x$")
    ax_c.set_ylabel("state")
    ax_c.set_title("Brio–Wu shock tube  ($t=0.1$)", fontsize=10, pad=3)
    ax_c.text(0.5, -0.30, "seven-wave MHD Riemann fan, $\\gamma=2$, $B_x=0.75$",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=7.5, color="0.35")
    ax_c.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # d. Kinetic heat-flux relaxation, with the flux limiter inset
    im = ax_d.pcolormesh(x_hf, t_hf, T_hf, cmap="viridis", shading="auto",
                         rasterized=True)
    ax_d.set_xlabel("$x$")
    ax_d.set_ylabel("$t$")
    ax_d.set_title("Kinetic heat-flux relaxation", fontsize=10, pad=3)
    ax_d.text(0.5, -0.30, "steep gradients are limited, not diffused",
              transform=ax_d.transAxes, ha="center", va="top",
              fontsize=7.5, color="0.35")
    cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("$T$", fontsize=7.5)

    inset = ax_d.inset_axes([0.55, 0.60, 0.42, 0.34])
    zz = np.linspace(-5 * z0, 5 * z0, 300)
    inset.plot(zz, -A_lim * np.tanh(zz / z0), color=ORANGE, lw=1.5)
    inset.axhline(-A_lim, color="0.6", lw=0.6, ls=":")
    inset.axhline(A_lim, color="0.6", lw=0.6, ls=":")
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_title(r"$C_{\rm loc}(z)$", fontsize=7)
    for sp in inset.spines.values():
        sp.set_linewidth(0.6)
    inset.patch.set_alpha(0.9)

    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        ax.text(-0.09, 1.06, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="right")

    fig.suptitle("Plasma systems in $examples/plasma/$", fontsize=12, y=0.975)

    out = ensure_outdir("results/PlasmaHero")
    fig.savefig(out / "plasma_hero.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / "plasma_hero.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGS]  Saved {out}/plasma_hero.png and .pdf")


if __name__ == "__main__":
    main()
