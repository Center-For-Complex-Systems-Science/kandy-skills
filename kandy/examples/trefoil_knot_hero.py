#!/usr/bin/env python3
"""Hero figure: Trefoil-selected Hopf fibers in R^3.

Generates a publication-quality standalone visualization of Hopf fibers
whose basepoints trace a (2,3)-torus knot path on S^2.  Each fiber is a
circle in S^3 that projects to a linked loop in R^3 via stereographic
projection.  The collection of fibers produces the characteristic
trefoil-linked structure of the Hopf fibration.

No axes, no background — just the geometry with glow for depth.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
RESULTS_DIR = "results/TrefoilKnot"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def hopf_fiber_parametric(theta, phi, m=800):
    """Hopf fiber over the S^2 basepoint at spherical coords (theta, phi)."""
    y1 = np.sin(theta) * np.cos(phi)
    y2 = np.sin(theta) * np.sin(phi)
    y3 = np.cos(theta)

    z1_mag = np.sqrt(max((1.0 + y3) / 2.0, 1e-12))
    z2 = complex(y1, y2) / (2.0 * z1_mag) if z1_mag > 1e-9 else 0j

    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    eit = np.exp(1j * t)

    z1_fiber = eit * z1_mag
    z2_fiber = eit * z2

    return np.column_stack([
        z1_fiber.real, z1_fiber.imag,
        z2_fiber.real, z2_fiber.imag,
    ]).astype(np.float32)


def stereo_s3_to_r3(x4):
    """Stereographic projection S^3 -> R^3 (north pole x4=1)."""
    denom = np.clip(1.0 - x4[:, 3], 1e-9, None)
    return x4[:, :3] / denom[:, None]


# ---------------------------------------------------------------------------
# Generate trefoil-selected Hopf fibers
# ---------------------------------------------------------------------------
N_FIBERS = 30
M_POINTS = 1200
P, Q = 2, 3
THETA_AMP = 0.40
PHI_AMP = 0.30

t = np.linspace(0, 2 * np.pi, N_FIBERS, endpoint=False)
cmap = plt.cm.twilight_shifted

fibers_r3 = []
fiber_colors = []

for i, ti in enumerate(t):
    theta = np.pi / 2 + THETA_AMP * np.sin(Q * ti)
    phi = P * ti + PHI_AMP * np.cos(Q * ti)

    fiber_s3 = hopf_fiber_parametric(theta, phi, m=M_POINTS)
    r3 = stereo_s3_to_r3(fiber_s3)
    fibers_r3.append(r3)
    fiber_colors.append(cmap(i / N_FIBERS))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for r3, col in zip(fibers_r3, fiber_colors):
    # Glow: multiple transparent wide passes
    ax.plot(r3[:, 0], r3[:, 1], r3[:, 2],
            lw=10.0, alpha=0.04, color=col, solid_capstyle="round")
    ax.plot(r3[:, 0], r3[:, 1], r3[:, 2],
            lw=7.0, alpha=0.08, color=col, solid_capstyle="round")
    ax.plot(r3[:, 0], r3[:, 1], r3[:, 2],
            lw=4.5, alpha=0.18, color=col, solid_capstyle="round")
    # Core
    ax.plot(r3[:, 0], r3[:, 1], r3[:, 2],
            lw=2.2, alpha=0.95, color=col, solid_capstyle="round")

# Tight framing — shrink limits to zoom the projection in
all_r3 = np.concatenate(fibers_r3, axis=0)
cx, cy, cz = all_r3.mean(axis=0)
half = np.abs(all_r3 - [cx, cy, cz]).max() * 0.72
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.set_zlim(cz - half, cz + half)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=15, azim=60)

ax.set_axis_off()

fig.subplots_adjust(left=-0.25, right=1.25, top=1.25, bottom=-0.25)
fig.savefig(f"{RESULTS_DIR}/trefoil_hero.png",
            dpi=300, bbox_inches="tight", facecolor="white",
            pad_inches=0)
fig.savefig(f"{RESULTS_DIR}/trefoil_hero.pdf",
            dpi=300, bbox_inches="tight", facecolor="white",
            pad_inches=0)
plt.close(fig)

print(f"Saved to {RESULTS_DIR}/trefoil_hero.png and .pdf")
