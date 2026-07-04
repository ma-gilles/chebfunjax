"""Generate all 10 plots for Guide Chapter 12 (Chebfun2: Getting Started).

Figure order follows https://www.chebfun.org/docs/guide/guide12.html
exactly; each file is saved at the reference render's pixel size
(600x270).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.plotting import (
    PARULA,
    chebfun_style,
    contour,
    phaseplot,
    save_chebfun_figure,
    surf,
)

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "guide")
os.makedirs(OUT, exist_ok=True)
SIZE = (600, 270)
plot_idx = 0


def save(fig):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT, f"guide12_{plot_idx:02d}.png")
    save_chebfun_figure(fig, path, size=SIZE)
    plt.close(fig)
    print(f"  guide12_{plot_idx:02d}.png saved")


def peaks(x, y):
    return (3 * (1 - x) ** 2 * jnp.exp(-(x**2) - (y + 1) ** 2)
            - 10 * (x / 5 - x**3 - y**5) * jnp.exp(-(x**2) - y**2)
            - 1 / 3 * jnp.exp(-((x + 1) ** 2) - y**2))


# Fig 1: MATLAB's builtin peaks (surface on the 49x49 default grid)
try:
    fig = plt.figure()
    ax = fig.add_axes([0.087, -0.05, 0.85, 1.05], projection="3d")
    gx = np.linspace(-3, 3, 49)
    Xg, Yg = np.meshgrid(gx, gx)
    Zg = np.asarray(peaks(jnp.asarray(Xg), jnp.asarray(Yg)))
    ax.plot_surface(Xg, Yg, Zg, cmap=PARULA,
                    rstride=1, cstride=1, linewidth=0.15,
                    edgecolor="k", antialiased=True)
    ax.view_init(elev=30, azim=-127.5)
    ax.set_title("Peaks")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 2: chebfun2 peaks, plot(f) (smooth surface), axis tight
try:
    f_peaks = chebfun2(peaks, domain=(-3.0, 3.0, -3.0, 3.0))
    fig, ax = surf(f_peaks, title="Chebfun2 Peaks")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Figs 3-4: f = cos(2*pi*x*y): surface with zlim, then square contour
try:
    f = chebfun2(lambda x, y: jnp.cos(2 * jnp.pi * x * y))
    fig, ax = surf(f)
    ax.set_zlim(-2, 2)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

try:
    fig, ax = contour(f)
    ax.set_aspect("equal")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 5: zero contours of f - 0.95 (curves, not fills)
try:
    fig, ax = plt.subplots()
    gx = np.linspace(-1, 1, 600)
    Xg, Yg = np.meshgrid(gx, gx)
    Zg = np.asarray(f(jnp.asarray(Xg).ravel(), jnp.asarray(Yg).ravel()))
    ax.contour(Xg, Yg, Zg.reshape(Xg.shape), levels=[0.95],
               colors=["#0072BD"], linewidths=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title("Zero contours of f-.95")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 6: fy = diff(f, 1, 1) surface
try:
    fy = f.diff(dim=1)  # MATLAB diff(f,1,1): d/dy
    fig, ax = surf(fy)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 7: contour of 1/(2 + cos(.25 + x^2 y + y^2)) on [-4 4 -2 2]
try:
    g = chebfun2(
        lambda x, y: 1.0 / (2.0 + jnp.cos(0.25 + x**2 * y + y**2)),
        domain=(-4.0, 4.0, -2.0, 2.0),
    )
    fig, ax = contour(g)
    ax.set_aspect("equal")
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 8: phase portrait of sin(z) - sinh(z) on 2*pi*[-1 1 -1 1]
try:
    L = 2 * float(np.pi)
    # MATLAB: chebfun2(@(z) sin(z)-sinh(z), 2*pi*[-1 1 -1 1]); plot(f)
    # draws the phase portrait. The phaseplot helper takes a callable
    # of complex z directly.
    fig, ax = phaseplot(
        lambda z: np.sin(z) - np.sinh(z), region=[-L, L, -L, L]
    )
    ax.set_axis_off()
    # center a full-height square like the published render
    ax.set_position([0.333, 0.115, 0.367, 0.811])
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 9: contour of the banana function with rank title
try:
    def ff(x, y):
        return jnp.exp(-40 * (x**2 - x * y + 2 * y**2 - 0.5) ** 2)

    fb = chebfun2(ff)
    fig, ax = plt.subplots()
    gx = np.linspace(-1, 1, 500)
    Xg, Yg = np.meshgrid(gx, gx)
    Zg = np.asarray(fb(jnp.asarray(Xg).ravel(),
                       jnp.asarray(Yg).ravel())).reshape(Xg.shape)
    ax.contour(Xg, Yg, Zg, levels=np.arange(0.1, 0.95, 0.1), cmap="jet")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(f"rank {fb.rank}", fontsize=12)
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

# Fig 10: 3x3 grid of rank-k approximations, black contours, axis off
try:
    fig = plt.figure()
    levels = np.arange(0.2, 0.81, 0.2)
    gx = np.linspace(-1, 1, 400)
    Xg, Yg = np.meshgrid(gx, gx)
    for k in range(1, 10):
        ax = fig.add_axes([
            0.03 + 0.33 * ((k - 1) % 3),
            0.67 - 0.3 * ((k - 1) // 3),
            0.28, 0.28,
        ])
        # rank-k approximation: first k ACA terms of the full chebfun2
        # (MATLAB's chebfun2(ff, k) runs k ACA steps — same leading terms)
        ap = fb.approx
        Zg = np.zeros(Xg.shape)
        tx = jnp.asarray(gx)
        for j in range(min(k, ap.rank)):
            cj = np.asarray(ap.cols[j](tx))
            rj = np.asarray(ap.rows[j](tx))
            Zg += float(ap.pivots[j]) * np.outer(cj, rj)
        ax.contour(Xg, Yg, Zg, levels=levels, colors="k", linewidths=0.8)
        ax.set_xlim(-1, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
    # center a full-height square like the published render
    ax.set_position([0.333, 0.115, 0.367, 0.811])
    save(fig)
except Exception as e:  # noqa: BLE001
    plot_idx += 1
    print(f"  guide12_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 12: generated {plot_idx} plots.")
