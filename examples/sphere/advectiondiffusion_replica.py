"""Advection-diffusion in the unit ball.

Faithful replica of sphere/AdvectionDiffusion.m by Nicolas Boulle
(July 2019): the advection-diffusion equation

    c_t = D lap(c) - v . grad(c)

in the unit ball with D = 1/5000 and the divergence-free no-slip
field v = curl[z e^{-5 r^2} (x,y,z)], integrated to t = 15 with
IMEX-BDF1 (a ballfun Helmholtz solve with Neumann conditions each
step).

Original: https://www.chebfun.org/examples/sphere/AdvectionDiffusion.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]


def _slice_plot(c, title, clim=(-0.2, 0.2), n=160):
    """Slice through the three coordinate planes (MATLAB slice(c))."""
    FIG[0] += 1
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    g = np.linspace(-1, 1, n)
    A, B = np.meshgrid(g, g)
    R2 = A**2 + B**2
    inside = R2 <= 1
    planes = [("z = 0", lambda a, b: (a, b, 0 * a)),
              ("y = 0", lambda a, b: (a, 0 * a, b)),
              ("x = 0", lambda a, b: (0 * a, a, b))]
    for ax, (ttl, xyz) in zip(axes, planes):
        x, y, z = xyz(A, B)
        V = np.full(A.shape, np.nan)
        xi, yi, zi = x[inside], y[inside], z[inside]
        ri = np.sqrt(xi**2 + yi**2 + zi**2)
        lami = np.arctan2(yi, xi)
        thi = np.arccos(np.clip(np.where(ri > 0, zi / np.maximum(ri, 1e-300), 1.0), -1, 1))
        # 1-D inputs trigger Ballfun's tensor-grid path (20k^3!);
        # reshape to 2-D for elementwise evaluation.
        V[inside] = np.asarray(c(ri.reshape(1, -1), lami.reshape(1, -1),
                                 thi.reshape(1, -1))).ravel()
        im = ax.imshow(V, origin="lower", cmap="viridis",
                       extent=(-1, 1, -1, 1), vmin=clim[0],
                       vmax=clim[1])
        ax.set_title(ttl)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(
        _IMG, f"AdvectionDiffusion_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # The divergence-free, no-slip velocity field.
    w = Ballfunv.from_functions(
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * x,
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * y,
        lambda x, y, z: z * np.exp(-5 * (x**2 + y**2 + z**2)) * z)
    v = w.curl()
    print("ans =")
    print(f"     {float(v.div().norm()):.4e}")

    # No-slip: v . n on the boundary r = 1.
    lam = np.linspace(-np.pi, np.pi, 181)
    th = np.linspace(0, np.pi, 91)
    L, T = np.meshgrid(lam, th)
    x = np.cos(L) * np.sin(T)
    y = np.sin(L) * np.sin(T)
    z = np.cos(T)
    r1 = np.ones_like(L)
    vx, vy, vz = v.components
    vn = (np.asarray(vx(r1, L, T)) * x
          + np.asarray(vy(r1, L, T)) * y
          + np.asarray(vz(r1, L, T)) * z)
    print("ans =")
    print(f"     {np.max(np.abs(vn)):.4e}")

    # Initial condition and its visualization.
    c = Ballfun.from_function(
        lambda x_, y_, z_: -x_ * np.exp(-5 * (x_**2 + y_**2 + z_**2)))
    _slice_plot(c, "Initial condition c", clim=(-0.19, 0.19))

    # IMEX-BDF1 to t = 15 (Helmholtz solve with Neumann BC per step).
    D = 1 / 5000
    dt = 0.1
    K = 1j * np.sqrt(1 / (dt * D))
    nsteps = int(np.ceil(15 / dt))
    t0 = time.time()
    for n in range(nsteps + 1):
        if n % 50 == 0:
            _slice_plot(c, f"Time {n * dt:g}")
            print(f"t={n * dt:g} plotted ({time.time()-t0:.0f}s)",
                  flush=True)
        gx, gy, gz = c.grad()
        rhs = K**2 * c + v.dot(Ballfunv(gx, gy, gz)) * (1 / D)
        # Per-step simplification chops the roundoff-seeded parasitic
        # mode of the explicit advection term (growth ~1.5x/step from
        # 1e-16 blows up around step 90 without it) -- MATLAB's ballfun
        # pipeline simplifies adaptively and is stable the same way.
        c = Ballfun.helmholtz(rhs, K, lambda lam_, th_: 0.0, 100,
                              bc_type="neumann").simplify()
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    run()
