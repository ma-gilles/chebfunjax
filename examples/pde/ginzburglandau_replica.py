"""Complex Ginzburg-Landau equation in 2D.

Faithful replica of pde/GinzburgLandau.m by Nick Trefethen (May
2016): the complex Ginzburg-Landau equation

    u_t = Lap u + u - (1 + 1.5i) u |u|^2

with spin2/ETDRK4 on [-50,50]^2: spirals from complex and real
initial data at t = 16, the beginnings of chaos at t = 48 (with the
diagonal symmetry preserved), chaos at t = 96, and the big-canvas
[-100,100]^2 two-spiral run to t = 30/60 plus the phase portrait
(complex-argument coloring).

Original: https://www.chebfun.org/examples/pde/GinzburgLandau.html
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

from matplotlib.colors import hsv_to_rgb

from chebfunjax.plotting import chebfun_style
from chebfunjax.spin.solver2d import spin2
from chebfunjax.spin.spinop2 import SpinOp2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')
FIG = [0]


def _gl(dom, t1, u0, npts, dt):
    op = SpinOp2(lin_coeffs=(1.0, 0, 0, 0, 0),
                 nonlin_vals=lambda u: u - (1 + 1.5j) * u * np.abs(u)**2,
                 n_vars=1, domain=dom, tspan=(0.0, t1), u0=u0,
                 is_real=False)
    return spin2(op, npts, dt, dealias=False)


def _plot(U, dom, phase=False):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    U = np.asarray(U)
    if phase:
        h = (np.angle(U) + np.pi) / (2 * np.pi)
        v = np.abs(U) / max(np.abs(U).max(), 1e-300)
        rgb = hsv_to_rgb(np.stack(
            [h.T, np.ones_like(h.T), np.clip(v.T, 0, 1)], axis=-1))
        ax.imshow(rgb, origin="lower",
                  extent=(dom[0], dom[1], dom[2], dom[3]))
    else:
        ax.imshow(U.real.T, origin="lower", cmap="viridis",
                  extent=(dom[0], dom[1], dom[2], dom[3]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"GinzburgLandau_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    dom = (-50.0, 50.0, -50.0, 50.0)

    def u1(x, y):
        return (1j * x + y) * np.exp(-.03 * (x**2 + y**2))

    def u2(x, y):
        return (x + y + 0j) * np.exp(-.03 * (x**2 + y**2))

    # 2. Non-chaotic spirals at t = 16.
    npts, dt = 80, 4 / 80
    t0 = time.time()
    _, _, _, U = _gl(dom, 16.0, u1, npts, dt)
    _plot(U, dom)
    _, _, _, U = _gl(dom, 16.0, u2, npts, dt)
    _plot(U, dom)
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}", flush=True)

    # 3. Beginnings of chaos at t = 48.
    t0 = time.time()
    _, _, _, U = _gl(dom, 48.0, u1, npts, dt)
    _plot(U, dom)
    _, _, _, U48 = _gl(dom, 48.0, u2, npts, dt)
    _plot(U48, dom)
    sym = (np.linalg.norm(np.asarray(U48).real
                          - np.asarray(U48).real.T)
           / np.linalg.norm(np.asarray(U48).real))
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}")
    print(f"diagonal symmetry error at t=48: {sym:.2e}", flush=True)

    # 4. Chaos at t = 96.
    t0 = time.time()
    _, _, _, U = _gl(dom, 96.0, u1, npts, dt)
    _plot(U, dom)
    _, _, _, U = _gl(dom, 96.0, u2, 128, 4 / 128)
    _plot(U, dom)
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}", flush=True)

    # 5. A bigger canvas: two spirals on [-100,100]^2, t = 30 and 60.
    dom2 = (-100.0, 100.0, -100.0, 100.0)

    def ub(x, y):
        return ((1j * (x - 8) + (y - 2))
                * np.exp(-.03 * ((x - 8)**2 + (y - 2)**2))
                + ((x + 8) - (y + 2) + 0j)
                * np.exp(-.03 * ((x + 8)**2 + (y + 2)**2)))

    t0 = time.time()
    npts, dt = 128, 8 / 128
    _, _, _, U30 = _gl(dom2, 30.0, ub, npts, dt)
    _plot(U30, dom2)
    _, _, _, U60 = _gl(dom2, 30.0, lambda x, y: U30, npts, dt)
    _plot(U60, dom2)
    _plot(U60, dom2, phase=True)
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}")


if __name__ == "__main__":
    run()
