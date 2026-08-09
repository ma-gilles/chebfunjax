"""Problems from Binous, Shaikh and Bellagi.

Faithful replica of temp/BinousShaikhBellagi.m (Trefethen, 2014):
transport-phenomena problems from Binous, Shaikh & Bellagi [1]
solved with chebop -- a split (nonlinear) BVP, two diffusion
problems via operator exponentials, the Falkner-Skan equation, and
the coupled convection system past an isothermal plate.

Original: https://www.chebfun.org/examples/temp/BinousShaikhBellagi.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'temp')
FIG = [0]


def _plot(xs, ys, title=None, ylim=None):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(xs, ys, lw=1.6)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"BinousShaikhBellagi_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Split boundary value problem: u u' - u'' = 1.
    N = Chebop(lambda x, u: u * u.diff() - u.diff(2), domain=(-1, 1))
    N.lbc = 0
    N.rbc = 2
    u = N.solve(1)
    xs = np.linspace(-1, 1, 600)
    _plot(xs, np.asarray(u(xs)), ylim=(0, 2))
    v = np.asarray(u(np.array([0.0, 1 / np.sqrt(2)])))
    print("ans =")
    print(f"   {float(v[0]):.15f}   {float(v[1]):.15f}")

    # 2. Diffusion problem: u_t = u_yy, expm at t = 0.0126.
    L = Chebop(lambda y, w: w.diff(2), domain=(0, 1))
    L.bc = 'dirichlet'
    u0 = chebfun(lambda y: np.ones_like(np.asarray(y)), domain=(0, 1))
    u1 = L.expm(0.0126, u0)
    ys = np.linspace(0, 1, 500)
    _plot(ys, np.asarray(u1(ys)))
    print("ans =")
    print(f"   {float(u1(np.array([0.5]))[0]):.15f}")

    # 4. Unsteady convection-diffusion (the published attempt).
    L = Chebop(lambda x, c: 0.49 * c.diff(2) - 2.5 * c.diff(),
               domain=(0, 10))
    L.lbc = 1
    L.rbc = 0
    c0 = chebfun(lambda x: np.zeros_like(np.asarray(x)), domain=(0, 10))
    u1 = L.expm(1.33, c0)
    xs = np.linspace(0, 10, 600)
    _plot(xs, np.asarray(u1(xs)))

    # 5. Falkner-Skan equation on [0, 4].
    N = Chebop(lambda x, f: f.diff(3) + f * f.diff(2)
               + (np.pi / 4) * (1 - f.diff()**2), domain=(0, 4))
    N.lbc = lambda f: [f, f.diff()]
    N.rbc = lambda f: f.diff() - 1
    f = N.solve(0)
    xs = np.linspace(0, 4, 600)
    _plot(xs, np.asarray(f.diff()(xs)))

    # 7. Convection past an isothermal plate: coupled (F, T) system.
    N = Chebop(lambda y, F, T: [F.diff(3) + 3 * F * F.diff()
                                - 2 * F.diff()**2 + T,
                                T.diff(2) + 30 * F * T.diff()],
               domain=(0, 2.5))
    N.lbc = lambda F, T: [F, F.diff(), T - 1]
    N.rbc = lambda F, T: [F.diff(), T]
    # From the zero guess our Newton lands on a spurious flat-T branch
    # (T = 1 except a layer at y = 2.5); seed the physical decaying
    # profile instead.
    N.init = [chebfun(lambda y: 0.2 * (1 - np.exp(-2 * y)) ** 2,
                      domain=(0, 2.5)),
              chebfun(lambda y: np.exp(-1.5 * y) * (1 - (y / 2.5) ** 4),
                      domain=(0, 2.5))]
    U = N.solve(0)
    F, T = U[0], U[1]
    ys = np.linspace(0, 2.5, 600)
    _plot(ys, np.asarray(T(ys)), title="temperature")
    _plot(ys, np.asarray(F.diff()(ys)), title="velocity")
    print(f"T(1) = {float(T(np.array([1.0]))[0]):.12f}, "
          f"F'(1) = {float(F.diff()(np.array([1.0]))[0]):.12f}")


if __name__ == "__main__":
    run()
