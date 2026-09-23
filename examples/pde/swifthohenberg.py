"""Swift-Hohenberg equation in 2D.

Translation of pde/SwiftHohenberg.m by Hadrien Montanelli (May
2017): the Swift-Hohenberg equation

    u_t = r u - (1 + Lap)^2 u + g u^2 - u^3

solved with spin2/ETDRK4: the preloaded 'sh' demo (convection rolls),
then a sine + five-Gaussian initial condition on [0, 20pi]^2 giving
spots (r = 0.01, g = 1), spirals (r = 0.7, g = 1) and stripes
(r = 0.1, g = 0), with a resolution-refinement error check.

Original: https://www.chebfun.org/examples/pde/SwiftHohenberg.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import chebfun2
from chebfunjax.operators.spinop2 import Spinop2, func2str, spin2
from chebfunjax.plotting import PARULA, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')
FIG = [0]


def _mat(v):
    return "[" + " ".join(f"{float(t):g}" for t in v) + "]"


def _disp_spinop2(S):
    """MATLAB's property display of a spinop2."""
    print("  spinop2 with properties:\n")
    print(f"     domain: {_mat(S.domain)}")
    print("       init: [InfxInf chebfun2]")
    print(f"        lin: {func2str(S.lin)}")
    print(f"     nonlin: {func2str(S.nonlin)}")
    print(f"      tspan: {_mat(S.tspan)}")
    print(f"    numVars: {S.numVars}")


def _trig2(u, dom):
    """The chebfun2(..., 'trig') spin2 returns (reshapeData.m)."""
    return chebfun2(lambda x, y: jnp.real(jnp.asarray(u(np.asarray(x),
                                                        np.asarray(y)))),
                    domain=dom, trig=True)


def _plot(u, dom):
    """plot(u), view(0,90), axis equal, axis off."""
    FIG[0] += 1
    xs = np.linspace(dom[0], dom[1], 300)
    ys = np.linspace(dom[2], dom[3], 300)
    X, Y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.pcolormesh(X, Y, np.real(np.asarray(u(X, Y))), cmap=PARULA,
                  shading="gouraud")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    _savefig(fig, os.path.join(_IMG, f"SwiftHohenberg_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    S = Spinop2("sh")
    print("S = ")
    _disp_spinop2(S)
    print()

    S.tspan = (0.0, 1200.0)
    N = 128
    dt = 1
    u = spin2(S, N, dt, "plot", "off", dealias=False)
    u = _trig2(u, S.domain)
    _plot(u, S.domain)

    print("u =")
    print(u.disp())

    dom = (0.0, 20 * np.pi, 0.0, 20 * np.pi)
    tspan = (0.0, 200.0)
    S = Spinop2(dom, tspan)
    S.lin = "@(u) -2*lap(u) - biharm(u)"
    r, g = 1e-2, 1

    def nonlin(r, g):
        return lambda u: (-1 + r) * u + g * u**2 - u**3
    S.nonlin = nonlin(r, g)

    pi = np.pi
    u0 = 1 / 20 * chebfun2(lambda x, y: jnp.cos(x) + jnp.sin(2 * x)
                           + jnp.sin(y) + jnp.cos(2 * y), domain=dom, trig=True)
    for cx, cy in [(5, 5), (5, 15), (15, 15), (15, 5), (10, 10)]:
        u0 = u0 + chebfun2(lambda x, y, cx=cx, cy=cy: jnp.exp(
            -((x - cx * pi)**2 + (y - cy * pi)**2)), domain=dom, trig=True)
    S.init = u0

    _plot(S.init, dom)

    u = spin2(S, 96, 2e-1, "plot", "off", dealias=False)
    _plot(u, dom)

    v = spin2(S, 128, 1e-1, "plot", "off", dealias=False)
    u2, v2 = _trig2(u, dom), _trig2(v, dom)
    error = float((u2 - v2).norm()) / float(v2.norm())
    print(f"Relative error: {error:1.2e}")

    r, g = 7e-1, 1
    S.nonlin = nonlin(r, g)
    u = spin2(S, 96, 2e-1, "plot", "off", dealias=False)
    _plot(u, dom)

    S.tspan = (0.0, 200.0)
    r, g = 1e-1, 0
    S.nonlin = nonlin(r, g)
    u = spin2(S, 100, 2e-1, "plot", "off", dealias=False)
    _plot(u, dom)


if __name__ == "__main__":
    run()
