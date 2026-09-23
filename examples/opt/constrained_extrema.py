"""Constrained extrema via composition.

Translation of opt/ConstrainedExtrema.m by Hrothgar
(October 2013): extrema of multivariate functions along constraint
curves and surfaces, computed by composing with parametrizations —
no Lagrange multipliers required.

Original: https://www.chebfun.org/examples/opt/ConstrainedExtrema.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import math
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import chebfun_style, contour, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.gallery2 import gallery2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    _savefig(fig, os.path.join(
        _IMG, f"ConstrainedExtrema_{FIG[0]:02d}.png"), size=(600, 268))
    plt.close(fig)


# --- MATLAB 'format long' display -------------------------------------

def _disp(name, M):
    """MATLAB format-long display of a real scalar/vector/matrix."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    print(f"{name} =")
    if np.all(M == np.round(M)):
        for row in M:
            print("".join(f"{int(v):6d}" for v in row))
        return
    if M.size == 1 and not 1e-3 <= abs(M[0, 0]) < 100:
        print(f"{M[0, 0]:26.15e}")
        return
    big = float(np.max(np.abs(M)))
    if big < 1e-3:
        e = math.floor(math.log10(big)) + 1
        print(f"   1.0e{e:+03d} *")
        M = M / 10.0**e
    for row in M:
        print("".join(f"{'0':>20}" if v == 0 else f"{v:20.15f}" for v in row))


def _col(v):
    return np.asarray(v, dtype=float).reshape(-1, 1)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Extrema of x^2 - y^2 on the unit circle
    f = cj.chebfun(lambda t: jnp.stack([jnp.cos(t), jnp.sin(t)], -1),
                   domain=(0.0, 2 * np.pi))
    g = cj.chebfun2(lambda x, y: x**2 - y**2)
    h = g(f)
    print("h =")
    print(repr(h))

    X, Y = h.minandmax(flag="local")
    _disp("Y", _col(Y))
    _disp("X", _col(X))

    X = f(jnp.asarray(X))
    _disp("X", X)

    # The SIAM 100-digit challenge function on the circle
    g = gallery2("challenge")
    h = g(f)
    (xmin, ymin), (xmax, ymax) = h.minandmax()
    Y = _col([ymin, ymax])
    Xh = _col([xmin, xmax])
    _disp("Y", Y)
    _disp("Xh", Xh)

    X = np.asarray(f(jnp.asarray(Xh[:, 0])))
    _disp("X", X)

    fig, ax = contour(g, levels=4, filled=True, line_color="k")
    th = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=2)
    ax.plot(X[:, 0], X[:, 1], "ko", mfc="k", ms=4)
    for (px, py), lab in zip(X, ("min", "max")):
        ax.text(px, py, "  " + lab, color="w", fontweight="bold",
                fontsize=12, va="center")
    ax.set_aspect("equal")
    _save(fig)

    fig, ax = plt.subplots()
    tt = np.linspace(0, 2 * np.pi, 8000)
    ax.plot(tt, np.asarray(h(jnp.asarray(tt))), lw=0.8)
    ax.plot(Xh[:, 0], Y[:, 0], "ko", mfc="k", ms=4)
    _save(fig)

    # Extrema of x + y + z on the surface z = x^3 + y^2
    g = chebfun3(lambda x, y, z: x + y + z, domain=(-2, 2, -2, 2, -2, 2))

    f = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                 lambda x, y: x**3 + y**2)
    # @chebfun2v/surf.m is surf(f(:,1), f(:,2), f(:,3))
    fig, ax = surf(*[Chebfun2(approx=c) for c in f.components])
    _save(fig)

    # h = g(f): chebfun3 composed with a 3-component chebfun2v
    h = Chebfun2.from_function(lambda x, y: g(*[f(x, y)[..., k]
                                                for k in range(3)]),
                               domain=f.domain)
    (Ymin, Ymax), X = h.minandmax2()
    X = np.asarray(X, dtype=float)
    _disp("Y", [[Ymin, Ymax]])
    _disp("X", X)

    Xmin = np.asarray(f(X[0, 0], X[0, 1]))
    Xmax = np.asarray(f(X[1, 0], X[1, 1]))
    _disp("Xmin", _col(Xmin))
    _disp("Xmax", _col(Xmax))

    _disp("ans", g(Xmin[0], Xmin[1], Xmin[2]))
    _disp("ans", g(Xmax[0], Xmax[1], Xmax[2]))

    # Extrema of x^3 + cos(5x) - y^2 on a rotated square
    f = Chebfun2v.from_functions(lambda x, y: x - y, lambda x, y: x + y,
                                 domain=(-1 / 2, 1 / 2, -1 / 2, 1 / 2))

    t = cj.chebfun(lambda t: t)
    bdry = (t - 1j).join(1 + 1j * t, 1j - t, -1 - 1j * t) / 2
    # fbdry = f(bdry): the chebfun2v on the complex boundary curve
    fbdry = cj.chebfun(
        lambda s: f(jnp.real(bdry(s)), jnp.imag(bdry(s))),
        domain=tuple(float(v) for v in bdry.domain.breakpoints))
    ss = np.linspace(float(bdry.domain.a), float(bdry.domain.b), 2000)
    vals = np.asarray(fbdry(jnp.asarray(ss)))
    fig, ax = plt.subplots()
    ax.plot(vals[:, 0], vals[:, 1], lw=0.8)
    ax.axis("equal")
    _save(fig)

    g = cj.chebfun2(lambda x, y: x**3 + jnp.cos(5 * x) - y**2)
    h = g(f)
    fig, ax = surf(h)
    (Ymin, Ymax), X = h.minandmax2()
    X = np.asarray(X, dtype=float)
    _disp("Y", [[Ymin, Ymax]])
    _disp("X", X)
    _save(fig)

    Xmin = np.asarray(f(X[0, 0], X[0, 1]))
    Xmax = np.asarray(f(X[1, 0], X[1, 1]))
    _disp("Xmin", _col(Xmin))
    _disp("Xmax", _col(Xmax))


if __name__ == "__main__":
    run()
