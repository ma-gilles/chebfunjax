"""The nullspace of a linear operator.

Translation of ode-eig/NullSpace.m by Nick Hale and Stefan
Guettel (December 2011): `null` computes orthonormal nullspace bases
of differential operators -- with no boundary conditions, with
incomplete boundary conditions, and with an exotic integral side
condition -- and an application: choosing the inhomogeneous Dirichlet
value minimizing the 2-norm of the solution of Lu = 1.

Original: https://www.chebfun.org/examples/ode-eig/NullSpace.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import math
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun, subspace
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')
FIG = [0]
LW = 1.6


# --- MATLAB 'format long' / chebfun display ---------------------------

def _num(x):
    """One MATLAB format-long scalar field."""
    if x == int(x) and abs(x) < 1e9:
        return f"{int(x):6d}"
    if 1e-3 <= abs(x) < 100:
        return f"{x:20.15f}"
    return f"{x:26.15e}"


def _disp_scalar(name, x):
    print(f"{name} =")
    print(_num(float(x)))


def _disp_matrix(name, M):
    """MATLAB format-long display of a real matrix (common scale factor
    when every entry is tiny)."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    print(f"{name} =")
    big = float(np.max(np.abs(M)))
    if np.all(M == np.round(M)):
        for row in M:
            print("".join(f"{int(v):6d}" for v in row))
        return
    if 0 < big < 1e-3:
        e = math.floor(math.log10(big)) + 1
        print(f"   1.0e{e:+03d} *")
        M = M / 10.0**e
    for row in M:
        print("".join(f"{'0':>20}" if v == 0 else f"{v:20.15f}" for v in row))


def _disp_quasi(name, cols):
    """MATLAB display of a chebfun quasimatrix (@chebfun/disp.m)."""
    print(f"{name} =")
    if len(cols) == 1:
        print(repr(cols[0]))
        return
    for k, f in enumerate(cols, 1):
        print(repr(f).replace("chebfun column (", f"chebfun column{k} (", 1))


def _gram(V):
    """V'*V for a quasimatrix."""
    return [[float(f.inner(g)) for g in V] for f in V]


def _fro(V):
    """norm(V) of a quasimatrix (MATLAB's default 'fro' norm)."""
    return math.sqrt(sum(float(f.norm()) ** 2 for f in V))


# --- plotting --------------------------------------------------------

def _plot(ax, f, style="-", **kw):
    a, b = float(f.domain.a), float(f.domain.b)
    xx = np.linspace(a, b, 2000)
    ax.plot(xx, np.asarray(f(xx)).real, style, lw=LW, **kw)


def _save(fig):
    FIG[0] += 1
    _savefig(fig, os.path.join(_IMG, f"NullSpace_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Simple example #1
    L = Chebop(lambda u: u.diff(2))
    v = [chebfun(lambda t: 0 * t + 1.0), chebfun(lambda t: t)]
    _disp_scalar("ans", _fro([L(f) for f in v]))

    V = L.null()
    _disp_quasi("V", V)
    fig, ax = plt.subplots()
    for f in V:
        _plot(ax, f)
    _save(fig)
    _disp_matrix("ans", _gram(V))
    _disp_scalar("ans", _fro([L(f) for f in V]))

    _disp_scalar("ans", subspace(v, V))

    # 2. Incomplete boundary conditions
    dom = (-np.pi, np.pi)
    L = Chebop(lambda x, u: (u.diff(2) + .1 * x * (1 - x**2) * u.diff()
                             + x.sin() * u), domain=dom)
    V = L.null()
    _disp_quasi("V", V)
    fig, ax = plt.subplots()
    for f in V:
        _plot(ax, f)
    _save(fig)
    _disp_matrix("ans", _gram(V))
    _disp_scalar("ans", _fro([L(f) for f in V]))

    L.lbc = 0.0
    L.rbc = None
    v = L.null()
    _disp_quasi("v", v)
    fig, ax = plt.subplots()
    _plot(ax, v[0])
    FIG[0] += 1
    _savefig(fig, os.path.join(_IMG, f"NullSpace_{FIG[0]:02d}.png"))
    _disp_matrix("ans", _gram(v))
    _disp_scalar("ans", _fro([L(f) for f in v]))
    v = v[0]

    _disp_scalar("ans", v(-np.pi))

    # 3. An application
    L.rbc = 0.0
    u = L.solve(1.0)
    _plot(ax, u, "--r")        # hold on, onto the previous figure
    _save(fig)

    E = chebfun(lambda c: (u + c * v).norm(2), domain=(-10.0, 10.0),
                vectorize=True, splitting=True)
    fig, ax = plt.subplots()
    _plot(ax, E)
    _save(fig)

    c_star, minE = E.min()
    _disp_scalar("minE", minE)
    _disp_scalar("c_star", c_star)
    u_star = u + float(c_star) * v
    _disp_quasi("u_star", [u_star])
    fig, ax = plt.subplots()
    _plot(ax, u_star)
    _save(fig)

    bc_star = u_star(np.pi)
    _disp_scalar("bc_star", bc_star)

    # 4. Exotic constraints
    dom = (-1.0, 1.0)
    L = Chebop(lambda x, u: .1 * u.diff(3) + x.sin() * u.diff(2) + u,
               domain=dom)
    L.bc = lambda x, u: u.sum() - u(0.0)
    V = L.null()
    _disp_quasi("V", V)
    fig, ax = plt.subplots()
    for f in V:
        _plot(ax, f)
    _save(fig)
    _disp_matrix("ans", _gram(V))

    _disp_matrix("ans", [[float(f.sum()) - float(f(0.0)) for f in V]])
    _disp_scalar("ans", max(float(L(f).norm(1)) for f in V))


if __name__ == "__main__":
    run()
