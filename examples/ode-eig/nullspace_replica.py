"""The nullspace of a linear operator.

Faithful replica of ode-eig/NullSpace.m by Nick Hale and Stefan
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
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun, subspace
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')
FIG = [0]


def _plot(fs, fname_extra=None, dashed=None):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    a = float(fs[0].domain.a)
    b = float(fs[0].domain.b)
    xx = np.linspace(a, b, 2000)
    for f in fs:
        ax.plot(xx, np.asarray(f(xx)).real, lw=1.6)
    if dashed is not None:
        ax.plot(xx, np.asarray(dashed(xx)).real, 'r--', lw=1.6)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"NullSpace_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gram(V):
    return np.array([[float((a * b).sum()) for b in V] for a in V])


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Simple example: L = d^2/dx^2 on [-1, 1].
    L = Chebop(lambda u: u.diff(2))
    one = chebfun(lambda t: 0 * t + 1.0)
    x = chebfun(lambda t: t)
    v = [one, x]
    print("ans =")
    print(f"     {max(float(L(f).norm(2)) for f in v):.4e}")

    V = L.null()
    for f in V:
        print(f)
    _plot(V)
    print("ans =")
    print(_gram(V))
    print("ans =")
    print(f"     {max(float(L(f).norm(2)) for f in V):.4e}")

    print("ans (subspace angle) =")
    print(f"     {float(subspace(v, V)):.4e}")

    # 2. Incomplete boundary conditions on [-pi, pi].
    dom = (-np.pi, np.pi)
    L = Chebop(lambda x_, u: (u.diff(2) + 0.1 * x_ * (1 - x_**2) * u.diff()
                              + x_.sin() * u), domain=dom)
    V = L.null()
    for f in V:
        print(f)
    _plot(V)
    print("ans =")
    print(_gram(V))
    print("ans =")
    print(f"     {max(float(L(f).norm(2)) for f in V):.4e}")

    L.lbc = 0.0
    Vn = L.null()
    vfun = Vn[0]
    print(vfun)
    _plot(Vn)
    print("ans =", _gram(Vn))
    print("ans =")
    print(f"     {float(L(vfun).norm(2)):.4e}")
    print("v(-pi) =")
    print(f"     {float(vfun(-np.pi)):.4e}")

    # 3. Application: minimal-norm inhomogeneous Dirichlet condition.
    L.rbc = 0.0
    u = L.solve(1.0)
    _plot(Vn, dashed=u)

    def Efun(c):
        c = np.atleast_1d(np.asarray(c, dtype=float))
        return np.array([float((u + float(ci) * vfun).norm(2))
                         for ci in c])

    E = chebfun(Efun, domain=(-10.0, 10.0), splitting=True)
    _plot([E])

    xmin, minE = None, None
    xs, ys = E.min("local")
    xs, ys = np.atleast_1d(np.asarray(xs)), np.atleast_1d(np.asarray(ys))
    j = int(np.argmin(ys))
    c_star, minE = float(xs[j]), float(ys[j])
    print("minE =")
    print(f"   {minE:.15f}")
    print("c_star =")
    print(f"   {c_star:.15f}")
    u_star = u + c_star * vfun
    print(u_star)
    _plot([u_star])

    print("bc_star =")
    print(f"   {float(u_star(np.pi)):.15f}")

    # 4. Exotic constraints: int(u) = u(0) on a 3rd-order operator.
    L = Chebop(lambda x_, u: 0.1 * u.diff(3) + x_.sin() * u.diff(2) + u,
               domain=(-1, 1))
    L.bc = lambda x_, u: u.sum() - u(0.0)
    V = L.null()
    for f in V:
        print(f)
    _plot(V)
    print("ans =")
    print(_gram(V))
    print("ans (sum(V) - V(0,:)) =")
    print([f"{float(f.sum()) - float(f(0.0)):.3e}" for f in V])
    print("ans (norm(L(V),1)) =")
    print(f"     {max(float(L(f).norm(1)) for f in V):.4e}")


if __name__ == "__main__":
    run()
