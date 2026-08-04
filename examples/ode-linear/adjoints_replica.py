"""Adjoints of linear operators.

Faithful replica of ode-linear/Adjoints.m by Yuji Nakatsukasa
(December 2016): formal adjoints and adjoint boundary conditions of
first- and second-order operators, verification of the bilinear
identity <v, Lu> = <L*v, u>, and the biorthogonality of eigenfunctions
of a nonnormal operator and its adjoint.

Original: https://www.chebfun.org/examples/ode-linear/Adjoints.html
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

import chebfunjax as cj
from chebfunjax.operators.adjoint import adjoint
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = cj.chebfun(lambda t: t, domain=(-1, 1))

    # First derivative with u(-1) = 0
    L = Chebop(lambda x, u: u.diff(), domain=(-1, 1))
    L.lbc = 0
    print("L =")
    print(repr(L))
    Ls = adjoint(L)
    print("Ls =")
    print(repr(Ls))

    u = (x + 1) * x.sin()
    v = (x - 1) * x.exp()
    lhs = float((v * u.diff()).sum())
    rhs = float((Ls.op(x, v) * u).sum())
    print("ans =")
    print(f"   {abs(lhs - rhs):.4g}")

    # Self-adjoint: u'' + u with Dirichlet conditions
    L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
    L.lbc = 0
    L.rbc = 0
    Ls = adjoint(L)
    print("Ls =")
    print(repr(Ls))

    # IVP: both conditions at the left end
    L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
    L.lbc = [1, 0]
    print("L =")
    print(repr(L))
    Ls = adjoint(L)
    print("Ls =")
    print(repr(Ls))

    # Just one boundary condition
    L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
    L.lbc = 1
    Ls = adjoint(L)
    print("Ls =")
    print(repr(Ls))

    # Variable coefficient: x u''
    L = Chebop(lambda x, u: x * u.diff(2), domain=(-1, 1))
    L.lbc = 0
    L.rbc = 0
    print("L =")
    print(repr(L))
    Ls = adjoint(L)
    print("Ls =")
    print(repr(Ls))

    u = (x**2 - 1) * x.sin()
    v = (x**2 - 1) * x.exp()
    lhs = float((v * (x * u.diff(2))).sum())
    rhs = float((Ls.op(x, v) * u).sum())
    print("ans =")
    print(f"   {abs(lhs - rhs):.4e}")

    # Nonnormal operator: eigenvalues and biorthogonality
    L = Chebop(lambda x, u: u.diff(2) - 20 * u.diff() + u,
               domain=(-1, 1))
    L.lbc = 0
    L.rbc = 0
    Ls = adjoint(L)
    lam, V = L.eigs(k=6, sigma="SM", return_eigenfunctions=True)
    lams, Vs = Ls.eigs(k=6, sigma="SM", return_eigenfunctions=True)
    lam = np.real(np.asarray(lam))
    lams = np.real(np.asarray(lams))
    order = np.argsort(lam)
    order_s = np.argsort(lams)
    lam, V = lam[order], [V[i] for i in order]
    lams, Vs = lams[order_s], [Vs[i] for i in order_s]
    print("ans =")
    for d, ds in zip(lam, lams):
        print(f" {d:9.4f} {ds:9.4f}")

    G = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            G[i, j] = float((Vs[i] * V[j]).sum())
    print("ans =")
    with np.printoptions(precision=4, suppress=False):
        print(G)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(-1, 1, 1000)
    for ii in range(2):
        vv = V[ii]
        if float(vv(jnp.array(0.9))) < 0:
            vv = -vv
        ax.plot(t, np.asarray(vv(t)), 'r', lw=1.4)
        vs = Vs[ii]
        if float(vs(jnp.array(-0.9))) < 0:
            vs = -vs
        ax.plot(t, np.asarray(vs(t)), 'b', lw=1.4)
    ax.text(-0.8, 2, "adjoint eigenfunctions", color='b')
    ax.text(0.3, 2, "eigenfunctions", color='r')
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Adjoints_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    n12 = float((V[1] * V[0]).sum()) / (
        float(V[1].norm()) * float(V[0].norm()))
    print("ans =")
    print(f"    {abs(n12):.4f}")


if __name__ == "__main__":
    run()
