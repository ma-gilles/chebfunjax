"""Krylov subspace methods for ODEs.

Faithful replica of ode-linear/Krylov.m by Alex Townsend and Marc
Aurele Gilles (November 2016): conjugate gradients, MINRES, and GMRES
applied directly to differential operators via the indefinite-integral
preconditioner of Gilles & Townsend.

Original: https://www.chebfun.org/examples/ode-linear/Krylov.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import gmres, minres, pcg
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Krylov_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = cj.chebfun(lambda t: t, domain=(-1, 1))

    # 1. Matrix CG warm-up
    n = 100
    h = 2.0 / (n + 1)
    e = np.ones(n)
    A = -(1 / h**2) * sp.diags([e, -2 * e, e], [-1, 0, 1],
                               shape=(n, n)).tocsr()
    b = np.ones(n)
    x_cg, _info = spla.cg(A, b, rtol=1e-12, maxiter=100)
    x_exact = spla.spsolve(A, b)
    print("error =")
    print(f"     {np.linalg.norm(x_cg - x_exact):.15e}")

    # 2. The collocation matrix is far from symmetric
    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    An = np.asarray(L.matrix(n))
    print("ans =")
    print(f"     {np.linalg.norm(An - An.T, 'fro'):.15e}")

    # 3. Variable coefficients: colloc vs operator pcg
    f = 1 / (1 + x**2)
    Lv = Chebop(lambda u: -((2 + (70 * np.pi * x).cos())
                            * u.diff()).diff() + (1 + x**12) * u,
                domain=(-1, 1))
    Lv.lbc = 3
    Lv.rbc = -5
    t0 = time.time()
    u_colloc = Lv.solve(f)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    t0 = time.time()
    u_cg = pcg(Lv, f)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    print("cg-vs-colloc =")
    print(f"     {float((u_cg - u_colloc).norm()):.3e}")

    # 4. pcg accuracy on the Poisson problem
    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    f1 = cj.chebfun(lambda t: jnp.ones_like(t), domain=(-1, 1))
    u_cg = pcg(L, f1)
    Lb = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    Lb.bc = 0
    print("error =")
    print(f"     {float((u_cg - Lb.solve(f1)).norm()):.15e}")

    # 5. An indefinite operator: eigenvalues straddle zero
    Le = Chebop(lambda u: -u.diff(2) - 100 * u, domain=(-1, 1))
    Le.bc = 0
    lam = np.sort(np.real(np.asarray(Le.eigs(k=6, sigma="SM"))))
    print("ans =")
    for v in lam:
        print(f" {v:11.6f}")

    # 6. minres on the indefinite operator
    fs = (13 * np.pi * abs(x)).sin()
    u_minres = minres(Le, fs, tol=1e-10, maxit=200)
    Le2 = Chebop(lambda u: -u.diff(2) - 100 * u, domain=(-1, 1))
    Le2.bc = 0
    u_col = Le2.solve(fs)
    print("error =")
    print(f"     {float((u_minres - u_col).norm()):.15e}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(-1, 1, 1200)
    ax.plot(t, np.asarray(u_minres(t)), lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.grid(True)
    _save(fig)

    # 7. gmres
    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    u_gmres = gmres(L, f1)
    print("error =")
    print(f"     {float((u_gmres - Lb.solve(f1)).norm()):.15e}")

    # 8. minres with a rough manufactured solution
    Lr = Chebop(lambda u: -((2 + (21 * np.pi * x).cos())
                            * u.diff()).diff() + u / (1 + x**2),
                domain=(-1, 1))
    Lr.bc = 0
    u_exact = (40 * np.pi * x).sin()
    fr = (-((2 + (21 * np.pi * x).cos()) * u_exact.diff()).diff()
          + u_exact / (1 + x**2))
    u_minres, flag, relres, it, resvec = minres(
        Lr, fr, tol=1e-10, maxit=300, full_output=True)
    print("error =")
    print(f"     {float((u_exact - u_minres).norm()):.15e}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    rv = np.asarray(resvec)
    ax.semilogy(np.arange(rv.size), rv / rv[0], lw=2)
    ax.set_xlabel("Iteration count")
    ax.set_ylabel("Relative residual")
    ax.set_title("Convergence of the operator MINRES method")
    ax.grid(True)
    _save(fig)

    # 9. A stiff problem with tight tolerance
    Ls = Chebop(lambda u: -1e-5 * u.diff(2) + u, domain=(-1, 1))
    Ls.bc = 0
    u_p, flag, relres, it, _rv = pcg(Ls, f1, tol=1e-13, maxit=1000,
                                     full_output=True)
    print("u_minres =")
    print(repr(u_p))
    print(f"flag = {flag}   iter = {it}")
    Ls2 = Chebop(lambda u: -1e-5 * u.diff(2) + u, domain=(-1, 1))
    Ls2.bc = 0
    print("error =")
    print(f"     {float((u_p - Ls2.solve(f1)).norm()):.3e}")

    # 10. Piecewise-smooth coefficients
    a = 2 + (5 * np.pi * x).cos().sign()
    c = -abs(x)
    Lp = Chebop(lambda u: -(a * u.diff()).diff() + c * u,
                domain=(-1, 1))
    Lp.bc = 2
    fp = -1e2 * (3 * np.pi * x).sin()
    u_p, _fl, relres, _it, _rv = minres(Lp, fp, tol=1e-10,
                                        maxit=300, full_output=True)
    print("relative_residual =")
    print(f"     {relres:.15e}")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u_p(t)), lw=2)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
