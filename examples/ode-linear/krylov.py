"""Krylov subspace methods for ODEs.

Translation of ode-linear/Krylov.m by Alex Townsend and Marc
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
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"Krylov_{FIG[0]:02d}.png"), size=(600, 269))
    plt.close(fig)


def _long(name, v):
    """MATLAB format-long display of a scalar."""
    print(f"{name} =")
    print(f"     {v:.15e}")


def _matrix_pcg(A, b, tol, maxit):
    """MATLAB pcg for a matrix, with its convergence message."""
    its = [0]

    def count(_xk):
        its[0] += 1
    x, _info = spla.cg(A, b, rtol=tol, maxiter=maxit, callback=count)
    relres = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    print(f"pcg converged at iteration {its[0]} to a solution with "
          f"relative residual {relres:.2g}.")
    return x


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    n = 100
    h = 2 / (n + 1)
    e = np.ones(n)
    A = -1 / h**2 * sp.diags([e, -2 * e, e], [-1, 0, 1], shape=(n, n)).tocsr()
    b = np.ones(n)

    x_cg = _matrix_pcg(A, b, 1e-12, 100)
    x_exact = spla.spsolve(A, b)
    _long("error", np.linalg.norm(x_cg - x_exact))

    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    A = np.asarray(L.matrix(n))

    _long("ans", np.linalg.norm(A - A.T, "fro"))

    x = cj.chebfun(lambda t: t)
    f = 1 / (1 + x**2)
    L = Chebop(lambda u: -((2 + (70 * np.pi * x).cos()) * u.diff()).diff()
               + (1 + x**12) * u, domain=(-1, 1))
    L.lbc = 3
    L.rbc = -5
    t0 = time.time()
    u_colloc = L.solve(f)                                    # noqa: F841
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    t0 = time.time()
    u_cg = pcg(L, f)                                         # noqa: F841
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    f = cj.chebfun(1.0)
    u_cg = pcg(L, f)
    _long("error", float((u_cg - L.solve(f)).norm()))  # error in CG solution

    L = Chebop(lambda u: -u.diff(2) - 100 * u, domain=(-1, 1))
    L.bc = 0
    lam = np.sort(np.real(np.asarray(L.eigs(sigma=0))))  # MATLAB default
    print("ans =")
    for v in lam:
        print(f"{v:20.15f}")

    f = (13 * np.pi * x.abs()).sin()
    u_minres = minres(L, f)
    u_colloc = L.solve(f)
    _long("error", float((u_minres - u_colloc).norm()))

    # Plot:
    fig, ax = plt.subplots(figsize=(6.0, 2.69))
    matlab_plot(u_minres, ax=ax, lw=2)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("u(x)", fontsize=16)
    ax.tick_params(labelsize=16)
    _save(fig)

    L = Chebop(lambda u: -u.diff(2) + u.diff(1) + u, domain=(-1, 1))
    L.bc = 0
    f = cj.chebfun(1.0)
    u_gmres = gmres(L, f)
    u_colloc = L.solve(f)
    _long("error", float((u_colloc - u_gmres).norm()))

    L = Chebop(lambda u: -((2 + (21 * np.pi * x).cos()) * u.diff(1)).diff(1)
               + u / (1 + x**2), domain=(-1, 1))
    L.bc = 0
    u_exact = (40 * np.pi * x).sin()
    f = L(u_exact)
    u_minres, flag, relres, it, resvec = minres(L, f, full_output=True)
    _long("error", float((u_exact - u_minres).norm()))

    # Plot:
    fig, ax = plt.subplots(figsize=(6.0, 2.69))
    rv = np.asarray(resvec)
    ax.semilogy(np.arange(1, rv.size + 1), rv / rv[0], lw=2)
    ax.set_xlabel("Iteration count", fontsize=16)
    ax.set_ylabel("Relative residual", fontsize=16)
    ax.set_title("Convergence of the operator MINRES method", fontsize=16)
    ax.tick_params(labelsize=16)
    _save(fig)

    L = Chebop(lambda u: -1e-5 * u.diff(2) + u, domain=(-1, 1))
    L.bc = 0
    f = cj.chebfun(1.0)
    u_minres, flag, relres, it, _ = pcg(L, f, 1e-13, 1000, full_output=True)
    print("u_minres =")
    print(repr(u_minres))
    print("flag =")
    print(f"{flag:6d}")
    _long("relres", relres)
    print("iter =")
    print(f"{it:6d}")
    u_colloc = L.solve(f)
    _long("error", float((u_minres - u_colloc).norm()))

    a = cj.chebfun(lambda t: 2 + jnp.sign(jnp.cos(5 * t * np.pi)),
                   splitting=True)
    c = cj.chebfun(lambda t: -jnp.abs(t), splitting=True)
    L = Chebop(lambda u: -(a * u.diff()).diff() + c * u, domain=(-1, 1))
    L.bc = 2
    f = -1e2 * (3 * np.pi * x).sin()
    u_minres = minres(L, f)
    _long("relative_residual",
          float((L(u_minres) - f).norm()) / float(f.norm()))

    fig, ax = plt.subplots(figsize=(6.0, 2.69))
    matlab_plot(a, ax=ax, lw=2, color="#0072BD", jumpline=":")
    matlab_plot(u_minres, ax=ax, lw=2, color="#D95319")
    lines = ax.get_lines()
    ax.legend([lines[0], lines[-1]],
              ["Variable coefficient a(x)", "Solution u(x)"], fontsize=16)
    ax.tick_params(labelsize=16)
    _save(fig)


if __name__ == "__main__":
    run()
