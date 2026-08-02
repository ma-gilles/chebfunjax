"""Convergence of the SOR iteration.

Faithful replica of linalg/SOR.m by Nick Trefethen (June 2011): the
spectral radius of the SOR iteration matrix as a chebfun of the
relaxation parameter omega, minimized to recover Young's exact
optimal omega = 2/(1 + sin(pi/N)).

Original: https://www.chebfun.org/examples/linalg/SOR.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

N = 11


def run():
    os.makedirs(_IMG, exist_ok=True)

    col = np.concatenate([[2.0, -1.0], np.zeros(N - 3)])
    A = sla.toeplitz(col)
    print("A =")
    with np.printoptions(formatter={'float': lambda v: f"{int(v):6d}"}):
        for row in A[:7]:
            print("  " + "".join(f"{int(v):6d}" for v in row))
        print("   ...")
    L = np.tril(A, -1)
    D = np.diag(np.diag(A))
    U = np.triu(A, 1)

    def rho(om_arr):
        om_arr = np.atleast_1d(np.asarray(om_arr, dtype=float))
        out = np.empty_like(om_arr)
        for i, om in enumerate(om_arr.ravel()):
            G = np.linalg.solve(D + om * L, (1 - om) * D - om * U)
            out.ravel()[i] = np.max(np.abs(np.linalg.eigvals(G)))
        return out.reshape(np.shape(om_arr))

    import jax.numpy as jnp

    def rho_op(x):
        return jnp.asarray(rho(np.asarray(x)))

    f = cj.chebfun(rho_op, domain=(1.0, 2.0), splitting=True)
    xs = np.linspace(1, 2, 800)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(f(xs)), 'b', lw=1.6)
    ax.grid(True)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("convergence factor")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SOR_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    pos, val = f.min()
    print("rho_opt =")
    print(f"   {float(val):.15f}")
    print("omega_opt =")
    print(f"   {float(pos):.15f}")

    omega_exact = 2 / (1 + np.sin(np.pi / N))
    print("omega_exact =")
    print(f"   {omega_exact:.15f}")
    print("rho_exact =")
    print(f"   {omega_exact - 1:.15f}")


if __name__ == "__main__":
    run()
