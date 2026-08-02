"""Resolvent norm on the imaginary axis.

Faithful replica of linalg/ResolventNorm.m by Nick Trefethen
(May 2011): the norm of the resolvent (zI-A)^{-1} along the
imaginary axis as a chebfun; its maximum is the reciprocal of the
distance to singularity, a key quantity of robust stability theory.

Original: https://www.chebfun.org/examples/linalg/ResolventNorm.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def _normfun_vals(A):
    n = A.shape[0]
    ident = np.eye(n)

    def nr(y_arr):
        y_arr = np.atleast_1d(np.asarray(y_arr, dtype=float))
        out = np.empty_like(y_arr)
        for i, y in enumerate(y_arr.ravel()):
            out.ravel()[i] = 1.0 / np.min(np.linalg.svd(
                1j * y * ident - A, compute_uv=False))
        return out.reshape(np.shape(y_arr))
    return nr


def _normfun(A, halfwidth=None):
    if halfwidth is None:
        halfwidth = 1.5 * np.linalg.norm(A, 2)
    nr = _normfun_vals(A)
    return cj.chebfun(lambda y: jnp.asarray(nr(np.asarray(y))),
                      domain=(-halfwidth, halfwidth))


def _fmt_eigs(w):
    for v in sorted(w, key=lambda z: (round(-abs(z.imag), 4),
                                      z.imag < 0)):
        sign = "+" if v.imag >= 0 else "-"
        print(f"  {v.real:8.4f} {sign} {abs(v.imag):.4f}i")


def run():
    os.makedirs(_IMG, exist_ok=True)

    A = np.array([[-1, 3, 5, 2], [-3, -2, 4, 6],
                  [-5, -4, -2, 1], [-2, -6, -1, 3]], dtype=float)
    print("A =")
    for row in A:
        print("  " + "".join(f"{int(v):6d}" for v in row))
    w = np.linalg.eigvals(A)
    print("ans =")
    _fmt_eigs(w)

    f = _normfun(A, halfwidth=25.0)
    xs = np.linspace(-25, 25, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(f(xs)), 'b', lw=1.6)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ResolventNorm_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    _, maxf = f.max()
    print("maxf =")
    print(f"   {float(maxf):.15f}")
    print("dist_sing =")
    print(f"   {1.0 / float(maxf):.15f}")

    B = np.array([
        [-3 - 2j, 1 + 1j, -1j, 0, -1 + 1j],
        [0, -2 - 3j, -1j, 1j, -2 - 1j],
        [1j, 0, -2 - 4j, -2 - 1j, 2 - 1j],
        [0, 1, 1j, -2 - 4j, 1j],
        [1 - 2j, 0, 1, 1, -2 - 3j]])
    print("ans =")
    for v in np.linalg.eigvals(B):
        sign = "+" if v.imag >= 0 else "-"
        print(f"  {v.real:8.4f} {sign} {abs(v.imag):.4f}i")
    fB = _normfun(B)
    hw = 1.5 * np.linalg.norm(B, 2)
    xsB = np.linspace(-hw, hw, 1200)
    _, mB = fB.max()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xsB, np.asarray(fB(xsB)), 'b', lw=1.6)
    ax.grid(True)
    ax.set_title(f"maximum = {float(mB):.4f}", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ResolventNorm_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    rs = np.random.RandomState(1)
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 10.5))
    for j in range(12):
        N = 6
        A = (rs.randn(N, N) + 1j * rs.randn(N, N)
             + 2j * np.diag(rs.randn(N)))
        abscissa = np.max(np.real(np.linalg.eigvals(A)))
        A = A - (abscissa + 0.25) * np.eye(N)
        g = _normfun(A)
        hw = 1.5 * np.linalg.norm(A, 2)
        xg = np.linspace(max(-hw, -10), min(hw, 10), 500)
        ax = axes.ravel()[j]
        ax.plot(xg, np.asarray(g(xg)), 'b', lw=1)
        ax.axis([-10, 10, 0, 8])
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ResolventNorm_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
