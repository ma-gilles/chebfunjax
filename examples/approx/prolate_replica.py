"""Prolate spheroidal wave functions.

Faithful replica of approx/Prolate.m by Nick Trefethen (May 2021):
eigenvalues and eigenfunctions of the band-limited kernel
exp(i c x t), whose eigenfunctions are the prolate spheroidal wave
functions.

Original: https://www.chebfun.org/examples/approx/Prolate.html
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
from chebfunjax.utils.pswf import pswf

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # The kernel K(x,t) = exp(i c x t) with c = 10 pi, and the unitary
    # DFT hidden inside it
    N = 10
    c = N * np.pi
    K10 = cj.chebfun2(lambda x, t: jnp.exp(1j * c * x * t))
    xx = np.arange(-N, N) / N
    A = np.asarray(K10(jnp.asarray(xx[:, None]), jnp.asarray(xx[None, :])))
    print("condA =")
    print(f"   {np.linalg.cond(A):.15f}")
    print("rankA =")
    print(f"    {np.linalg.matrix_rank(A)}")

    xs = np.linspace(-1, 1, 300)
    X, T = np.meshgrid(xs, xs)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.pcolormesh(X, T, np.real(np.asarray(
        K10(jnp.asarray(X), jnp.asarray(T)))), shading="auto")
    ax.plot(np.repeat(xx, len(xx)), np.tile(xx, len(xx)), '.w', ms=3)
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{Re}(K(x,t))$", fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Prolate_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Eigenvalues: |lambda| plateau at 1/sqrt(5), then super-exponential
    # decay past the c/pi transition
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    lam10 = np.sort(np.abs(np.asarray(K10.eig())))[::-1]
    ax.semilogy(np.arange(1, len(lam10) + 1), lam10, '.', ms=11)
    K20 = cj.chebfun2(lambda x, t: jnp.exp(2j * c * x * t))
    lam20 = np.sort(np.abs(np.asarray(K20.eig())))[::-1]
    ax.semilogy(np.arange(1, len(lam20) + 1), lam20, '.', ms=11)
    ax.grid(True)
    ax.set_ylim(1e-15, 100)
    ax.text(20, 4, r"$c = 10\pi$", ha="center")
    ax.text(40, 4, r"$c = 20\pi$", ha="center")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Prolate_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("lamabs =")
    for v in lam10[:14]:
        print(f"   {v:.15f}")

    # Eigenfunctions of the c = 4 pi kernel
    c4 = 4 * np.pi
    K4 = cj.chebfun2(lambda x, t: jnp.exp(1j * c4 * x * t))
    lam, V, xg = K4.eig(return_functions=True)
    lam = np.asarray(lam)
    V = np.real(np.asarray(V))
    xg = np.asarray(xg)
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 8.8))
    order = np.argsort(xg)
    for j in range(8):
        ax = axes[j // 2, j % 2]
        vj = V[:, j] / np.max(np.abs(V[:, j]))
        ax.plot(xg[order], vj[order], lw=1)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        L = lam[j]
        lamstr = (f"{L.real:.4g}" if abs(L.real) > abs(L.imag)
                  else f"{L.imag:.4g}i")
        ax.set_title(rf"$\lambda_{j+1} = {lamstr}$", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Prolate_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The pswf command
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0))
    for ax, (n_, c_) in zip(axes, ((10, 200), (50, 200))):
        x_grid, P, _ = pswf(n_, c_)
        ax.plot(np.asarray(x_grid), np.asarray(P).ravel(), lw=1)
        ax.set_title(f"pswf({n_}, {c_})", fontsize=11)
        ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Prolate_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
