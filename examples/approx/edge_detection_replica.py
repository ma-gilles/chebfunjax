"""Edge detection.

Faithful replica of approx/EdgeDetection.m by Nick Trefethen (July
2019): Chebfun's edge detector recovers the 21 kinks of
|exp(x) sin(10 pi x)| to machine precision, and locates the kinks of
an eigenvalue-abscissa function of a random matrix pencil.

Original: https://www.chebfun.org/examples/approx/EdgeDetection.html
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
from chebfunjax.utils.fov import fov

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Field of values of a random-ish matrix (MATLAB rng(1) randn draws
    # are ziggurat-based and not bit-reproducible outside MATLAB; the
    # figure is qualitative).
    rs = np.random.RandomState(1)
    d = np.sort(rs.standard_normal(20)) + 1j * rs.standard_normal(20)
    A = np.diag(d).astype(complex)
    A[:10, :10] += np.diag(np.ones(9), 1)
    W, _ = fov(A)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(np.real(W), np.imag(W), 'k', lw=1.1)
    ax.plot(d.real, d.imag, '.r', ms=12)
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EdgeDetection_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The 21 kinks of |exp(x) sin(10 pi x)|, found by splitting on
    f = cj.chebfun(lambda x: jnp.abs(jnp.exp(x) * jnp.sin(10 * jnp.pi * x)),
                   splitting=True)
    xs = np.linspace(-1, 1, 4000)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.2)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EdgeDetection_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    ends = np.array(sorted(float(b) for b in f.domain.breakpoints))
    print("ans =")
    for e in ends:
        print(f"  {e:.15f}")
    true_edges = np.arange(-1, 1.05, 0.1)
    maxerr = float(np.max(np.abs(ends - true_edges)))
    print("maxerr =")
    print(f"     {maxerr:.15e}")

    f2 = cj.chebfun(lambda x: jnp.abs(jnp.exp(x) * jnp.sin(10 * jnp.pi * x)),
                    domain=true_edges.tolist())
    print("ans =")
    print(f"     {float((f - f2).norm(2)):.15e}")

    # Eigenvalue abscissa of a matrix pencil (rng(0) randn draws are
    # likewise MATLAB-specific; kink structure is what replicates).
    rs = np.random.RandomState(0)
    B = rs.standard_normal((20, 20))
    C = rs.standard_normal((20, 20))

    def abscissa_vals(t):
        arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = [float(np.max(np.real(np.linalg.eigvals(
            (1 - tv) * B + tv * C)))) for tv in arr.ravel()]
        return jnp.asarray(out, dtype=jnp.float64).reshape(arr.shape)

    g = cj.chebfun(abscissa_vals, domain=(0.0, 1.0), splitting=True)
    brk = np.array(sorted(float(b) for b in g.domain.breakpoints))[1:-1]
    print("breakpts =")
    for e in brk:
        print(f"   {e:.15f}")

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ts = np.linspace(0, 1, 1500)
    ax.plot(ts, np.asarray(g(jnp.asarray(ts))), lw=1.3)
    ax.plot(brk, np.asarray(g(jnp.asarray(brk))), '.r', ms=12)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EdgeDetection_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    gd = g.diff()
    for lo, hi in zip(list(g.domain.breakpoints)[:-1],
                      list(g.domain.breakpoints)[1:]):
        tt = np.linspace(float(lo) + 1e-9, float(hi) - 1e-9, 200)
        ax.plot(tt, np.asarray(gd(jnp.asarray(tt))), 'b', lw=1.1)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EdgeDetection_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Larger splitLength: only the genuine kinks survive
    g2 = cj.chebfun(abscissa_vals, domain=(0.0, 1.0), splitting=True,
                    split_length=1000)
    brk2 = np.array(sorted(float(b) for b in g2.domain.breakpoints))[1:-1]
    print("breakpts2 =")
    for e in brk2:
        print(f"   {e:.15f}")

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(ts, np.asarray(g2(jnp.asarray(ts))), lw=1.3)
    if len(brk2):
        ax.plot(brk2, np.asarray(g2(jnp.asarray(brk2))), '.r', ms=12)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "EdgeDetection_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
