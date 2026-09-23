"""Edge detection.

Translation of approx/EdgeDetection.m by Nick Trefethen (July
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
from chebfunjax.chebfun1d.fov import fov
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


# One colour per chebfun, dotted jump lines (MATLAB @chebfun/plot).
_PW = dict(color="#0072BD", jumpline=":")


def _save(fig, k):
    fig.set_facecolor("white")
    fig.set_size_inches(6.0, 2.7)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"EdgeDetection_{k:02d}.png"))
    plt.close(fig)


def _col(v):
    """MATLAB format-long display of a column vector."""
    for e in v:
        print(f"{e:20.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Field of values of a random-ish matrix (MATLAB rng(1) randn draws
    # are ziggurat-based and not bit-reproducible outside MATLAB; the
    # figure is qualitative).
    rs = np.random.RandomState(1)
    d = np.sort(rs.standard_normal(20)) + 1j * rs.standard_normal(20)
    A = np.diag(d).astype(complex)
    A[:10, :10] += np.diag(np.ones(9), 1)
    W, W2, _ = fov(A, line_segments=True)
    fig, ax = plt.subplots()
    matlab_plot(W, 'k', ax=ax)
    matlab_plot(W2, 'k', ax=ax)
    ax.plot(d.real, d.imag, '.r', ms=10)
    ax.set_axis_off()
    _save(fig, 1)

    # The 21 kinks of |exp(x) sin(10 pi x)|, found by splitting on
    f = cj.chebfun(lambda x: jnp.abs(jnp.exp(x) * jnp.sin(10 * jnp.pi * x)),
                   splitting=True)
    fig, ax = plt.subplots()
    matlab_plot(f, ax=ax, **_PW)
    _save(fig, 2)

    ends = np.array([float(b) for b in f.domain.breakpoints])
    print("ans =")
    _col(ends)
    true_edges = np.arange(-10, 11) / 10
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
    brk = np.array([float(b) for b in g.domain.breakpoints])[1:-1]
    fig, ax = plt.subplots()
    matlab_plot(g, ax=ax, **_PW)
    ax.grid(True)
    print("breakpts =")
    _col(brk)
    ax.plot(brk, np.asarray(g(jnp.asarray(brk))), '.r', ms=10)
    _save(fig, 3)

    fig, ax = plt.subplots()
    matlab_plot(g.diff(), ax=ax, **_PW)
    ax.grid(True)
    _save(fig, 4)

    # Larger splitLength: only the genuine kinks survive
    g2 = cj.chebfun(abscissa_vals, domain=(0.0, 1.0), splitting=True,
                    split_length=1000)
    brk2 = np.array([float(b) for b in g2.domain.breakpoints])[1:-1]
    fig, ax = plt.subplots()
    matlab_plot(g2, ax=ax, **_PW)
    ax.grid(True)
    print("breakpts2 =")
    _col(brk2)
    ax.plot(brk2, np.asarray(g2(jnp.asarray(brk2))), '.r', ms=10)
    _save(fig, 5)


if __name__ == "__main__":
    run()
