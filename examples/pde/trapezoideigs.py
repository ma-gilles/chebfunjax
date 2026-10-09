"""Eigenvalues of a trapezoidal drum.

Translation of pde/TrapezoidEigs.m by Nick Trefethen (November
2014): Laplace eigenvalues of the trapezoid (0,0)-(1,0)-(1,1)-(-1,1)
by the method of particular solutions -- expanding eigenfunctions in
sin(4j theta/3) J_{4j/3}(lambda r) terms, sampling the two remaining
boundary segments, and locating the near-singular lambda by the
minimal singular value, scanned as a chebfun over [3, 7] with
splitting on for n = 4..7.

Original: https://www.chebfun.org/examples/pde/TrapezoidEigs.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.besselj import besselj

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')


def boundary_matrix(lam, n):
    """Source 3n-by-n boundary matrix, batched over lambda inputs."""
    z = jnp.concatenate([
        -1 + 1j + 2*jnp.arange(1, 2*n+1)/(2*n+1),
        1 + 1j*jnp.arange(1, n+1)/(n+1)])
    r, theta = jnp.abs(z), jnp.angle(z)
    lam = jnp.asarray(lam, dtype=jnp.float64).reshape(-1)
    return jnp.stack([
        besselj(4*j/3, lam[:, None]*r[None, :])
        * jnp.sin(4*j*theta/3)[None, :]
        for j in range(1, n+1)], axis=-1)


def trapfun(lam, n):
    """Minimal singular value, retaining the input lambda shape."""
    lam = jnp.asarray(lam, dtype=jnp.float64)
    matrix = boundary_matrix(lam, n)
    return jnp.linalg.svd(matrix, compute_uv=False)[..., -1].reshape(lam.shape)


def run():
    # Align physical output resolution to the reference PNGs (72dpi).
    # Historical typography beyond this resolution remains unqualified.
    os.makedirs(_IMG, exist_ok=True)

    # The trapezoid itself.
    zv = jnp.array([0, 1, 1 + 1j, -1 + 1j])
    fig, ax = plt.subplots(figsize=(6, 2.7))
    ax.set_position([.13, .11, .775, .815])
    ax.fill(zv.real, zv.imag, color=(.7, .7, 1), edgecolor='black')
    ax.text(0.25, 0.55, "?", fontsize=30)
    ax.set_xlim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "TrapezoidEigs_01.png"), dpi=72.0)
    plt.close(fig)

    dom = (3.0, 7.0)
    fignum = 1
    for n in range(4, 8):
        t0 = time.perf_counter()
        f = chebfun(lambda lam, _n=n: trapfun(lam, _n), domain=dom,
                    splitting=True)
        t = time.perf_counter() - t0
        # Python public min returns (locations, values), opposite MATLAB.
        xs, _values = f.min('local')
        xs = xs[xs > dom[0]]
        fignum += 1
        fig, ax = plt.subplots(figsize=(6, 2.7))
        ax.set_position([.13, .11, .775, .815])
        matlab_plot(f, ax=ax, linewidth=1.6)
        ax.grid(True)
        ax.set_title(f"n = {n}     time ={t:4.1f} secs.", fontsize=10)
        ax.set_xlabel("first three minima: lam = "
                      f"{xs[0]:8.5f}, {xs[1]:8.5f}, {xs[2]:8.5f}",
                      fontsize=10)
        fig.set_facecolor("white")
        _savefig(fig, os.path.join(
            _IMG, f"TrapezoidEigs_{fignum:02d}.png"), dpi=72.0)
        plt.close(fig)


if __name__ == "__main__":
    run()
