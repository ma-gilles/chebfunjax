"""Integrating Tj(x)*Tk(y) over the unit disk.

Faithful replica of quad/TjTkDisk.m by Mikael Slevinsky, Nick
Trefethen, and Klaus Wang (May 2016): the integrals of products of
Chebyshev polynomials over the unit disk vanish unless j and k are
both even and differ by 0 or 2, with closed-form values.

Original: https://www.chebfun.org/examples/quad/TjTkDisk.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def _T(n):
    """Chebyshev polynomial T_n as a callable on [-1, 1]."""
    return lambda x: jnp.cos(n * jnp.arccos(jnp.clip(x, -1.0, 1.0)))


def _disp_long(v, name):
    print(f"{name} =")
    print(f"  {v:.15f}" if v < 0 else f"   {v:.15f}")


def _disp_short_matrix(M, exact_zero=False):
    print("I =")
    for row in M:
        cells = []
        for v in row:
            if exact_zero and v == 0:
                cells.append(f"{'0':>10}")
            else:
                cells.append(f"{v:10.4f}")
        print("".join(cells))


def _disk_integral(f):
    """Integral of f(r, t) over the unit disk via nested chebfuns."""
    def fr(r):
        circ = cj.chebfun(lambda t: f(r, t) + 0.0 * t,
                          domain=(0.0, 2 * np.pi), trig=True)
        return r * float(circ.sum())

    def radial_vals(r):
        arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
        vals = [fr(float(ri)) for ri in arr.ravel()]
        return jnp.asarray(vals, dtype=jnp.float64).reshape(arr.shape)

    radial = cj.chebfun(radial_vals, domain=(0.0, 1.0))
    return float(radial.sum())


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Section 1: contour plot of T8(x) T16(y) on the disk
    T8, T16 = _T(8), _T(16)
    s = np.linspace(-1, 1, 160)
    xx, yy = np.meshgrid(s, s)
    ff = np.array(T8(jnp.asarray(xx)) * T16(jnp.asarray(yy)))
    ff[xx**2 + yy**2 > 1] = np.nan
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.contour(s, s, ff, levels=10, linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), 'k', lw=1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TjTkDisk_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Numerical confirmation that the integral of 1 is pi:
    I = _disk_integral(lambda r, t: 1.0 + 0 * t)
    _disp_long(I, "I")
    _disp_long(np.pi, "Iexact")

    # f(r,t) = r^2: the integral is pi/2
    I = _disk_integral(lambda r, t: r**2 + 0 * t)
    _disp_long(I, "I")
    _disp_long(np.pi / 2, "Iexact")

    # f(r,t) = r^2 cos^2(t): the integral is pi/4
    I = _disk_integral(lambda r, t: r**2 * jnp.cos(t) ** 2)
    _disp_long(I, "I")
    _disp_long(np.pi / 4, "Iexact")

    # Section 2: numerically computed integrals for j,k = 0,2,...,10
    t0 = time.time()
    M = np.zeros((6, 6))
    for j in range(0, 11, 2):
        Tj = _T(j)
        for k in range(0, j + 1, 2):
            Tk = _T(k)
            f = lambda r, t: Tk(r * jnp.cos(t)) * Tj(r * jnp.sin(t))  # noqa: E731
            M[j // 2, k // 2] = _disk_integral(f)
    M = M + np.tril(M, -1).T
    _disp_short_matrix(M)
    elapsed = time.time() - t0
    print("time_elapsed_in_seconds =")
    print(f"    {elapsed:.4f}")

    # Section 3: analytic expressions reproduce the matrix
    A = np.zeros((6, 6))
    A[0, 0] = np.pi
    A[1, 0] = -np.pi / 2
    for k in range(2, 11, 2):
        A[k // 2, k // 2] = np.pi * (-1) ** (k // 2) / (2 - 2 * k**2)
    for k in range(2, 9, 2):
        A[1 + k // 2, k // 2] = -np.pi * (-1) ** (k // 2) / (4 * k + 4)
    A = A + np.tril(A, -1).T
    _disp_short_matrix(A, exact_zero=True)


if __name__ == "__main__":
    run()
