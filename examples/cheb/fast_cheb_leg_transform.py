"""The fast Chebyshev-Legendre transform.

Translation of cheb/FastChebyshevLegendreTransform.m by Nick Hale and Alex
Townsend (August 2013): converting between Chebyshev and Legendre
expansions.  The coefficient-comparison sections replicate exactly;
the two large-N timing demos (N ~ 24000-32000) require the O(N log N)
transform of Hale & Townsend, which chebfunjax has not ported (its
cheb2leg/leg2cheb are quadratic-cost; ledgered), and are described
rather than run.

Original: https://www.chebfun.org/examples/cheb/FastChebyshevLegendreTransform.html
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
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.transforms import cheb2leg

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Runge-type function
    f = cj.chebfun(lambda x: 1.0 / (1 + 1000 * (x - 0.1) ** 2))
    c_cheb = np.asarray(f.coeffs)
    c_leg = np.asarray(cheb2leg(f.coeffs))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.abs(c_leg) + 1e-30, 'xr', ms=4,
                label="Legendre coefficients")
    ax.semilogy(np.abs(c_cheb) + 1e-30, '.b', ms=8,
                label="Chebyshev coefficients")
    ax.legend()
    ax.set_xlabel("n", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, "FastChebyshevLegendreTransform_01.png"))
    plt.close(fig)
    print(f"Runge-type: N = {len(f)}")

    # |x - .1|^(7/4): algebraic decay with a half-power gap between the
    # two coefficient families
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = cj.chebfun(lambda x: jnp.abs(x - 0.1) ** (7.0 / 4),
                       max_length=2**13)
    N = len(g)
    c_cheb = np.asarray(g.coeffs)
    c_leg = np.asarray(cheb2leg(g.coeffs))
    nn = np.arange(1, N + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.abs(c_leg) + 1e-30, 'xr', ms=4,
                label="Legendre coefficients")
    ax.semilogy(np.abs(c_cheb) + 1e-30, '.b', ms=8,
                label="Chebyshev coefficients")
    ax.semilogy(nn, nn ** (-7.0 / 4 - 1 + 0.5), 'k--', lw=1.6,
                label=r"$O(n^{-2.25})$")
    ax.semilogy(nn, nn ** (-7.0 / 4 - 1), 'k--', lw=1.6,
                label=r"$O(n^{-2.75})$")
    ax.set_xlim(0, N)
    ax.set_xlabel("n", fontsize=12)
    ax.legend()
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, "FastChebyshevLegendreTransform_02.png"))
    plt.close(fig)
    print(f"|x-.1|^(7/4): N = {N}")

    # --- Fast evaluation of Legendre expansions ---
    import time

    from scipy.sparse import diags as _diags
    from scipy.sparse.linalg import spsolve as _spsolve

    from chebfunjax.utils.quadrature import chebpts
    from chebfunjax.utils.transforms import coeffs2vals, leg2cheb

    t = .999j
    f = cj.chebfun(lambda x: 1 / jnp.sqrt(1 - 2 * x * t + t ** 2))  # generating function
    N = len(f)
    print(f"No. of evaluation points = {N}")
    t0 = time.time()                                                # evaluate f
    c_leg = t ** np.arange(N)                                        # via Legendre coeffs
    cheb_vals = np.asarray(coeffs2vals(leg2cheb(jnp.asarray(c_leg))))  # and time it...
    print(f"Evaluation time = {time.time() - t0:.2f}s")
    xk = np.asarray(chebpts(N, kind=2))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(xk, np.abs(np.asarray(f(jnp.asarray(xk))) - cheb_vals) + 1e-300)
    ax.set_title("Absolute error", fontsize=12)
    ax.axis([-1, 1, 1e-16, 1e-12])
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, "FastChebyshevLegendreTransform_03.png"))
    plt.close(fig)

    # --- Computing Legendre coefficients by a spectral method ---
    t0 = time.time()
    w = 10000 * np.pi                                  # solution is cos(wx)
    f = cj.chebfun(lambda x: jnp.cos(w * x))
    N = len(f)
    print("N =")
    print(f"       {N}")
    n_ = np.arange(N, dtype=float)
    D1 = _diags([np.ones(N - 1)], [1], shape=(N, N))
    D2 = 3 * D1                                        # diff operators
    S1 = _diags([.5 / (n_ + .5), -.5 / (n_[:N - 2] + .5)], [0, 2], shape=(N, N))
    S2 = _diags([1.5 / (n_ + 1.5), -1.5 / (n_[:N - 2] + 1.5)], [0, 2], shape=(N, N))
    A = (D2 @ D1 + w ** 2 * S2 @ S1).tolil()           # u''(x) + w^2 u = 0
    A[N - 2, :] = (-1.0) ** n_                         # left bc
    A[N - 1, :] = np.ones(N)                           # right bc
    b = np.zeros(N)
    b[N - 2] = float(f(jnp.asarray(-1.0)))
    b[N - 1] = float(f(jnp.asarray(1.0)))              # rhs
    P = _diags([np.r_[np.arange(1, N - 1, dtype=float), 1.0, 1.0]], [0])  # preconditioner
    c_leg = _spsolve((P @ A).tocsc(), P @ b)           # solve
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    t0 = time.time()
    c_cheb = np.asarray(leg2cheb(jnp.asarray(c_leg)))
    u = cj.chebfun(jnp.asarray(c_cheb), coeffs=True)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    xz = np.linspace(-0.001, 0.001, 2000)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(xz, np.asarray(u(jnp.asarray(xz))), lw=1.2)
    ax.set_title("Computed solution (zoomed in)", fontsize=12)
    ax.set_xlabel("x", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, "FastChebyshevLegendreTransform_04.png"))
    plt.close(fig)
    print("ans =")
    print(f"     {np.max(np.abs(c_cheb - np.asarray(f.coeffs))):.15e}")
    print("ans =")
    print(f"     {float((u - f).norm()):.15e}")


if __name__ == "__main__":
    run()
