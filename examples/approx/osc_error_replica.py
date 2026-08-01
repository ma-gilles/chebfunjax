"""Error curves for polynomial approximations.

Faithful replica of approx/OscError.m by Mohsin Javed (October 2013):
error curves of the best (minimax), Chebyshev-truncation, Legendre
least-squares, and interpolation approximations of degree 4.

Original: https://www.chebfun.org/examples/approx/OscError.html
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
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 3000)


def run():
    os.makedirs(_IMG, exist_ok=True)

    fop = lambda x: jnp.exp(x) + 0.5 * jnp.sin(2 * jnp.pi * x)  # noqa: E731
    f = cj.chebfun(fop, n=10)
    fv = np.asarray(f(jnp.asarray(XS)))

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, fv, lw=2)
    ax.set_title("Function to be approximated", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OscError_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    n = 4
    res = minimax(lambda x: f(x), n)
    p = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    pv = np.asarray(p(jnp.asarray(XS)))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, fv, lw=2)
    ax.plot(XS, pv, 'r-.', lw=2)
    ax.set_title("Function and its best approximation", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OscError_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Error curves of four approximations
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(XS, fv - pv, 'r', lw=2, label=r"$L_\infty$-Best")
    maxError = float(np.max(np.abs(fv - pv)))
    for y in (maxError, -maxError, 0.0):
        ax.plot([-1, 1], [y, y], 'k--', lw=1.5)

    # Chebyshev truncation of the series to degree n
    c = np.asarray(f.coeffs)[:n + 1]
    fn = cj.chebfun(jnp.asarray(c), coeffs=True)
    ax.plot(XS, fv - np.asarray(fn(jnp.asarray(XS))), 'k', lw=2,
            label=r"$L_2$-Cheb")

    # Legendre least squares
    pLn = f.polyfit(n)
    ax.plot(XS, fv - np.asarray(pLn(jnp.asarray(XS))), 'b', lw=2,
            label=r"$L_2$-Legn")

    # Interpolation in n+1 Chebyshev points
    pn = cj.chebfun(lambda x: f(x), n=n + 1)
    ax.plot(XS, fv - np.asarray(pn(jnp.asarray(XS))), 'g', lw=2,
            label="Interp")
    ax.set_title("Approximation error", fontsize=12)
    ax.legend()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OscError_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"maxError = {maxError:.6f}")


if __name__ == "__main__":
    run()
