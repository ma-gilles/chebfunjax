"""Convergence rates for functions of fractional smoothness.

Faithful replica of cheb/Convergence.m by Alex Townsend (October
2010): interpolation of |x|^pi converges at the algebraic rate n^-pi,
and of sin(|x|^(x+5.5)) at n^-5.5.

Original: https://www.chebfun.org/examples/cheb/Convergence.html
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')

XS = np.linspace(-1, 1, 30001)
NN = 2 * np.round(2.0 ** np.arange(0, 7.5, 0.5)).astype(int)


def _study(fop, rate, rate_label, text_xy, title, fname):
    fv = np.asarray(fop(jnp.asarray(XS)))
    ee = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in NN:
            fn = cj.chebfun(fop, n=int(n))
            ee.append(np.max(np.abs(fv
                                    - np.asarray(fn(jnp.asarray(XS))))))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.loglog(NN, NN.astype(float) ** (-rate), 'r', lw=1.2)
    ax.loglog(NN, ee, '.', ms=12)
    ax.grid(True)
    ax.set_xlabel("no. of interpolation points")
    ax.set_ylabel("max Error")
    ax.text(*text_xy, rate_label, fontsize=12)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{title}: final err {ee[-1]:.3e} vs n^-{rate:g} = "
          f"{NN[-1]**(-rate):.3e}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    _study(lambda x: jnp.abs(x) ** jnp.pi, np.pi, r"$n^{-\pi}$",
           (10, 1e-4), "Convergence for fractional differentiable function",
           "Convergence_repl_01.png")

    _study(lambda x: jnp.sin(jnp.abs(x) ** (x + 5.5)), 5.5,
           r"$n^{-5.5}$", (10, 3e-8),
           "Convergence for a trigonometric function",
           "Convergence_repl_02.png")


if __name__ == "__main__":
    run()
