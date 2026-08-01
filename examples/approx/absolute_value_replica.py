"""Absolute value approximations by rationals.

Faithful replica of approx/AbsoluteValue.m by Nick Trefethen (May
2011): Newton's method for r^2 = x^2 generates type (2^k, 2^k)
rational approximations to |x|, with maximum error exactly 2^-k after
k steps but much faster convergence away from x = 0.

Original: https://www.chebfun.org/examples/approx/AbsoluteValue.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _newton_grid(x, r, fname, label):
    """Six Newton steps in a 3x2 subplot grid; returns final r."""
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 6.2))
    xs = np.linspace(-1, 1, 1200)
    for k in range(6):
        ax = axes[k // 2, k % 2]
        rv = np.asarray(r(jnp.asarray(xs)))
        ax.plot(xs, rv, lw=1.6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True)
        err = float((r - x.abs()).norm(np.inf))
        n = len(r)
        print(f"k={k}: error={err:4.1e}   {label} = {n}")
        ax.set_title(f"error={err:4.1e}   {label} = {n}", fontsize=12)
        r = (r**2 + x**2) / (2 * r)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r


def _error_plot(x, r, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    xs = np.linspace(-1, 1, 20001)
    err = np.abs(np.asarray(r(jnp.asarray(xs))) - np.abs(xs))
    ax.semilogy(xs, np.maximum(err, 1e-19), lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.grid(True)
    ax.set_xlabel("x", fontsize=12)
    ax.set_title("Error", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Newton iteration for r^2 = x^2, no breakpoint
    x = cj.chebfun(lambda t: t)
    r = cj.chebfun(lambda t: 1.0 + 0 * t)
    print("-- no breakpoint --")
    _newton_grid(x, r, "AbsoluteValue_repl_01.png", "len")

    # With a breakpoint at x = 0 the lengths stay modest
    x = cj.chebfun(lambda t: t, domain=[-1.0, 0.0, 1.0])
    r = cj.chebfun(lambda t: 1.0 + 0 * t, domain=[-1.0, 0.0, 1.0])
    print("-- breakpoint at 0 --")
    r = _newton_grid(x, r, "AbsoluteValue_repl_02.png", "length")

    # Error after six steps
    _error_plot(x, r, "AbsoluteValue_repl_03.png")

    # Six more steps
    print("-- six more steps --")
    r = _newton_grid(x, r, "AbsoluteValue_repl_04.png", "length")

    # Error after twelve steps
    _error_plot(x, r, "AbsoluteValue_repl_05.png")


if __name__ == "__main__":
    run()
