"""Absolute value approximations by rationals II.

Faithful replica of approx/AbsoluteValueScaled.m by Yuji Nakatsukasa
(July 2012): the scaled Newton iteration for sign(x) yields Zolotarev
best rational approximations, giving uniformly small error for |x| —
unlike the plain Newton iteration of approx/AbsoluteValue.

Original: https://www.chebfun.org/examples/approx/AbsoluteValueScaled.html
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

EPS = float(np.finfo(np.float64).eps)


def _semilogy_cheb(ax, f, xs, floor=1e-19, **kw):
    v = np.abs(np.asarray(f(jnp.asarray(xs))))
    ax.semilogy(xs, np.maximum(v, floor), **kw)


def run():
    os.makedirs(_IMG, exist_ok=True)
    xs = np.linspace(-1, 1, 20001)
    dom = [-1.0, 0.0, 1.0]
    kmax = 5

    # Plain Newton iteration from approx/AbsoluteValue
    x = cj.chebfun(lambda t: t, domain=dom)
    r = cj.chebfun(lambda t: 1.0 + 0 * t, domain=dom)
    for k in range(kmax + 1):
        r = (r**2 + x**2) / (2 * r)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _semilogy_cheb(ax, (r - x.abs()) + EPS, xs, lw=1.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.grid(True)
    ax.set_xlabel("x", fontsize=12)
    ax.set_title("Error", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AbsoluteValueScaled_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scaled Newton for sign(x), then |x| = x / sign(x).
    # The intermediate iterates (t*r + 1/(t*r))/2 are rational functions
    # with a pole at x = 0, so the recurrence is evaluated pointwise
    # (mathematically identical to the chebfun arithmetic in MATLAB).
    b = 1e-3

    def scaled_newton_vals(xv, ksteps):
        rv = np.asarray(xv, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = 1 / np.sqrt(b)
            for k in range(ksteps + 1):
                if k > 0:
                    t = np.sqrt(2 / (t + 1 / t))
                rv = ((t * rv) + 1.0 / (t * rv)) / 2
            out = np.asarray(xv) / rv
        return np.where(np.asarray(xv) == 0.0, 0.0, out)

    rs_vals = scaled_newton_vals(xs, kmax)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _semilogy_cheb(ax, (r - x.abs()) + EPS, xs, lw=1.6, label="Newton")
    ax.semilogy(xs, np.maximum(np.abs(rs_vals - np.abs(xs)), 1e-19),
                lw=1.6, color="r", label="scaled Newton")
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.grid(True)
    ax.set_xlabel("x", fontsize=12)
    ax.set_title("Error", fontsize=12)
    ax.legend(loc="best")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AbsoluteValueScaled_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Errors of the scaled iteration for varying k
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _semilogy_cheb(ax, r - x.abs(), xs, lw=1.6, label="Newton k=5")
    colork = ['k', 'm', 'g', 'r']
    for k in range(2, kmax + 1):
        vals = scaled_newton_vals(xs, k)
        ax.semilogy(xs, np.maximum(np.abs(vals - np.abs(xs)), 1e-19),
                    lw=1.6, color=colork[k - 2], label=f"s-Newton k={k}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(1e-18, 10)
    ax.grid(True)
    ax.legend(loc="best")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AbsoluteValueScaled_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # scalar sanity: max errors of the two final approximants
    print("newton_err =")
    print(f"   {float((r - x.abs()).norm(np.inf)):.6e}")
    print("scaled_err =")
    print(f"   {float(np.max(np.abs(rs_vals - np.abs(xs)))):.6e}")


if __name__ == "__main__":
    run()
