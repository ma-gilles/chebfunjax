"""Spike integral.

Faithful replica of quad/SpikeIntegral.m by Nick Hale: integrating a
function with four spikes of widths down to 1e-3.  The global adaptive
construction resolves it (length and integral match the published page
digit-for-digit); the splitting-mode variant is affected by a ledgered
splitting defect on narrow smooth spikes and is shown with its
measured (incorrect) value for honesty.

Original: https://www.chebfun.org/examples/quad/SpikeIntegral.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')

sech = lambda z: 1 / jnp.cosh(z)


def _f(x):
    return (sech(10 * (x - 0.2)) ** 2 + sech(100 * (x - 0.4)) ** 4
            + sech(1000 * (x - 0.6)) ** 6
            + sech(1000 * (x - 0.8)) ** 8)


def run():
    os.makedirs(_IMG, exist_ok=True)
    ff = cj.chebfun(_f, domain=[0, 1])
    print("length =")
    print(f"   {len(ff)}")
    xs = np.linspace(0, 1, 4000)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, np.asarray(ff(jnp.asarray(xs))), "b", lw=1.6)
    ax.grid(True)
    ax.set_title("Spike function", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SpikeIntegral_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    zx = np.linspace(0.795, 0.805, 1200)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.semilogy(zx, np.maximum(np.asarray(ff(jnp.asarray(zx))), 1e-30),
                "b", lw=1.6)
    ax.grid(True)
    ax.set_title("Zoom, on semilogy axes", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "SpikeIntegral_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    print(f"   {float(ff.sum()):.15f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ffs = cj.chebfun(_f, domain=[0, 1], splitting=True,
                         min_samples=100)
    print("splitting pieces =")
    print(f"   {len(ffs.funs)}")
    print("splitting sum (KNOWN DEFECT, see ledger) =")
    print(f"   {float(ffs.sum()):.15f}")
    return True


if __name__ == "__main__":
    run()
