"""Spike integral.

Translation of quad/SpikeIntegral.m by Nick Hale: integrating a
function with four spikes of widths down to 1e-3.  The global adaptive
construction resolves it; with splitting on, the default minimum sample
count misses the narrowest spike, and minSamples = 100 catches it.

Original: https://www.chebfun.org/examples/quad/SpikeIntegral.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')

sech = lambda z: 1 / jnp.cosh(z)


def _f(x):
    return (sech(10 * (x - 0.2)) ** 2 + sech(100 * (x - 0.4)) ** 4
            + sech(1000 * (x - 0.6)) ** 6
            + sech(1000 * (x - 0.8)) ** 8)


def _plot(ff, title, fname):
    xs = np.linspace(0, 1, 4000)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, np.asarray(ff(jnp.asarray(xs))), "b", lw=1.6)
    ax.grid(True)
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname), size=(600, 270))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.simplefilter("ignore")
    ff = cj.chebfun(_f, domain=[0, 1])
    print("ff =")
    print(repr(ff))
    _plot(ff, "Spike function", "SpikeIntegral_01.png")

    zx = np.linspace(0.795, 0.805, 1200)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.semilogy(zx, np.maximum(np.asarray(ff(jnp.asarray(zx))), 1e-30),
                "b", lw=1.6)
    ax.grid(True)
    ax.set_title("Zoom, on semilogy axes", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "SpikeIntegral_02.png"), size=(600, 270))
    plt.close(fig)

    t0 = time.time()
    ff = cj.chebfun(_f, domain=[0, 1])
    print("ans =")
    print(f"   {float(ff.sum()):.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    ff = cj.chebfun(_f, domain=[0, 1], splitting=True)
    print("ff =")
    print(repr(ff))
    _plot(ff, "Unresolved spike function with splitting on",
          "SpikeIntegral_03.png")

    ff = cj.chebfun(_f, domain=[0, 1], splitting=True, min_samples=100)
    print("ff =")
    print(repr(ff))
    _plot(ff, "Resolved spike function with splitting on",
          "SpikeIntegral_04.png")

    t0 = time.time()
    ff = cj.chebfun(_f, domain=[0, 1], splitting=True, min_samples=100)
    print("ans =")
    print(f"   {float(ff.sum()):.15f}")
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
