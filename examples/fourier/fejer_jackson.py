"""Fejer-Jackson inequality.

Translation of fourier/FejerJackson.m by Nick Trefethen, July 2015:
partial sums f_n(x) = sum_{k=1}^n sin(kx)/k are positive on (0, pi)
(Fejer-Jackson), their min/max are shown for n = 32, 128, 512, and the
lengths of cheb vs trig representations are compared.

Original: https://www.chebfun.org/examples/fourier/FejerJackson.html
Copyright 2015 by The University of Oxford and The Chebfun Developers.
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
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _fnx(n):
    ks = jnp.arange(n, 0, -1, dtype=jnp.float64)

    def op(x):
        xx = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
        return jnp.sum(jnp.sin(xx[..., None] * ks) / ks, axis=-1)
    return op


def _fn(n):
    return cj.chebfun(_fnx(n), domain=[0.0, np.pi])


def _save(fig, stem):
    fig.set_facecolor("white")
    fig.set_size_inches(6.1, 2.76)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, stem + ".png"), size=(610, 276))
    plt.close(fig)


def _pi_yticks(ax):
    ax.set_yticks(np.pi * np.array([0, .25, .5]), ["0", r"$\pi/4$", r"$\pi/2$"])


def _plot_fn(f, n, stem, lw, g=None):
    (_, fmin), (_, fmax) = f.minandmax()
    fig, ax = plt.subplots()
    matlab_plot(f if g is None else g, ax=ax, lw=lw)
    if g is None:
        ax.axis([-.1, 3.3, 0, 2])
        ax.set_xticks(np.pi * np.array([0, .5, 1]), ["0", r"$\pi/2$", r"$\pi$"])
    else:
        ax.axis([0, .2, 0, 2])
    _pi_yticks(ax)
    ax.grid(True)
    ax.set_title(f"Min and max of f{n}:  {fmin:9.6f}, {fmax:9.6f}",
                 fontsize=12)
    _save(fig, stem)


def run():
    os.makedirs(_IMG, exist_ok=True)
    f32 = _fn(32)
    _plot_fn(f32, 32, "FejerJackson_01", 1.6)
    f128 = _fn(128)
    _plot_fn(f128, 128, "FejerJackson_02", 1)
    f512 = _fn(512)
    _plot_fn(f512, 512, "FejerJackson_03", 1, g=f512.restrict(0, 0.2))

    nn = np.arange(10, 501, 10)
    ln = [len(_fn(int(n))) for n in nn]
    fig, ax = plt.subplots()
    ax.plot(nn, ln, ".", ms=10)
    ax.set_title("Length of chebfuns", fontsize=12)
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("length(fn(n))", fontsize=10)
    _save(fig, "FejerJackson_04")

    lntrig = [len(cj.chebfun(_fnx(int(n)), domain=[0.0, 2 * np.pi],
                             trig=True)) for n in nn]
    fig, ax = plt.subplots()
    ax.plot(nn, ln, ".", ms=10, label="cheb")
    ax.plot(nn, lntrig, "or", ms=4, mfc="none", label="trig")
    ax.set_title("Length of chebfuns, both cheb and trig", fontsize=12)
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("length(fn(n))", fontsize=10)
    ax.legend(loc="upper left")
    _save(fig, "FejerJackson_05")


if __name__ == "__main__":
    run()
