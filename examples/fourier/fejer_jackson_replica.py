"""Fejer-Jackson inequality.

Faithful replica of fourier/FejerJackson.m by Nick Trefethen, July 2015:
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
from chebfunjax.plotting import chebfun_style

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
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_fn(f, n, xlim, stem, xs=None):
    if xs is None:
        xs = np.linspace(xlim[0], xlim[1], 2000)
    (xmin, fmin), (xmax, fmax) = f.minandmax()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 2)
    ax.grid(True)
    ax.set_title(f"Min and max of f{n}:  {fmin:9.6f}, {fmax:9.6f}",
                 fontsize=12)
    _save(fig, stem)
    print(f"f{n}: min {fmin:.6f} max {fmax:.6f} length {len(f)}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    f32 = _fn(32)
    _plot_fn(f32, 32, (-0.1, 3.3), "FejerJackson_repl_01")
    f128 = _fn(128)
    _plot_fn(f128, 128, (-0.1, 3.3), "FejerJackson_repl_02")
    f512 = _fn(512)
    _plot_fn(f512, 512, (0.0, 0.2), "FejerJackson_repl_03",
             xs=np.linspace(0, 0.2, 2000))

    nn = np.arange(10, 501, 10)
    ln = [len(_fn(int(n))) for n in nn]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(nn, ln, ".", ms=10)
    ax.set_title("Length of chebfuns", fontsize=12)
    ax.set_xlabel("n")
    ax.set_ylabel("length(fn(n))")
    _save(fig, "FejerJackson_repl_04")

    lntrig = [len(cj.chebfun(_fnx(int(n)), domain=[0.0, 2 * np.pi],
                             trig=True)) for n in nn]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(nn, ln, ".", ms=10, label="cheb")
    ax.plot(nn, lntrig, "or", ms=4, label="trig")
    ax.set_title("Length of chebfuns, both cheb and trig", fontsize=12)
    ax.set_xlabel("n")
    ax.set_ylabel("length(fn(n))")
    ax.legend(loc="upper left")
    _save(fig, "FejerJackson_repl_05")
    print("length ratio at n=500:", ln[-1] / lntrig[-1])
    return True


if __name__ == "__main__":
    run()
