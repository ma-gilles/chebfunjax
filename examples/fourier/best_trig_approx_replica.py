"""Best trigonometric approximation with trigremez.

Faithful replica of fourier/BestTrigApprox.m by Mohsin Javed and Nick
Trefethen, February 2015: best (minimax) trigonometric approximations
and their equioscillating error curves for a smooth periodic function,
a spiky non-smooth function, and a detrended piecewise-constant
integral.

Original: https://www.chebfun.org/examples/fourier/BestTrigApprox.html
Copyright 2015 by The University of Oxford and The Chebfun Developers.
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _save(fig, stem):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _pair_plots(f, p, err, dom, stem_base, deg):
    xs = np.linspace(dom[0], dom[1], 3000)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), "k", lw=1.0)
    ax.plot(xs, np.asarray(p(jnp.asarray(xs))), "r", lw=1.2)
    ax.set_title("Function (black) and best trigonometric "
                 "approximation (red)")
    _save(fig, stem_base + "a")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    e = np.asarray(f(jnp.asarray(xs))) - np.asarray(p(jnp.asarray(xs)))
    ax.plot(xs, e, lw=1.0)
    ax.plot([dom[0], dom[1]], [err, err], "--k", lw=1.0)
    ax.plot([dom[0], dom[1]], [-err, -err], "--k", lw=1.0)
    ax.set_ylim(-5 * err, 5 * err)
    ax.set_title(f"Degree {deg} trigonometric error curve")
    _save(fig, stem_base + "b")
    print(f"deg {deg}: err = {err:.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.exp(jnp.sin(2 * x) + jnp.cos(3 * x)),
                   domain=[-np.pi, np.pi], trig=True)
    p, err, *_ = cj.trigremez(f, 5)
    _pair_plots(f, p, float(err), (-np.pi, np.pi),
                "BestTrigApprox_repl_1", 5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f2 = cj.chebfun(
            lambda x: 10 * jnp.abs(x) + jnp.sin(20 * np.pi * x)
            + 10 * jnp.exp(-50 * (x - 0.1) ** 2), splitting=True)
    p2, err2, *_ = cj.trigremez(f2, 8)
    _pair_plots(f2, p2, float(err2), (-1.0, 1.0),
                "BestTrigApprox_repl_2", 8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = cj.chebfun(lambda x: jnp.sign(jnp.sin(20 * jnp.exp(x))),
                       splitting=True).cumsum()
    x = cj.chebfun(lambda t: t)
    m = (float(g(jnp.asarray([1.0]))[0])
         - float(g(jnp.asarray([-1.0]))[0])) / 2
    y = m * (x - 1.0) + float(g(jnp.asarray([1.0]))[0])
    f3 = g - y
    p3, err3, *_ = cj.trigremez(f3, 15)
    _pair_plots(f3, p3, float(err3), (-1.0, 1.0),
                "BestTrigApprox_repl_3", 15)
    return True


if __name__ == "__main__":
    run()
