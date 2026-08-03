"""Resampling random variables.

Faithful replica of stats/ResamplingRandomVariables.m by Toby
Driscoll (December 2011): transforming uniform samples into samples
of the von Mises and logit-normal distributions by inverting the
cumulative distribution function.

Original: https://www.chebfun.org/examples/stats/ResamplingRandomVariables.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ResamplingRandomVariables_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dense_inverse(cdf, a, b, n=20000):
    """Monotone dense-grid inverse of a cdf chebfun on [a, b]."""
    xg = np.linspace(a, b, n)
    ug = np.asarray(cdf(xg))
    ug, idx = np.unique(ug, return_index=True)
    xg = xg[idx]
    return lambda u: np.interp(np.asarray(u), ug, xg)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    rs = np.random.RandomState(5489)

    # von Mises distribution
    kappa = 1.5
    f = cj.chebfun(lambda x: jnp.exp(kappa * jnp.cos(x)),
                   domain=(-np.pi, np.pi))
    density = f * (1.0 / float(f.sum()))
    cdf = density.cumsum()
    xs = np.linspace(-np.pi, np.pi, 800)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(density(xs)), lw=1.6, label="density")
    ax.plot(xs, np.asarray(cdf(xs)), lw=1.6, label="distribution")
    ax.axis([-np.pi, np.pi, 0, 1])
    ax.legend(loc="upper left")
    ax.set_title("von Mises distribution", fontsize=12)
    ax.grid(True)
    _save(fig)

    cdfinv = _dense_inverse(cdf, -np.pi, np.pi)
    us = np.linspace(1e-4, 1 - 1e-4, 500)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(us, cdfinv(us), lw=1.6)
    ax.set_title("Inverse of von Mises distribution", fontsize=12)
    ax.grid(True)
    _save(fig)

    u = rs.rand(10**4)
    x = cdfinv(u)
    count, bins = np.histogram(x, 36)
    centers = (bins[:-1] + bins[1:]) / 2
    countn = count / np.sum(count * (centers[1] - centers[0]))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(centers, countn, width=centers[1] - centers[0],
           color=(0.3, 0.5, 0.8), edgecolor='k', lw=0.3)
    ax.plot(xs, np.asarray(density(xs)), 'r', lw=1.6)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_title("Sampled points and the original density",
                 fontsize=12)
    _save(fig)

    # logit-normal distribution
    sig = 1.11

    def ln_op(x):
        return (jnp.exp(-(jnp.log(x / (1 - x)))**2 / (2 * sig**2))
                / (x * (1 - x)))

    eps_ = 1e-8
    density2 = cj.chebfun(ln_op, domain=(eps_, 1 - eps_))
    density2 = density2 * (1.0 / float(density2.sum()))
    cdf2 = density2.cumsum()
    xs2 = np.linspace(eps_, 1 - eps_, 900)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs2, np.asarray(density2(xs2)), lw=1.6, label="density")
    ax.plot(xs2, np.asarray(cdf2(xs2)), lw=1.6,
            label="distribution")
    ax.legend(loc="upper left")
    ax.set_title("logit-normal distribution", fontsize=12)
    ax.grid(True)
    _save(fig)

    # invert on [0.5, 1-1e-3], using symmetry for the lower half
    cdfinv2 = _dense_inverse(cdf2, 0.5, 1 - 1e-3, n=40000)
    us2 = np.linspace(float(cdf2(0.5)), float(cdf2(1 - 1e-3)), 500)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(us2, cdfinv2(us2), lw=1.6)
    ax.set_title("Inverse of the logit-normal distribution",
                 fontsize=12)
    ax.grid(True)
    _save(fig)

    u = rs.rand(10**4)
    flag = u < 0.5
    u[flag] = 1 - u[flag]
    x = cdfinv2(u)
    x[flag] = 1 - x[flag]
    count, bins = np.histogram(x, 36)
    centers = (bins[:-1] + bins[1:]) / 2
    countn = count / np.sum(count * (centers[1] - centers[0]))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(centers, countn, width=centers[1] - centers[0],
           color=(0.3, 0.5, 0.8), edgecolor='k', lw=0.3)
    ax.plot(xs2, np.asarray(density2(xs2)), 'r', lw=1.6)
    ax.set_xlim(0, 1)
    ax.set_title("Sampled points and the original density",
                 fontsize=12)
    _save(fig)

    missing = 1 - float(cdf2(1 - 1e-3))
    print("missing =")
    print(f"     {missing:.15e}")


if __name__ == "__main__":
    run()
