"""The bivariate normal distribution.

Translation of stats/BivariateNormalDistribution.m by Alex
Townsend (March 2013): the joint pdf as a chebfun2, its integral,
marginal distribution, and conditional pdf — each checked against
the closed form.

Original: https://www.chebfun.org/examples/stats/BivariateNormalDistribution.html
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
from chebfunjax.plotting import PARULA, chebfun_style, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]


def _format_axes(fig, number):
    ax = fig.axes[0]
    ax.tick_params(labelsize=9)
    if number < 3:
        ax.set_title(ax.get_title(), fontsize=10.5, fontweight="bold")
        ax.set_xticks(np.arange(-10, 11, 2))
        ax.tick_params(direction="in", top=True, right=True)
        if number == 1:
            ax.set_yticks(np.arange(-10, 11, 2))
            for collection in ax.collections:
                collection.set_linewidth(0.5)
        else:
            ax.set_yticks(np.arange(0, 0.401, 0.05))
            for line in ax.lines:
                line.set_linewidth(1.0)
    else:
        ax.set_xticks(np.arange(-2, 3))
        ax.set_yticks(np.arange(-2, 3))
        ax.set_zticks(np.arange(0, 0.51, 0.1))


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _format_axes(fig, FIG[0])
    _savefig(fig, os.path.join(
        _IMG, f"BivariateNormalDistribution_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    mu1 = mu2 = 0.0
    s1 = s2 = 1.0
    rho = 0.5
    d = (-10.0, 10.0, -10.0, 10.0)

    def z(x, y):
        return ((x - mu1)**2 / s1**2
                - 2 * rho * (x - mu1) * (y - mu2) / (s1 * s2)
                + (y - mu2)**2 / s2**2)

    p = cj.chebfun2(
        lambda x, y: 1 / (2 * jnp.pi * s1 * s2
                          * jnp.sqrt(1 - rho**2))
        * jnp.exp(-z(x, y) / (2 * (1 - rho**2))), domain=d)

    xs = np.linspace(-10, 10, 240)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(p(jnp.asarray(X), jnp.asarray(Y)))
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.13, 0.11, 0.775, 0.815])
    ax.contour(X, Y, Z, levels=np.arange(0.001, 0.2, 0.01), cmap=PARULA)
    ax.set_title("Bivariate normal distribution", fontsize=14)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    _save(fig)
    print(f"Integral of pdf {float(p.sum2()):1.16f}")

    px = p.sum()   # source default: integrate over y
    xg = np.linspace(-10, 10, 800)
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.13, 0.11, 0.775, 0.815])
    ax.plot(xg, np.asarray(px(xg)).ravel(), lw=1.6)
    ax.set_title("Marginal distribution", fontsize=14)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.4)
    _save(fig)
    exact = cj.chebfun(
        lambda x: 1 / (jnp.sqrt(2 * jnp.pi) * s1)
        * jnp.exp(-(x - mu1)**2 / s1**2 / 2), domain=d[:2])
    err = float((px - exact.T).norm())
    print(f"Error of marginal = {err:1.3e}")

    # conditional pdf on a smaller domain
    d2 = (-2.0, 2.0, -2.0, 2.0)
    fy = cj.chebfun2(lambda x, y: p(x, y) / px.T(x), domain=d2)
    fig, ax = surf(fy)
    _save(fig)

    x0 = np.pi / 6
    mu = mu1 + s1 / s2 * rho * (x0 - mu2)
    sigmasq = (1 - rho**2) * s1**2
    exact_c = cj.chebfun(
        lambda y: 1 / jnp.sqrt(2 * jnp.pi * sigmasq)
        * jnp.exp(-(y - mu)**2 / sigmasq / 2), domain=d2[:2])
    errc = float((fy(x0, ":") - exact_c).norm())
    print(f"Error in conditional pdf is {errc:1.5e}")
    return {"integral": float(p.sum2()), "marginal_error": err,
            "conditional_error": errc, "joint_rank": p.approx.rank,
            "conditional_rank": fy.approx.rank,
            "marginal_length": len(px)}


if __name__ == "__main__":
    run()
