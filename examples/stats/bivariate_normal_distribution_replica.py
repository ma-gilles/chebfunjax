"""The bivariate normal distribution.

Faithful replica of stats/BivariateNormalDistribution.m by Alex
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
        _IMG, f"BivariateNormalDistribution_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
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

    xs = np.linspace(-4, 4, 240)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(p(jnp.asarray(X), jnp.asarray(Y)))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.contour(X, Y, Z, levels=np.arange(0.001, 0.2, 0.01))
    ax.set_title("Bivariate normal distribution", fontsize=14)
    ax.set_aspect("equal")
    _save(fig)
    print(f"Integral of pdf {float(p.sum2()):1.16f}")

    px = p.sum(dim=1)   # marginal over y
    xg = np.linspace(-10, 10, 800)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xg, np.asarray(px(xg)).ravel(), lw=1.6)
    ax.set_title("Marginal distribution", fontsize=14)
    ax.grid(True)
    _save(fig)
    exact = (lambda x: 1 / (np.sqrt(2 * np.pi) * s1)
             * np.exp(-(x - mu1)**2 / s1**2 / 2))
    gx, gw = np.polynomial.legendre.leggauss(1000)
    xq = 10 * gx
    dv = np.asarray(px(xq)).ravel() - exact(xq)
    err = np.sqrt(np.sum(gw * dv**2) * 10)
    print(f"Error of marginal = {err:1.3e}")

    # conditional pdf on a smaller domain
    d2 = (-2.0, 2.0, -2.0, 2.0)
    fy = cj.chebfun2(
        lambda x, y: (1 / (2 * jnp.pi * s1 * s2
                           * jnp.sqrt(1 - rho**2))
                      * jnp.exp(-z(x, y) / (2 * (1 - rho**2))))
        / (1 / (jnp.sqrt(2 * jnp.pi) * s1)
           * jnp.exp(-(x - mu1)**2 / s1**2 / 2)), domain=d2)
    xs2 = np.linspace(-2, 2, 200)
    X2, Y2 = np.meshgrid(xs2, xs2)
    Z2 = np.asarray(fy(jnp.asarray(X2), jnp.asarray(Y2)))
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X2, Y2, Z2, cmap="viridis")
    ax.set_title("Conditional pdf", fontsize=13)
    _save(fig)

    x0 = np.pi / 6
    mu = mu1 + s1 / s2 * rho * (x0 - mu2)
    sigmasq = (1 - rho**2) * s1**2
    exact_c = (lambda y: 1 / np.sqrt(2 * np.pi * sigmasq)
               * np.exp(-(y - mu)**2 / sigmasq / 2))
    yq = 2 * gx
    dv = (np.asarray(fy(np.full_like(yq, x0), yq)).ravel()
          - exact_c(yq))
    errc = np.sqrt(np.sum(gw * dv**2) * 2)
    print(f"Error in conditional pdf is {errc:1.5e}")


if __name__ == "__main__":
    run()
