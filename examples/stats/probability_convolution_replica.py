"""Convolutions of probability distributions.

Faithful replica of stats/ProbabilityConvolution.m by Nick Hale
(December 2012): sums of independent random variables have densities
given by convolutions — normal*normal, gamma*gamma, exp*exp, each
verified against the closed-form result, plus a piecewise example
with Heaviside steps.

Original: https://www.chebfun.org/examples/stats/ProbabilityConvolution.html
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
from scipy.special import gamma as _gamma

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
        _IMG, f"ProbabilityConvolution_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pw(ax, f, color, lw=1.6):
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 300)
        ax.plot(t, np.asarray(f(t)), color, lw=lw)


def _norm2_diff(f, g, a, b):
    """L2 norm of (f - g) on [a, b] via fine Gauss quadrature."""
    xg, wg = np.polynomial.legendre.leggauss(2000)
    xq = a + (b - a) * (xg + 1) / 2
    d = np.asarray(f(xq)) - np.asarray(g(xq))
    return np.sqrt(np.sum(wg * d**2) * (b - a) / 2)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    def normal(mu, s, dom):
        return cj.chebfun(
            lambda x: jnp.exp(-0.5 * (x - mu)**2 / s**2)
            / (s * jnp.sqrt(2 * jnp.pi)), domain=dom)

    dom = (-1.2, 1.2)
    s1, m1 = 0.1, 0.1
    N1 = normal(m1, s1, dom)
    s2, m2 = 0.11, -0.3
    N2 = normal(m2, s2, dom)
    N3 = N1.conv(N2)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _plot_pw(ax, N1, 'b')
    _plot_pw(ax, N2, 'r')
    _plot_pw(ax, N3, 'k')
    ax.set_xlim(-1.5, 1.5)
    _save(fig)
    N4 = normal(m1 + m2, np.sqrt(s1**2 + s2**2), dom)
    print("ans =")
    print(f"     {_norm2_diff(N4, N3, *dom):.15e}")

    def gamma_dist(k, t, dom):
        return cj.chebfun(
            lambda x: x**(k - 1) * jnp.exp(-x / t)
            / (t**k * _gamma(k)), domain=dom)

    dom = (0.0, 5.0)
    k1, t = 2, 0.3
    G1 = gamma_dist(k1, t, dom)
    k2 = 1
    G2 = gamma_dist(k2, t, dom)
    G3 = G1.conv(G2)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _plot_pw(ax, G1, 'b')
    _plot_pw(ax, G2, 'r')
    _plot_pw(ax, G3, 'k')
    ax.set_xlim(0, 5)
    _save(fig)
    G4 = gamma_dist(k1 + k2, t, dom)
    print("ans =")
    print(f"     {_norm2_diff(G4, G3, *dom):.15e}")

    dom = (0.0, 40.0)
    lam = 0.25
    E = cj.chebfun(lambda x: lam * jnp.exp(-lam * x), domain=dom)
    E2 = E.conv(E)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _plot_pw(ax, E, 'b')
    _plot_pw(ax, E2, 'k')
    ax.set_xlim(0, 40)
    _save(fig)
    E3 = gamma_dist(2, 1 / lam, dom)
    print("ans =")
    print(f"     {_norm2_diff(E3, E2, *dom):.15e}")

    # piecewise: mixtures on top of Heaviside steps
    rs = np.random.RandomState(5489)
    dom = (-2.0, 2.0)

    def heaviside_at(c, up=True):
        lo = cj.chebfun(lambda x: 0.0 * x + (0.0 if up else 1.0),
                        domain=(dom[0], c))
        hi = cj.chebfun(lambda x: 0.0 * x + (1.0 if up else 0.0),
                        domain=(c, dom[1]))
        return lo.join(hi)

    F = heaviside_at(0.0, up=True)
    G = heaviside_at(0.5, up=False)
    for _k in range(10):
        F = F + normal(rs.randn(), 2 * rs.rand(), dom)
        G = G + normal(rs.randn(), 2 * rs.rand(), dom)
    F = F * (1.0 / float(F.sum()))
    G = G * (1.0 / float(G.sum()))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _plot_pw(ax, F, 'b')
    _plot_pw(ax, G, 'r')
    ax.set_xlim(-4, 4)
    _save(fig)

    t0 = time.time()
    h = F.conv(G)
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _plot_pw(ax, h, 'k')
    ax.set_xlim(-4, 4)
    _save(fig)
    print(f"sum h = {float(h.sum()):.15f}")


if __name__ == "__main__":
    run()
