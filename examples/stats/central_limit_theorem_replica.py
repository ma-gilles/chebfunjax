"""Central limit theorem.

Faithful replica of stats/CentralLimitTheorem.m by Nick Trefethen
(June 2015): repeated convolution of a triangular distribution
converges to a Gaussian, and repeated convolution of a Bernoulli
delta-train gives the binomial distribution.

Original: https://www.chebfun.org/examples/stats/CentralLimitTheorem.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

AX = (-3, 3, -0.2, 1.2)
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"CentralLimitTheorem_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pw(ax, f, scale_dom=None, **kw):
    bps = [float(v) for v in f.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 200)
        ax.plot(t, np.asarray(f(t)), 'b', **kw)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    X = cj.chebfun(lambda x: 0.0 * x, domain=(-3.0, -4 / 3)).join(
        cj.chebfun(lambda x: (4 / 3 + x) / 2, domain=(-4 / 3, 2 / 3)),
        cj.chebfun(lambda x: 0.0 * x, domain=(2 / 3, 3.0)))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot_pw(ax, X, lw=1.6)
    ax.axis(AX)
    ax.grid(True)
    ax.set_title("Distribution of X", fontsize=12)
    _save(fig)

    t = cj.chebfun(lambda s: s, domain=(-3.0, 3.0))
    mu = float((t * X).sum())
    variance = float((t**2 * X).sum())
    print("mu =")
    print(f"    {mu:.15e}")
    print("variance =")
    print(f"   {variance:.15f}")
    sigma = np.sqrt(variance)

    def gauss_vals(s, sig):
        return np.exp(-0.5 * (s / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    xs = np.linspace(-3, 3, 600)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot_pw(ax, X, lw=1.6)
    ax.plot(xs, gauss_vals(xs, sigma), 'r', lw=1.4)
    ax.axis(AX)
    ax.grid(True)
    ax.set_title("Distribution of X compared with normal distribution",
                 fontsize=12)
    _save(fig)

    X2 = X.conv(X)
    S2 = (X2 * float(np.sqrt(2))).new_domain(
        (-3 * np.sqrt(2), 3 * np.sqrt(2)))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot_pw(ax, S2, lw=1.6)
    ax.plot(xs, gauss_vals(xs, sigma), 'r', lw=1.4)
    ax.axis(AX)
    ax.grid(True)
    ax.set_title("Renormalized distribution of (X+X)/2", fontsize=12)
    _save(fig)

    X3 = X2.conv(X)
    S3 = (X3 * float(np.sqrt(3))).new_domain(
        (-3 * np.sqrt(3), 3 * np.sqrt(3)))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    _plot_pw(ax, S3, lw=1.6)
    ax.plot(xs, gauss_vals(xs, sigma), 'r', lw=1.4)
    ax.axis(AX)
    ax.grid(True)
    ax.set_title("Renormalized distribution of (X+X+X)/3",
                 fontsize=12)
    _save(fig)

    # coin tosses: Bernoulli delta train -> binomial
    x = cj.chebfun(lambda s: s, domain=(-1.0, 2.0))
    p, q = 0.6, 0.4
    p1 = cj.dirac(x - 0) * q + cj.dirac(x - 1) * p

    def _plot_deltas(f, xlim, title):
        FIG[0] += 1
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        for loc, mag in getattr(f, "deltas", ()):
            ax.plot([loc, loc], [0, mag], 'b', lw=2)
            ax.plot(loc, mag, '^b', ms=7)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, None)
        ax.grid(True)
        ax.set_title(title, fontsize=11)
        fig.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(os.path.join(
            _IMG, f"CentralLimitTheorem_repl_{FIG[0]:02d}.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_deltas(p1, (-0.1, 1.1),
                 "Probability distribution for getting a head "
                 "in a single toss")
    p2 = p1.conv(p1)
    _plot_deltas(p2, (-0.1, 2.1),
                 "Probability distribution for number of heads "
                 "in two tosses")
    print("ans =")
    print(f"     {float(p2.sum()):g}")

    n = 10
    pn = p2
    for _k in range(3, n + 1):
        pn = pn.conv(p1)
    _plot_deltas(pn, (-0.5, 10.5), "The binomial distribution")
    print("ans =")
    print(f"   {float(pn.sum()):.15f}")
    mu_b = n * p
    sigma_b = np.sqrt(n * p * q)
    print("mu =")
    print(f"     {mu_b:g}")
    print("sigma =")
    print(f"   {sigma_b:.15f}")

    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for loc, mag in getattr(pn, "deltas", ()):
        ax.plot([loc, loc], [0, mag], 'b', lw=2)
        ax.plot(loc, mag, '^b', ms=7)
    xg = np.linspace(-1, 11, 500)
    ax.plot(xg, gauss_vals(xg - mu_b, sigma_b), 'r', lw=1.4)
    ax.grid(True)
    ax.set_xlabel(f"{n} Tosses, p = {p:2.1f}, "
                  f"expected value = {int(n*p)}")
    ax.set_title("The binomial distribution compared with the "
                 "normal distribution", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"CentralLimitTheorem_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
