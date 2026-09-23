"""A Bayesian gradebook.

Translation of stats/BayesianGradebook.m by Toby Driscoll
(August 2014): tracking belief about a student's ability theta via
Bayesian updates of a prior on [0,1], compared with the traditional
running average.

Original: https://www.chebfun.org/examples/stats/BayesianGradebook.html
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
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]
THETA = cj.chebfun(lambda t: t, domain=(0.0, 1.0))


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"BayesianGradebook_{FIG[0]:02d}.png"))
    plt.close(fig)


def E(f, prob):
    return float((f * prob).sum())


def Var(f, prob):
    mu = E(f, prob)
    return float((((f - mu) ** 2) * prob).sum())


def phi(mu, sigma):
    return cj.chebfun(
        lambda t: jnp.exp(-((t - mu) / sigma) ** 2 / 2),
        domain=(0.0, 1.0))


def qfun(sigma):
    """q = chebfun(@(theta) sum(phi(theta, sigma)), [0 1], 'vectorize')."""
    return cj.chebfun(
        lambda t: jnp.asarray([float(phi(float(v), sigma).sum())
                               for v in np.atleast_1d(t)]),
        domain=(0.0, 1.0))


def _row_long(v):
    """MATLAB format-long display of a 4-vector (80-column wrap)."""
    print("  Columns 1 through 3")
    print("".join(f"{t:20.15f}" for t in v[:3]))
    print("  Column 4")
    print(f"{v[3]:20.15f}")


def bayes(scores, prior, sigma, q):
    belief = [prior]
    m = len(scores)
    trad = np.cumsum(scores) / np.arange(1, m + 1)
    Mu, Sig2, Mode = [], [], []
    for k in range(m):
        lik = phi(scores[k], sigma) / q
        b = belief[-1] * lik
        b = b * (1.0 / float(b.sum()))
        belief.append(b)
        Mu.append(E(THETA, b))
        Sig2.append(Var(THETA, b))
        pos, _ = b.max()
        Mode.append(float(pos))
    print(f"Method       {'m-3':>6} {'m-2':>6} {'m-1':>6} {'m':>6}")
    print("-" * 48)
    print("Traditional   " + " ".join(f"{v:6.3f}"
                                      for v in trad[m - 4:m]))
    print("Bayes Mode    " + " ".join(f"{v:6.3f}"
                                      for v in Mode[m - 4:m]))
    print("Bayes Mean    " + " ".join(f"{v:6.3f}"
                                      for v in Mu[m - 4:m]))
    print("Std dev       " + " ".join(f"{np.sqrt(v):6.3f}"
                                      for v in Sig2[m - 4:m]))
    return belief


def _plot_belief(belief):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(belief, ax=ax, lw=2)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$P(\theta|x)$")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"BayesianGradebook_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    prior = phi(0.7, 0.3)
    prior = prior * (1.0 / float(prior.sum()))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(prior, ax=ax, lw=2)
    _save(fig)

    sigma = 0.06
    q = qfun(sigma)

    scores = np.array([0.55, 0.67, 0.62, 0.66])
    belief = bayes(scores, prior, sigma, q)
    _plot_belief(belief)

    scores2 = 0.3 + scores
    print("scores =")
    _row_long(scores2)
    belief = bayes(scores2, prior, sigma, q)
    _plot_belief(belief)

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(1 / q, ax=ax, lw=2)
    _save(fig)

    scores3 = scores2.copy()
    scores3[0] = 0.72
    print("scores =")
    _row_long(scores3)
    bayes(scores3, prior, sigma, q)

    sigma = 0.15
    q = qfun(sigma)
    scores4 = np.array([0.88, 0.90, 0.46, 0.86, 0.93,
                        0.61, 0.95, 0.89, 0.84, 0.76])
    bayes(scores4, prior, sigma, q)


if __name__ == "__main__":
    run()
