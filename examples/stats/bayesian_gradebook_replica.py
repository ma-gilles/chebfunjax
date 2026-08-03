"""A Bayesian gradebook.

Faithful replica of stats/BayesianGradebook.m by Toby Driscoll
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
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

FIG = [0]
THETA = cj.chebfun(lambda t: t, domain=(0.0, 1.0))


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"BayesianGradebook_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
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
    """q(theta) = int_0^1 phi(x; theta, sigma) dx, closed form."""
    def q(t):
        t = np.asarray(t, dtype=float)
        return (sigma * np.sqrt(np.pi / 2)
                * (erf((1 - t) / (sigma * np.sqrt(2)))
                   + erf(t / (sigma * np.sqrt(2)))))
    return cj.chebfun(lambda t: jnp.asarray(q(np.asarray(t))),
                      domain=(0.0, 1.0))


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
    xs = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for b in belief:
        ax.plot(xs, np.asarray(b(xs)), lw=1.6)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$P(\theta|x)$")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"BayesianGradebook_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    prior = phi(0.7, 0.3)
    prior = prior * (1.0 / float(prior.sum()))
    xs = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(prior(xs)), lw=2)
    ax.grid(True)
    _save(fig)

    sigma = 0.06
    q = qfun(sigma)

    scores = np.array([0.55, 0.67, 0.62, 0.66])
    belief = bayes(scores, prior, sigma, q)
    _plot_belief(belief)

    scores2 = 0.3 + scores
    print("scores =")
    print("  " + "  ".join(f"{v:.15f}" for v in scores2))
    belief = bayes(scores2, prior, sigma, q)
    _plot_belief(belief)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, 1.0 / np.asarray(q(xs)), lw=2)
    ax.grid(True)
    _save(fig)

    scores3 = scores2.copy()
    scores3[0] = 0.72
    print("scores =")
    print("  " + "  ".join(f"{v:.15f}" for v in scores3))
    bayes(scores3, prior, sigma, q)

    sigma = 0.15
    q = qfun(sigma)
    scores4 = np.array([0.88, 0.90, 0.46, 0.86, 0.93,
                        0.61, 0.95, 0.89, 0.84, 0.76])
    bayes(scores4, prior, sigma, q)


if __name__ == "__main__":
    run()
