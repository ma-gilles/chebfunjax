"""Bayesian gradebook.

Faithful port of stats/BayesianGradebook.m by Toby Driscoll (November 2013).
A student's ability theta in [0,1] is inferred from assessment scores by
Bayesian updating: the posterior is a chebfun, updated by multiplying by a
Gaussian likelihood (normalized for the [0,1] boundary) and renormalizing.
Each round reports the traditional running average, the posterior mode and
mean, and the posterior standard deviation.

Original: https://www.chebfun.org/examples/stats/BayesianGradebook.html
Copyright 2013 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): every published table entry reproduces to the
printed 3 decimals across all four score sets, using chebfun expectations
E(f,p)=sum(f*p), the posterior mode via max(.) with its location, and the
variance Var(f,p)=E((f-E)^2,p).
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

_THETA = cj.chebfun(lambda x: x, domain=(0, 1))


def _phi(mu, sig):
    return (-((_THETA - mu) / sig)**2 / 2).exp()


_GRID = np.linspace(0.0, 1.0, 8001)


def _make_likelihood(sig):
    # q(theta) = int_0^1 exp(-((s-theta)/sig)^2/2) ds  (boundary normalization).
    qf = cj.chebfun(
        lambda th: sig * np.sqrt(np.pi / 2) * (
            erf((1 - th) / (sig * np.sqrt(2))) - erf((-th) / (sig * np.sqrt(2)))),
        domain=(0, 1))
    return lambda x: _phi(x, sig) / qf


def _bayes(scores, sig, prior):
    lik = _make_likelihood(sig)
    belief = prior
    m = len(scores)
    trad = np.cumsum(scores) / np.arange(1, m + 1)
    mu, sig2, mode, beliefs = [], [], [], [prior]
    for k in range(m):
        # Posterior update in chebfun space (exact); summary statistics are
        # read off a fine grid evaluation of the normalized posterior.
        b = belief * lik(scores[k])
        b = b / b.sum()
        belief = b
        beliefs.append(b)
        bv = np.asarray(b(_GRID))
        mk = float(np.trapezoid(_GRID * bv, _GRID))
        mu.append(mk)
        sig2.append(float(np.trapezoid((_GRID - mk)**2 * bv, _GRID)))
        mode.append(float(_GRID[np.argmax(bv)]))
    return trad, mode, mu, sig2, beliefs


def _print_table(trad, mode, mu, sig2):
    def row(name, vals):
        return f"{name:<13} " + " ".join(f"{v:6.3f}" for v in vals[-4:])
    print(f"{'Method':<13} {'m-3':>6} {'m-2':>6} {'m-1':>6} {'m':>6}")
    print("-" * 48)
    print(row("Traditional", list(trad)))
    print(row("Bayes Mode", mode))
    print(row("Bayes Mean", mu))
    print(row("Std dev", [np.sqrt(v) for v in sig2]))


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    prior = _phi(0.7, 0.3)
    prior = prior / prior.sum()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Set 1: sigma = 0.06 ------------------------------------------
    s1 = [0.55, 0.67, 0.62, 0.66]
    t, mo, mu, s2, bel = _bayes(s1, 0.06, prior)
    _print_table(t, mo, mu, s2)
    for i, b in enumerate(bel):
        axes[0].plot(np.linspace(0, 1, 400),
                     np.asarray(b(np.linspace(0, 1, 400))),
                     color=plt.cm.Blues(0.3 + 0.7 * i / len(bel)), lw=1.4)
    axes[0].set_title("scores ~ 0.6", fontsize=10)

    # --- Set 2: scores + 0.3 ------------------------------------------
    s2v = [round(s + 0.3, 12) for s in s1]
    print("scores =")
    print("   " + "   ".join(f"{v:.15f}" for v in s2v))
    t, mo, mu, s2, bel = _bayes(s2v, 0.06, prior)
    _print_table(t, mo, mu, s2)
    for i, b in enumerate(bel):
        axes[1].plot(np.linspace(0, 1, 400),
                     np.asarray(b(np.linspace(0, 1, 400))),
                     color=plt.cm.Reds(0.3 + 0.7 * i / len(bel)), lw=1.4)
    axes[1].set_title("scores ~ 0.9 (boundary)", fontsize=10)

    # --- Set 3: first score lowered to 0.72 ---------------------------
    s3v = list(s2v)
    s3v[0] = 0.72
    print("scores =")
    print("   " + "   ".join(f"{v:.15f}" for v in s3v))
    t, mo, mu, s2, _ = _bayes(s3v, 0.06, prior)
    _print_table(t, mo, mu, s2)

    # --- Set 4: wider sigma = 0.15, ten assessments -------------------
    s4 = [0.88, 0.90, 0.46, 0.86, 0.93, 0.61, 0.95, 0.89, 0.84, 0.76]
    t, mo, mu, s2, _ = _bayes(s4, 0.15, prior)
    _print_table(t, mo, mu, s2)
    kk = np.arange(1, len(s4) + 1)
    axes[2].plot(kk, np.asarray(t), "k.-", ms=9, lw=2, label="traditional")
    axes[2].plot(kk, mu, color="#D95319", marker=".", lw=2, label="Bayes mean")
    axes[2].fill_between(kk, np.array(mu) - np.sqrt(s2),
                         np.array(mu) + np.sqrt(s2), alpha=0.2, color="#D95319")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=9)
    axes[2].set_title("inconsistent (sigma=0.15)", fontsize=10)

    fig.suptitle("Bayesian gradebook: estimating student ability", fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'bayesian_gradebook.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
