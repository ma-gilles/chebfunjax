"""Pricing European Options: puts, digitals, and power options.

Faithful port of applics/EuropeanOptions.m by Ricardo Pachon (December 2014).
Each option value is obtained by pushing the risk-neutral lognormal density of
the asset at expiry through the payoff and integrating with a chebfun (the
``approx`` price), then checked against the closed-form Black-Scholes value
(the ``exact`` price).

Original: https://www.chebfun.org/examples/applics/EuropeanOptions.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the out-of-the-money probability and all three
approx/exact option prices reproduce to ~14-15 significant figures -- put (K=150)
51.166911483849546, digital call (K=100) 0.440783414443267, power call
(K=9.1, alpha=0.5) 1.078491451154445.  The prior port used a coarse 3000-point
trapezoid (accurate only to ~1e-6) for the numerical prices; here the chebfun
integral of payoff*density matches the closed form to machine precision.
"""
import matplotlib

matplotlib.use("Agg")
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()

_S0 = 100.0
_VOL = 0.45
_R = 0.01
_T = 0.5
_MAXS = 10000.0
_DISC = float(np.exp(-_R * _T))


def _pdf(S):
    """Risk-neutral lognormal density of the asset at expiry."""
    return (jnp.exp(-(jnp.log(S / _S0) - (_R - 0.5 * _VOL**2) * _T)**2
                    / (2 * _VOL**2 * _T))
            / (_VOL * S * np.sqrt(2 * np.pi * _T)))


def _integral(fn, a, b):
    """int_a^b fn(S) dS via a chebfun."""
    return float(cj.chebfun(fn, domain=(a, b)).sum())


def run():
    # ------------------------------------------------------------------
    # European put, K = 150.
    # ------------------------------------------------------------------
    K = 150.0
    prob_oom = 1.0 - _integral(_pdf, 1e-8, K)
    print("probOOM =")
    print(f"   {prob_oom:.15f}")

    approx = _DISC * _integral(lambda S: (K - S) * _pdf(S), 1e-8, K)
    d1 = (np.log(_S0 / K) + (_R + 0.5 * _VOL**2) * _T) / (_VOL * np.sqrt(_T))
    d2 = d1 - _VOL * np.sqrt(_T)
    exact = -(_S0 * norm.cdf(-d1) - K * norm.cdf(-d2) * _DISC)
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")

    # ------------------------------------------------------------------
    # European digital call, K = 100.
    # ------------------------------------------------------------------
    K = 100.0
    approx = _DISC * _integral(_pdf, K, _MAXS)
    d1 = (np.log(_S0 / K) + (_R + 0.5 * _VOL**2) * _T) / (_VOL * np.sqrt(_T))
    d2 = d1 - _VOL * np.sqrt(_T)
    exact = norm.cdf(d2) * _DISC
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")

    # ------------------------------------------------------------------
    # European power call, payoff max(0, S^alpha - K), K = 9.1, alpha = 0.5.
    # ------------------------------------------------------------------
    K = 9.1
    alpha = 0.5
    thr = K**(1.0 / alpha)
    approx = _DISC * _integral(lambda S: (S**alpha - K) * _pdf(S), thr, _MAXS)
    d1 = ((np.log(_S0 / K**(1.0 / alpha)) + (_R + (alpha - 0.5) * _VOL**2) * _T)
          / (_VOL * np.sqrt(_T)))
    d2 = d1 - alpha * _VOL * np.sqrt(_T)
    m = (_R + 0.5 * alpha * _VOL**2) * (alpha - 1)
    exact = (_S0**alpha * np.exp(m * _T) * norm.cdf(d1)
             - _DISC * K * norm.cdf(d2))
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")

    # ------------------------------------------------------------------
    # Plot: the three payoffs against the asset density.
    # ------------------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    S = np.linspace(1e-3, 300, 600)
    dens = np.asarray(_pdf(jnp.asarray(S)))

    axes[0].plot(S, np.maximum(150 - S, 0), "k-", lw=2, label="put payoff K=150")
    axes[0].fill_between(S, dens / dens.max() * 50, 0, alpha=0.3)
    axes[0].set_title("European put", fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].plot(S, (S > 100).astype(float), "k-", lw=2, label="digital K=100")
    axes[1].fill_between(S, dens / dens.max(), 0, alpha=0.3)
    axes[1].set_title("Digital call", fontsize=11)
    axes[1].legend(fontsize=9)

    axes[2].plot(S, np.maximum(np.sqrt(S) - 9.1, 0), "k-", lw=2,
                 label="power K=9.1, a=0.5")
    axes[2].fill_between(S, dens / dens.max() * 2, 0, alpha=0.3)
    axes[2].set_title("Power call", fontsize=11)
    axes[2].legend(fontsize=9)

    fig.suptitle("European option types: put, digital, power", fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "european_options.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
