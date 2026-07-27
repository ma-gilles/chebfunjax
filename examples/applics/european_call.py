"""Pricing of a European Call option.

Prices a European call by building the risk-neutral lognormal density as a
Chebfun and integrating it against the payoff -- spectral quadrature, not
Monte Carlo.  Faithful port of applics/EuropeanCall.m by Ricardo Pachon
(November 2014).

Original MATLAB: https://www.chebfun.org/examples/applics/EuropeanCall.html
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    # MATLAB: S0=100; K=60; T=0.5; vol=0.45; r=0.01; RHS=10000;
    S0, K, T, vol, r, RHS = 100.0, 60.0, 0.5, 0.45, 0.01, 10000.0

    # Risk-neutral lognormal density (chebfun on [0, RHS]).
    def lognHnd(S):
        return jnp.exp(-(jnp.log(S / S0) - (r - 0.5 * vol**2) * T)**2
                       / (2 * vol**2 * T)) / (vol * S * jnp.sqrt(2 * np.pi * T))

    lognPDF = cj.chebfun(lognHnd, domain=(0.0, RHS))
    print(f"sum(lognPDF) = {float(lognPDF.sum()):.15f}")

    # probOOM = lognCDF(K): probability of expiring out-of-the-money.
    lognCDF = lognPDF.cumsum()
    probOOM = float(lognCDF(jnp.array(K)))
    print(f"probOOM = {probOOM:.15f}")

    # Payoff PDF: the in-the-money density shifted by K, plus a Dirac mass of
    # weight probOOM at S=0 for the OOM outcome.  Total probability is 1.
    # MATLAB: OOM = 2*probOOM*dirac(x); payoffPDF = exp(-r*T)*(OOM+ITM); the
    # boundary Dirac integrates to probOOM.  (chebfunjax conv/sum do not
    # propagate Dirac deltas, so we add the OOM mass explicitly.)
    ITM = cj.chebfun(lambda S: lognHnd(S + K), domain=(0.0, RHS))
    x = cj.chebfun(lambda x: x, domain=(0.0, RHS))
    total_prob = probOOM + float(ITM.sum())
    print(f"sum(OOM + ITM) = {total_prob:.15f}")

    # Option price = exp(-r*T) * sum(x .* payoffPDF).  The OOM Dirac at x=0
    # contributes x*dirac(x)=0, so only the ITM part enters.
    approx = float(np.exp(-r * T) * (x * ITM).sum())
    print(f"approx = {approx:.15f}")

    # Black-Scholes closed form.
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    exact = S0 * norm.cdf(d1) - K * norm.cdf(d2) * np.exp(-r * T)
    print(f"exact = {exact:.15f}")

    # --- Plot ------------------------------------------------------------
    Sp = np.linspace(1e-3, 300.0, 600)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(Sp, np.asarray(lognPDF(jnp.array(Sp))), color='#0072BD', lw=2)
    axes[0].axvline(K, color='#77AC30', ls='--', label=f'K={K:.0f}')
    axes[0].set_title('Risk-neutral density', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[1].plot(Sp, np.maximum(Sp - K, 0.0), 'k-', lw=2)
    axes[1].axvline(K, color='#77AC30', ls='--')
    axes[1].set_title('Call payoff max(S-K,0)', fontsize=11)
    fig.suptitle(f'European call (S0={S0:.0f}, sigma={vol}, T={T})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "european_call.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("european_call: done")
    return True


if __name__ == "__main__":
    run()
