"""Pricing of a European Call option.

Faithful replica of applics/EuropeanCall.m (Pachon, 2014): the price
of a European call as the risk-neutral expectation of the discounted
payoff, computed with chebfuns of the lognormal density on [0, 10000]
-- PDF evolution, the OOM probability from the CDF, the payoff
distribution (Dirac at 0 with the OOM weight + shifted ITM density),
and the comparison against the Black-Scholes formula.

Original: https://www.chebfun.org/examples/applics/EuropeanCall.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')

K = 60
T = 0.5
RHSPLOT = 200
RHS = 10000
MU = 0.075
VOL = 0.45
S0 = 100
R = 0.01


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"EuropeanCall_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def logn(S, t, drift):
    S = np.asarray(S, dtype=float)
    return (np.exp(-(np.log(S / S0) - (drift - 0.5 * VOL**2) * t)**2
                   / (2 * VOL**2 * t))
            / (VOL * S * np.sqrt(2 * np.pi * t)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # The call payoff.
    Ss = np.linspace(0, RHSPLOT, 800)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(Ss, np.maximum(0, Ss - K), 'k', lw=1.6)
    ax.set_xlabel("S", fontsize=14)
    ax.set_ylabel("V(S)", fontsize=14)
    ax.set_ylim(-20, 140)
    ax.grid(True)
    _save(fig, 1)

    # Lognormal PDFs under the original measure at t = 0.05..1.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    tvec = np.arange(0.05, 1.0001, 0.05)
    Sp = np.linspace(1e-8, RHSPLOT, 1200)
    for t in tvec:
        if np.isclose(t, tvec[0]):
            col, lw = 'r', 2
        elif np.isclose(t, tvec[-1]):
            col, lw = 'b', 2
        else:
            col, lw = 'k', 1
        ax.plot(Sp, logn(Sp, t, MU), col, lw=lw)
    ax.set_ylim(0, 0.05)
    ax.set_yticks(np.arange(0, 0.051, 0.01))
    ax.set_xlabel("S_t", fontsize=14)
    ax.grid(True)
    _save(fig, 2)

    lognPDF = chebfun(lambda S: logn(np.maximum(S, 1e-300), 1.0, MU),
                      domain=(1e-12, RHS))
    print("ans =")
    print(f"   {float(lognPDF.sum()):.15f}")

    # Risk-neutral measure at t = T.
    lognT = chebfun(lambda S: logn(np.maximum(S, 1e-300), T, R),
                    domain=(1e-12, RHS))

    # Moneyness picture.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pdfp = logn(Sp, T, R)
    ax.fill_between(Sp[Sp <= K], pdfp[Sp <= K], color=(0.9, 0.3, 0.4))
    ax.fill_between(Sp[Sp >= K], pdfp[Sp >= K], color=(0.3, 0.9, 0.4))
    ax.plot(Sp, pdfp, 'k', lw=1.6)
    ax.plot(Ss, 1e-4 * np.maximum(0, Ss - K), 'b--', lw=1.6)
    ax.set_ylim(-0.001, 0.015)
    ax.set_yticks([])
    ax.set_xlabel("S_T", fontsize=14)
    _save(fig, 3)

    # Probability of expiring out-of-the-money, from the CDF.
    lognCDF = lognT.cumsum()
    probOOM = float(lognCDF(np.array([float(K)]))[0])
    print("probOOM =")
    print(f"   {probOOM:.15f}")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    cdfp = np.asarray(lognCDF(Sp))
    ax.plot(Sp, cdfp, lw=1.6)
    ax.plot([K, K], [0, probOOM], 'r--', lw=1.6)
    ax.plot([0, K], [probOOM, probOOM], 'r--', lw=1.6)
    ax.set_ylim(-.1, 1.1)
    ax.set_xlim(0, RHSPLOT)
    ax.set_xlabel("S_T", fontsize=14)
    ax.grid(True)
    _save(fig, 4)

    # PDF of the discounted payoff: Dirac at 0 (weight probOOM) + the
    # shifted ITM density.
    ITM = chebfun(lambda y: logn(np.maximum(y + K, 1e-300), T, R),
                  domain=(0, RHS))
    disc = np.exp(-R * T)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    yv = np.linspace(0, RHSPLOT, 1000)
    pv = disc * np.asarray(ITM(yv))
    ax.fill_between(yv, pv, color=(0.3, 0.9, 0.4))
    ax.plot(yv, pv, 'k', lw=1.6)
    ax.plot([0, 0], [0, 0.014], 'r', lw=2)  # delta line at 0
    ax.set_xlim(-10, RHSPLOT)
    ax.set_xlabel(r"$e^{-rT}V(S_T)$", fontsize=14)
    ax.grid(True)
    _save(fig, 5)

    # Total (undiscounted) mass: delta weight + ITM integral.
    total = probOOM + float(ITM.sum())
    print("ans =")
    print(f"   {total:.15f}")

    # Expected value = call price.
    x = chebfun(lambda y: y, domain=(0, RHS))
    approx = disc * float((x * ITM).sum())
    print(f"approx = {approx:.15f}")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.fill_between(yv, pv, color=(0.3, 0.9, 0.4))
    ax.plot(yv, pv, 'k', lw=1.6)
    ax.plot([approx, approx], [0, 0.04], 'b--', lw=1.6)
    ax.set_xlim(-10, RHSPLOT)
    ax.set_xlabel(r"$e^{-rT}V(S_T)$", fontsize=14)
    ax.grid(True)
    _save(fig, 6)

    # Black-Scholes formula.
    d1 = (np.log(S0 / K) + (R + 0.5 * VOL**2) * T) / (VOL * np.sqrt(T))
    d2 = d1 - VOL * np.sqrt(T)
    exact = norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-R * T)
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")


if __name__ == "__main__":
    run()
