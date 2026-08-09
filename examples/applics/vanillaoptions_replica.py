"""Exploring Vanilla Options.

Faithful replica of applics/VanillaOptions.m (Pachon, 2014):
Black-Scholes call/put prices as chebfuns of the underlying --
price profiles vs maturity, put-call parity hedging strategies with
their maximum instant losses, implicit/time value, non-zero rates,
the chebfun2 price surface, and the early-exercise boundary from
the roots of a chebfun2.

Original: https://www.chebfun.org/examples/applics/VanillaOptions.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')
FIG = [0]

K = 100
VOL = 0.45


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"VanillaOptions_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def vanilla(S, Kk, T, vol, r, W):
    S = np.maximum(np.asarray(S, dtype=float), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / Kk) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = (np.log(S / Kk) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        return W * (S * norm.cdf(W * d1)
                    - Kk * norm.cdf(W * d2) * np.exp(-r * T))


def payoff(S, Kk, W):
    return np.maximum(0, W * (np.asarray(S, dtype=float) - Kk))


def _minloss(err_fn, lo=0.0, hi=300.0, n=3000):
    ss = np.linspace(lo, hi, n)
    vals = err_fn(ss)
    i = int(np.argmin(vals))
    res = minimize_scalar(err_fn,
                          bounds=(max(lo, ss[i] - 1), min(hi, ss[i] + 1)),
                          method="bounded",
                          options={"xatol": 1e-10})
    return float(res.fun), float(res.x)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ---- zero interest rate ----
    r = 0.0
    Ss = np.linspace(1e-8, 350, 2000)

    fig, (axp, axc) = plt.subplots(1, 2, figsize=(11.6, 5.2))
    axp.plot(Ss, payoff(Ss, K, -1), lw=1.6)
    axc.plot(Ss, payoff(Ss, K, +1), lw=1.6)
    for T in 2.0 ** np.arange(-1, 9):
        axp.plot(Ss, vanilla(Ss, K, T, VOL, r, -1), 'k', lw=0.9)
        axc.plot(Ss, vanilla(Ss, K, T, VOL, r, +1), 'k', lw=0.9)
    axp.plot(Ss, vanilla(Ss, K, 1000, VOL, r, -1), 'r', lw=1.2)
    axc.plot(Ss, vanilla(Ss, K, 1000, VOL, r, +1), 'r', lw=1.2)
    for ax, tt in [(axp, 'put'), (axc, 'call')]:
        ax.set_xticks(np.arange(0, 351, 50))
        ax.set_title(tt, fontsize=18)
        ax.set_xlabel('S', fontsize=14)
        ax.set_xlim(0, 250)
        ax.set_ylim(-10, 250)
        ax.grid(True)
    _save(fig)

    # ---- put-call parity hedges, r = 0 ----
    Sh = np.linspace(1e-8, 300, 3000)

    def call(s):
        return vanilla(s, 100, .5, VOL, 0.0, +1)

    def err2(s):
        return (vanilla(s, 105.5, .25, VOL, 0.0, -1) + s - K) - call(s)

    def err3(s):
        return (1.03 * vanilla(s, 94.5, .75, VOL, 0.0, -1)
                + s - K) - call(s)

    def err1(s):
        return (vanilla(s, 100.0, .50, VOL, 0.0, -1) + s - K) - call(s)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(Sh, err1(Sh), 'b', lw=1.6, label='1.00 put/same K/same T')
    ax.plot(Sh, err2(Sh), 'r', lw=1.6,
            label='1.00 put/higher K/ shorter T')
    ax.plot(Sh, err3(Sh), 'k', lw=1.6,
            label='1.03 put/lower K/ longer T')
    ax.set_ylim(-6, 8)
    ax.set_title('Imperfect put-call parity relation, r=0')
    ax.set_xlabel('S', fontsize=14)
    ax.set_ylabel('profit/loss (instant)', fontsize=14)
    ax.legend()
    ax.grid(True)
    _save(fig)

    ls2, as2 = _minloss(err2)
    ls3, as3 = _minloss(err3)
    print(f"Max loss stgy 2: {ls2:.4f} at {as2:.4f}")
    print(f"Max loss stgy 3: {ls3:.4f} at {as3:.4f}")

    # ---- implicit and time value, r = 0 ----
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(11.6, 5.2))
    for T in 2.0 ** np.arange(-1, 9):
        axp.plot(Ss, vanilla(Ss, K, T, VOL, 0.0, -1)
                 - payoff(Ss, K, -1), 'k', lw=0.9)
        axc.plot(Ss, vanilla(Ss, K, T, VOL, 0.0, +1)
                 - payoff(Ss, K, +1), 'k', lw=0.9)
    axp.plot(Ss, vanilla(Ss, K, 1000, VOL, 0.0, -1)
             - payoff(Ss, K, -1), 'r', lw=1.2)
    axc.plot(Ss, vanilla(Ss, K, 1000, VOL, 0.0, +1)
             - payoff(Ss, K, +1), 'r', lw=1.2)
    for ax, tt in [(axp, 'put'), (axc, 'call')]:
        ax.set_xticks(np.arange(0, 351, 50))
        ax.set_title(tt, fontsize=14)
        ax.set_xlabel('S', fontsize=14)
        ax.set_ylim(-30, 110)
        ax.grid(True)
    _save(fig)

    # ---- non-zero rates ----
    r = 0.015
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(11.6, 5.2))
    for T in 2.0 ** np.arange(-1, 5):
        axp.plot(Ss, vanilla(Ss, K, T, VOL, r, -1)
                 - payoff(Ss, K, -1), 'k', lw=0.9)
        axc.plot(Ss, vanilla(Ss, K, T, VOL, r, +1)
                 - payoff(Ss, K, +1), 'k', lw=0.9)
    for ax, tt in [(axp, 'put'), (axc, 'call')]:
        ax.set_xticks(np.arange(0, 351, 50))
        ax.set_title(tt, fontsize=14)
        ax.set_xlabel('S', fontsize=14)
        ax.set_xlim(0, 350)
        ax.set_ylim(-30, 110)
        ax.grid(True)
    _save(fig)

    # ---- chebfun2 price surfaces ----
    fig = plt.figure(figsize=(11.6, 5.4))
    for j, rr in enumerate([0.0, 0.015], start=1):
        put2 = Chebfun2.from_function(
            lambda s, t: vanilla(s, K, np.maximum(t, 1e-3), VOL, rr, -1),
            domain=(1e-6, 200, 0.001, 200))
        ax = fig.add_subplot(1, 2, j, projection="3d")
        sg = np.linspace(0, 200, 120)
        tg = np.linspace(0.001, 200, 120)
        SG, TG = np.meshgrid(sg, tg)
        ax.plot_surface(SG, TG, np.asarray(put2(SG, TG)),
                        cmap="viridis", rstride=1, cstride=1,
                        linewidth=0)
        ax.set_zlim(0, 100)
        ax.set_xlabel('S')
        ax.set_ylabel('T')
        ax.set_title('put, r=0' if rr == 0 else 'put, r=1.5%',
                     fontsize=14)
        ax.view_init(20, -60)
    _save(fig)

    # ---- early-exercise boundary via chebfun2 roots ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for rr in np.arange(0.0005, 0.0551, 0.010):
        put2 = Chebfun2.from_function(
            lambda t, s: (vanilla(s, K, np.maximum(t, 1e-3), VOL, rr, -1)
                          - payoff(s, K, -1)),
            domain=(0.001, 25, 1e-6, 100))
        for c in put2.roots():
            tt = np.linspace(float(c.domain.a), float(c.domain.b), 400)
            z = np.asarray(c(tt))
            ax.plot(np.real(z), np.imag(z), lw=1.6)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Asset level', fontsize=14)
    ax.set_xlabel('Time to maturity', fontsize=14)
    ax.set_title('Asset level at which time value of a put becomes '
                 'negative', fontsize=12)
    ax.grid(True)
    _save(fig)

    # ---- put-call parity with r = 5% ----
    r = 0.05
    loan = K * np.exp(-r * 0.5)

    def call5(s):
        return vanilla(s, 100, .5, VOL, r, +1)

    def e1(s):
        return (vanilla(s, 100.0, .50, VOL, r, -1) + s - loan) - call5(s)

    def e2(s):
        return (vanilla(s, 105.5, .25, VOL, r, -1) + s - loan) - call5(s)

    def e3(s):
        return (1.03 * vanilla(s, 95., .75, VOL, r, -1)
                + s - loan) - call5(s)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(Sh, e1(Sh), 'b', lw=1.6, label='1.00 put/same K/same T')
    ax.plot(Sh, e2(Sh), 'r', lw=1.6,
            label='1.00 put/higher K/ shorter T')
    ax.plot(Sh, e3(Sh), 'k', lw=1.6, label='1.03 put/lower K/ longer T')
    ax.set_ylim(-6, 8)
    ax.set_title('Imperfect put-call parity relation, r = 5%')
    ax.set_xlabel('S', fontsize=14)
    ax.set_ylabel('profit/loss (instant)', fontsize=14)
    ax.legend()
    ax.grid(True)
    _save(fig)

    ls2, as2 = _minloss(e2)
    ls3, as3 = _minloss(e3)
    print(f"Max loss stgy 2: {ls2:.4f} at {as2:.4f}")
    print(f"Max loss stgy 3: {ls3:.4f} at {as3:.4f}")


if __name__ == "__main__":
    run()
