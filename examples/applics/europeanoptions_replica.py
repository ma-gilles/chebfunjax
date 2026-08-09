"""Pricing other European Options: Puts, Digitals, Powers.

Faithful replica of applics/EuropeanOptions.m (Pachon, 2014): the
chebfun payoff-distribution pricing method applied to a put, a
digital (cash-or-nothing) call, and a power call, each compared
against its closed-form Black-Scholes price.

Original: https://www.chebfun.org/examples/applics/EuropeanOptions.html
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
FIG = [0]

S0 = 100
VOL = 0.45
R = 0.01
T = 0.5
MAXS = 10000


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"EuropeanOptions_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def logn(S):
    S = np.maximum(np.asarray(S, dtype=float), 1e-300)
    return (np.exp(-(np.log(S / S0) - (R - 0.5 * VOL**2) * T)**2
                   / (2 * VOL**2 * T))
            / (VOL * S * np.sqrt(2 * np.pi * T)))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    lognPDF = chebfun(lambda S: logn(S), domain=(1e-12, MAXS))
    lognCDF = lognPDF.cumsum()

    ss = np.linspace(1e-6, 200, 1200)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ss, logn(ss), 'k', lw=1.6)
    ax.set_ylim(0, 0.015)
    ax.set_yticks(np.arange(0, 0.0151, 0.003))
    ax.set_xlabel(r'$S_T$', fontsize=14)
    ax.grid(True)
    _save(fig)

    # ---- European put ----
    K = 150.0
    maxV = K
    Sp = np.linspace(0, 250, 1000)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(Sp, np.maximum(0, K - Sp), 'k', lw=1.6)
    ax.set_xlabel('S', fontsize=14)
    ax.set_ylabel('V(S)', fontsize=14)
    ax.set_ylim(-10, K)
    ax.grid(True)
    _save(fig)

    probOOM = 1 - float(lognCDF(np.array([K]))[0])
    print("probOOM =")
    print(f"   {probOOM:.15f}")

    ITM = chebfun(lambda S: logn(K - S), domain=(0, maxV))
    disc = np.exp(-R * T)
    xk = chebfun(lambda y: np.asarray(y), domain=(0, maxV))
    approx = disc * float((xk * ITM).sum())
    print(f"approx = {approx:.15f}")

    yv = np.linspace(0, maxV, 900)
    pv = disc * np.asarray(ITM(yv))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(yv, pv, color=(0.3, 0.9, 0.4))
    ax.plot(yv, pv, 'k', lw=1.6)
    ax.plot([0, 0], [0, 0.06], 'r', lw=2)
    ax.plot([approx, approx], [0, 0.025], 'b--', lw=1.6)
    ax.set_xlim(-10, K)
    ax.set_ylim(0, 0.08)
    ax.set_xlabel(r'$e^{-rT}V(S_T)$', fontsize=14)
    ax.grid(True)
    _save(fig)

    d1 = (np.log(S0 / K) + (R + 0.5 * VOL**2) * T) / (VOL * np.sqrt(T))
    d2 = d1 - VOL * np.sqrt(T)
    W = -1
    exact = W * (S0 * norm.cdf(W * d1)
                 - K * norm.cdf(W * d2) * np.exp(-R * T))
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")

    # ---- Digital call ----
    K = 100.0
    Sp = np.linspace(0, 200, 1000)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.step([0, K, 200], [0, 0, 1], 'k', lw=1.6, where='post')
    ax.set_xlabel('S', fontsize=14)
    ax.set_ylabel('V(S)', fontsize=14)
    ax.set_ylim(-.1, 1.1)
    ax.grid(True)
    _save(fig)

    probOOM = float(lognCDF(np.array([K]))[0])
    probITM = 1 - probOOM
    approx = np.exp(-R * T) * probITM  # E[x * (deltas at 0 and 1)]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot([0, 0], [0, probOOM], 'r', lw=3)
    ax.plot([1, 1], [0, probITM], 'g', lw=3)
    ax.plot([approx, approx], [0, 0.3], 'b--', lw=1.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel(r'$e^{-rT}V(S_T)$', fontsize=14)
    ax.grid(True)
    _save(fig)
    print(f"approx = {approx:.15f}")

    d1 = (np.log(S0 / K) + (R + 0.5 * VOL**2) * T) / (VOL * np.sqrt(T))
    d2 = d1 - VOL * np.sqrt(T)
    exact = norm.cdf(d2) * np.exp(-R * T)
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")

    # ---- Power call ----
    K = 9.1
    alpha = 0.5
    alphainv = 1 / alpha
    Sp = np.linspace(0, 1000, 2000)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(Sp, np.maximum(0, Sp**alpha - K), 'k', lw=1.6)
    ax.set_xlabel('S', fontsize=14)
    ax.set_ylabel('V(S)', fontsize=14)
    ax.grid(True)
    _save(fig)

    maxV = 50.0
    ITM = chebfun(
        lambda S: logn((S + K)**alphainv)
        * np.abs(alphainv * (S + K)**(alphainv - 1)),
        domain=(0, maxV))
    xk = chebfun(lambda y: np.asarray(y), domain=(0, maxV))
    approx = np.exp(-R * T) * float((xk * ITM).sum())
    print(f"approx = {approx:.15f}")

    yv = np.linspace(0, 10, 900)
    pv = np.exp(-R * T) * np.asarray(ITM(yv))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(yv, pv, color=(0.3, 0.9, 0.4))
    ax.plot(yv, pv, 'k', lw=1.6)
    ax.plot([0, 0], [0, 0.35], 'r', lw=2)
    ax.plot([approx, approx], [0, 0.3], 'b--', lw=1.6)
    ax.set_xlim(-.5, 10)
    ax.set_xlabel(r'$e^{-rT}V(S_T)$', fontsize=14)
    ax.grid(True)
    _save(fig)

    d1 = ((np.log(S0 / K**alphainv) + (R + (alpha - 0.5) * VOL**2) * T)
          / (VOL * np.sqrt(T)))
    d2 = d1 - alpha * VOL * np.sqrt(T)
    m = (R + 0.5 * alpha * VOL**2) * (alpha - 1)
    exact = (S0**alpha * np.exp(m * T) * norm.cdf(d1)
             - np.exp(-R * T) * K * norm.cdf(d2))
    print(f"exact  = {exact:.15f}")
    print(f"approx = {approx:.15f}")


if __name__ == "__main__":
    run()
