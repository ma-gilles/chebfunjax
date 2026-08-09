"""Exponential, logistic, and Gompertz growth.

Faithful replica of applics/Gompertz.m (Driscoll, 2015): population
growth IVPs P' = P * pcrate(P) on [0, 25] with P(0) = 0.2 solved with
chebop -- exponential (constant per-capita rate), logistic (carrying
capacity 6), and Gompertz (log rate), with the per-capita-rate
comparison plot.

Original: https://www.chebfun.org/examples/applics/Gompertz.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'applics')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Gompertz_repl_{k:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _solve(pcrate):
    N = Chebop(lambda t, P: P.diff() - P * pcrate(P), domain=(0, 25))
    N.lbc = 0.2
    return N.solve(0)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    ts = np.linspace(0, 25, 800)

    # Exponential growth: constant per-capita rate.
    expo = _solve(lambda P: 0.5)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(ts, np.asarray(expo(ts)), lw=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.set_title("Exponential growth")
    ax.grid(True)
    _save(fig, 1)

    # Logistic model: carrying capacity 6.
    logi = _solve(lambda P: 0.5 * (6 - P) / 5.8)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(ts, np.asarray(expo(ts)), lw=1.6, label="exp")
    ax.plot(ts, np.asarray(logi(ts)), lw=1.6, label="logistic")
    ax.set_ylim(0, 8)
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.legend()
    ax.grid(True)
    _save(fig, 2)

    # Per-capita growth rates of the two limited models.
    Ps = np.linspace(0.2, 6, 400)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(Ps, 0.5 * (6 - Ps) / 5.8, lw=1.6)
    ax.plot(Ps, 0.5 * np.log(6 / Ps) / np.log(6 / 0.2), lw=1.6)
    ax.set_xlabel("P")
    ax.set_ylabel("P'/P")
    ax.grid(True)
    _save(fig, 3)

    # Gompertz model.
    def gomp_rate(P):
        if hasattr(P, "log"):
            return 0.5 * (6 / P).log() / np.log(6 / 0.2)
        return 0.5 * np.log(6 / P) / np.log(6 / 0.2)

    gomp = _solve(gomp_rate)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(ts, np.asarray(expo(ts)), lw=1.6, label="exp")
    ax.plot(ts, np.asarray(logi(ts)), lw=1.6, label="logistic")
    ax.plot(ts, np.asarray(gomp(ts)), lw=1.6, label="Gompertz")
    ax.set_ylim(0, 8)
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.legend()
    ax.grid(True)
    _save(fig, 4)

    for name, sol, ref in [
        ("exponential P(25)", expo, 0.2 * np.exp(0.5 * 25)),
        ("logistic P(25)", logi, None),
        ("Gompertz P(25)", gomp, None),
    ]:
        v = float(sol(np.array([25.0]))[0])
        line = f"{name} = {v:.12g}"
        if ref is not None:
            line += f"  (exact {ref:.12g})"
        print(line)


if __name__ == "__main__":
    run()
