"""Fractional calculus in Chebfun.

Faithful replica of integro/FracCalc.m by Nick Hale (October 2010):
half-derivatives and fractional derivatives/integrals via
``diff(f, alpha)`` and ``cumsum(f, alpha)``.

Original: https://www.chebfun.org/examples/integro/FracCalc.html
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'integro')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"FracCalc_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _vals(f, t):
    return np.asarray(f(jnp.asarray(t))).ravel()


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = cj.chebfun(lambda t: t, domain=(0, 4))
    t = np.linspace(1e-8, 4, 800)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(t, _vals(x, t), lw=1.2, label="x")
    ax.plot(t, _vals(x.diff(), t), lw=1.2, label="x'")
    ax.plot(t, _vals(x.cumsum(), t), lw=1.2, label="x^2/2")
    ax.axis([0, 4, 0, 4])
    ax.set_xlabel("x", fontsize=10)
    ax.set_title("The function 'x' with its derivative and "
                 "antiderivative", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True)
    _save(fig)

    xp05 = x.diff(0.5)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(t, _vals(x, t), lw=1.2, label="x")
    ax.plot(t, _vals(x.diff(), t), lw=1.2, label="x'")
    ax.plot(t, _vals(x.cumsum(), t), lw=1.2, label="x^2/2")
    ax.plot(t, _vals(xp05, t), lw=1.2,
            label=r"$d^{1/2}x\,/\,dx^{1/2}$")
    ax.axis([0, 4, 0, 4])
    ax.set_xlabel("x", fontsize=10)
    ax.set_title("The function 'x' and its half-derivative",
                 fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True)
    _save(fig)

    f = cj.chebfun(lambda s: 2 * jnp.sqrt(s / jnp.pi),
                   domain=(0, 4), exps=[0.5, 0])
    print("ans =")
    print(f"   {float((f - xp05).norm(jnp.inf))}")

    # Fractional derivatives of x, alpha = 0:0.1:1
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    alphas = np.arange(0.0, 1.05, 0.1)
    for a in alphas:
        u = x if a == 0 else x.diff(float(a))
        ax.plot(t, _vals(u, t), lw=1.2, label=f"{a:.1f}")
    ax.set_title("Fractional derivatives of x", fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(r"$d^a x / dx^a$", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    _save(fig)

    # Fractional derivatives of sin(x) on [0, 20]
    u0 = cj.chebfun(jnp.sin, domain=(0, 20))
    t20 = np.linspace(1e-8, 20, 1600)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    salphas = np.sqrt(2) * np.arange(0, 11, 2) / 17
    for a in salphas:
        u = u0 if a == 0 else u0.diff(float(a))
        ax.plot(t20, _vals(u, t20), lw=1.2, label=f"{a:.5f}")
    ax.set_title("Fractional derivatives of sin(x)", fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(r"$d^a \sin(x) / dx^a$", fontsize=10)
    ax.legend(fontsize=8)
    ax.axis([-0.5, np.pi, 0.0, 1.01])
    ax.grid(True)
    _save(fig)

    # Half-integrals of x^k, k = 1..10 on [0, 1]
    x1 = cj.chebfun(lambda s: s, domain=(0, 1))
    t1 = np.linspace(1e-8, 1, 500)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for k in range(1, 11):
        u = (x1 ** k).cumsum(0.5)
        ax.plot(t1, _vals(u, t1), lw=1.2, label=str(k))
    ax.set_title("Half-integrals of x^k for k = 1, ..., 10",
                 fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    _save(fig)

    # Fractional integrals of exp(x)-1
    g = cj.chebfun(lambda s: jnp.exp(s) - 1, domain=(0, 1))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for a in alphas:
        u = g if a == 0 else g.cumsum(float(a))
        ax.plot(t1, _vals(u, t1), lw=1.2, label=f"{a:.1f}")
    ax.set_title("Fractional integrals of exp(x)-1", fontsize=10)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
