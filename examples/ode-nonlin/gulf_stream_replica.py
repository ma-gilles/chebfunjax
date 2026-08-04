"""A Gulf Stream model.

Faithful replica of ode-nonlin/GulfStream.m by Nick Trefethen
(November 2010, after a model of Stommel): the third-order nonlinear
boundary-value problem

    u''' - lambda ((u')^2 - u u'') - u + 1 = 0,  x in [0, 35],

with stress-free conditions u(0) = u''(0) = 0 and u(35) = 1, plus the
conserved quantity I = int (u'')^2 - 3 lambda u u' u'' = 1/2.

Original: https://www.chebfun.org/examples/ode-nonlin/GulfStream.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, plotcoeffs

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
X = 35.0
LAM = -0.1


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"GulfStream_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _op(u):
    return (u.diff(3) - LAM * (u.diff(1)**2 - u * u.diff(2))
            - u + 1)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    N = Chebop(_op, domain=(0, X))
    N.lbc = lambda u: [u, u.diff(2)]       # stress-free
    N.rbc = 1
    u, info = N.solvebvp(0.0)

    t = np.linspace(0, X, 2000)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=1.6, label="u")
    ax.plot(t, np.asarray(u.diff()(t)), lw=1.6, label="u'")
    ax.plot(t, np.asarray(u.diff(2)(t)), lw=1.6, label="u''")
    ax.axis([0, 20, -1, 1.5])
    ax.set_xlabel("x")
    ax.legend(loc="lower right")
    ax.set_title("Slippery or stress-free b. c.")
    ax.grid(True)
    _save(fig)

    print("N_residual =")
    print(f"     {float(_op(u).norm()):.15e}")
    print("lbc_residuals =")
    print(f"   {float(u(jnp.array(0.0))):.15e}  "
          f"{float(u.diff(2)(jnp.array(0.0))):.15e}")
    print("rbc_residual =")
    print(f"    {float(u(jnp.array(X))) - 1:.15e}")

    nd = info["normDelta"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.semilogy(np.arange(1, len(nd) + 1), nd, 'm*-', lw=1.4)
    ax.set_ylim(1e-16, 1e1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("norm of Newton update")
    ax.grid(True)
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    plotcoeffs(u, ax=ax)
    ax.grid(True)
    _save(fig)

    I = float((u.diff(2)**2
               - 3 * LAM * (u * u.diff() * u.diff(2))).sum())
    print("I =")
    print(f"   {I:.15f}")
    print("I_error =")
    print(f"     {abs(I - 0.5):.15e}")
    print("total_time_for_this_example =")
    print(f"   {time.time() - t0:.15f}")


if __name__ == "__main__":
    run()
