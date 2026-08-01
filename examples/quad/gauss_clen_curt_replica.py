"""Gauss and Clenshaw-Curtis quadrature.

Faithful replica of quad/GaussClenCurt.m by Nick Trefethen: for a
wiggly function, chebfun's sum, Clenshaw-Curtis at the chebfun length,
and Gauss quadrature all agree; the two rules' convergence curves are
compared.

Original: https://www.chebfun.org/examples/quad/GaussClenCurt.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.quadrature import chebpts, chebweights, legpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def run():
    os.makedirs(_IMG, exist_ok=True)
    f = lambda x: x * jnp.sin(2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x))))
    fc = cj.chebfun(f)
    xs = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, np.asarray(fc(jnp.asarray(xs))), lw=1.0)
    ax.set_title("Function f", fontsize=16)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "GaussClenCurt_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    Icheb = float(fc.sum())
    print("Ichebfun =")
    print(f"   {Icheb:.15f}")
    N = len(fc)
    print("Npts =")
    print(f"   {N}")
    s = np.asarray(chebpts(N))
    w = np.asarray(chebweights(N))
    print("Iclenshawcurtis =")
    print(f"   {float(w @ np.asarray(f(jnp.asarray(s)))):.15f}")
    sg, wg = (np.asarray(v) for v in legpts(N))
    print("Igauss =")
    print(f"   {float(wg @ np.asarray(f(jnp.asarray(sg)))):.15f}")

    NN = np.arange(10, 501, 10)
    errG, errC = [], []
    for Npts in NN:
        sg, wg = (np.asarray(v) for v in legpts(int(Npts)))
        errG.append(abs(float(wg @ np.asarray(f(jnp.asarray(sg))))
                        - Icheb))
        s = np.asarray(chebpts(int(Npts)))
        w = np.asarray(chebweights(int(Npts)))
        errC.append(abs(float(w @ np.asarray(f(jnp.asarray(s)))) - Icheb))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(NN, np.maximum(errG, 1e-18), ".-", ms=8, label="Gauss")
    ax.semilogy(NN, np.maximum(errC, 1e-18), ".-r", ms=8,
                label="Clenshaw-Curtis")
    ax.set_ylim(1e-18, 1)
    ax.grid(True)
    ax.set_xlabel("Npts", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Gauss and Clenshaw-Curtis", fontsize=16)
    ax.legend(loc="lower left")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "GaussClenCurt_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    run()
