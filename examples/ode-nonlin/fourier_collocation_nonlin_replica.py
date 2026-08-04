"""Fourier collocation for nonlinear periodic ODEs.

Faithful replica of ode-nonlin/FourierCollocationNonLin.m by Hadrien
Montanelli (December 2014): the nonlinear periodic problem

    u' - u cos(u) = cos(4x),   x in [0, 2pi],

which has (at least) two solutions, found from two different initial
guesses.

Original: https://www.chebfun.org/examples/ode-nonlin/FourierCollocationNonLin.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

DOM = (0, 2 * np.pi)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = cj.chebfun(lambda x: jnp.cos(4 * x), domain=DOM)
    t = np.linspace(DOM[0], DOM[1], 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    N = Chebop(lambda u: u.diff() - u * u.cos(), domain=DOM)
    N.bc = "periodic"
    N.init = cj.chebfun(lambda x: jnp.cos(x), domain=DOM)
    u = N.solve(f)
    print("u =")
    print(repr(u))
    ax.plot(t, np.asarray(N.init(t)), '--b', lw=2)
    ax.plot(t, np.asarray(u(t)), 'b', lw=2)
    print("ans =")
    print(f"     {float((u.diff() - u*u.cos() - f).norm(jnp.inf)):.15e}")

    N = Chebop(lambda u: u.diff() - u * u.cos(), domain=DOM)
    N.bc = "periodic"
    N.init = cj.chebfun(lambda x: jnp.sin(x)**2, domain=DOM)
    v = N.solve(f)
    print("v =")
    print(repr(v))
    ax.plot(t, np.asarray(N.init(t)), '--g', lw=2)
    ax.plot(t, np.asarray(v(t)), 'g', lw=2)
    print("ans =")
    print(f"     {float((v.diff() - v*v.cos() - f).norm(jnp.inf)):.15e}")

    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "FourierCollocationNonLin_repl_01.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
