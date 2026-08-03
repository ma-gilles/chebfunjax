"""Probability exercises: normal and near-normal distributions.

Faithful replica of stats/NormalExercises.m by Jie Gao and Nick
Trefethen (June 2013): the probability that a normal random variable
falls in [mu-1, mu+1], computed with a chebfun on the unbounded real
line, and the same computation for the non-smooth density
exp(-|x-mu|^{5/4}).

Original: https://www.chebfun.org/examples/stats/NormalExercises.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'stats')

MU, SIGMA = 2.0, 1.0
FIG = [0]


def _area_plot(f, color):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    xs_fill = np.linspace(1, 3, 200)
    ax.fill_between(xs_fill, np.asarray(f(xs_fill)), color=color)
    xs = np.linspace(-1, 6, 600)
    ax.plot(xs, np.asarray(f(xs)), 'k', lw=1.6)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"NormalExercises_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = cj.chebfun(
        lambda x: 1 / (SIGMA * jnp.sqrt(2 * jnp.pi))
        * jnp.exp(-0.5 * ((x - MU) / SIGMA) ** 2),
        domain=(-np.inf, np.inf))
    fint = f.cumsum()
    p = float(fint(3.0)) - float(fint(1.0))
    print("p =")
    print(f"   {p:.15f}")
    _area_plot(f, (0.3, 0.9, 0.4))

    # a heavier-tailed, non-smooth cousin: exp(-|x-mu|^(5/4)).
    # By symmetry p = int_0^1 e^{-t^{5/4}} dt / int_0^inf; the
    # integrand's fractional kink at t = 0 is resolved by splitting
    # on a finite domain (the tail beyond t = 40 is below e^{-100}).
    h = cj.chebfun(lambda t: jnp.exp(-jnp.abs(t) ** 1.25),
                   domain=(0.0, 40.0), splitting=True)
    total = 2 * float(h.sum())
    H = h.cumsum()
    p2 = 2 * float(H(1.0)) / total
    print("p =")
    print(f"   {p2:.15f}")

    g = cj.chebfun(
        lambda x: jnp.exp(-jnp.abs((x - MU) / SIGMA) ** 1.25)
        / total, domain=(-6.0, 10.0))
    _area_plot(g, (0.9, 0.3, 0.4))


if __name__ == "__main__":
    run()
