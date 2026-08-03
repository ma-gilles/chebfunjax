"""Nonstandard 'boundary' conditions.

Faithful replica of ode-linear/NonstandardBCs.m by Asgeir Birkisson
(October 2011): the ODE u'' + x^2 u = 1 with u(-1) = 1 and a second
condition that is NOT a boundary condition: a zero-mean condition,
a prescribed mean, a weighted integral, an interior value u(0) = 1/2,
and an interior derivative u'(0) = 1.

Original: https://www.chebfun.org/examples/ode-linear/NonstandardBCs.html
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
_XF = Chebfun.identity(Domain((-1.0, 1.0)))


def _save_plot(u):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.linspace(-1, 1, 900)
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"NonstandardBCs_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make():
    N = Chebop(lambda x, u: u.diff(2) + x**2 * u, domain=(-1, 1))
    N.lbc = 1
    return N


def _resid(u):
    return float((u.diff(2) + _XF**2 * u - 1).norm())


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # sum(u) = 0
    N = _make()
    N.bc = lambda x, u: u.sum()
    u = N.solve(1.0)
    _save_plot(u)
    print(f"Residual of differential equation: {_resid(u):.5g}")
    print("Residual of left BC:               "
          f"{abs(float(u(jnp.array(-1.0))) - 1):.5g}")
    print("Residual of interior condition:    "
          f"{abs(float(u.sum())):.5g}")

    # mean(u) = 1
    N = _make()
    N.bc = lambda x, u: u.sum() / 2 - 1
    u = N.solve(1.0)
    print("Residual of Interior condition: "
          f"{abs(float(u.sum()) / 2 - 1):.5g}")

    # weighted integral sum(sin(4 pi x) u) = 0
    N = _make()
    N.bc = lambda x, u: ((4 * jnp.pi * x).sin() * u).sum()
    u = N.solve(1.0)
    _save_plot(u)
    print(f"Residual of differential equation: {_resid(u):.5g}")
    print("Residual of left BC:               "
          f"{abs(float(u(jnp.array(-1.0))) - 1):.5g}")
    print("Residual of interior condition:    "
          f"{abs(float(((4 * jnp.pi * _XF).sin() * u).sum())):.5g}")

    # interior value u(0) = 1/2
    N = _make()
    N.bc = lambda x, u: u(0.0) - 0.5
    u = N.solve(1.0)
    _save_plot(u)
    print(f"Residual of differential equation: {_resid(u):.5g}")
    print("Residual of left BC:               "
          f"{abs(float(u(jnp.array(-1.0))) - 1):.5g}")
    print("Residual of interior condition:    "
          f"{abs(float(u(jnp.array(0.0))) - 0.5):.5g}")

    # interior derivative u'(0) = 1
    N = _make()
    N.bc = lambda x, u: u.diff()(0.0) - 1
    u = N.solve(1.0)
    _save_plot(u)
    print(f"Residual of differential equation: {_resid(u):.5g}")
    print("Residual of left BC:               "
          f"{abs(float(u(jnp.array(-1.0))) - 1):.5g}")
    print("Residual of interior condition:    "
          f"{abs(float(u.diff()(jnp.array(0.0))) - 1):.5g}")


if __name__ == "__main__":
    run()
