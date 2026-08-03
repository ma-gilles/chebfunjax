"""The catenary by variational Newton iteration.

Faithful replica of opt/Catenary.m by Toby Driscoll (October 2010,
revised 2016): minimizing the surface-of-revolution energy
J[y] = int y sqrt(1+y'^2) by Newton's method in function space,
each step solving the accessory (Jacobi) equation as a chebop BVP.

Original: https://www.chebfun.org/examples/opt/Catenary.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

DOM = (-1.0, 1.0)


def J(y):
    yp = y.diff()
    return float((y * (1 + yp**2).sqrt()).sum())


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    alpha, beta = np.cosh(-1), np.cosh(1)
    y = cj.chebfun(lambda x: alpha + (beta - alpha) * (x + 1) / 2,
                   domain=DOM)
    print("startJ =")
    print(f"   {J(y):.15f}")

    xs = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    iterates = [np.asarray(y(xs))]

    for _k in range(5):
        yp = y.diff()
        s = (1 + yp**2).sqrt()
        # analytic first and second variations of f = y*sqrt(1+yp^2)
        f1 = s
        f2 = y * yp / s
        f12 = yp / s
        f22 = y / (s**3)

        N = Chebop(
            lambda x, u, _f22=f22, _f12=f12: (
                (_f22 * u.diff()).diff() + _f12.diff() * u),
            domain=DOM, lbc=0.0, rbc=0.0)
        u = N.solve(f1 - f2.diff())
        y = y + u
        print("nextJ =")
        print(f"   {J(y):.15f}")
        iterates.append(np.asarray(y(xs)))

    y_exact = np.cosh(xs)
    err = np.max(np.abs(iterates[-1] - y_exact))
    print("ans =")
    print(f"     {err:.15e}")
    print()
    print(f"  final J[y]: {J(y):.16f}")
    Jexact = J(cj.chebfun(lambda x: jnp.cosh(x), domain=DOM))
    print(f"optimal J[y]: {Jexact:.16f}")

    ax.plot(xs, y_exact, 'r--', lw=2, label="exact cosh(x)")
    for i, v in enumerate(iterates):
        ax.plot(xs, v, lw=0.9,
                label=f"iterate {i}" if i in (0, 5) else None)
    ax.legend()
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Catenary_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
