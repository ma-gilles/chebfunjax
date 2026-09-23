"""The catenary by variational Newton iteration.

Translation of opt/Catenary.m by Toby Driscoll (October 2010,
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')

DOM = (-1.0, 1.0)


def f(y, yp):
    return y * (1 + yp**2).sqrt()


def J(y):
    return float(f(y, y.diff()).sum())


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    alpha, beta = np.cosh(-1), np.cosh(1)
    y0 = cj.chebfun(jnp.asarray([alpha, beta]), domain=DOM)
    startJ = J(y0)
    print("startJ =")
    print(f"   {startJ:.15f}")

    # The integrand as a chebfun2 F(y, y') and its partial derivatives
    # (diff(F, [1 0]) is d/dy: the chebfun2 x-direction, dim=2).
    dom2 = (1.0, 2.0, -2.0, 4.0)
    F = cj.chebfun2(lambda y, yp: y * jnp.sqrt(1 + yp**2), domain=dom2)
    F1 = F.diff(2, 1)
    F2 = F.diff(1, 1)
    F11 = F.diff(2, 2)
    F12 = F.diff(2, 1).diff(1, 1)
    F22 = F.diff(1, 2)

    y = y0
    for _k in range(5):
        # The first and second variations of f.
        yp = y.diff()
        f1 = F1(y, yp)
        f2 = F2(y, yp)
        f11 = F11(y, yp)
        f12 = F12(y, yp)
        f22 = F22(y, yp)
        # The next Newton step solves the accessory equation.
        N = Chebop(
            lambda x, u, f22=f22, c=f12.diff() - f11: (
                (f22 * u.diff()).diff() + c * u),
            domain=DOM, lbc=0.0, rbc=0.0)
        u = N.solve(f1 - f2.diff())
        y = y + u
        nextJ = J(y)
        print("nextJ =")
        print(f"   {nextJ:.15f}")

    y_exact = cj.chebfun(lambda x: jnp.cosh(x), domain=DOM)
    print("ans =")
    print(f"     {float((y - y_exact).norm()):.15e}")
    print(f"\n  final J[y]: {J(y):.16f}")
    print(f"optimal J[y]: {J(y_exact):.16f}")

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    y_exact.plot(ax=ax, color='r', linestyle='--', linewidth=2,
                 label='exact')
    y.plot(ax=ax, color='k', linewidth=1, label='computed')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Solution to the catenary problem', fontsize=9)
    ax.set_xlabel('x', fontsize=10.5)
    ax.set_ylabel('y(x)', fontsize=10.5)
    ax.legend(loc='upper center')
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Catenary_01.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
