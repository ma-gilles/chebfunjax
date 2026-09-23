"""The Blasius function.

Translation of ode-nonlin/Blasius.m by Hrothgar (October 2013):
the Blasius boundary-layer equation

    2u''' + u u'' = 0,  u(0) = u'(0) = 0,  u'(L) = 1  (L = 11),

its wall shear a = u''(0), displacement constant b, Taylor
coefficients, and the singularity that defeats a solve on a domain
extending left of the origin.

Original: https://www.chebfun.org/examples/ode-nonlin/Blasius.html
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

from chebfunjax import chebfun, poly
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Blasius_{k:02d}.png"))


def _col_long(v):
    """MATLAB format-long display of a column, with the common
    ``1.0e-NN *`` scale factor MATLAB factors out of small entries."""
    v = np.asarray(v, dtype=float)
    big = float(np.max(np.abs(v)))
    e = int(np.floor(np.log10(big))) if big > 0 else 0
    if e < -3 or e >= 3:
        print(f"   1.0e{e:+03d} *")
        v = v / 10.0**e
    for t in v:
        print(f"{t:20.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 11.0)
    op = lambda u: 2 * u.diff(3) + u * u.diff(2)  # noqa: E731

    def bc(x, u):
        return [u(0.0), u.diff()(0.0), u.diff()(dom[1]) - 1]
    N = Chebop(op, domain=dom)
    N.bc = bc
    u = N.solve(0.0)
    print("u =")
    print(repr(u))

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(u, 'k', ax=ax)
    ax.set_title("The Blasius function")
    _save(fig, 1)

    print("op_residual =")                    # Residual of the ODE
    print(f"     {float(op(u).norm()):.15e}")
    print("bc_residuals =")                   # Residuals of the BCs
    _col_long([float(r) for r in bc(0, u)])

    a_exact = 0.33205733621519630
    a_computed = float(u.diff(2)(jnp.array(0.0)))
    print("ans =")
    print(f"    {a_exact - a_computed:.15e}")

    x = chebfun(lambda t: t, domain=dom)
    matlab_plot(x, 'r--', ax=ax)
    _save(fig, 2)
    plt.close(fig)

    b_exact = -1.720787657520503
    b_computed = float((u - x)(jnp.array(dom[1])))
    print("ans =")
    print(f"    {b_exact - b_computed:.15e}")

    coeffs = poly(u)
    print("ans =")
    _col_long(coeffs[::-1][:6])

    N2 = Chebop(op, domain=(-5.6, 11.0))
    N2.bc = bc
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        v = N2.solve(0.0)
    if any("Newton" in str(w.message) for w in wlist):
        print("Warning: Newton iteration failed.")
        print("Please try supplying a better initial guess via the .init "
              "field")
        print("of the chebop. ")
    print("v =")
    print(repr(v))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    matlab_plot(v, 'k-', ax=ax)
    ax.set_xlim(-5.7, 11)
    ax.set_title("A singularity of the Blasius function")
    _save(fig, 3)
    plt.close(fig)


if __name__ == "__main__":
    run()
