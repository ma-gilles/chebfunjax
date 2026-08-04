"""The Blasius function.

Faithful replica of ode-nonlin/Blasius.m by Hrothgar (October 2013):
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 11.0)
    op = lambda u: 2 * u.diff(3) + u * u.diff(2)  # noqa: E731
    N = Chebop(op, domain=dom)
    N.bc = lambda x, u: [u(0.0), u.diff()(0.0),
                         u.diff()(11.0) - 1]
    u = N.solve(0.0)
    print("u =")
    print(repr(u))

    t = np.linspace(0, 11, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), 'k', lw=1.6)
    ax.plot(t, t, 'r--', lw=1.2)
    ax.set_title("The Blasius function")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Blasius_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("op_residual =")
    print(f"     {float(op(u).norm()):.15e}")
    print("bc_residuals =")
    for v in (float(u(jnp.array(0.0))),
              float(u.diff()(jnp.array(0.0))),
              float(u.diff()(jnp.array(11.0))) - 1):
        print(f"  {v:.6e}")

    a_exact = 0.33205733621519630
    a_computed = float(u.diff(2)(jnp.array(0.0)))
    print("ans =")
    print(f"    {a_exact - a_computed:.15e}")

    b_exact = -1.720787657520503
    b_computed = float(u(jnp.array(11.0))) - 11.0
    print("ans =")
    print(f"    {b_exact - b_computed:.15e}")

    # Taylor coefficients at 0 (monomial basis)
    cheb = np.polynomial.chebyshev.Chebyshev(
        np.asarray(u.funs[0].tech.coeffs), domain=[0, 11])
    mono = cheb.convert(kind=np.polynomial.Polynomial)
    c = mono.coef
    print("ans =")
    for k in range(6):
        print(f"  {c[k]:18.15f}")

    # A domain crossing the singularity at -5.69...: Newton fails,
    # exactly as on the published page.
    N2 = Chebop(op, domain=(-5.6, 11.0))
    N2.bc = lambda x, u: [u(0.0), u.diff()(0.0),
                          u.diff()(11.0) - 1]
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            v = N2.solve(0.0)
            print("v =")
            print(repr(v))
        except Exception as e:
            print(f"solve failed: {type(e).__name__}")
        for w in wlist:
            if "Newton" in str(w.message) or "iterations" in str(
                    w.message):
                print(f"Warning: {w.message}")
                break


if __name__ == "__main__":
    run()
