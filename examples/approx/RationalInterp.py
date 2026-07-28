"""Rational interpolation, robust and non-robust.

Faithful port of approx/RationalInterp.m by Nick Trefethen (August 2011).
Rational interpolants r = p/q of a function in Chebyshev points via
``ratinterp``: for cos(exp(x)) the type-(n,n) interpolation error decreases
super-algebraically, and for exp(x) the type-(8,8) interpolant develops
near-cancelling spurious pole/zero pairs (Froissart doublets).

Original: https://www.chebfun.org/examples/approx/RationalInterp.html
Copyright 2011 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the type-(n,n) interpolation-error table for
cos(exp(x)) reproduces the published values -- (1,1) 2.46e-01, (2,2) 7.32e-03,
(4,4) 6.11e-06, (5,5) 4.16e-07, (6,6) 6.19e-09.  The (3,3) case has a pole in
[-1,1], so its sup-norm error is infinite (published Inf) -- unverifiable.

The spurious-zero/degree portion (Example 3) is a documented ratinterp
behaviour gap: MATLAB's ratinterp reduces the exact denominator degree to
nu=4 for exp(x) at type (8,8) (and reports two spurious zeros), whereas our
ratinterp keeps nu=8, so the reported degree_of_q and the spurious
zeros/poles differ.  Ledger backlog: ratinterp exact-degree reduction.
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
from chebfunjax.utils.ratapprox import ratinterp

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')
os.makedirs(_OUTDIR, exist_ok=True)


def run():
    # ------------------------------------------------------------------
    # Type-(n,n) rational interpolation error for cos(exp(x)).
    # ------------------------------------------------------------------
    f = cj.chebfun(lambda x: jnp.cos(jnp.exp(x)), domain=(-1, 1))
    xx = np.linspace(-1, 1, 8000)
    fx = np.asarray(f(xx))
    print("    (n,n)       Error ")
    for n in range(1, 7):
        r = ratinterp(f, n, n)[0]
        err = float(np.max(np.abs(fx - np.asarray(r(xx)))))
        print(f"    ({n},{n})     {err:7.2e}")

    # ------------------------------------------------------------------
    # Type-(8,8) interpolant of exp(x): spurious pole/zero pairs.
    # ------------------------------------------------------------------
    g = cj.chebfun(lambda x: jnp.exp(x), domain=(-1, 1))
    _, pc, qc, mu, nu = ratinterp(g, 8, 8)[:5]
    p = cj.Chebfun.from_coeffs(jnp.asarray(pc, dtype=jnp.complex128))
    q = cj.Chebfun.from_coeffs(jnp.asarray(qc, dtype=jnp.complex128))
    print("degree_of_p =")
    print(f"     {int(mu)}")
    zeros = np.sort_complex(np.asarray(p.roots()))
    print("spurious_zeros =")
    for z in zeros:
        print(f"   {z.real:.15f}")
    print("degree_of_q =")
    print(f"     {int(nu)}")
    poles = np.sort_complex(np.asarray(q.roots()))
    print("spurious_poles =")
    for z in poles:
        print(f"   {z.real:.15f}")

    # ------------------------------------------------------------------
    # Plot: the type-(8,8) interpolant of exp(x) through 17 Chebyshev points.
    # ------------------------------------------------------------------
    r88 = ratinterp(g, 8, 8)[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xx, np.asarray(r88(xx)).real, "m", lw=1.6, label="p/q")
    xc = np.cos(np.pi * np.arange(17) / 16)
    ax.plot(xc, np.exp(xc), ".k", ms=10, label="exp at Cheb pts")
    ax.legend(fontsize=9)
    ax.set_title("type (8,8) rational interpolant of exp(x)", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "RationalInterp.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
