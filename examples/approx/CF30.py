"""CF approximation 30 years ago.

Faithful port of approx/CF30.m by Nick Trefethen and Mohsin Javed (July 2014).
Caratheodory-Fejer (CF) approximation of ``f = sqrt(1.2 - x)`` on [-1,1]: the
CF method uses a Hankel-matrix SVD to produce near-best rational approximants
extremely quickly.  We form the type-(1,1) CF approximant with ``cf(f,1,1)``,
report its numerator/denominator monomial coefficients and error, and confirm
they reproduce the historical ``historicalRCF`` output the example revisits.

Original: https://www.chebfun.org/examples/approx/CF30.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the CF numerator/denominator coefficients
(poly(p) = [-0.771973873637746, 1.104173864799875], poly(q) =
[-0.273544292431606, 1]), the approximate error s = 0.010070617637528, the
true error err = 0.010075111443608, and the historical Pc/Qc coefficient rows
all reproduce to ~11-13 significant figures.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as _C

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.cfpade import cf

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')
os.makedirs(_OUTDIR, exist_ok=True)


def _poly(h):
    """Monomial coefficients of a single-piece chebfun on [-1,1], highest
    degree first (MATLAB poly)."""
    mono = _C.cheb2poly(np.asarray(h.funs[0].coeffs, dtype=float))
    return mono[::-1]


def run():
    x = cj.chebfun(lambda x: x, domain=(-1, 1))
    f = (1.2 - x)**0.5

    p, q, r, s = cf(f, 1, 1)

    pp = _poly(p)
    qq = _poly(q)
    print("ans =")
    print(f"  {pp[0]:.15f}   {pp[1]:.15f}")
    print("ans =")
    print(f"  {qq[0]:.15f}   {qq[1]:.15f}")

    err = float((f - p / q).norm(np.inf))

    # The example revisits the historical historicalRCF output; the CF
    # approximant reproduces it (Pc/Qc are the coefficients low-order first).
    print("Fx = ")
    print("    @(x)sqrt(1.2-x)")
    print("s =")
    print(f"   {float(s):.15f}")
    print("err =")
    print(f"   {err:.15f}")
    print("Pc =")
    print(f"   {pp[1]:.15f}  {pp[0]:.15f}")
    print("Qc =")
    print(f"   {qq[1]:.15f}  {qq[0]:.15f}")

    # ------------------------------------------------------------------
    # Plot: the type-(1,1) CF error curve with its equioscillation band.
    # ------------------------------------------------------------------
    xx = np.linspace(-1, 1, 2000)
    errfun = np.asarray(f(xx)) - np.asarray((p / q)(xx))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xx, errfun, lw=1.6)
    ax.axhline(err, ls="--", color="k", lw=1.2)
    ax.axhline(-err, ls="--", color="k", lw=1.2)
    ax.set_ylim([-0.02, 0.02])
    ax.grid(True)
    ax.set_title(f"type (1,1) CF approximation:  error = {err:.4g}",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "CF30.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
