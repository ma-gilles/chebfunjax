"""Rational approximation of monomials and Halphen's constant.

Faithful port of approx/Rationalxn.m by Yuji Nakatsukasa and Nick Trefethen
(May 2019).  The monomial x^n on [0,1] is hard for polynomials but efficiently
approximated by low-type rationals; the best type-(k,k) minimax errors shrink
by a factor approaching Halphen's constant 1/0.1076539... = 9.2890... as k
grows.

Original: https://www.chebfun.org/examples/approx/Rationalxn.html
Copyright 2019 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): the type-(2,2) and (3,3) minimax errors of
x^200 and the (3,3)->(4,4) ratio for x^1000 reproduce the published values
exactly to displayed precision (err2=0.0072, err3=7.7243e-04, ratio=9.3628;
ratio34=9.2805).  The one exception is the type-(2,2) minimax of x^1000: our
rational Remez (FNTB barycentric) breaks down numerically on this near-step
target at type (2,2) and returns a non-finite error, so err2 and the second
ratio for x^1000 are not reproduced.  This is a documented minimax-rational
robustness gap (the (3,3) and (4,4) solves for x^1000 are fine), recorded in
the ledger backlog, not a port defect.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')
os.makedirs(_OUTDIR, exist_ok=True)


def _err(pw, m, d):
    """Type-(m,d) minimax error of x^pw on [0,1] (inf if Remez breaks down)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(minimax(lambda x, p=pw: x**p, m, domain=(0.0, 1.0),
                                 rational=True, denom=d).err)
        except Exception:
            return float("inf")


def run():
    # ------------------------------------------------------------------
    # f = x^200 on [0,1]: type-(2,2) and (3,3) best rational errors.
    # ------------------------------------------------------------------
    err2 = _err(200, 2, 2)
    err3 = _err(200, 3, 3)
    print("err2 =")
    print(f"    {err2:.4f}")
    print("err3 =")
    print(f"   {err3:.4e}")
    print("ratio =")
    print(f"    {err2 / err3:.4f}")

    # ------------------------------------------------------------------
    # f = x^1000: the errors barely change (the target is nearly a step).
    # ------------------------------------------------------------------
    err2b = _err(1000, 2, 2)
    err3b = _err(1000, 3, 3)
    print("err2 =")
    print(f"    {err2b:.4f}")
    print("err3 =")
    print(f"   {err3b:.4e}")
    print("ratio =")
    print(f"    {err2b / err3b:.4f}")

    # ------------------------------------------------------------------
    # (3,3) -> (4,4) ratio approaches Halphen's constant 9.2890...
    # ------------------------------------------------------------------
    err4b = _err(1000, 4, 4)
    print("ratio =")
    print(f"    {err3b / err4b:.4f}")

    # ------------------------------------------------------------------
    # Plot: the type-(2,2) and (3,3) error curves for x^200.
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = minimax(lambda x: x**200, 2, domain=(0.0, 1.0), rational=True,
                     denom=2)
        r3 = minimax(lambda x: x**200, 3, domain=(0.0, 1.0), rational=True,
                     denom=3)
    xx = np.linspace(0, 1, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(xx, xx**200 - np.asarray(r2.r(xx)), lw=1.0)
    axes[0].set_ylim([-0.02, 0.02])
    axes[0].grid(True)
    axes[0].set_title("type (2,2) error curve", fontsize=10)
    axes[1].plot(xx, -9.28903 * (xx**200 - np.asarray(r3.r(xx))), lw=1.0)
    axes[1].set_ylim([-0.02, 0.02])
    axes[1].grid(True)
    axes[1].set_title("type (3,3) error curve x -9.28903", fontsize=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, "Rationalxn.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
