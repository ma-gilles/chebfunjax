"""Rational approximation of monomials.

Faithful replica of approx/Rationalxn.m by Yuji Nakatsukasa and Nick
Trefethen (May 2018): successive type (k,k) rational minimax errors
for x^n on [0,1] decrease by Halphen's constant 9.28903...

Original: https://www.chebfun.org/examples/approx/Rationalxn.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(0, 1, 4000)


def _errplot(fvals, rvals, title, fname, scale=1.0):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, scale * (fvals - rvals), lw=1.3)
    ax.grid(True)
    ax.set_ylim(-0.02, 0.02)
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f200 = lambda x: x**200  # noqa: E731
    fv = XS**200
    r2 = minimax(f200, 2, rational=True, denom=2, domain=(0.0, 1.0))
    _errplot(fv, np.asarray(r2.r(XS)), "Type (2,2) error curve",
             "Rationalxn_repl_01.png")
    r3 = minimax(f200, 3, rational=True, denom=3, domain=(0.0, 1.0))
    _errplot(fv, np.asarray(r3.r(XS)),
             "Type (3,3) error curve multiplied by -9.28903",
             "Rationalxn_repl_02.png", scale=-9.28903)
    print("err2 =")
    print(f"    {r2.err:.4f}")
    print("err3 =")
    print(f"   {r3.err:.4e}")
    print("ratio =")
    print(f"    {r2.err/r3.err:.4f}")

    f1000 = lambda x: x**1000  # noqa: E731
    # The default AAA-Lawson initial reference degenerates for this
    # extremely localized function (MATLAB's minimax needed its own
    # re-initialization fallback here too); seed the (2,2) solve from
    # the x^200 reference mapped by x -> x^(1/5), since
    # x^1000 = (x^5)^200.
    xk200 = np.asarray(r2.xk, dtype=np.float64)
    init = np.sort(np.clip(xk200 ** (1.0 / 5.0), 0.0, 1.0))
    init[0] = 0.0
    e2 = minimax(f1000, 2, rational=True, denom=2, domain=(0.0, 1.0),
                 init_xk=init).err
    e3 = minimax(f1000, 3, rational=True, denom=3, domain=(0.0, 1.0)).err
    print("err2 =")
    print(f"    {e2:.4f}")
    print("err3 =")
    print(f"   {e3:.4e}")
    print("ratio =")
    print(f"    {e2/e3:.4f}")

    e4 = minimax(f1000, 4, rational=True, denom=4, domain=(0.0, 1.0)).err
    print("ratio =")
    print(f"    {e3/e4:.4f}")


if __name__ == "__main__":
    run()
