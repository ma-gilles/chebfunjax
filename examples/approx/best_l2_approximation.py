"""Best L2 polynomial approximation.

Translation of approx/BestL2Approximation.m by Alex Townsend
(October 2013): best L2 approximations via normalized Legendre
expansion, cheb2leg truncation, and the polyfit command, with the
|x| convergence-rate study.

Original: https://www.chebfun.org/examples/approx/BestL2Approximation.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import legpoly
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.transforms import cheb2leg, leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

LW = 1.6
FS = 16

# Output of MATLAB's `help chebfun/polyfit` as shown on the page.
POLYFIT_HELP = """\
 POLYFIT   Fit polynomial to a CHEBFUN.
    F = POLYFIT(Y, N) returns a CHEBFUN F corresponding to the polynomial of
    degree N that fits the CHEBFUN Y in the least-squares sense.

    If Y is a global polynomial of degree n then this code has an O(n (log n)^2)
    complexity. If Y is piecewise polynomial then it has an O(n^2) complexity.

    F = POLYFIT(X, Y, N, D), where D is a DOMAIN object, returns a CHEBFUN F on
    the domain D which corresponds to the polynomial of degree N that fits the
    data (X, Y) in the least-squares sense. X should be a real-valued column
    vector and Y should be a matrix with size(Y,1) = size(X,1).

    F = POLYFIT(Y, N) where Y is represented as a periodic TRIGFUN object
    returns the degree N trigonometric polynomial fit of length 2N+1.

    Note CHEBFUN/POLYFIT does not not support more than one output argument in
    the way that MATLAB/POLYFIT does.

  See also INTERP1.
"""


def _plot_pair(f, pn, title, fname, fs):
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    f.plot(ax=ax, linewidth=LW)
    pn.plot(ax=ax, color="r", linewidth=LW)
    ax.set_title(title, fontsize=0.75 * fs)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # help chebfun/polyfit
    print(POLYFIT_HELP, end="")

    # Best L2 approximation to |x| of degree 5 from the normalized
    # Legendre-Vandermonde quasimatrix: cleg = P'*f, pn = P*cleg.
    n = 5
    x = cj.chebfun(lambda t: t)
    f = cj.abs(x)
    P = legpoly(np.arange(n + 1), (-1.0, 1.0), 'norm')
    cleg = P.H * f
    # P*cleg (quasimatrix times vector): combine the columns.
    pn = cj.chebfun((P.coeffs @ cleg).ravel(), coeffs=True)
    _plot_pair(f, pn, r"Best $L^2$ approximation to $|x|$ of degree 5",
               "BestL2Approximation_01.png", 16)

    # Runge function via cheb2leg truncation.
    n = 10
    f = 1 / (1 + 25 * x**2)
    ccheb = f.coeffs
    cleg = cheb2leg(ccheb)
    cleg = cleg[:n + 1]
    ccheb = leg2cheb(cleg)
    pn = cj.chebfun(ccheb, coeffs=True)
    _plot_pair(f, pn, r"Best $L^2$ approx to Runge function of degree 10",
               "BestL2Approximation_02.png", 14)

    # The same thing with polyfit.
    pn = f.polyfit(n)
    _plot_pair(f, pn, r"Best $L^2$ approx to Runge function of degree 10",
               "BestL2Approximation_03.png", 14)

    # A large-degree fit of a sharper Runge function.
    n = 10000
    f = 1 / (1 + 1e6 * x**2)
    t0 = time.time()
    pn = f.polyfit(n)
    t = time.time() - t0
    print(f"L^2 error is {float((f - pn).norm()):1.3e}")
    print(f"L^2 approximation of degree {n} in t = {t:1.3f}")

    # Convergence for |x|: rate n^{-3/2}.
    f = cj.abs(x)
    nn = 10 ** np.arange(0, 4)
    err = [float((f - f.polyfit(int(k))).norm()) for k in nn]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.loglog(nn, err, 'k.-', lw=LW, ms=10,
              label=r"$\|\ |x| - p_n\ \|_2$")
    ax.loglog(nn, nn.astype(float) ** (-1.5), 'k--', lw=LW,
              label=r"$n^{-3/2}$")
    ax.legend()
    ax.set_xlabel("n", fontsize=0.75 * FS)
    ax.set_ylabel(r"$\|\ |x| - p_n\ \|_2$", fontsize=0.75 * FS)
    ax.set_title(r"Convergence of $\|\ |x| - p_n\ \|_2$", fontsize=0.75 * FS)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "BestL2Approximation_04.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
