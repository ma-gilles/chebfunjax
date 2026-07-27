"""Central Limit Theorem via convolution.

A triangular density X is convolved with itself; the renormalised sums
converge to a Gaussian.  A discrete coin-toss density (a pair of Dirac
masses) is convolved ten times to give the binomial distribution.
Faithful port of stats/CentralLimitTheorem.m.

Original: https://www.chebfun.org/examples/stats/CentralLimitTheorem.html
Authors: Nick Trefethen and Mohsin Javed, July 2012
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
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.plotting import chebfun_style

chebfun_style()


def _triangular_X():
    """MATLAB X = chebfun({0, '(4/3+x)/2', 0}, [-3 -4/3 2/3 3]).

    Built from explicit per-piece Chebfuns (a jnp.where keyed on the
    breakpoints flips under float rounding of the mapped Chebyshev nodes,
    which spuriously fails adaptive construction)."""
    pL = cj.chebfun(0.0, domain=(-3.0, -4.0 / 3.0))
    pM = cj.chebfun(lambda x: (4.0 / 3.0 + x) / 2.0, domain=(-4.0 / 3.0, 2.0 / 3.0))
    pR = cj.chebfun(0.0, domain=(2.0 / 3.0, 3.0))
    return Chebfun(funs=pL.funs + pM.funs + pR.funs,
                   domain=Domain((-3.0, -4.0 / 3.0, 2.0 / 3.0, 3.0)))


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/stats')
    os.makedirs(outdir, exist_ok=True)

    # --- Triangular density and its moments -------------------------------
    # MATLAB: t = chebfun('t',[-3 3]); mu = sum(t*X); variance = sum(t^2*X)
    X = _triangular_X()
    t = cj.chebfun(lambda t: t, domain=(-3.0, 3.0))
    mu = float((t * X).sum())
    variance = float(((t * t) * X).sum())
    print(f"mu = {mu:.15e}")
    print(f"variance = {variance:.15f}")

    sigma = np.sqrt(variance)
    xs = np.linspace(-3, 3, 600)
    gauss = np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # Renormalised self-convolutions of X (chebfun conv), rescaled to [-3, 3].
    X2 = X.conv(X)
    X3 = X2.conv(X)

    def _renorm(F, k):
        s = np.sqrt(k)
        a, b = float(F.domain.a) / s, float(F.domain.b) / s
        xr = np.linspace(a, b, 600)
        yr = s * np.asarray(F(jnp.array(xr * s)))
        return xr, np.real(yr)

    fig, axes = plt.subplots(1, 3)
    Xx = np.linspace(-3, 3, 600)
    for ax, (data, title) in zip(axes, [
        ((Xx, np.asarray(X(jnp.array(Xx)))), 'Distribution of X'),
        (_renorm(X2, 2), 'Renormalized (X+X)/sqrt2'),
        (_renorm(X3, 3), 'Renormalized (X+X+X)/sqrt3'),
    ]):
        ax.plot(data[0], np.real(data[1]), color='#0072BD', lw=2)
        ax.plot(xs, gauss, color='#D95319', ls='--', lw=2)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Central Limit Theorem', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'central_limit_theorem.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Coin toss: convolution of Dirac point masses ---------------------
    # MATLAB: p1 = q*dirac(x-0) + p*dirac(x-1); p2 = conv(p1,p1); sum(p2); ...
    # p1 is a pair of Dirac masses (weights q, p at 0, 1).  Convolving Dirac
    # trains is the discrete convolution of their point masses, so the k-fold
    # self-convolution is the binomial distribution and each total mass is 1.
    # (chebfunjax's conv() integrates the smooth part only and does not yet
    # propagate Dirac deltas, so the point masses are convolved directly.)
    p = 0.6
    q = 1.0 - p
    p1 = np.array([q, p])          # masses at x = 0, 1
    p2 = np.convolve(p1, p1)       # conv(p1, p1)
    print(f"sum(p2) = {float(np.sum(p2)):.15g}")

    n = 10
    pn = p2
    for _k in range(3, n + 1):
        pn = np.convolve(pn, p1)
    print(f"sum(pn) = {float(np.sum(pn)):.15f}")

    # MATLAB: mu = n*p; sigma = sqrt(n*p*q)
    mu_binom = n * p
    sigma_binom = np.sqrt(n * p * q)
    print(f"mu = {mu_binom:.15g}")
    print(f"sigma = {sigma_binom:.15f}")

    print("central_limit_theorem: done")
    return True


if __name__ == "__main__":
    run()
