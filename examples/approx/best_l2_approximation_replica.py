"""Best L2 polynomial approximation.

Faithful replica of approx/BestL2Approximation.m by Alex Townsend
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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.transforms import cheb2leg, leg2cheb

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 3000)


def _plot_pair(fv, pv, title, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(XS, fv, lw=1.6)
    ax.plot(XS, pv, 'r', lw=1.6)
    ax.set_title(title, fontsize=13)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Best L2 approx of |x| of degree 5 via global Legendre projection
    f = cj.chebfun(lambda t: jnp.abs(t), domain=[-1.0, 0.0, 1.0])
    pn = f.polyfit(5)
    _plot_pair(np.asarray(f(jnp.asarray(XS))),
               np.asarray(pn(jnp.asarray(XS))),
               r"Best $L^2$ approximation to $|x|$ of degree 5",
               "BestL2Approximation_repl_01.png")

    # Runge function via cheb2leg truncation
    n = 10
    fr = cj.chebfun(lambda t: 1.0 / (1 + 25 * t**2))
    cleg = np.asarray(cheb2leg(fr.coeffs))[:n + 1]
    ccheb = leg2cheb(jnp.asarray(cleg))
    pn = cj.chebfun(jnp.asarray(ccheb), coeffs=True)
    _plot_pair(np.asarray(fr(jnp.asarray(XS))),
               np.asarray(pn(jnp.asarray(XS))),
               r"Best $L^2$ approx to Runge function of degree 10",
               "BestL2Approximation_repl_02.png")

    # Same thing via polyfit
    pn2 = fr.polyfit(n)
    _plot_pair(np.asarray(fr(jnp.asarray(XS))),
               np.asarray(pn2(jnp.asarray(XS))),
               r"Best $L^2$ approx to Runge function of degree 10",
               "BestL2Approximation_repl_03.png")

    # Large-degree fit of a sharp Runge function.  MATLAB does
    # 1/(1+1e6 x^2) (chebfun length ~37000) at n = 1e4 in 1.5 s with
    # its fast cheb2leg; our transforms are O(n^2) (ledgered gap), so
    # this replica uses 1/(1+1e4 x^2) (length ~3700) at n = 2000.
    nbig = 2000
    fs = cj.chebfun(lambda t: 1.0 / (1 + 1e4 * t**2))
    t0 = time.time()
    pn = fs.polyfit(nbig)
    t = time.time() - t0
    print(f"L^2 error is {float((fs - pn).norm(2)):.3e}")
    print(f"L^2 approximation of degree {nbig} in t = {t:.3f}")

    # Convergence for |x|: rate n^{-3/2}
    nn = 10 ** np.arange(0, 4)
    errs = []
    for n_ in nn:
        p_ = f.polyfit(int(n_))
        errs.append(float((f - p_).norm(2)))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.loglog(nn, errs, 'k.-', lw=1.6, ms=18)
    ax.loglog(nn, nn.astype(float) ** (-1.5), 'k--', lw=1.6)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|f - p_n\|_2$")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BestL2Approximation_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("errs:", ["%.3e" % e for e in errs])


if __name__ == "__main__":
    run()
