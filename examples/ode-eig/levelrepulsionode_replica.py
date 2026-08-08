"""Avoided crossings for ODE eigenvalues.

Faithful replica of ode-eig/LevelRepulsionODE.m by Abi Gopal and Nick
Trefethen (March 2017): the first six eigenvalues of the self-adjoint
fourth-order operator

    L(t) u = u'''' + t u'' + exp(x/20) u,   clamped u = u' = 0 at +-5,

tracked as chebfuns in t over [4.4, 5.4] (eps 1e-4), showing two pairs
of curves that nearly cross but not quite -- eigenvalue repulsion.

Original: https://www.chebfun.org/examples/ode-eig/LevelRepulsionODE.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _L(a):
    N = Chebop(lambda x, u, _a=a: (u.diff(4) + _a * u.diff(2)
                                   + (x / 20).exp() * u), domain=(-5, 5))
    N.lbc = lambda u: [u, u.diff()]
    N.rbc = lambda u: [u, u.diff()]
    return N


_CACHE = {}


def eigL(t):
    # MATLAB's default eigs selection keeps the LOWEST modes: at
    # t ~ 5.31 the bottom near-degenerate pair reaches |lam| = 5.3
    # while a higher mode sits at +4.68, so a pure smallest-magnitude
    # pick swaps set membership mid-interval and puts a jump (plus
    # global Gibbs ringing) into every E_k(t).  Emulate MATLAB by
    # taking 8 by magnitude and keeping the 6 with smallest real part.
    t = float(t)
    if t not in _CACHE:
        lam = np.sort(np.asarray(_L(t).eigs(k=8)).real)
        _CACHE[t] = lam[:6]
    return _CACHE[t]


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    d = (4.4, 5.4)
    t0 = time.time()
    E = []
    for k in range(6):
        E.append(chebfun(
            lambda tt, _k=k: np.array([eigL(t)[_k] for t in
                                       np.atleast_1d(tt)],
                                      dtype=np.float64),
            domain=d, eps=1e-4))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xx = np.linspace(*d, 600)
    for Ek in E:
        ax.plot(xx, np.asarray(Ek(xx)), lw=1.8)
    ax.grid(True)
    ax.set_title("Eigenvalues of L(t)")
    ax.set_xlabel("t")
    ax.set_ylim(-6, 2)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LevelRepulsionODE_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
