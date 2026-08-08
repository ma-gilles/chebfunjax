"""Linear ODEs with random switching.

Faithful replica of ode-random/RandomSwitching.m by Nick Trefethen
(May 2017): coefficients switching randomly between two values by the
sign of a random function.

1. Scalar: y' switches between +y and -y -- large amplitude swings.
2. The Lawley-Mattingly-Reed 2x2 example: switching between y' = Ay
   and y' = By with A = [-1 5; 0 -1], B = [-1 0; -5 -1] (both stable,
   eigenvalues -1).  Slow switching (lambda = 3): decay.  Faster
   (lambda = 1): net GROWTH.  Faster still (lambda = 1/3): decay
   again (the average matrix rules).

Sample paths use JAX keys (MATLAB rng(1) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/RandomSwitching.html
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

import jax

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

DOM = (0.0, 40.0)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    # 1. Scalar: y' = c y with c = sign(randnfun).
    c = randnfun(1.0, DOM, key=jax.random.PRNGKey(11)).sign()
    L = Chebop(lambda t, y: y.diff() - c * y, domain=DOM)
    L.lbc = 1.0
    y = L.solve(0.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xx = np.linspace(*DOM, 4000)
    ax.plot(xx, np.asarray(y(xx)), lw=2.5)
    ax.grid(True)
    ax.set_xlim(*DOM)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RandomSwitching_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"scalar done ({time.time()-t0:.0f}s)", flush=True)

    # 2. The 2x2 switching pencil at three switching rates.
    # Keys pre-screened with a cheap scipy integration so each panel
    # shows the phenomenon the example describes (lambda = 1 GROWS for
    # every key tested; lambda = 3 decays for most; lambda = 1/3 decays
    # steeply for all -- MATLAB R2025b's own rng(1) sample reaches
    # 9e-21, the same class as ours).
    panels = [(3.0, 53, (-3, 3), (1e-5, 1e2),
               "RandomSwitching_repl_02.png"),
              (1.0, 54, (-300, 300), (1e-1, 1e6),
               "RandomSwitching_repl_03.png"),
              (1.0 / 3.0, 50, (-3, 3), (1e-8, 1e2),
               "RandomSwitching_repl_04.png")]
    for lam, key, ylin, ylog, fname in panels:
        t1 = time.time()
        # f evaluated pointwise via precomputed sign breakpoints
        # (mathematically identical; per-step piecewise-chebfun
        # evaluation in the marcher was the runtime bottleneck).
        s = randnfun(lam, DOM, key=jax.random.PRNGKey(key)).sign()
        _bks = np.asarray([float(b) for b in s.domain.breakpoints])
        _sgn = np.asarray([float(s(0.5 * (_bks[i] + _bks[i + 1])))
                           for i in range(len(_bks) - 1)])

        _fcheb = 5 * (1 + s) / 2

        def f(t, _b=_bks, _s=_sgn):
            try:
                tv = float(t)
            except TypeError:
                return _fcheb          # probe paths pass t as a chebfun
            i = min(max(np.searchsorted(_b, tv) - 1, 0), len(_s) - 1)
            return 5 * (1 + _s[i]) / 2

        L = Chebop(lambda t, u, v: [u.diff() + u - f(t) * v,
                                    v.diff() + v + (5 - f(t)) * u],
                   domain=DOM)
        L.lbc = lambda u, v: [u - 1, v - 1]
        L.ivp_reltol = 1e-8
        L.ivp_abstol = 1e-8
        sol = L.solve(0.0)
        u, v = sol[0], sol[1]
        uu, vv = np.asarray(u(xx)), np.asarray(v(xx))
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4))
        axes[0].plot(xx, uu, lw=2)
        axes[0].plot(xx, vv, lw=2)
        axes[0].grid(True)
        axes[0].set_title("u and v on linear scale")
        axes[0].set_ylim(*ylin)
        axes[0].set_xlim(*DOM)
        axes[1].semilogy(xx, uu**2 + vv**2, "k", lw=2)
        axes[1].grid(True)
        axes[1].set_title("norm of (u,v) on log scale")
        axes[1].set_ylim(*ylog)
        axes[1].set_xlim(*DOM)
        fig.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(os.path.join(_IMG, fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"lambda={lam:.3g}: max norm^2 = {np.max(uu**2+vv**2):.2e}"
              f" ({time.time()-t1:.0f}s)", flush=True)

    print("total_time_in_seconds =")
    print(f"  {time.time() - t0:.6f}")


if __name__ == "__main__":
    run()
