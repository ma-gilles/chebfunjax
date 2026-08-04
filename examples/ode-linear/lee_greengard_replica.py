"""Lee & Greengard ODE test problems.

Faithful replica of ode-linear/LeeGreengardODEs.m by Nick Trefethen
(December 2016): the six challenging linear BVPs from Lee & Greengard
(1997) — viscous shock, Bessel with nu=100 on [0,600], the Airy-type
turning point, two interior-turning-point problems on piecewise
domains, and a cusp — each solved for two values of the small
parameter.

Original: https://www.chebfun.org/examples/ode-linear/LeeGreengardODEs.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"LeeGreengardODEs_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pair(make, eps_pow, color, ylim=None):
    fig, axs = plt.subplots(2, 1, figsize=(9.2, 6.6))
    for k, ax in zip((1, 2), axs):
        ep = 10.0 ** (-eps_pow * k)
        t0 = time.time()
        u = make(ep)
        el = time.time() - t0
        dom = (float(u.domain.a), float(u.domain.b))
        t = np.linspace(dom[0], dom[1], 2000)
        ax.plot(t, np.asarray(u(t)), color=color, lw=1.2)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True)
        ax.set_title(f"Ep = {ep:5.1e}   Length ={len(u):4d}   "
                     f"Time ={el:6.3f}", fontsize=10)
        print(f"ep={ep:.1e} len={len(u)} time={el:.1f}s", flush=True)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Viscous shock: ep u'' + 2x u' = 0, u(-1)=-1, u(1)=1
    def shock(ep):
        N = Chebop(lambda x, u: ep * u.diff(2) + 2 * x * u.diff(),
                   domain=(-1, 1))
        N.lbc = -1
        N.rbc = 1
        return N.solve(0.0)
    _pair(shock, 2, 'm', ylim=(-1.4, 1.4))

    # 2. Bessel: x^2 u'' + x u' + (x^2 - nu^2) u = 0 on [0, 600]
    nu = 100
    N = Chebop(lambda x, u: x**2 * u.diff(2) + x * u.diff()
               + (x**2 - nu**2) * u, domain=(0, 600))
    N.lbc = 0
    N.rbc = 1
    t0 = time.time()
    u = N.solve(0.0)
    el = time.time() - t0
    print(f"bessel len={len(u)} time={el:.1f}s", flush=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    t = np.linspace(0, 600, 6000)
    ax.plot(t, np.asarray(u(t)), lw=1.0)
    ax.grid(True)
    ax.set_title(f"nu = {nu}   Length ={len(u):4d}   Time ={el:6.3f}",
                 fontsize=10)
    _save(fig)

    # 3. Turning point: ep u'' - x u = 0, u(-1)=1, u(1)=1
    def airy(ep):
        N = Chebop(lambda x, u: ep * u.diff(2) - x * u,
                   domain=(-1, 1))
        N.lbc = 1
        N.rbc = 1
        return N.solve(0.0)
    _pair(airy, 3, 'r')

    # 4. Two turning points: ep u'' + (x^2 - 0.25) u = 0
    def twoturn(ep):
        N = Chebop(lambda x, u: ep * u.diff(2) + (x**2 - 0.25) * u,
                   domain=(-1, 1))
        N.lbc = 1
        N.rbc = 2
        return N.solve(0.0)
    _pair(twoturn, 3, (0, 0.7, 0))

    # 5. Boundary layers at an interior point (piecewise domain)
    def interior(ep):
        N = Chebop(lambda x, u: ep * u.diff(2) + x * u.diff()
                   - 0.5 * u, domain=(-1, 0, 1))
        N.lbc = 1
        N.rbc = 2
        return N.solve(0.0)
    _pair(interior, 3, (1, 0.5, 0.5))

    # 6. Cusp: ep u'' - x u' + u = 0 (piecewise domain)
    def cusp(ep):
        N = Chebop(lambda x, u: ep * u.diff(2) - x * u.diff() + u,
                   domain=(-1, 0, 1))
        N.lbc = 1
        N.rbc = 2
        return N.solve(0.0)
    _pair(cusp, 3, 'b')


if __name__ == "__main__":
    run()
