"""An Allen-Cahn equation with continuation.

Faithful replica of ode-nonlin/AllenCahn.m: the steady Allen-Cahn
problem

    Eps u'' + u - u^3 = sin(x),   u(0) = 1, u(10) = -1,

solved first at Eps = 2 and then continued down through
1, 0.5, 0.2, 0.1, 0.03, 0.01, 0.003, each solve starting from the
previous solution. As Eps shrinks the solution develops ever sharper
interior layers.

Original: https://www.chebfun.org/examples/ode-nonlin/AllenCahn.html
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

from chebfunjax import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
DOM = (0.0, 10.0)


def _save_solution(u, eps, secs):
    FIG[0] += 1
    xx = np.linspace(*DOM, 4000)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(xx, np.asarray(u(xx)), lw=1.6)
    ax.set_title(f"Eps = {eps:5.1e}    length(u) = {len(u)}"
                 f"    time = {secs:3.1f} secs", fontsize=14)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"AllenCahn_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Eps = {eps:7.3g}   length(u) = {len(u):4d}   "
          f"time = {secs:5.1f}s", flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = chebfun(lambda x: np.sin(x), domain=DOM)

    eps = 2.0
    N = Chebop(lambda u, _e=eps: _e * u.diff(2) + u - u**3, DOM, 1, -1)
    t0 = time.time()
    u = N.solve(f)
    _save_solution(u, eps, time.time() - t0)

    for eps in (1, 0.5, 0.2, 0.1, 0.03, 0.01, 0.003):
        N = Chebop(lambda u, _e=eps: _e * u.diff(2) + u - u**3, DOM, 1, -1)
        N.init = u
        t0 = time.time()
        u = N.solve(f)
        _save_solution(u, eps, time.time() - t0)


if __name__ == "__main__":
    run()
