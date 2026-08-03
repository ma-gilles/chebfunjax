"""A linear initial-value problem.

Faithful replica of ode-linear/LinearIVP.m (Nick Trefethen): solve
u'' + u = 0 on [0, 100] with u(0) = 1, u'(0) = 0 via chebop
backslash, recovering cos(x) over 16 periods.

Original: https://www.chebfun.org/examples/ode-linear/LinearIVP.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    x = cj.chebfun(lambda t: t, domain=(0, 100))
    L = Chebop(lambda x, u: u.diff(2) + u, domain=(0, 100))
    L.lbc = lambda u: [u - 1, u.diff()]
    u = L.solve(0.0)
    err = float((u - x.cos()).norm(jnp.inf))
    print(f"error = {err:7.2e}")

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.linspace(0, 100, 2000)
    ax.plot(t, np.asarray(u(t)), lw=1.6)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("cos(x)", fontsize=12)
    ax.set_title(f"Solution of IVP for cos(x) -- error = {err:7.2e}",
                 fontsize=14)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "LinearIVP_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
