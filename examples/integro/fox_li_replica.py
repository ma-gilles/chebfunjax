"""Eigenvalues of the Fox-Li integral operator.

Faithful replica of integro/FoxLi.m by Toby Driscoll and Nick
Trefethen (October 2010): the largest 80 eigenvalues of the Fox-Li
operator of laser theory with Fresnel number F = 64*pi,

    (Lu)(x) = sqrt(i F / pi) * int_{-1}^{1} exp(-i F (x-s)^2) u(s) ds,

which spiral in toward the origin from near the unit circle.

Original: https://www.chebfun.org/examples/integro/FoxLi.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.integral import fred_eigs
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'integro')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    F = 64 * np.pi
    K = lambda x, s: jnp.exp(-1j * F * (x - s)**2)  # noqa: E731
    t0 = time.time()
    lam = np.asarray(fred_eigs(K, k=80, which="LM",
                               scale=np.sqrt(1j * F / np.pi)))
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), '--r', lw=1.4)
    ax.plot(lam.real, lam.imag, 'k.', ms=9)
    ax.set_title("largest 80 eigenvalues of Fox-Li operator",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.axis(1.05 * np.array([-1, 1, -1, 1]))
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "FoxLi_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
