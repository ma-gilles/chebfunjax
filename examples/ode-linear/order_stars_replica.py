"""Order stars.

Faithful replica of ode-linear/OrderStars.m by Nick Trefethen
(December 2011): the order star of the type (2,3) Pade approximant
r(z) to exp(z), traced as the level curve |r(z) e^{-z}| = 1 via
chebfun2 roots.

Original: https://www.chebfun.org/examples/ode-linear/OrderStars.html
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
from scipy.special import factorial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.ratapprox import padeapprox

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    c = 1.0 / factorial(np.arange(19))
    out = padeapprox(c, 2, 3)
    r = out[0] if isinstance(out, tuple) else out

    def smash_op(x, y):
        z = np.asarray(x) + 1j * np.asarray(y)
        v = np.asarray(r(z.ravel())).reshape(z.shape) * np.exp(-z)
        return jnp.asarray(np.tanh(np.abs(v)**2) / np.tanh(1))

    d = (-6.0, 6.0, -6.0, 6.0)
    f = cj.chebfun2(smash_op, domain=d)
    star = (f - 1).roots()

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    t = np.linspace(-1, 1, 1200)
    for curve in star:
        zv = np.asarray(curve(jnp.asarray(t)))
        ax.plot(zv.real, zv.imag, 'k', lw=1.6)
    ax.axis(d)
    ax.set_aspect("equal")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OrderStars_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
