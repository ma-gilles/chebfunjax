"""The perimeter of an ellipse.

Faithful replica of geom/Ellipse.m by Kuan Xu (October 2012): the
arc length of an ellipse with axes 1/pi and 0.8/pi.

Original: https://www.chebfun.org/examples/geom/Ellipse.html
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 2 * np.pi)
    x = cj.chebfun(lambda t: (0.5 / jnp.pi) * jnp.cos(t),
                   domain=dom)
    y = cj.chebfun(lambda t: (0.4 / jnp.pi) * jnp.sin(t),
                   domain=dom)
    ts = np.linspace(*dom, 500)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(np.asarray(x(ts)), np.asarray(y(ts)), '-', lw=2)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Ellipse_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    speed = (x.diff()**2 + y.diff()**2).sqrt()
    arc_length = float(speed.abs().sum())
    print("arc_length =")
    print(f"   {arc_length:.15f}")
    print("exact 0.90277992777219")


if __name__ == "__main__":
    run()
