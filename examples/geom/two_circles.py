"""Overlap of two circles.

The boundary arcs are built as Chebfuns with splitting on (each has a
square-root branch point at an endpoint); their intersection points come
from roots() and the lens area from a Chebfun integral.  Faithful port of
geom/TwoCircles.m.

Original: https://www.chebfun.org/examples/geom/TwoCircles.html
Author: Nick Trefethen, May 2016
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
from chebfunjax.plotting import chebfun_style

chebfun_style()


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # MATLAB:
    #   bigcircle    = chebfun(@(x) sqrt(4-(x-1).^2), 'splitting', 'on');
    #   littlecircle = chebfun(@(x) 2-sqrt(1-(x+1).^2), [-1,0], 'splitting','on');
    bigcircle = cj.chebfun(lambda x: jnp.sqrt(jnp.maximum(4 - (x - 1)**2, 0.0)),
                           splitting=True)
    littlecircle = cj.chebfun(
        lambda x: 2 - jnp.sqrt(jnp.maximum(1 - (x + 1)**2, 0.0)),
        domain=(-1.0, 0.0), splitting=True)

    # Intersection points: x = roots(bigcircle{-1,0} - littlecircle)
    diff = bigcircle.restrict(-1.0, 0.0) - littlecircle
    x = np.sort(np.asarray(diff.roots()).ravel())
    x = x[(x > -1.0) & (x < 0.0)]
    x1, x2 = float(x[0]), float(x[1])
    print(f"x1 = {x1:.15f}")
    print(f"x2 = {x2:.15f}")

    # Lens area: sum(bigcircle{x1,x2} - littlecircle{x1,x2})
    area = float((bigcircle.restrict(x1, x2) - littlecircle.restrict(x1, x2)).sum())
    print(f"area = {area:.15f}")

    # Exact: acos(5*sqrt(2)/8) + 4*acos(11*sqrt(2)/16) - sqrt(7)/2
    exact = (np.arccos(5 * np.sqrt(2) / 8) + 4 * np.arccos(11 * np.sqrt(2) / 16)
             - np.sqrt(7) / 2)
    print(f"exact = {exact:.15f}")

    # --- Plot -----------------------------------------------------------
    fig, ax = plt.subplots()
    xb = np.linspace(-1.0, 1.0, 400)
    xl = np.linspace(-1.0, 0.0, 200)
    ax.plot(xb, np.asarray(bigcircle(jnp.array(xb))), color='#0072BD', lw=2,
            label='big circle')
    ax.plot(xl, np.asarray(littlecircle(jnp.array(xl))), 'k-', lw=2,
            label='little circle')
    xf = np.linspace(x1, x2, 200)
    ax.fill_between(xf, np.asarray(littlecircle(jnp.array(xf))),
                    np.asarray(bigcircle(jnp.array(xf))),
                    color='#D95319', alpha=0.4, label=f'lens ~ {area:.4f}')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title('Overlap of two circles', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'two_circles.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("two_circles: done")
    return True


if __name__ == "__main__":
    run()
