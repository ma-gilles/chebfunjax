"""Global optimization: the Rosenbrock function.

The 2D minimum is found by minimising over vertical slices: fminx(x) =
min_y f(x,y) is built as a 1D Chebfun (splitting on), and its own minimum is
the global one.  Faithful port of opt/Rosenbrock.m by Nick Trefethen
(October 2010).

Original: https://www.chebfun.org/examples/opt/Rosenbrock.html

Output-parity note (measured): the min VALUES match the published output
to full precision (minf part2 -0.969232500643148 exact); minimiser
LOCATIONS agree to ~1e-9 -- the sqrt(eps) conditioning floor of a flat
quadratic minimum, scheme-dependent.  Our splitting finds the two
published breakpoints exactly (-0.635872022371398, 0.210237104254783)
plus one extra; and the published part-1 minf, a roundoff-scale
-1.6e-14 printed in format long e, carries a displayed-precision
tolerance of 5e-30 that no reimplementation can meet.  Classified
scheme-dependent DIFF, not a defect.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


def _slice_min(f2d, x0, ydom):
    """min_y f(x0, y) evaluated over an array of x0 (MATLAB @(x0) min(chebfun(...)))."""
    x0 = np.asarray(x0)
    out = np.array([
        float(cj.chebfun(lambda y, xi=float(xv): f2d(xi, y), domain=ydom).min()[1])
        for xv in np.ravel(x0)
    ])
    return jnp.asarray(out.reshape(x0.shape))


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/opt')
    os.makedirs(outdir, exist_ok=True)

    # --- 1. The classic Rosenbrock function ------------------------------
    # MATLAB: f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;
    #   fminx = chebfun(@(x0) min(chebfun(@(y) f(x0,y),[-1 3])),[-1.5 1.5],'splitting','on');
    #   [minf,minx] = min(fminx); [minf,miny] = min(chebfun(@(y) f(minx,y),[-1 3]));
    def f1(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    fminx = cj.chebfun(lambda x: _slice_min(f1, x, (-1.0, 3.0)),
                       domain=(-1.5, 1.5), splitting=True)
    minx, minf = fminx.min()
    print(f"minf = {float(minf):.15e}")
    print(f"minx = {float(minx):.15f}")
    cy = cj.chebfun(lambda y: f1(float(minx), y), domain=(-1.0, 3.0))
    miny, _ = cy.min()
    print(f"miny = {float(miny):.15f}")

    # --- 2. Function with several local minima ---------------------------
    # MATLAB: f = @(x,y) exp(x-2*x.^2-y.^2).*sin(6*(x+y+x.*y.^2));
    def f2(x, y):
        return jnp.exp(x - 2 * x**2 - y**2) * jnp.sin(6 * (x + y + x * y**2))

    fminx2 = cj.chebfun(lambda x: _slice_min(f2, x, (-1.0, 1.0)),
                        domain=(-1.0, 1.0), splitting=True)
    print("fminx.ends =", "  ".join(f"{b:.15f}"
                                    for b in fminx2.domain.breakpoints))
    minx2, minf2 = fminx2.min()
    print(f"minf = {float(minf2):.15f}")
    print(f"minx = {float(minx2):.15f}")
    cy2 = cj.chebfun(lambda y: f2(float(minx2), y), domain=(-1.0, 3.0))
    miny2, _ = cy2.min()
    print(f"miny = {float(miny2):.15f}")

    # --- Plot the two slice-minimum curves -------------------------------
    fig, axes = plt.subplots(1, 2)
    xx1 = np.linspace(-1.5, 1.5, 300)
    axes[0].plot(xx1, np.asarray(fminx(jnp.array(xx1))), color='#0072BD', lw=1.6)
    axes[0].set_title('min_y (1-x)^2+100(y-x^2)^2', fontsize=10)
    xx2 = np.linspace(-1.0, 1.0, 300)
    axes[1].plot(xx2, np.asarray(fminx2(jnp.array(xx2))), color='#D95319', lw=1.6)
    axes[1].set_title('min_y exp(...)sin(...)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rosenbrock.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("rosenbrock: done")
    return True


if __name__ == "__main__":
    run()
