"""Polynomial level curve of constant width.

A degree-8 polynomial p(x,y) whose zero set is a curve of constant width
(like the British 50p coin).  The boundary is extracted as the zero curve of
a chebfun2 (roots), then its width across five directions and its perimeter
are measured.  Faithful port of geom/ConstantWidth.m.

Original: https://www.chebfun.org/examples/geom/ConstantWidth.html
Author: Nick Trefethen, May 2022
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()


def _p(x, y):
    r2 = x**2 + y**2
    xy = x**2 - 3 * y**2
    return (r2**4 - 45 * r2**3 - 41283 * r2**2 + 7950960 * r2
            + 16 * xy**3 + 48 * r2 * xy**2
            + x * xy * (16 * r2**2 - 5544 * r2 + 266382) - 373248000)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # MATLAB: pc = chebfun2(p,[-11 11 -11 11]); r = roots(pc);
    pc = chebfun2(_p, domain=(-11.0, 11.0, -11.0, 11.0))
    r = pc.roots()[0]  # the (single) closed zero curve, a complex chebfun

    # Width across five directions: max(real(a*r)) - min(real(a*r)).
    print("theta/pi     width")
    for theta in np.pi * np.arange(5) / 5:
        a = complex(np.exp(1j * theta))
        rp = (r * a).real()
        width = float(rp.max()[1]) - float(rp.min()[1])
        print(f"{theta / np.pi:8.5f} {width:12.8f}")

    # y=0 slice polynomial, evaluated at -8 and 10 (both roots -> 0).
    def p1(x):
        return (x**8 + 16 * x**7 + 19 * x**6 - 5544 * x**5 - 41283 * x**4
                + 266382 * x**3 + 7950960 * x**2 - 373248000)
    print(f"p(-8) = {p1(-8.0):.6g}")
    print(f"p(10) = {p1(10.0):.6g}")

    # Perimeter = norm(diff(r), 1) = arc length.
    print(f"perimeter = {float(r.diff().norm(1)):.15f}")

    # --- Plot the curve ---------------------------------------------------
    tt = np.linspace(r.domain.a, r.domain.b, 2000)
    z = np.asarray(r(jnp.array(tt)))
    fig, ax = plt.subplots()
    ax.fill(np.real(z), np.imag(z), color=[0.722, 0.451, 0.20])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Curve of constant width', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'constant_width.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("constant_width: done")
    return True


if __name__ == "__main__":
    run()
