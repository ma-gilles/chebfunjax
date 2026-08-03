"""A curve of constant width that is not a circle.

Faithful replica of geom/ConstantWidth.m by Nick Trefethen
(February 2020, after Rabinowitz): a degree-8 algebraic curve of
constant width 18, from the zero contour of a chebfun2 — width
checks in five directions and the perimeter.

Original: https://www.chebfun.org/examples/geom/ConstantWidth.html
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

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    def r2(x, y):
        return x**2 + y**2

    def xy(x, y):
        return x**2 - 3 * y**2

    def p(x, y):
        return (r2(x, y)**4 - 45 * r2(x, y)**3
                - 41283 * r2(x, y)**2 + 7950960 * r2(x, y)
                + 16 * xy(x, y)**3 + 48 * r2(x, y) * xy(x, y)**2
                + x * xy(x, y) * (16 * r2(x, y)**2
                                  - 5544 * r2(x, y) + 266382)
                - 373248000)

    pc = cj.chebfun2(p, domain=(-11, 11, -11, 11))
    curves = pc.roots()
    # gather the boundary as complex samples
    rs = []
    for c in curves:
        bps = [float(v) for v in c.domain.breakpoints]
        t = np.linspace(bps[0], bps[-1], 2000)
        rs.append(np.asarray(c(t)))
    r = np.concatenate(rs)

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    ax.fill(r.real, r.imag, color=(0.722, 0.451, 0.20))
    ax.axis([-12, 12, -12, 12])
    ax.set_aspect("equal")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ConstantWidth_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("theta/pi     width")
    for k in range(5):
        theta = np.pi * k / 5
        a = np.exp(1j * theta)
        width = np.max((a * r).real) - np.min((a * r).real)
        print(f"{theta/np.pi:8.5f} {width:12.8f}")

    poly = lambda x: (x**8 + 16 * x**7 + 19 * x**6  # noqa: E731
                      - 5544 * x**5 - 41283 * x**4
                      + 266382 * x**3 + 7950960 * x**2
                      - 373248000)
    print("ans =")
    print(f"     {poly(-8.0):g}")
    print("ans =")
    print(f"     {poly(10.0):g}")

    # perimeter = arc length of the boundary chebfun(s)
    per = 0.0
    for c in curves:
        dc = c.diff()
        bps = [float(v) for v in c.domain.breakpoints]
        t = np.linspace(bps[0], bps[-1], 4000)
        v = np.abs(np.asarray(dc(t)))
        per += np.trapezoid(v, t)
    print("perimeter =")
    print(f"  {per:.15f}")
    print("time_for_this_example =")
    print(f"   {time.time()-t0:.6f}")


if __name__ == "__main__":
    run()
