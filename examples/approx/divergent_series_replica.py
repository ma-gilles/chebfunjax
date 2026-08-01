"""Summing a divergent series.

Faithful replica of approx/DivergentSeries.m by Nick Trefethen (May
2011): Euler's divergent series 1 - 1 + 4 - 36 + 576 - ... summed via
the Borel-type integral f(x) = int_0^inf exp(-t)/(1+xt) dt, and via
Pade approximation of the series coefficients.

Original: https://www.chebfun.org/examples/approx/DivergentSeries.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import math
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.ratapprox import padeapprox

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # The integral of exp(-t)/(1+t) over [0, inf)
    g = cj.chebfun(lambda t: jnp.exp(-t) / (1 + t),
                   domain=[0.0, np.inf])
    print("ans =")
    print(f"   {float(g.sum()):.15f}")

    # f(x) = int_0^inf exp(-t)/(1+x t) dt as a chebfun in x
    def ff_scalar(x):
        h = cj.chebfun(lambda t: jnp.exp(-t) / (1 + x * t),
                       domain=[0.0, np.inf])
        return float(h.sum())

    def ff(x):
        arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        vals = [ff_scalar(float(v)) for v in arr.ravel()]
        return jnp.asarray(vals, dtype=jnp.float64).reshape(arr.shape)

    f = cj.chebfun(ff, domain=(0.0, 5.0))
    xs = np.linspace(0, 5, 1200)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, np.asarray(f(jnp.asarray(xs))), lw=1.4)
    ax.set_title("The integral f as a function of parameter x",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "DivergentSeries_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Taylor coefficients at 0 are (-1)^j (j!)^2
    for j in range(7):
        fj = f.diff(j) if j > 0 else f
        v = float(fj(jnp.asarray(0.0)))
        should = (-1) ** j * math.factorial(j) ** 2
        print(f"{v:21.12f}  (should be {should:7.0f})")

    # f(1) sums Euler's divergent series in the Borel sense
    print("ans =")
    print(f"   {float(f(jnp.asarray(1.0))):.15f}")

    # Pade approximation of the series coefficients
    c = np.array([(-1.0) ** k * math.factorial(k) for k in range(11)])
    r, *_ = padeapprox(c, 5, 5)
    print("ans =")
    print(f"   {float(np.real(r(np.asarray([1.0]))[0])):.15f}")

    # And at x = 0.5:
    print("ans =")
    print(f"   {float(f(jnp.asarray(0.5))):.15f}")
    print("ans =")
    print(f"   {float(np.real(r(np.asarray([0.5]))[0])):.15f}")


if __name__ == "__main__":
    run()
