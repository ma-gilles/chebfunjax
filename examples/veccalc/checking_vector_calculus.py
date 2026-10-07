"""Checking vector calculus.

Translation of veccalc/CheckingVectorCalculus.m by Alex Townsend,
March 2013: the parallelogram law for chebfun2v norms, the gradient
theorem for line integrals, a closed-curve integral of a gradient
field, and curl(grad f) = 0.

Original: https://www.chebfun.org/examples/veccalc/CheckingVectorCalculus.html
Copyright 2013 by The University of Oxford and The Chebfun Developers.
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
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'veccalc')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # -- Introduction: two source construction forms -----------------
    F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x),
                                 lambda x, y: jnp.sin(y))
    f = chebfun2(lambda x, y: jnp.sin(x))
    g = chebfun2(lambda x, y: jnp.sin(y))
    G = Chebfun2v(components=[f.approx, g.approx])

    # -- Parallelogram law -------------------------------------------
    F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x * y),
                                 lambda x, y: jnp.sin(x * y))
    G = Chebfun2v.from_functions(lambda x, y: x + y,
                                 lambda x, y: 1 + x + y)
    nF, nG = float(F.norm()), float(G.norm())
    lhs = 2 * nF ** 2 + 2 * nG ** 2
    rhs = float((F + G).norm()) ** 2 + float((F - G).norm()) ** 2
    print("ans =")
    print(f"     {abs(lhs - rhs):.15e}")

    # -- Gradient theorem --------------------------------------------
    f = chebfun2(lambda x, y: jnp.sin(2 * x) + x * y ** 2)
    F = f.gradient()
    C = cj.chebfun(lambda t: t * jnp.exp(100j * t),
                   domain=[0.0, np.pi / 10])
    v = float(np.real(np.asarray(F.integral(C))))
    ends = (float(np.asarray(f(np.pi / 10, 0.0)))
            - float(np.asarray(f(0.0, 0.0))))
    print("ans =")
    print(f"     {abs(v - ends):.15e}")

    # -- Closed curve: integral of a gradient field is zero ----------
    circ = lambda p: cj.chebfun(
        lambda x: jnp.exp(2j * p * np.pi * x + 0.8j), domain=[-1.0, 1.0])
    C = (circ(1) + circ(3) / 1.5 + circ(8) / 3.5) / 2
    v = float(np.real(np.asarray(F.integral(C))))
    print("v =")
    print(f"    {v:.15e}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig, ax = F.quiver(ax=ax, n_pts=12, autoscale_factor=0.5)
    fig, ax = C.plot(ax=ax, color="r")
    # MATLAB's default axes rectangle for the source 600-by-400 figure.
    # Both plotting helpers apply tight_layout; restore the source rectangle.
    ax.set_position((0.13, 0.11, 0.775, 0.815))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "CheckingVectorCalculus_01.png"),
             size=(600, 400))
    plt.close(fig)

    # -- curl(grad f) = 0 --------------------------------------------
    # MATLAB norm(curl(grad(f))) is the continuous Frobenius/L2 norm.
    # The two-component Python curl returns a scalar SeparableApprox.
    cg = Chebfun2(approx=f.gradient().curl())
    print("ans =")
    print(f"     {float(cg.norm()):.15e}")
    return True


if __name__ == "__main__":
    run()
