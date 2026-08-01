"""Checking vector calculus.

Faithful replica of veccalc/CheckingVectorCalculus.m by Alex Townsend,
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'veccalc')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # -- Parallelogram law -------------------------------------------
    F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x * y),
                                 lambda x, y: jnp.sin(x * y))
    G = Chebfun2v.from_functions(lambda x, y: x + y,
                                 lambda x, y: 1 + x + y)
    nF, nG = float(F.norm()), float(G.norm())
    FpG = Chebfun2v(components=[
        (Chebfun2(approx=a) + Chebfun2(approx=b)).approx
        for a, b in zip(F.components, G.components)])
    FmG = Chebfun2v(components=[
        (Chebfun2(approx=a) - Chebfun2(approx=b)).approx
        for a, b in zip(F.components, G.components)])
    lhs = 2 * nF ** 2 + 2 * nG ** 2
    rhs = float(FpG.norm()) ** 2 + float(FmG.norm()) ** 2
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
    C = (circ(1) + circ(3) * (1 / 1.5) + circ(8) * (1 / 3.5)) * 0.5
    v = float(np.real(np.asarray(F.integral(C))))
    print("v =")
    print(f"    {v:.15e}")

    xs = np.linspace(-1, 1, 600)
    cv = np.asarray(C(jnp.asarray(xs)))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xq = np.linspace(-1, 1, 12)
    Xq, Yq = np.meshgrid(xq, xq)
    U = np.asarray(Chebfun2(approx=F.components[0])(
        jnp.asarray(Xq.ravel()), jnp.asarray(Yq.ravel()))).reshape(Xq.shape)
    V = np.asarray(Chebfun2(approx=F.components[1])(
        jnp.asarray(Xq.ravel()), jnp.asarray(Yq.ravel()))).reshape(Xq.shape)
    ax.quiver(Xq, Yq, U, V, color="b")
    ax.plot(np.real(cv), np.imag(cv), "r", lw=1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "CheckingVectorCalculus_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- curl(grad f) = 0 --------------------------------------------
    cg = F.curl()
    xs2 = np.linspace(-0.95, 0.95, 30)
    vals = np.asarray(Chebfun2(approx=cg.approx if hasattr(cg, "approx")
                               else cg)(jnp.asarray(xs2),
                                        jnp.asarray(xs2))) \
        if not isinstance(cg, Chebfun2) else np.asarray(
            cg(jnp.asarray(xs2), jnp.asarray(xs2)))
    print("ans =")
    print(f"     {float(np.max(np.abs(vals))):.15e}")
    return True


if __name__ == "__main__":
    run()
