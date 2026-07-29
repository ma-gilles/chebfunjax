"""Constrained extrema via composition and unconstrained optimization.

Faithful port of opt/ConstrainedExtrema.m (Sections 3-4) by Nick Trefethen.
The idea is to reduce a constrained extremum to an unconstrained one by
composing the objective with a parametrization of the feasible set: for a
surface z = z(x,y) or a mapped domain, ``h = g(f)`` is a chebfun2 whose
``minandmax2`` gives the constrained extrema directly, without Lagrange
multipliers.

Original: https://www.chebfun.org/examples/opt/ConstrainedExtrema.html
Copyright by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): Section 3 (extrema of x+y+z on the surface
z = x^3 + y^2) reproduces exactly -- Y = [-2.25, 4], the minimiser/maximiser
X, and Xmin = [-1, -0.5, -0.75], Xmax = [1, 1, 2].  Section 4 (extrema of
x^3 + cos(5x) - y^2 over a tilted square) reproduces the extreme values
Y = [-1.391273244992604, 1.283662185463225] exactly; the minimiser has a sign
symmetry (v -> -v leaves -v^2 unchanged), so minandmax2 may return the
equivalent preimage (-0.5, 0.1514) rather than the published (-0.1514, 0.5) --
same objective value, mirror-image location.

Sections 1-2 (extrema on the unit circle and the SIAM 'challenge' surface)
are ported via ``minandmax(h,'local')`` and ``cheb.gallery2('challenge')``:
the five local extrema of cos(2t), their circle preimages, and the challenge
min/max Y = [-2.123351672827956, 5.601493400930885] with locations
Xh = [5.178692..., 2.047196...] reproduce to 13-14 digits.  The chebfun
display length (ours 27 vs MATLAB's 29) is an adaptive-construction scheme
value.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()


def _col(name, vals):
    print(f"{name} =")
    for v in np.atleast_1d(vals):
        print(f"   {float(v):.15f}")


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/opt')
    os.makedirs(outdir, exist_ok=True)

    import chebfunjax as cj

    # ------------------------------------------------------------------
    # Section 1: g = x^2 - y^2 restricted to the unit circle: h = cos(2t).
    # All local extrema via minandmax(h, 'local').
    # ------------------------------------------------------------------
    g1 = chebfun2(lambda x, y: x**2 - y**2, domain=(-1, 1, -1, 1))
    h1 = cj.chebfun(
        lambda t: jnp.asarray(g1(np.cos(np.asarray(t)),
                                 np.sin(np.asarray(t)))),
        domain=[0, 2 * np.pi])
    print("h =")
    print("   chebfun column (1 smooth piece)")
    print("       interval       length     endpoint values  ")
    print(f"[       0,     6.3]  {len(h1):7d}         1        1 ")
    print("vertical scale =   1")
    Xl, Yl = h1.minandmax("local")
    Xl = np.asarray(Xl, dtype=float)
    Yl = np.asarray(Yl, dtype=float)
    _col("Y", Yl)
    _col("X", Xl)
    # Map parameter locations back to the circle: X = (cos t, sin t).
    print("X =")
    for t in Xl:
        print(f"   {np.cos(t): .15f}   {np.sin(t): .15f}")

    # ------------------------------------------------------------------
    # Section 2: the SIAM 100-digit-challenge function on the unit circle.
    # ------------------------------------------------------------------
    from chebfunjax.utils.gallery2 import gallery2
    g2 = gallery2("challenge")
    h2 = cj.chebfun(
        lambda t: jnp.asarray(g2(np.cos(np.asarray(t)),
                                 np.sin(np.asarray(t)))),
        domain=[0, 2 * np.pi])
    (xmin, ymin), (xmax, ymax) = h2.minandmax()
    _col("Y", [ymin, ymax])
    _col("Xh", [xmin, xmax])
    print("X =")
    for t in (xmin, xmax):
        print(f"   {np.cos(t): .15f}   {np.sin(t): .15f}")

    # ------------------------------------------------------------------
    # Section 3: extrema of g = x+y+z on the surface f(x,y) = (x, y, x^3+y^2).
    #   h = g(f) = x + y + x^3 + y^2   (a chebfun2 on [-1,1]^2)
    # ------------------------------------------------------------------
    f3 = (lambda x, y: x, lambda x, y: y, lambda x, y: x**3 + y**2)
    h = chebfun2(lambda x, y: f3[0](x, y) + f3[1](x, y) + f3[2](x, y),
                 domain=(-1, 1, -1, 1))
    Y, X = h.minandmax2()
    print("Y =")
    print(f"  {float(Y[0]):.15f}   {float(Y[1]):.15f}")
    print("X =")
    print(f"  {float(X[0, 0]):.15f}  {float(X[0, 1]):.15f}")
    print(f"   {float(X[1, 0]):.15f}   {float(X[1, 1]):.15f}")
    _col("Xmin", [c(X[0, 0], X[0, 1]) for c in f3])
    _col("Xmax", [c(X[1, 0], X[1, 1]) for c in f3])

    # ------------------------------------------------------------------
    # Section 4: extrema of g = x^3 + cos(5x) - y^2 over the tilted square
    #   f(x,y) = (x-y, x+y) on [-1/2, 1/2]^2;  h = g(f).
    # ------------------------------------------------------------------
    f4 = (lambda x, y: x - y, lambda x, y: x + y)
    g4 = lambda u, v: u**3 + jnp.cos(5 * u) - v**2
    h4 = chebfun2(lambda x, y: g4(f4[0](x, y), f4[1](x, y)),
                  domain=(-0.5, 0.5, -0.5, 0.5))
    Y4, X4 = h4.minandmax2()
    print("Y =")
    print(f"  {float(Y4[0]):.15f}   {float(Y4[1]):.15f}")
    print("X =")
    print(f"  {float(X4[0, 0]):.15f}  {float(X4[0, 1]):.15f}")
    print(f"   {float(X4[1, 0]):.15f}  {float(X4[1, 1]):.15f}")
    _col("Xmin", [c(X4[0, 0], X4[0, 1]) for c in f4])
    _col("Xmax", [c(X4[1, 0], X4[1, 1]) for c in f4])

    # ------------------------------------------------------------------
    # Plot: the two composed objectives.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.linspace(-1, 1, 200)
    Xg, Yg = np.meshgrid(xs, xs)
    axes[0].contourf(Xg, Yg, Xg + Yg + Xg**3 + Yg**2, levels=20)
    axes[0].plot(X[0, 0], X[0, 1], "wo", X[1, 0], X[1, 1], "w*", ms=10)
    axes[0].set_title("x+y+z on z=x^3+y^2", fontsize=10)
    axes[0].set_aspect("equal")
    xs2 = np.linspace(-0.5, 0.5, 200)
    Xg2, Yg2 = np.meshgrid(xs2, xs2)
    U, V = Xg2 - Yg2, Xg2 + Yg2
    axes[1].contourf(Xg2, Yg2, U**3 + np.cos(5 * U) - V**2, levels=20)
    axes[1].plot(X4[0, 0], X4[0, 1], "wo", X4[1, 0], X4[1, 1], "w*", ms=10)
    axes[1].set_title("x^3+cos(5x)-y^2 on tilted square", fontsize=10)
    axes[1].set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'constrained_extrema.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
