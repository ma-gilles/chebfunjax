"""The fractal structure of the Lorenz attractor, via rational approximation.

Faithful replica of ode-nonlin/LorenzAttractor.m: integrate the Lorenz
system to t = 5 at tight tolerance, then use ratinterp to study the
complex-t singularity structure of the trajectory components. The poles
of the rational approximants line up just above and below the real axis,
and the three components agree on where they are.

Original: https://www.chebfun.org/examples/ode-nonlin/LorenzAttractor.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import ratinterp
from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
D = (0.0, 5.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"LorenzAttractor_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rat_eval(a, b, z):
    """p(z)/q(z) on complex z from Chebyshev coefficients on D."""
    mid, h = 0.5 * (D[0] + D[1]), 0.5 * (D[1] - D[0])
    s = (np.asarray(z) - mid) / h
    from numpy.polynomial import chebyshev as C
    return C.chebval(s, np.asarray(a)) / C.chebval(s, np.asarray(b))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    def fun(t, u):
        return np.array([10 * (u[1] - u[0]),
                         28 * u[0] - u[1] - u[0] * u[2],
                         u[0] * u[1] - (8 / 3) * u[2]])

    u = ode113(fun, D, np.array([-14.0, -15.0, 20.0]),
               rtol=1e-13, atol=1e-13)
    u1, u2, u3 = u

    # 3D trajectory
    t = np.linspace(*D, 20000)
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot(np.asarray(u1(t)), np.asarray(u2(t)), np.asarray(u3(t)),
            lw=1.0)
    ax.view_init(elev=20, azim=20 - 90)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-40, 40)
    ax.set_zlim(5, 45)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.set_zlabel("z(t)")
    ax.set_title("A 3D Trajectory of the Lorenz Attractor", fontsize=14)
    _save(fig)

    # the components as scalar functions of t
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for f in (u1, u2, u3):
        ax.plot(t, np.asarray(f(t)), lw=1.2)
    ax.grid(True)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t), y(t), z(t)")
    ax.set_title("Solution to the Lorenz Attractor as Scalar Functions",
                 fontsize=14)
    _save(fig)

    # rational approximants and their poles
    rats = []
    for f, m, NN in ((u1, 221, 444), (u2, 241, 484), (u3, 236, 473)):
        rh, a, b, mu, nu, poles, res = ratinterp(
            f, m, 40, NN, None, 1e-12, domain=D)
        rats.append((a, b, np.asarray(poles)))

    xx = np.linspace(-0.5, 5.5, 200)
    yy = np.linspace(-0.5, 0.5, 200)
    XX, YY = np.meshgrid(xx, yy)
    zz = XX + 1j * YY
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.4))
    for k, (a, b, poles) in enumerate(rats):
        ax = axes[k]
        ax.contour(xx, yy, np.abs(_rat_eval(a, b, zz)),
                   levels=np.arange(0, 151, 5))
        ax.grid(True)
        ax.set_title(f"r{k + 1}(t)")
        ax.plot(poles.real, poles.imag, "xk", markersize=9)
        ax.plot([0, 5], [0, 0], "k", lw=1.6)
    _save(fig)

    # the pole table: the three components agree on the singularities
    p1, p2, p3 = (np.sort_complex(p[np.argsort(np.abs(p.imag))][:10])
                  for p in (r[2] for r in rats))
    diff = np.sort(np.max(np.abs(
        np.array([p1 - p2, p1 - p3, p2 - p3])), axis=0))
    print("   poles in x         poles in y         poles in z"
          "         max. difference")
    for k in range(10):
        print(f"   {p1[k].real:6.4f} {p1[k].imag:+.4f}i"
              f"   {p2[k].real:6.4f} {p2[k].imag:+.4f}i"
              f"   {p3[k].real:6.4f} {p3[k].imag:+.4f}i"
              f"   {diff[k]:6.4f}")
    print()
    print("half differences:")
    print(np.round(0.5 * diff, 4))

    # with tol = 0: all 40 poles, spurious ones included
    rh, a, b, mu, nu, poles, res = ratinterp(
        u1, 221, 40, 444, None, 0.0, domain=D)
    pol = np.asarray(poles)
    pol = pol[np.argsort(pol.real)]
    print("\npoles (tol = 0, first 20):")
    for z in pol[:20]:
        print(f"   {z.real:7.4f} {z.imag:+.4f}i")

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(pol.real, pol.imag, "xk", markersize=9)
    ax.plot([0, 5], [0, 0], "k", lw=1.6)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)
    ax.set_title("poles with tol = 0: spurious poles appear on the axis")
    _save(fig)


if __name__ == "__main__":
    run()
