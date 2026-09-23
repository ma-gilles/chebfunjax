"""The Rayleigh quotient on the sphere.

Faithful replica of sphere/RayleighQuotientExample.m: the eigenvalues
of a random symmetric 3x3 matrix A recovered by maximizing the
Rayleigh quotient q = x'Ax over the sphere -- lambda1 from max2(q),
lambda2 from the max of q restricted (as a trig chebfun) to the great
circle orthogonal to the first eigenvector, and lambda3 a quarter
turn further along that circle.

A uses a numpy seed (MATLAB's rng(52509) is not reproducible); every
printed error against eig(A) is a sample-independent identity.

Original: https://www.chebfun.org/examples/sphere/RayleighQuotientExample.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.spherefun.spherefun import Spherefun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(52509)
    A = 10 * (2 * rng.random((3, 3)) - 1)
    A = 0.5 * (A + A.T)

    def q_fn(lam, th):
        x = np.cos(lam) * np.sin(th)
        y = np.sin(lam) * np.sin(th)
        z = np.cos(th)
        return (A[0, 0] * x * x + A[1, 1] * y * y + A[2, 2] * z * z
                + 2 * A[0, 1] * x * y + 2 * A[0, 2] * x * z
                + 2 * A[1, 2] * y * z)

    q = Spherefun.from_function(q_fn)

    # Surface plot of the quadratic form.
    n = 260
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    V = q_fn(L, T)
    fig, ax = plt.subplots(figsize=(7.0, 5.6),
                           subplot_kw={"projection": "3d"})
    xs, ys, zs = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T),
                  np.cos(T))
    vmax = np.max(np.abs(V))
    ax.plot_surface(xs, ys, zs, facecolors=plt.cm.viridis(
        (V + vmax) / (2 * vmax)), rstride=2, cstride=2, linewidth=0,
        antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "RayleighQuotientExample_repl_01.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)

    lambda1, loc = q.max2()
    lambda1 = float(lambda1)
    loc = np.asarray(loc, dtype=float).ravel()
    print("lambda1 =")
    print(f"   {lambda1:.15f}")
    lamA = np.sort(np.linalg.eigvalsh(A))[::-1]
    print("error =")
    print(f"     {abs(lamA[0] - lambda1):.15e}")

    # Great circle orthogonal to the first eigenvector.
    l0, t0 = loc[0], loc[1]

    def xp(t):
        return np.cos(l0) * np.cos(t0) * np.cos(t) - np.sin(l0) * np.sin(t)

    def yp(t):
        return np.sin(l0) * np.cos(t0) * np.cos(t) + np.cos(l0) * np.sin(t)

    def zp(t):
        return -np.sin(t0) * np.cos(t)

    def q_cart(x, y, z):
        return (A[0, 0] * x * x + A[1, 1] * y * y + A[2, 2] * z * z
                + 2 * A[0, 1] * x * y + 2 * A[0, 2] * x * z
                + 2 * A[1, 2] * y * z)

    f = chebfun(lambda t: q_cart(xp(t), yp(t), zp(t)),
                domain=(-np.pi, np.pi), trig=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    tt = np.linspace(-np.pi, np.pi, 1000)
    ax.plot(tt, np.asarray(f(tt)), lw=2)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "RayleighQuotientExample_repl_02.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)

    locf, lambda2 = f.max()            # (x_max, f_max)
    lambda2, locf = float(lambda2), float(locf)
    print("lambda2 =")
    print(f"   {lambda2:.15f}")
    print("error =")
    print(f"     {abs(lamA[1] - lambda2):.15e}")

    lambda3 = float(f(locf + np.pi / 2))
    print("lambda3 =")
    print(f"  {lambda3:.15f}")
    print("error =")
    print(f"     {abs(lamA[2] - lambda3):.15e}")


if __name__ == "__main__":
    run()
