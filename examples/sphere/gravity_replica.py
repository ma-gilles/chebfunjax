"""Gravitational force from a spherical shell.

Faithful replica of sphere/Gravity.m: Newton's theorem -- a uniform
spherical shell attracts an exterior point as if its mass were
concentrated at the center.  With the point X = (-1, -1.1, -0.2)
(|X| = 1.5) and shell density 1/(4 pi), the force integral over the
sphere equals 1/1.5^2 = 4/9 exactly.

Original: https://www.chebfun.org/examples/sphere/Gravity.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.spherefun.spherefun import Spherefun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')

X = np.array([-1.0, -1.1, -0.2])


def r_fn(lam, th):
    x = np.cos(lam) * np.sin(th)
    y = np.sin(lam) * np.sin(th)
    z = np.cos(th)
    return np.sqrt((X[0] - x)**2 + (X[1] - y)**2 + (X[2] - z)**2)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    print("ans =")
    print(f"   {np.linalg.norm(X):.15f}")

    r = Spherefun.from_function(r_fn)
    print("min_distance =")
    print(f"   {float(r.min2()[0]):.15f}")
    print("max_distance =")
    print(f"   {float(r.max2()[0]):.15f}")

    # Distance contours on the sphere with the point mass.
    n = 300
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    R = r_fn(L, T)
    fig, ax = plt.subplots(figsize=(7.0, 5.6),
                           subplot_kw={"projection": "3d"})
    xs = np.cos(L) * np.sin(T)
    ys = np.sin(L) * np.sin(T)
    zs = np.cos(T)
    vmax = R.max()
    ax.plot_surface(xs, ys, zs, facecolors=plt.cm.viridis(R / vmax),
                    rstride=2, cstride=2, linewidth=0,
                    antialiased=False, alpha=0.9)
    ax.plot([X[0]], [X[1]], [X[2]], '.r', markersize=18)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(35, -10)
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Gravity_repl_01.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("force_exact =")
    print(f"   {1 / 1.5**2:.15f}")
    rho = 1 / (4 * np.pi)
    print("rho =")
    print(f"   {rho:.15f}")

    Xn = X / np.linalg.norm(X)

    def force_fn(lam_, th_):
        x = np.cos(lam_) * np.sin(th_)
        y = np.sin(lam_) * np.sin(th_)
        z = np.cos(th_)
        rr = r_fn(lam_, th_)
        num = (Xn[0] * (X[0] - x) + Xn[1] * (X[1] - y)
               + Xn[2] * (X[2] - z))
        return rho * num / rr**3

    force = Spherefun.from_function(force_fn)
    print("force =")
    print(f"   {float(force.sum2()):.15f}")


if __name__ == "__main__":
    run()
