"""Arc length of complex paths.

Translation of complex/ComplexArcLength.m by Kuan Xu (October
2012): arc lengths of the keyhole contour (whole and per piece) and of
a flower curve, with equal-arclength points found by rootfinding.

Original: https://www.chebfun.org/examples/complex/ComplexArcLength.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')


def _save(fig, k):
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"ComplexArcLength_{k:02d}.png"),
             size=(600, 270))


def _show_row(name, v, per_line=3):
    """MATLAB format long display of a row vector, split into columns."""
    print(f"{name} =")
    v = list(v)
    for i in range(0, len(v), per_line):
        j = min(i + per_line, len(v))
        print(f"  Columns {i + 1} through {j}" if j - i > 1
              else f"  Column {j}")
        print("".join(f"{x:20.15f}" for x in v[i:j]))


def run():
    os.makedirs(_IMG, exist_ok=True)

    r, R, e = 0.2, 2, 0.1
    t = cj.chebfun('t', domain=[0, 1])                      # parameter
    c = [-R + e * 1j, -r + e * 1j, -r - e * 1j, -R - e * 1j]

    def cpow(a):
        # a.^t for a complex scalar a (our Chebfun has no __rpow__)
        return cj.exp(t * np.log(a))

    z = (c[0] + t * (c[1] - c[0])).join(                  # top of the keyhole
        c[1] * cpow(c[2]) / cpow(c[1]),                    # inner circle
        c[2] + t * (c[3] - c[2]),                          # bottom of the keyhole
        c[3] * cpow(c[0]) / cpow(c[3]))                    # outer circle
    lw = 1.6
    fig, ax = plt.subplots()
    # one chebfun, one colour (matlab_plot cycles colours per piece of a
    # complex piecewise chebfun)
    matlab_plot(z, ax=ax, lw=lw, color="C0")
    ax.set_aspect("equal")
    _save(fig, 1)
    plt.close(fig)

    L = z.arc_length()
    print("L =")
    print(f"{L:20.15f}")

    z = [c[0] + t * (c[1] - c[0]),                         # Top of the keyhole
         c[1] * cpow(c[2]) / cpow(c[1]),                   # Inner circle
         c[2] + t * (c[3] - c[2]),                         # Bottom of the keyhole
         c[3] * cpow(c[0]) / cpow(c[3])]                   # Outer circle
    L = [zk.arc_length() for zk in z]
    _show_row("L", L)

    t = cj.chebfun('t', domain=[0, 1])
    s = cj.exp(1j * 2 * np.pi * t) * (0.5 * cj.sin(8 * np.pi * t) ** 2 + 0.5)
    fig, ax = plt.subplots()
    matlab_plot(s, ax=ax, lw=lw)
    ax.set_aspect("equal")
    _save(fig, 2)

    L = s.arc_length()
    print("L =")
    print(f"{L:20.15f}")

    N = 64
    h = L / N
    T = np.zeros(N)

    t0 = time.perf_counter()
    len_ = abs(s.diff()).cumsum()
    for k in range(1, N):
        T[k] = float((len_ - k * h).roots()[0])
    print(f"Elapsed time is {time.perf_counter() - t0:.6f} seconds.")

    P = np.asarray(s(jnp.asarray(T)))
    # MS = 16: MATLAB draws '.' at about a third of MarkerSize,
    # matplotlib at about half, hence 12
    ax.plot(P.real, P.imag, ".r", markersize=12)
    _save(fig, 3)

    print("ans =")
    print(f"{len(len_):12d}")

    t0 = time.perf_counter()
    g = len_.inv()
    print(f"Elapsed time is {time.perf_counter() - t0:.6f} seconds.")

    Q = np.asarray(s(g(jnp.arange(N) * h)))
    ax.plot(Q.real, Q.imag, "ok", markersize=8, markerfacecolor="none")
    _save(fig, 4)
    plt.close(fig)


if __name__ == "__main__":
    run()
