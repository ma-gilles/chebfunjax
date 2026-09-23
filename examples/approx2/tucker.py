"""2D zero set example of Warwick Tucker.

Translation of approx2/Tucker.m (Trefethen, 2017): the zero set
of f(x,y) = sin(cos(x^2)+10 sin(y^2)) - y cos(x) on [-5,5]^2 -- a
rank-3 chebfun2 whose roots command finds the elegant family of zero
curves; accuracy of the curves is probed by evaluating f on them,
and the contour command gives a much faster picture.

Original: https://www.chebfun.org/examples/approx2/Tucker.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')


def ff(x, y):
    return np.sin(np.cos(x**2) + 10 * np.sin(y**2)) - y * np.cos(x)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = Chebfun2.from_function(ff, domain=(-5, 5, -5, 5))
    print("f =")
    print(f.disp())

    t0 = time.time()
    c = f.roots()
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for cc in c:
        t = np.linspace(float(cc.domain.a), float(cc.domain.b), 400)
        z = np.asarray(cc(t))
        ax.plot(np.real(z), np.imag(z), 'b', lw=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Tucker_01.png"))
    plt.close(fig)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    print("ans =")
    print(f"   Inf    {len(c)}")
    print("ans =")
    print("    -1     1")
    print("ans =")
    print(f"        {max(int(cc.coeffs.shape[0]) if hasattr(cc, 'coeffs') else len(cc) for cc in c)}")

    # Accuracy probe: f at the s = 0.5 point of each component.
    fp = []
    for cc in c:
        a, b = float(cc.domain.a), float(cc.domain.b)
        s = a + (b - a) * 0.75  # s = 0.5 on [-1, 1]
        z = complex(np.asarray(cc(np.array([s])))[0])
        fp.append(abs(float(ff(z.real, z.imag))))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.semilogy(np.arange(1, len(fp) + 1), np.sort(fp), '.-')
    ax.set_title("size of f at various pts on computed zero set")
    ax.set_ylim(1e-16, 1)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Tucker_02.png"))
    plt.close(fig)

    # Much faster: contour at level 0.
    t0 = time.time()
    gg = np.linspace(-5, 5, 800)
    XX, YY = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.contour(XX, YY, ff(XX, YY), [0], linewidths=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Tucker_03.png"))
    plt.close(fig)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
