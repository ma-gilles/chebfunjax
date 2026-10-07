"""A keyhole contour integral.

Translation of complex/KeyholeContour.m by Nick Trefethen and
Nick Hale (October 2010): integrating log(x)tanh(x) around a keyhole
contour to get 4i*pi*log(pi/2).

Original: https://www.chebfun.org/examples/complex/KeyholeContour.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = lambda z: cj.log(z) * cj.tanh(z)  # noqa: E731
    r, R, e = 0.2, 2.0, 0.1
    c = [complex(-R, e), complex(-r, e), complex(-r, -e),
         complex(-R, -e)]
    s = cj.chebfun(lambda s: s, domain=(0.0, 1.0))
    z = (c[0] + s * (c[1] - c[0])).join(
        c[1] * c[2] ** s / c[1] ** s,
        c[2] + s * (c[3] - c[2]),
        c[3] * c[0] ** s / c[3] ** s,
    )

    # Plot the joined complex Chebfun, then overlay the branch cut.
    fig, ax = plt.subplots(figsize=(5.1, 3.88))
    matlab_plot(z, ax=ax, linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("A keyhole contour in the complex plane", fontsize=11)
    ax.plot([-2.6, 0], [0, 0], "-r", lw=0.5)
    ax.set_xlim(-2.6, 2.6)
    ax.tick_params(labelsize=9, width=0.5, length=3, direction="in")
    # MATLAB default normalized axes position; equal aspect contracts the
    # wider box to the source contour's data ratio.
    ax.set_position([0.13, 0.11, 0.775, 0.815])
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "KeyholeContour_01.png"), size=(510, 388))
    plt.close(fig)

    # Source sequence: integral of f(z) times the contour derivative.
    I = complex(np.asarray((f(z) * z.diff()).sum()))
    Iexact = 4j * np.pi * np.log(np.pi / 2)
    print("I =")
    print(f"  {I.real:.15f} + {I.imag:.15f}i")
    print("Iexact =")
    print(f"  {Iexact.real:.15f} + {Iexact.imag:.15f}i")
    print("error =")
    print(f"     {abs(I - Iexact):.15e}")


if __name__ == "__main__":
    run()
