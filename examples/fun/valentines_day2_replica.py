"""Another Valentine's Day heart.

Faithful replica of fun/ValentinesDay2.m by Alex Townsend
(February 2013): a 3D heart surface (the same parametrization as
VolumeOfHeart) with a scribbled greeting wrapped around it.

Original: https://www.chebfun.org/examples/fun/ValentinesDay2.html
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
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = np.linspace(0, 1, 90)
    th = np.linspace(0, 4 * np.pi, 180)
    T, TH = np.meshgrid(t, th)
    X = np.sin(np.pi * T) * np.cos(TH / 2)
    Y = 0.7 * np.sin(np.pi * T) * np.sin(TH / 2)
    Z = ((T - 1) * (-49 + 50 * T + 30 * T * np.cos(TH)
                    + np.cos(2 * TH)) / (-25 + np.cos(TH)**2))
    C = np.sin(10 * X) * np.cos((Y - 0.1)**2) + (Z + 1)

    fig = plt.figure(figsize=(8.2, 7.6))
    ax = fig.add_subplot(projection="3d")
    norm = plt.Normalize(C.min(), C.max())
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.hot(norm(C)),
                    rstride=2, cstride=2, linewidth=0)
    S = scribble("Happy Valentines Day!")
    bps = [float(v) for v in S.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        u = np.linspace(a, b, 12)
        z = np.asarray(S(u))
        ax.plot(1.1 * np.cos(2.5 * (z.real + 1)),
                0.8 * np.sin(2.5 * (z.real + 1)),
                1.5 * z.imag - 1.05, 'k', lw=2)
    # MATLAB view(180, 6): matplotlib azimuth is offset by -90 deg
    ax.view_init(elev=6, azim=90)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-2, 0.1333)
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ValentinesDay2_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
