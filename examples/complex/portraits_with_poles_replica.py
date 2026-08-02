"""Phase portraits for functions with poles.

Faithful replica of complex/PortraitsWithPoles.m by Nick Trefethen
(May 2017): the smash trick g = f/(1+|f|^2) makes pole-bearing
functions representable while preserving their phase.

Original: https://www.chebfun.org/examples/complex/PortraitsWithPoles.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

D = 1.5
XS = np.linspace(-D, D, 520)
X, Y = np.meshgrid(XS, XS)
ZZ = X + 1j * Y
FIG = [0]


def smash(v):
    with np.errstate(all="ignore"):
        g = v / (1 + np.abs(v) ** 2)
    return np.where(np.isnan(g), 0.0, g)


def portrait(V, title):
    FIG[0] += 1
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(plt.cm.hsv(H), origin="lower", extent=(-D, D, -D, D),
              aspect="equal")
    ax.set_title(title, fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"PortraitsWithPoles_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # z^2 (z^3 - 1): five zeros, no poles
    portrait(ZZ**2 * (ZZ**3 - 1), "z^2 (z^3 - 1)")

    # z^-2 (z^3 - 1): a double pole at 0 (colors circulate backwards)
    portrait(smash(ZZ**-2 * (ZZ**3 - 1)), "z^{-2} (z^3 - 1), smashed")

    # tan((3+3i)z): strings of zeros and poles along a rotated line
    portrait(smash(np.tan((3 + 3j) * ZZ)), "tan((3+3i)z), smashed")
    print("portraits:", FIG[0])


if __name__ == "__main__":
    run()
