"""Rouche's theorem.

Faithful replica of complex/RoucheTheorem.m by Anthony Austin
(November 2012): illustrating Rouche's theorem on the unit circle for
sin(z) vs z and for a degree-7 polynomial vs its dominant term 15z^3.

Original: https://www.chebfun.org/examples/complex/RoucheTheorem.html
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

T = np.linspace(0, 2 * np.pi, 3000)
Z = np.exp(1j * T)
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"RoucheTheorem_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # f = z, g = sin(z)
    f = Z
    g = np.sin(Z)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(T, np.abs(f), lw=1.4)
    ax.plot(T, np.abs(f - g), lw=1.4)
    ax.set_title("|f| (above) and |f - g| (below) on the unit circle",
                 fontsize=12)
    ax.set_xlabel("t")
    ax.axis([0, 2 * np.pi, 0, 1.1])
    _save(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot(f.real, f.imag, lw=1.4)
    ax.plot(g.real, g.imag, lw=1.4)
    ax.set_title("Images of the unit circle under f and g", fontsize=12)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True)
    ax.set_aspect("equal")
    _save(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    q = g / f
    ax.plot(q.real, q.imag, lw=1.4)
    ax.set_title("Image of the unit circle under g/f", fontsize=12)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis([0, 1.5, -0.5, 0.5])
    ax.grid(True)
    ax.set_aspect("equal")
    _save(fig)

    # f = 15z^3, g = z^7 - 2z^5 + 15z^3 - z + 1
    f = 15 * Z**3
    g = Z**7 - 2 * Z**5 + 15 * Z**3 - Z + 1
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(T, np.abs(f), lw=1.4)
    ax.plot(T, np.abs(f - g), lw=1.4)
    ax.set_title("|f| (above) and |f - g| (below) on the unit circle",
                 fontsize=12)
    ax.set_xlabel("t")
    ax.axis([0, 2 * np.pi, 0, 16])
    _save(fig)

    r = np.roots([1, 0, -2, 0, 15, 0, -1, 1])
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(Z.real, Z.imag, lw=1.2)
    ax.plot(r.real, r.imag, 'o', ms=7, mfc='none')
    ax.set_title("Roots of g", fontsize=12)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis([-1.5, 1.5, -1.5, 1.5])
    ax.grid(True)
    ax.set_aspect("equal")
    _save(fig)
    inside = np.sum(np.abs(r) < 1)
    print(f"roots of g inside the unit circle: {inside} (Rouche: 3)")

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot(f.real, f.imag, lw=1.4)
    ax.plot(g.real, g.imag, lw=1.4)
    ax.set_title("Images of the unit circle under f and g", fontsize=12)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True)
    ax.set_aspect("equal")
    _save(fig)


if __name__ == "__main__":
    run()
