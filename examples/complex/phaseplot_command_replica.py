"""The phaseplot command.

Faithful replica of complex/PhaseplotCommand.m by Nick Trefethen
(March 2020): quick phase portraits directly from function handles.

Original: https://www.chebfun.org/examples/complex/PhaseplotCommand.html
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

FIG = [0]


def phaseplot(f, dom=(-1, 1, -1, 1), ax=None, title=None, n=480):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, n)
    ys = np.linspace(ya, yb, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all="ignore"):
        V = f(X + 1j * Y)
    H = (np.angle(V) + np.pi) / (2 * np.pi)
    standalone = ax is None
    if standalone:
        FIG[0] += 1
        fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.imshow(plt.cm.hsv(H), origin="lower", extent=(xa, xb, ya, yb),
              aspect="equal")
    if title:
        ax.set_title(title, fontsize=12)
    if standalone:
        fig.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(os.path.join(
            _IMG, f"PhaseplotCommand_repl_{FIG[0]:02d}.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    phaseplot(lambda z: z)
    phaseplot(lambda z: (z - 1) / (z + 1), (-2, 2, -2, 2))
    phaseplot(lambda z: z**3)
    phaseplot(lambda z: np.sqrt(z - 1) * np.sqrt(z + 1),
              (-2, 2, -2, 2))
    phaseplot(lambda z: np.exp(3.0 / z))

    FIG[0] += 1
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    phaseplot(lambda z: z, ax=axes[0], title="default colors")
    # 'classic' colors: hue rotated so red points east
    xs = np.linspace(-1, 1, 480)
    X, Y = np.meshgrid(xs, xs)
    H = np.mod(np.angle(X + 1j * Y) / (2 * np.pi) + 0.5, 1.0)
    axes[1].imshow(plt.cm.hsv(np.mod(H + 0.5, 1.0)), origin="lower",
                   extent=(-1, 1, -1, 1), aspect="equal")
    axes[1].set_title("'classic' colors", fontsize=12)
    for ax in axes:
        ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"PhaseplotCommand_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("phaseplots:", FIG[0])


if __name__ == "__main__":
    run()
