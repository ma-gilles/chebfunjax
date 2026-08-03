"""Encrypting a message with chebfuns.

Faithful replica of fun/Encryption.m by Nick Trefethen
(December 2011): adding a key scribble to a message scribble
encrypts it; subtracting decrypts; and a nonlinear scrambling via
exp(1.5i z) is undone with unwrap(log(.)).

Original: https://www.chebfun.org/examples/fun/Encryption.html
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

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Encryption_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pieces(cf, n=12):
    bps = [float(v) for v in cf.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, n)
        yield np.asarray(cf(t))


def _plot(zsegs, color):
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    for z in zsegs:
        ax.plot(z.real, z.imag, color, lw=1.6)
    ax.set_aspect("equal")
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    message = scribble("This is the message")
    key = scribble("Aardvarks eat ants")
    _plot(_pieces(message), 'b')
    _plot(_pieces(key), 'r')

    enc = message + key
    _plot(_pieces(enc), 'm')
    message2 = enc - key
    _plot(_pieces(message2), 'b')

    # nonlinear scrambling and its inverse (per continuous stroke)
    scr = [np.exp(1.5j * z) for z in _pieces(enc, 40)]
    _plot(scr, 'g')
    keysegs = list(_pieces(key, 40))
    dec = []
    for z, k in zip(scr, keysegs):
        w = (np.log(np.abs(z)) + 1j * np.unwrap(np.angle(z)))
        dec.append(w / 1.5j - 1 - k)
    _plot(dec, 'b')


if __name__ == "__main__":
    run()
