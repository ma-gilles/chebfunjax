"""Procrustes shape analysis.

Faithful replica of geom/Procrustes.m by Alex Townsend
(August 2011): comparing shapes (a frisbee and a pebble) by
translating, scaling, and rotating complex-valued curves, then
measuring the L2 distance.

Original: https://www.chebfun.org/examples/geom/Procrustes.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import matplotlib.pyplot as plt

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]
TS = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
DT = 2 * np.pi / len(TS)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Procrustes_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mean(fv):
    return np.mean(fv)


def _norm(fv):
    return np.sqrt(np.sum(np.abs(fv)**2) * DT)


def shape_analysis(fv, gv):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    axes[0, 0].plot(fv.real, fv.imag, 'r', lw=2)
    axes[0, 0].plot(gv.real, gv.imag, 'k', lw=2)
    axes[0, 0].set_title("Original", fontsize=13)
    fv = fv - _mean(fv)
    gv = gv - _mean(gv)
    axes[0, 1].plot(fv.real, fv.imag, 'r', lw=2)
    axes[0, 1].plot(gv.real, gv.imag, 'k', lw=2)
    axes[0, 1].set_title("After translation", fontsize=13)
    fv = fv / _norm(fv)
    gv = gv / _norm(gv)
    axes[1, 0].plot(fv.real, fv.imag, 'r', lw=2)
    axes[1, 0].plot(gv.real, gv.imag, 'k', lw=2)
    axes[1, 0].set_title("After scaling", fontsize=13)
    jf = int(np.argmax(np.abs(fv)))
    jg = int(np.argmax(np.abs(gv)))
    rotf = np.angle(fv[jf])
    rotg = np.angle(gv[jg])
    fv = np.exp(-1j * rotf) * np.roll(fv, -jf)
    gv = np.exp(-1j * rotg) * np.roll(gv, -jg)
    axes[1, 1].plot(fv.real, fv.imag, 'r', lw=2)
    axes[1, 1].plot(gv.real, gv.imag, 'k', lw=2)
    axes[1, 1].set_title("After aligning", fontsize=13)
    for ax in axes.ravel():
        ax.set_aspect("equal")
    _save(fig)
    return fv, gv


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    t = TS
    fv = 3 * (1.5 * np.cos(t) + 1j * np.sin(t))
    gv = np.exp(1j * np.pi / 3) * (
        1 + np.cos(t) + 1.5j * np.sin(t)
        + 0.125 * (1 + 1.5j) * np.sin(3 * t)**2)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot(fv.real, fv.imag, 'r', lw=2)
    ax.plot(gv.real, gv.imag, 'k', lw=2)
    ax.set_aspect("equal")
    ax.set_title("Frisbee and pebble", fontsize=13)
    _save(fig)

    f1, g1 = shape_analysis(fv, gv)
    print("ans =")
    print(f"   {_norm(f1 - g1):.15f}")

    gv2 = np.exp(-1j * np.pi / 3) * (
        1 + np.cos(2 * np.pi - t) - 1.5j * np.sin(2 * np.pi - t)
        + 0.125 * (1 - 1.5j) * np.sin(3 * (2 * np.pi - t))**2)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot(gv.real, gv.imag, 'r', lw=2)
    ax.plot(gv2.real, gv2.imag, 'k', lw=2)
    ax.set_aspect("equal")
    ax.set_title("Pebble and its reflection", fontsize=13)
    _save(fig)
    f2, g2 = shape_analysis(gv, gv2)
    print("ans =")
    print(f"   {_norm(f2 - g2):.15f}")


if __name__ == "__main__":
    run()
