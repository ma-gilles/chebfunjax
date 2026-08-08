"""Parity partitioning a spherefun.

Faithful replica of sphere/SpherefunPartition.m by Behnam Hashemi
(November 2016): partition splits a spherefun into its
even/pi-periodic and odd/pi-anti-periodic parts, whose CDR columns
and rows carry the corresponding parities; the integral lives
entirely in the even/periodic part.

Original: https://www.chebfun.org/examples/sphere/SpherefunPartition.html
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
FIG = [0]


def _plot_sf(F, title="", n=220, ax=None, fig=None):
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    V = np.asarray(F(L.ravel(), T.ravel())).reshape(L.shape)
    X, Y, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T), np.cos(T))
    vmax = max(np.max(np.abs(V)), 1e-300)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(
        (V + vmax) / (2 * vmax)), rstride=1, cstride=1, linewidth=0,
        antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)


def _save(fig, name):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"SpherefunPartition_repl_{FIG[0]:02d}.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = Spherefun.from_function(lambda lam, th:
        0.5 + np.sinh(5 * (np.cos(lam) * np.sin(th))
                      * (np.sin(lam) * np.sin(th)) * np.cos(th))
        * np.cos(np.cos(lam) * np.sin(th)
                 - np.sin(lam) * np.sin(th) + 2 * np.cos(th)))
    print("f rank:", f.rank)
    fig = plt.figure(figsize=(6.6, 6.2))
    ax = fig.add_subplot(projection="3d")
    _plot_sf(f, ax=ax, fig=fig)
    _save(fig, "01")

    fep, foa = f.partition()
    print("fep rank:", fep.rank)
    print("foa rank:", foa.rank)
    print("err =")
    print(f"     {float((fep + foa - f).norm()):g}")

    fig = plt.figure(figsize=(10.6, 5.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _plot_sf(fep, "even/periodic part", ax=ax, fig=fig)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_sf(foa, "odd/anti-periodic part", ax=ax, fig=fig)
    _save(fig, "02")

    # Columns and rows of the two parts carry the parities.
    th = np.linspace(-np.pi, np.pi, 1200)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for c in fep.cols:
        ax.plot(th, np.asarray(c(th)), lw=1.0)
    ax.grid(True)
    ax.set_title("Columns of the even part of f")
    _save(fig, "03")

    lamg = np.linspace(-np.pi, np.pi, 1200)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for r in fep.rows:
        ax.plot(lamg, np.asarray(r(lamg)), lw=1.0)
    ax.grid(True)
    ax.set_title(r"Rows of the $\pi$-periodic part of f")
    _save(fig, "04")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for c in foa.cols:
        ax.plot(th, np.asarray(c(th)), lw=1.0)
    ax.grid(True)
    ax.set_title("Columns of the odd part of f")
    _save(fig, "05")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for r in foa.rows:
        ax.plot(lamg, np.asarray(r(lamg)), lw=1.0)
    ax.grid(True)
    ax.set_title(r"Rows of the $\pi$-anti-periodic part of f")
    _save(fig, "06")

    print("sum_f =")
    print(f"   {float(f.sum2()):.15f}")
    print("sum_foa =")
    print(f"     {float(foa.sum2()):g}")
    print("sum_fep =")
    print(f"   {float(fep.sum2()):.15f}")


if __name__ == "__main__":
    run()
