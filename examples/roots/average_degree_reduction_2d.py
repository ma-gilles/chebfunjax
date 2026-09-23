"""The average degree reduction of subdivision (2D).

Translation of roots/AverageDegreeReduction2D.m by Alex Townsend
(August 2013): the tau parameter for domain subdivision in bivariate
rootfinding, computed from chebfun2 coefficient matrices on
subsquares.

Original: https://www.chebfun.org/examples/roots/AverageDegreeReduction2D.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

TOL = 1e-14
FIG = [0]


def _coeff_len(g):
    """MATLAB ``find(max(abs(rot90(X,2))) < tol, 1, 'last')`` on
    ``X = chebcoeffs2(g)``: the (1-based) last column of the doubly
    reversed coefficient matrix whose maximum is below tol."""
    X = np.rot90(np.asarray(g.chebcoeffs2()), 2)
    idx = np.where(np.max(np.abs(X), axis=0) < TOL)[0]
    return int(idx[-1] + 1) if idx.size else 0


def compute_tau(op, N):
    g = cj.chebfun2(op)
    L = _coeff_len(g)
    x = np.linspace(-1, 1, 2**N + 1)
    tot = 0
    for j in range(len(x) - 1):
        for k in range(len(x) - 1):
            g = cj.chebfun2(op, domain=(x[j], x[j + 1],
                                        x[k], x[k + 1]))
            tot += _coeff_len(g)
    avg = tot / (len(x) - 1) ** 2
    tau = (avg / L) ** (1.0 / N)
    print(f"Tau = {tau:.5f}")


def subdivision_diagram(op):
    FIG[0] += 1
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.8))
    for levels in range(4):
        fs = round(14 - 2.5 * levels)
        ax = axes.ravel()[levels]
        x = np.linspace(-1, 1, 2**levels + 1)
        if levels > 0:
            for xv in x:
                ax.plot([-1, 1], [xv, xv], 'k-', lw=1)
                ax.plot([xv, xv], [-1, 1], 'k-', lw=1)
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', lw=1)
        for j in range(len(x) - 1):
            for k in range(len(x) - 1):
                g = cj.chebfun2(op, domain=(x[j], x[j + 1],
                                            x[k], x[k + 1]))
                ln = _coeff_len(g)
                ax.text(np.mean(x[j:j + 2]) - 0.1,
                        np.mean(x[k:k + 2]), f"{ln}", fontsize=fs)
        ax.axis(list(1.05 * np.array([-1, 1, -1, 1])))
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"AverageDegreeReduction2D_{FIG[0]:02d}.png"), size=(600, 480))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    M = 2000
    op = lambda x, y: jnp.sin(M * x) * jnp.sin(M * y)  # noqa: E731
    compute_tau(op, 2)
    subdivision_diagram(op)

    M = 20
    op = lambda x, y: jnp.sin(M * (x - y))  # noqa: E731
    compute_tau(op, 2)
    subdivision_diagram(op)
    m, n = cj.chebfun2(op).length()
    vals = max(m, n) / 2.0 ** np.arange(0, 2, 0.5)
    print("ans =")
    print("  Columns 1 through 3")
    print("".join(f"{v:20.15f}" for v in vals[:3]))
    print("  Column 4")
    print(f"{vals[3]:20.15f}")

    a, b = 1, 100
    op = lambda x, y: 1.0 / ((b - a) / 2 * ((x + 1) + (y + 1))  # noqa: E731
                             + 2 * a)
    subdivision_diagram(op)

    m = []
    eps_ = np.finfo(float).eps
    for bb in [100, 50, 25, 17.5]:
        r = bb / 1
        B = (r + 3) / (r - 1)
        m.append(int(np.ceil(
            np.log(4 / (r - 1) / eps_ / np.sqrt(B**2 - 1))
            / np.log(B + np.sqrt(B**2 - 1)))))
    print("ans =")
    for v in m:
        print(f"{v:6d}")


if __name__ == "__main__":
    run()
