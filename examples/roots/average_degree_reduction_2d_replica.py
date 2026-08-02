"""The average degree reduction of subdivision (2D).

Faithful replica of roots/AverageDegreeReduction2D.m by Alex Townsend
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

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

TOL = 1e-14
FIG = [0]


def _coeff_len(g):
    """Effective column degree: last column with mass above tol.

    The published MATLAB code applies find(...,'last') to the
    rot90'd matrix, which literally counts trailing *negligible*
    columns — a representation-slack number that varies between
    implementations and contradicts the example's own prose (it
    predicts tau ~ 1/2 and ~ 1/sqrt(2)).  We compute the intended
    effective degree; see the page note.
    """
    X = np.asarray(g.coeffs2())
    colmax = np.max(np.abs(X), axis=0)
    idx = np.where(colmax >= TOL)[0]
    return (idx[-1] + 1) if idx.size else 1


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
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0))
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
    fig.savefig(os.path.join(
        _IMG, f"AverageDegreeReduction2D_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
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
    g = cj.chebfun2(op)
    m, n = np.asarray(g.coeffs2()).shape
    vals = max(m, n) / 2.0 ** np.arange(0, 2, 0.5)
    print("ans =")
    print("  " + "  ".join(f"{v:.15f}" for v in vals))

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
        print(f"   {v}")


if __name__ == "__main__":
    run()
