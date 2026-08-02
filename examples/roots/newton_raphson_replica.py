"""Newton's method.

Faithful replica of roots/NewtonRaphson.m by Kuan Xu (October 2012):
Newton iterations on f = x^3 - 3x^2 + 2 with a tangent-line
visualization, quadratic convergence at a simple root, and cubic
convergence at the middle root where f''(1) = 0.

Original: https://www.chebfun.org/examples/roots/NewtonRaphson.html
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
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')


def run():
    os.makedirs(_IMG, exist_ok=True)
    dom = (-3.0, 3.0)
    f = cj.chebfun(lambda x: x**3 - 3 * x**2 + 2, domain=dom)
    fprime = f.diff()
    xs = np.linspace(*dom, 800)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(f(xs)), 'b', lw=2)
    ax.plot(dom, [0, 0], 'k')
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "NewtonRaphson_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    r = np.asarray(f.roots())
    print("ans =")
    for v in r:
        print(f"  {v:.15f}")

    d = float(f.abs().max()[1])
    tol = 1e-8
    xold = -2.0
    x = []
    i = 0
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xs2 = np.linspace(-2.5, 0, 400)
    ax.plot(xs2, np.asarray(f(xs2)), 'b', lw=2)
    ax.set_xlim(-2.5, 0)
    ax.plot(dom, [0, 0], 'k')
    while d > tol:
        x.append(xold)
        fx = float(f(xold))
        xnew = xold - fx / float(fprime(xold))
        d = abs(xnew - xold)
        ax.plot(xold, fx, 'ok')
        ax.text(xold - 0.05, 1.2, f"$x_{{{i}}}$", fontsize=12)
        ax.plot(xold, 0, '.k', ms=12)
        ax.plot([xold, xold], [0, fx], '--k', lw=2)
        ax.plot([xold, xnew], [fx, 0], '-.k', lw=2)
        xold = xnew
        i += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "NewtonRaphson_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("root1 =")
    print(f"  {xnew:.15f}")

    res = np.abs(np.array(x) - xnew)
    print("iterations     Logarithm of the step size")
    for i, v in enumerate(np.log(res), 1):
        print(f"{i:5d}  {v:25.8f}")

    d = float(f.abs().max()[1])
    xold = 0.5
    x = []
    while d > tol:
        x.append(xold)
        xnew = xold - float(f(xold)) / float(fprime(xold))
        d = abs(xnew - xold)
        xold = xnew
    print("root2 =")
    print(f"   {xnew:.15f}")
    res = np.abs(np.array(x) - xnew)
    print("iterations     Logarithm of the step size")
    for i, v in enumerate(np.log(res), 1):
        print(f"{i:5d}  {v:25.8f}")


if __name__ == "__main__":
    run()
