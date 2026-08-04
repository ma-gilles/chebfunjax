"""Phase portraits of linear dynamical systems.

Faithful replica of ode-linear/DynamicalSystems.m by Grady Wright
(October 2014): phase planes of u' = Au for ten 2x2 matrices —
unstable/stable nodes, center, spirals, saddle, degenerate cases —
with quiver fields and ode45 trajectories, plus the trace-determinant
stability diagram lettered with scribble.

Original: https://www.chebfun.org/examples/ode-linear/DynamicalSystems.html
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
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"DynamicalSystems_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_eig(A):
    lam, EV = np.linalg.eig(np.asarray(A, dtype=float))
    if np.iscomplexobj(lam) and np.any(np.abs(lam.imag) > 0):
        order = np.argsort(lam.imag, kind="stable")
        lam, EV = lam[order], EV[:, order]
    print("eigenvalues of A:")
    if np.iscomplexobj(lam) and np.any(np.abs(lam.imag) > 0):
        for i, v in enumerate(lam):
            print(f"  Column {i+1}")
            print(f"  {v.real:.15f} {'-' if v.imag < 0 else '+'} "
                  f"{abs(v.imag):.15f}i")
    else:
        print("   " + "   ".join(f"{v.real:g}" for v in lam))
    print("eigenvectors of A:")
    if np.iscomplexobj(EV) and np.any(np.abs(EV.imag) > 0):
        for j in range(EV.shape[1]):
            print(f"  Column {j+1}")
            for i in range(EV.shape[0]):
                v = EV[i, j]
                print(f"  {v.real:.15f} "
                      f"{'-' if v.imag < 0 else '+'} "
                      f"{abs(v.imag):.15f}i")
    else:
        for row in EV.real:
            print("  " + "  ".join(f"{v:18.15f}" for v in row))


def _system(A, T, initvals, title, extra=None, short=None):
    A = np.asarray(A, dtype=float)
    _print_eig(A)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    g = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(g, g)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    ax.quiver(X, Y, U, V, color='b')
    ax.set_aspect("equal")

    def rhs(t, y):
        return A @ y

    for iv in initvals:
        s = solve_ivp(rhs, T, iv, rtol=1e-9, atol=1e-11,
                      dense_output=True)
        tt = np.linspace(T[0], T[1], 500)
        yy = s.sol(tt)
        ax.plot(yy[0], yy[1], 'r', lw=2)
        ax.plot(iv[0], iv[1], 'r.', ms=12)
    for iv in (short or []):
        Ts = (T[0], T[0] + (T[1] - T[0]) * 2 / 3)
        s = solve_ivp(rhs, Ts, iv, rtol=1e-9, atol=1e-11,
                      dense_output=True)
        tt = np.linspace(Ts[0], Ts[1], 500)
        yy = s.sol(tt)
        ax.plot(yy[0], yy[1], 'r', lw=2)
        ax.plot(iv[0], iv[1], 'r.', ms=12)
    if extra:
        extra(ax)
    ax.plot(0, 0, 'k.', ms=14)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(title, fontsize=14)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    _system([[2, -2], [0, 1]], (0, 3),
            [[.1, .05], [-.1, -.05], [-.1, 0], [.1, 0]],
            "The origin is an unstable fixed point",
            short=[[.1, .1], [-.1, -.1]])

    _system([[-1, 3], [0, -3]], (0, 6),
            [[1, -2/3], [-1, 2/3], [.5, -1], [-.5, 1], [1, 0],
             [-1, 0]],
            "The origin is a stable fixed point")

    _system([[2, -2], [3, -2]], (0, 5),
            [[.2, 0], [.5, 0]],
            "The origin is a center")

    # Trace-determinant stability diagram with scribble lettering
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    rt = cj.chebfun(lambda x: 2 * jnp.sqrt(x), domain=(0, 1),
                    splitting=True)
    tt = np.linspace(0, 1, 400)
    ax.plot([-1, 1], [0, 0], lw=1.6)
    ax.plot([0, 0], [-2, 2], lw=1.6)
    ax.plot(tt, np.asarray(rt(tt)), 'b', lw=1.6)
    ax.plot(tt, -np.asarray(rt(tt)), 'b', lw=1.6)
    labels = []
    s1 = scribble("stable")
    s2 = scribble("unstable")
    s3 = scribble("saddles")
    s4 = scribble("spirals")
    for s, off in ((s3, -0.5 + 1j), (s3, -0.5 - 1j),
                   (s2, 0.4 + 1.8j), (s1, 0.4 - 1.8j),
                   (s2, 0.6 + 0.8j), (s4, 0.6 + 0.6j),
                   (s1, 0.6 - 0.6j), (s4, 0.6 - 0.8j)):
        bps = [float(v) for v in s.domain.breakpoints]
        for aa, bb in zip(bps[:-1], bps[1:]):
            u = np.linspace(aa, bb, 8)
            z = 0.3 * np.asarray(s(u)) + off
            ax.plot(z.real, z.imag, 'k', lw=1)
    ax.set_title("Stability of linear dynamical systems", fontsize=14)
    ax.set_xlabel("det(A)", fontsize=14)
    ax.set_ylabel("tr(A)", fontsize=14)
    _save(fig)

    _system([[2, -2], [8, 1]], (0, 2),
            [[.1, .1], [-.1, -.1], [.1, -.1], [-.1, .1]],
            "The origin is an unstable spiral")

    _system([[-.5, -2], [2, -.2]], (0, 10),
            [[0, 1], [1, 0], [0, -1], [-1, 0]],
            "The origin is a stable spiral")

    _system([[1, 1], [4, -2]], (0, 2),
            [[-.1, 1], [-.5, 1], [.1, -1], [.6, -1]],
            "The origin is a saddle point",
            extra=lambda ax: (
                ax.plot([-0.275, 0.275], [1.1, -1.1], 'k', lw=2),
                ax.plot([-1.1, 1.1], [-1.1, 1.1], 'k', lw=2)))

    _system([[1, 1], [-2, -2]], (0, 2),
            [[-.6, 1], [-.2, 1], [.2, 1], [.7, -1], [.3, -1],
             [-.1, -1]],
            "A line of stable fixed points",
            extra=lambda ax: ax.plot([-1, 1], [1, -1], 'k-', lw=2))

    _system([[1, 2], [1, 2]], (0, 2),
            [[0, .05], [-.5, .3], [-1, .55], [1, -.55], [0, -.05],
             [.5, -.3]],
            "A line of unstable fixed points",
            extra=lambda ax: ax.plot([-1, 1], [.5, -.5], 'k', lw=2),
            short=[[-.5, .2], [.5, -.2]])

    _system([[1, 4], [-1, -3]], (0, 4),
            [[-1, .5], [1, -.5], [-.9, 1], [-.5, 1], [.9, -1],
             [.5, -1], [1, -.75], [-1, .75]],
            "A stable node and collinear eigendirections")

    _system([[-1, 5/2], [-5/2, 4]], (0, 2),
            [[.1, .1], [-.1, -.1], [.5, .35], [.1, -.1], [-.1, .1],
             [-.5, -.35]],
            "An unstable node and collinear eigendirections")


if __name__ == "__main__":
    run()
