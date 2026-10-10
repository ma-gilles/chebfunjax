"""Phase portraits of linear dynamical systems.

Translation of ode-linear/DynamicalSystems.m by Georges Klein
(March 2013): phase planes of u' = Au for ten 2x2 matrices —
unstable/stable nodes, center, spirals, saddle, degenerate cases —
with quiver fields and ode45 trajectories, plus the trace-determinant
stability diagram lettered with scribble.

Original URL: https://www.chebfun.org/examples/ode-linear/DynamicalSystems.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.plotting import CHEBFUN_RC, chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
# Native R2025b/7574c77 headless graphics capture reports 72 screen pixels/inch.
# Keep literal point sizes and pixel canvas; historical website font metrics
# remain a separate qualification.
_EXPORT_DPI = 72.0


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"DynamicalSystems_{FIG[0]:02d}.png"), size=(500, 400), dpi=_EXPORT_DPI, layout="matlab")
    plt.close(fig)


def _print_eig(A):
    lam, EV = jnp.linalg.eig(jnp.asarray(A, dtype=jnp.float64))
    if jnp.iscomplexobj(lam) and jnp.any(jnp.abs(lam.imag) > 0):
        order = jnp.argsort(lam.imag, stable=True)
        lam, EV = lam[order], EV[:, order]
    print("eigenvalues of A:")
    if jnp.iscomplexobj(lam) and jnp.any(jnp.abs(lam.imag) > 0):
        for i, v in enumerate(lam):
            print(f"  Column {i+1}")
            print(f"  {v.real:.15f} {'-' if v.imag < 0 else '+'} "
                  f"{abs(v.imag):.15f}i")
    else:
        print("   " + "   ".join(f"{v.real:g}" for v in lam))
    print("eigenvectors of A:")
    if jnp.iscomplexobj(EV) and jnp.any(jnp.abs(EV.imag) > 0):
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


def _system(A, T, initvals, title, extra=None, short=None, short_scale=2/3,
            origin_marker=True, identity=None):
    A = jnp.asarray(A, dtype=jnp.float64)
    _print_eig(A)
    # Native A*g: sequential scalar products/addition of identity components.
    if identity is None:
        identity = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
    components = [Chebfun2(approx=component) for component in identity.components]
    field = Chebfun2v([
        (float(A[row, 0])*components[0] + float(A[row, 1])*components[1]).approx
        for row in range(2)])
    fig, ax = plt.subplots(figsize=(5, 4))
    field.quiver(ax=ax, color='b', linewidth=2)
    ax.set_aspect("equal")

    for iv in initvals:
        _, trajectory = field.ode45(T, iv)
        matlab_plot(trajectory, 'r', ax=ax, linewidth=2)
        initial = complex(trajectory(jnp.asarray(0.)))
        ax.plot(initial.real, initial.imag, 'r.', ms=20)
    for iv in (short or []):
        Ts = tuple(value * short_scale for value in T)
        _, trajectory = field.ode45(Ts, iv)
        matlab_plot(trajectory, 'r', ax=ax, linewidth=2)
        initial = complex(trajectory(jnp.asarray(0.)))
        ax.plot(initial.real, initial.imag, 'r.', ms=20)
    if extra:
        extra(ax)
    if origin_marker:
        ax.plot(0, 0, 'k.', ms=20)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    identity = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
    system = partial(_system, identity=identity)

    system([[2, -2], [0, 1]], (0, 3),
            [[.1, .05], [-.1, -.05], [-.1, -.05], [-.1, 0], [.1, 0]],
            "The origin is an unstable fixed point",
            short=[[.1, .1], [-.1, -.1]])

    system([[-1, 3], [0, -3]], (0, 6),
            [[1, -2/3], [-1, 2/3], [.5, -1], [-.5, 1], [1, 0],
             [-1, 0]],
            "The origin is a stable fixed point")

    system([[2, -2], [3, -2]], (0, 5),
            [[.2, 0], [.5, 0]],
            "The origin is a center")

    # Trace-determinant stability diagram with scribble lettering
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    rt = cj.chebfun(lambda x: 2 * jnp.sqrt(x), domain=(0, 1),
                    splitting=True)
    ax.plot([-1, 1], [0, 0], lw=1.6)
    ax.plot([0, 0], [-2, 2], lw=1.6)
    matlab_plot(rt, 'b', -rt, 'b', ax=ax, linewidth=1.6)
    s1 = .3*scribble("stable")
    s2 = .3*scribble("unstable")
    s3 = .3*scribble("saddles")
    s4 = .3*scribble("spirals")
    colors = CHEBFUN_RC['axes.prop_cycle'].by_key()['color']
    for index, (s, off) in enumerate(((s3, -0.5 + 1j), (s3, -0.5 - 1j),
                   (s2, 0.4 + 1.8j), (s1, 0.4 - 1.8j),
                   (s2, 0.6 + 0.8j), (s4, 0.6 + 0.6j),
                   (s1, 0.6 - 0.6j), (s4, 0.6 - 0.8j))):
        matlab_plot(s+off, ax=ax, color=colors[index % len(colors)], linewidth=1)
    ax.set_title("Stability of linear dynamical systems", fontsize=14, fontweight="bold")
    ax.set_xlabel("det(A)", fontsize=14)
    ax.set_ylabel("tr(A)", fontsize=14)
    _save(fig)

    system([[2, -2], [8, 1]], (0, 2),
            [[.1, .1], [-.1, -.1], [.1, -.1], [-.1, .1]],
            "The origin is an unstable spiral")

    system([[-.5, -2], [2, -.2]], (0, 10),
            [[0, 1], [1, 0], [0, -1], [-1, 0]],
            "The origin is a stable spiral")

    system([[1, 1], [4, -2]], (0, 2),
            [[-.1, 1], [-.5, 1], [.1, -1], [.6, -1]],
            "The origin is a saddle point",
            extra=lambda ax: (
                ax.plot([-0.275, 0.275], [1.1, -1.1], 'k', lw=2),
                ax.plot([-1.1, 1.1], [-1.1, 1.1], 'k', lw=2)),
            origin_marker=False)

    system([[1, 1], [-2, -2]], (0, 2),
            [[-.6, 1], [-.2, 1], [.2, 1], [.7, -1], [.3, -1],
             [-.1, -1]],
            "A line of stable fixed points",
            extra=lambda ax: ax.plot([-1, 1], [1, -1], 'k-', lw=2),
            origin_marker=False)

    system([[1, 2], [1, 2]], (0, 2),
            [[0, .05], [-.5, .3], [-1, .55], [1, -.55], [0, -.05],
             [.5, -.3]],
            "A line of unstable fixed points",
            extra=lambda ax: ax.plot([-1, 1], [.5, -.5], 'k', lw=2),
            short=[[-.5, .2], [.5, -.2]], short_scale=1/2,
            origin_marker=False)

    system([[1, 4], [-1, -3]], (0, 4),
            [[-1, .5], [1, -.5], [-.9, 1], [-.5, 1], [.9, -1],
             [.5, -1], [1, -.75], [-1, .75]],
            "A stable node and collinear eigendirections")

    system([[-1, 5/2], [-5/2, 4]], (0, 2),
            [[.1, .1], [-.1, -.1], [.5, .35], [.1, -.1], [-.1, .1],
             [-.5, -.35]],
            "An unstable node and collinear eigendirections")


if __name__ == "__main__":
    run()
