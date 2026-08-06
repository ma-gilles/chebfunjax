"""A droplet sitting on a surface.

Faithful replica of ode-nonlin/Droplets.m: the shape of an axisymmetric
sessile drop, from the Young-Laplace equation written in arclength form
as a first-order system for the radius R, height U, surface angle Psi
and the (constant) arclength scale L.

Three cases: borderline wetting (Psi_b = -pi/2), no wetting
(Psi_b = -pi), and a drop of prescribed volume, where the contact radius
b becomes a fifth unknown -- a scalar parameter rather than a function.

Original: https://www.chebfun.org/examples/ode-nonlin/Droplets.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
BLUE = (0.3, 0.4, 1.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Droplets_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _op(t, R, U, Psi, L, *rest):
    return [R.diff() - L * Psi.cos(),
            U.diff() - L * Psi.sin(),
            R * Psi.diff() + L * Psi.sin() - R * L * U,
            L.diff()]


def _draw(R, U, title, xlim=None):
    s = np.linspace(-1, 1, 3000)
    r = np.asarray(R(s))
    u = np.asarray(U(s))
    u = u - u.min()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    lo, hi = (xlim if xlim is not None
              else (-(abs(r).max() + 1), abs(r).max() + 1))
    ax.plot([lo, hi], [0, 0], "k", lw=1.5)
    ax.fill(r, u, color=BLUE)
    ax.plot(r, u, "k", lw=1)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_start = time.time()
    t = chebfun(lambda t: t, domain=(-1, 1))
    b = 3.0

    # --- borderline wetting ------------------------------------------
    Psib = -np.pi / 2
    N = Chebop(_op, domain=(-1, 1))
    N.bc = lambda t, R, U, Psi, L: [R(-1.0) + b, R(1.0) - b,
                                    Psi(-1.0) + Psib, Psi(1.0) - Psib]
    N.init = [b * t, 1 + 0 * t, t * Psib, 2 * b + 0 * t]
    R, U, Psi, L = N.solve(0.0)
    _draw(R, U, "borderline wetting", (-b - 1, b + 1))

    # --- no wetting ---------------------------------------------------
    Psib = -np.pi
    N = Chebop(_op, domain=(-1, 1))
    N.bc = lambda t, R, U, Psi, L: [R(-1.0) + b, R(1.0) - b,
                                    Psi(-1.0) + Psib, Psi(1.0) - Psib]
    N.init = [b * t, 1 + 0 * t, t * Psib, 2 * b + 0 * t]
    R, U, Psi, L = N.solve(0.0)
    _draw(R, U, "no wetting", (-b - 1, b + 1))

    print(f"Elapsed time is {time.time() - t_start:.6f} seconds.")
    print("ans =")
    print(f"   {len(R):5d}{len(U):6d}{len(Psi):6d}")

    # The drop volume, from the shape at the contact line.
    vol = np.pi * b * (2 * np.sin(Psib) - b * float(U(np.float64(1.0))))
    print("ans =")
    print(f"  {vol:.15f}")

    # --- prescribed volume: the contact radius becomes an unknown ----
    v0 = 10.0
    N = Chebop(lambda t, R, U, Psi, ell, bb: _op(t, R, U, Psi, ell),
               domain=(-1, 1))
    N.bc = lambda t, R, U, Psi, ell, bb: [
        R(-1.0) + bb, R(1.0) - bb,
        Psi(-1.0) + Psib, Psi(1.0) - Psib,
        np.pi * bb * (2 * np.sin(Psib) - bb * U(1.0)) - v0]
    bg = v0 ** (1 / 3) / 2
    N.init = [bg * t, bg * (-2 + (t * Psib).cos()), t * Psib,
              2 * bg + 0 * t, bg + 0 * t]
    R, U, Psi, ell, bb = N.solve(0.0)
    b_val = float(bb(np.float64(0.0)))
    _draw(R, U, "prescribed volume",
          (np.floor(np.asarray(R(np.linspace(-1, 1, 3000))).min()),
           np.ceil(np.asarray(R(np.linspace(-1, 1, 3000))).max())))

    got = np.pi * b_val * (2 * np.sin(Psib)
                           - b_val * float(U(np.float64(1.0))))
    print(f"contact radius b = {b_val:.12f}")
    print(f"recovered volume = {got:.12f}   (prescribed {v0})")


if __name__ == "__main__":
    run()
