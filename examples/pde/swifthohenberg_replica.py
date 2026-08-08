"""Swift-Hohenberg equation in 2D.

Faithful replica of pde/SwiftHohenberg.m by Hadrien Montanelli (May
2017): the Swift-Hohenberg equation

    u_t = r u - (1 + Lap)^2 u + g u^2 - u^3

solved with spin2/ETDRK4.  Section 1 runs the preloaded 'sh' demo
(r = 0.1, g = 0, random init) to t = 1200 (convection rolls); section
2 uses a deterministic sine + five-Gaussian init on [0, 20pi]^2 to
produce spots (r = 0.01, g = 1), spirals (r = 0.7, g = 1), and
stripes (r = 0.1, g = 0), plus the published resolution-refinement
error check.

Original: https://www.chebfun.org/examples/pde/SwiftHohenberg.html
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
from chebfunjax.spin.solver2d import spin2
from chebfunjax.spin.spinop2 import SpinOp2
from chebfunjax.utils.random import randnfun2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')
FIG = [0]


def _plot(U, dom):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(np.asarray(U).T, origin="lower", cmap="viridis",
              extent=(dom[0], dom[1], dom[2], dom[3]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"SwiftHohenberg_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. The preloaded demo (r = 0.1, g = 0, random init) to t = 1200.
    # MATLAB's preloaded 'sh': u0 = randnfun2(4, dom, 'trig')
    # NORMALIZED to inf-norm 1 -- without the normalization the
    # explicit u^3 step is unstable at dt = 1 (u0 ~ +-3 -> u^3 ~ 30).
    dom = (0.0, 50.0, 0.0, 50.0)
    f0 = randnfun2(4.0, dom, seed=7, trig=True)
    xg = np.linspace(0, 50, 256, endpoint=False)
    Xg, Yg = np.meshgrid(xg, xg, indexing="ij")
    nrm = np.max(np.abs(np.asarray(f0(Xg, Yg))))
    op = SpinOp2(lin_coeffs=(-2.0, -1.0, 0.0, 0.0, 0.0),
                 nonlin_vals=lambda u: -0.9 * u - u**3,
                 n_vars=1, domain=dom, tspan=(0.0, 1200.0),
                 u0=lambda x, y: np.asarray(f0(x, y)) / nrm)
    x, y, t, U = spin2(op, 128, 1.0, dealias=False)
    _plot(U, dom)
    print("demo done", flush=True)

    # 2. Spots, spirals, stripes on [0, 20pi]^2.
    P = 20 * np.pi
    dom = (0.0, P, 0.0, P)

    def u0(x, y):
        g = (np.cos(x) + np.sin(2 * x) + np.sin(y)
             + np.cos(2 * y)) / 20
        for cx, cy in [(5, 5), (5, 15), (15, 15), (15, 5), (10, 10)]:
            g = g + np.exp(-((x - cx * np.pi)**2 + (y - cy * np.pi)**2))
        return g

    def sh_op(r, g, t1):
        return SpinOp2(lin_coeffs=(-2.0, -1.0, 0.0, 0.0, 0.0),
                       nonlin_vals=lambda u:
                       (-1 + r) * u + g * u**2 - u**3,
                       n_vars=1, domain=dom, tspan=(0.0, t1), u0=u0)

    # initial condition
    xg = np.linspace(0, P, 400)
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    _plot(u0(X, Y), dom)

    # spots (r = 0.01, g = 1)
    x, y, t, U = spin2(sh_op(1e-2, 1.0, 200.0), 96, 2e-1, dealias=False)
    _plot(U, dom)
    print("spots done", flush=True)

    # refinement check
    x2, y2, t2, V = spin2(sh_op(1e-2, 1.0, 200.0), 128, 1e-1,
                          dealias=False)

    def _spectral_interp(W, M):
        n = W.shape[0]
        c = np.fft.fftshift(np.fft.fft2(W)) / n**2
        C = np.zeros((M, M), dtype=complex)
        s0 = (M - n) // 2
        C[s0:s0 + n, s0:s0 + n] = c
        return np.real(np.fft.ifft2(np.fft.ifftshift(C))) * M**2

    err = (np.linalg.norm(_spectral_interp(U, 128) - V)
           / np.linalg.norm(V))
    print(f"Relative error: {err:1.2e}")

    # spirals (r = 0.7, g = 1)
    x, y, t, U = spin2(sh_op(7e-1, 1.0, 200.0), 96, 2e-1, dealias=False)
    _plot(U, dom)
    print("spirals done", flush=True)

    # stripes (r = 0.1, g = 0)
    x, y, t, U = spin2(sh_op(1e-1, 0.0, 200.0), 100, 2e-1,
                       dealias=False)
    _plot(U, dom)
    print("stripes done", flush=True)


if __name__ == "__main__":
    run()
