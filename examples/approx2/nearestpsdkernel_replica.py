"""Nearest positive semidefinite kernel.

Faithful replica of approx2/NearestPSDKernel.m (Hashemi, 2016):
the PSD part of a symmetric kernel is obtained by dropping the
negative-eigenvalue terms of its spectral expansion, computed from
the chebfun2 SVD with the sign trick relating singular functions to
eigenfunctions.  The Gaussian-bump kernels use the EXACT rng(1) and
rng(3) center streams dumped from MATLAB R2025b.

Original: https://www.chebfun.org/examples/approx2/NearestPSDKernel.html
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

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')
FIG = [0]

# MATLAB rng(1): 2*rand-1, 20 draws (R2025b)
A20 = [-0.165955990594852, 0.4406489868843162, -0.99977125036531023,
       -0.39533485473632046, -0.70648821836577391, -0.8153228104624044,
       -0.62747957724465819, -0.30887854591390451, -0.20646505153866013,
       0.07763346800671389, -0.16161097119341039, 0.37043900079351899,
       -0.59109550053696513, 0.75623487278189083, -0.94522481360414767,
       0.34093502035680445, -0.16539039526574606, 0.11737965689150331,
       -0.71922612280953246, -0.60379702183024242]
# MATLAB rng(3): (2*rand-1, 2*rand-1), 10 pairs (R2025b)
B10 = [(0.10159580514915101, 0.41629564523620965),
       (-0.41819052217411135, 0.02165521039532603),
       (0.7858939086953094, 0.79258617786687613),
       (-0.74882937907232749, -0.58551424372362648),
       (-0.89706559339834024, -0.11838031269872706),
       (-0.94024757824286609, -0.086333551210577841),
       (0.29828809522952149, -0.44302543470404943),
       (0.35250980396026255, 0.18172563483270165),
       (-0.95203623524566927, 0.11770817598176397),
       (-0.48149510618506919, -0.16979760597986071)]

_TQ, _WQ = np.polynomial.legendre.leggauss(700)


def _ip(u, v, a, b):
    t = 0.5 * (b - a) * _TQ + 0.5 * (b + a)
    return float(0.5 * (b - a)
                 * np.sum(_WQ * np.asarray(u(t)) * np.asarray(v(t))))


def nearest_psd(K):
    """PSD part of a symmetric chebfun2 (subfunction nearestPSD):
    KHat = U*diag(max(Lambda,0))*V', assembled directly as a CDR."""
    import jax.numpy as jnp

    from chebfunjax.chebfun2d.separable_approx import SeparableApprox

    xa, xb, ya, yb = K.domain
    U, S, V = K.svd(full=True)
    s = np.array([_ip(U[i], V[i], ya, yb) for i in range(len(U))])
    lam = np.sign(s) * np.asarray(S)
    keep = [i for i in range(len(U)) if not lam[i] < 0]
    return Chebfun2(approx=SeparableApprox(
        cols=[U[i].funs[0].tech for i in keep],
        rows=[V[i].funs[0].tech for i in keep],
        pivots=jnp.asarray([lam[i] for i in keep], dtype=jnp.float64),
        domain=K.domain))


def _display(name, F):
    xa, xb, ya, yb = F.domain
    cv = [float(F(np.array([x]), np.array([y]))[0])
          for (x, y) in [(xa, ya), (xb, ya), (xa, yb), (xb, yb)]]
    g = np.linspace(xa, xb, 151)
    X, Y = np.meshgrid(g, g)
    vs = float(np.max(np.abs(np.asarray(F(X, Y)))))
    print(f"{name} =")
    print("   chebfun2 object")
    print("       domain                 rank       corner values")
    print(f"[{xa:4.0f},{xb:4.0f}] x [{ya:4.0f},{yb:4.0f}]"
          f"     {int(F.rank):4d}     "
          f"[{cv[0]:.2g} {cv[1]:.2g} {cv[2]:.2g} {cv[3]:.2g}]")
    print(f"vertical scale = {vs:.2g}")


def _pair_plot(K, KH, t1, t2, surface):
    FIG[0] += 1
    xa, xb, _, _ = K.domain
    g = np.linspace(xa, xb, 200)
    X, Y = np.meshgrid(g, g)
    if surface:
        fig = plt.figure(figsize=(11.0, 4.8))
        for i, (F, tt) in enumerate([(K, t1), (KH, t2)], start=1):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            ax.plot_surface(X, Y, np.asarray(F(X, Y)), cmap="viridis",
                            rstride=2, cstride=2, linewidth=0)
            ax.set_title(tt)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
        for ax, (F, tt) in zip(axes, [(K, t1), (KH, t2)]):
            ax.contourf(X, Y, np.asarray(F(X, Y)), 10)
            ax.set_aspect("equal")
            ax.set_title(tt)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"NearestPSDKernel_repl_{FIG[0]:02d}.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- multiquadric (conditionally positive definite: -mq) ---
    c = 0.01

    def mq(x, y):
        return -np.sqrt((x**2 + y**2) + c**2)

    K = Chebfun2.from_function(mq, domain=(-2, 2, -2, 2))
    _display("K", K)
    KHat = nearest_psd(K)
    _display("KHat", KHat)
    _pair_plot(K, KHat, "Multiquadric kernel", "Nearest PSD kernel",
               surface=False)

    # --- PSD sum of 20 diagonal Gaussian bumps ---
    gam = 50

    def kpsd(x, y):
        out = np.zeros(np.broadcast(x, y).shape)
        for x0 in A20:
            out = out + np.exp(-gam * ((x - x0)**2 + (y - x0)**2))
        return out

    K = Chebfun2.from_function(kpsd)
    _display("K", K)
    KHat = nearest_psd(K)
    _display("KHat", KHat)
    _pair_plot(K, KHat, "A PSD sum of 20 bumps", "Nearest PSD kernel",
               surface=True)
    _pair_plot(K, KHat, "A PSD sum of 20 bumps", "Nearest PSD kernel",
               surface=False)

    g = np.linspace(-1, 1, 220)
    X, Y = np.meshgrid(g, g)
    diff = float(np.max(np.abs(np.asarray(K(X, Y)) -
                               np.asarray(KHat(X, Y)))))
    print("ans =")
    print(f"     {diff:.2g}  (fro-norm of K - KHat at plot precision)")

    # --- indefinite sum of off-diagonal symmetric bump pairs ---
    gam = 100

    def kind(x, y):
        out = np.zeros(np.broadcast(x, y).shape)
        for (x0, y0) in B10:
            out = (out
                   + np.exp(-gam * ((x - x0)**2 + (y - y0)**2))
                   + np.exp(-gam * ((x - y0)**2 + (y - x0)**2)))
        return out

    K = Chebfun2.from_function(kind)
    _display("K", K)
    KHat = nearest_psd(K)
    _display("KHat", KHat)
    _pair_plot(K, KHat, "An indefinite sum of bumps",
               "Nearest PSD kernel", surface=True)
    _pair_plot(K, KHat, "An indefinite sum of bumps",
               "Nearest PSD kernel", surface=False)

    # --- waffle ---
    def waffle(x, y):
        return 1 / (1 + 1e3 * ((x**2 - .25)**2 * (y**2 - .25)**2))

    K = Chebfun2.from_function(waffle)
    _display("K", K)
    KHat = nearest_psd(K)
    _display("KHat", KHat)
    _pair_plot(K, KHat, "Waffle", "Nearest PSD kernel", surface=True)
    _pair_plot(K, KHat, "Waffle", "Nearest PSD kernel", surface=False)


if __name__ == "__main__":
    run()
