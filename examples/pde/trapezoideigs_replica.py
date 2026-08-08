"""Eigenvalues of a trapezoidal drum.

Faithful replica of pde/TrapezoidEigs.m by Nick Trefethen (November
2014): Laplace eigenvalues of the trapezoid (0,0)-(1,0)-(1,1)-(-1,1)
by the method of particular solutions -- expanding eigenfunctions in
sin(4j theta/3) J_{4j/3}(lambda r) terms, sampling the two remaining
boundary segments, and locating the near-singular lambda by the
minimal singular value, scanned as a chebfun over [3, 7] with
splitting on for n = 4..7.

Original: https://www.chebfun.org/examples/pde/TrapezoidEigs.html
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
from scipy.special import jv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')


def trapfun(lam, n):
    """Minimal singular value of the boundary-sampling matrix."""
    z = np.concatenate([
        -1 + 1j + 2 * np.arange(1, 2 * n + 1) / (2 * n + 1),
        1 + 1j * np.arange(1, n + 1) / (n + 1)])
    r, th = np.abs(z), np.angle(z)
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.zeros(lam.shape)
    for k, lk in enumerate(lam):
        A = np.stack([jv(4 * j / 3, lk * r) * np.sin(4 * j * th / 3)
                      for j in range(1, n + 1)], axis=1)
        out[k] = np.linalg.svd(A, compute_uv=False)[-1]
    return out


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # The trapezoid itself.
    zv = np.array([0, 1, 1 + 1j, -1 + 1j])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.fill(zv.real, zv.imag, color=(.7, .7, 1))
    ax.text(0.25, 0.5, "?", fontsize=30)
    ax.set_xlim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TrapezoidEigs_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    dom = (3.0, 7.0)
    fignum = 1
    for n in range(4, 8):
        t0 = time.time()
        f = chebfun(lambda lam, _n=n: trapfun(lam, _n), domain=dom,
                    splitting=True)
        t = time.time() - t0
        # Prominence-filtered minima: at large n the sigma_min curve
        # develops a noise-level plateau whose jitter creates spurious
        # local minima; the physical eigenvalue dips are sharp V's.
        from scipy.signal import find_peaks
        xxm = np.linspace(*dom, 8000)
        fv = np.asarray(f(xxm))
        pk, _ = find_peaks(-fv, prominence=0.02 * np.max(fv))
        xs = xxm[pk]
        xs = xs[xs > dom[0]]
        fignum += 1
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        xx = np.linspace(*dom, 2000)
        ax.plot(xx, np.asarray(f(xx)), lw=1.6)
        ax.grid(True)
        ax.set_title(f"n = {n}     time ={t:5.1f} secs.", fontsize=10)
        ax.set_xlabel("first three minima: lam = "
                      f"{xs[0]:8.5f}, {xs[1]:8.5f}, {xs[2]:8.5f}",
                      fontsize=10)
        fig.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(os.path.join(
            _IMG, f"TrapezoidEigs_repl_{fignum:02d}.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"n={n}: lam = {xs[0]:.5f}, {xs[1]:.5f}, {xs[2]:.5f}"
              f" ({t:.1f}s)", flush=True)


if __name__ == "__main__":
    run()
