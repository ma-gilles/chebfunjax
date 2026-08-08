"""Random trajectory on a sphere.

Faithful replica of ode-random/RandomOnASphere.m by Kevin Burrage and
Nick Trefethen (May 2017): the random linear system

    u' = f A u + g B u + h C u

with skew-symmetric A, B, C conserves |u|, so the trajectory wanders
on the unit sphere.  Two runs on [0, 100]: lambda = 0.5 and 0.125.

Sample paths use JAX keys (MATLAB rng(0) not reproducible).

Original: https://www.chebfun.org/examples/ode-random/RandomOnASphere.html
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

import jax

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-random')

DOM = (0.0, 100.0)


class _FastTrig:
    """Fast exact evaluator of a trig chebfun's Fourier series."""

    def __init__(self, fn, n=8192):
        self._orig = fn
        a, b = float(fn.domain.a), float(fn.domain.b)
        self.a, self.L = a, b - a
        xs = a + self.L * np.arange(n) / n
        vals = np.asarray(fn(xs), dtype=float)
        self.c = np.fft.rfft(vals) / n
        self.k = np.arange(len(self.c))

    def __call__(self, t):
        try:
            t = float(t)
        except TypeError:
            # Probe paths call the op with t as a chebfun: hand back
            # the coefficient as a function.
            return self._orig
        th = 2 * np.pi * (t - self.a) / self.L
        z = np.exp(1j * self.k * th)
        return float(np.real(self.c[0])
                     + 2 * np.real(np.dot(self.c[1:], z[1:])))


BROWN = (0.5, 0.25, 0.12)


def _panel(lam, key0, lw, fname, tol=None):
    t0 = time.time()
    rng = np.random.default_rng(key0)
    u0 = rng.standard_normal(3)
    u0 = u0 / np.linalg.norm(u0)
    # Coefficients evaluated pointwise as f(t) -- mathematically the
    # same operator as MATLAB's f*y, through a precomputed numpy trig
    # series (per-step chebfun evaluation in the marcher was the
    # runtime bottleneck: hours instead of minutes).
    f = _FastTrig(randnfun(lam, DOM, key=jax.random.PRNGKey(key0)))
    g = _FastTrig(randnfun(lam, DOM, key=jax.random.PRNGKey(key0 + 1)))
    h = _FastTrig(randnfun(lam, DOM, key=jax.random.PRNGKey(key0 + 2)))
    L = Chebop(lambda t, x, y, z: [
        x.diff() - f(t) * y - g(t) * z,
        y.diff() + f(t) * x - h(t) * z,
        z.diff() + g(t) * x + h(t) * y], domain=DOM)
    L.lbc = lambda x, y, z: [x - u0[0], y - u0[1], z - u0[2]]
    if tol is not None:
        # MATLAB: cheboppref.setDefaults('ivpAbsTol',1e-6,'ivpRelTol',1e-6)
        L.ivp_reltol = tol
        L.ivp_abstol = tol
    sol = L.solve(0.0)
    x, y, z = sol[0], sol[1], sol[2]
    tt = np.linspace(*DOM, 12000)
    xx, yy, zz = (np.asarray(x(tt)), np.asarray(y(tt)),
                  np.asarray(z(tt)))
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    print(f"lambda={lam}: ({time.time()-t0:.0f}s) "
          f"radius drift {np.max(np.abs(r - 1)):.2e}", flush=True)

    fig = plt.figure(figsize=(7.6, 7.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot3D(xx, yy, zz, color=BROWN, lw=lw)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    _panel(0.5, 10, 1.4, "RandomOnASphere_repl_01.png")
    _panel(0.125, 20, 1.0, "RandomOnASphere_repl_02.png", tol=1e-6)


if __name__ == "__main__":
    run()
