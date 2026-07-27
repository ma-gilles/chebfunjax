"""Rose curves.

A rose curve is the complex chebfun cos(m/n t)(cos t + i sin t) on
[0, 2*pi*lcm(m,n)].  Because it is periodic, a trigonometric (Fourier)
representation needs about pi/2 times fewer coefficients than a Chebyshev
representation of the same curve.  Faithful port of geom/RoseCurves.m.

Original: https://www.chebfun.org/examples/geom/RoseCurves.html
Author: Hrothgar, June 2014
"""

import math

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()


def rose_curve(m, n, trig=True):
    """MATLAB roseCurve(m,n): complex chebfun on [0, 2*pi*lcm(m,n)]."""
    L = math.lcm(m, n)
    dom = (0.0, 2.0 * math.pi * L)
    fn = lambda t: jnp.cos(m / n * t) * jnp.cos(t) + 1j * jnp.cos(m / n * t) * jnp.sin(t)
    return cj.chebfun(fn, domain=dom, trig=trig)


def _length(f):
    return sum(len(p.tech.coeffs) for p in f.funs)


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # MATLAB:
    #   m = 50; n = 51;
    #   f = roseCurve(m, n);                          % 'trig'
    #   g = chebfun(@(x) f(x), [0, 2*pi*lcm(m,n)]);   % Chebyshev
    #   length(g) ./ length(f)                        % ~ pi/2
    #   pi/2
    m, n = 50, 51
    f = rose_curve(m, n, trig=True)
    g = rose_curve(m, n, trig=False)
    ratio = _length(g) / _length(f)
    print(f"length(g)/length(f) = {ratio:.15f}")
    print(f"pi/2 = {math.pi / 2:.15f}")

    # A gallery of rose curves for m, n = 1..N (offset on the complex plane).
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')
    N = 6
    for mm in range(1, N + 1):
        for nn in range(1, N + 1):
            fc = rose_curve(mm, nn, trig=True)
            tt = np.linspace(fc.domain.a, fc.domain.b, 4000)
            z = np.asarray(fc(jnp.array(tt)))
            offset = 2.5 * mm - 2.5j * nn
            ax.plot(np.real(z) + offset.real, np.imag(z) + offset.imag,
                    'k-', linewidth=0.8)
    fig.suptitle('Rose curves cos(m/n·t)·e^{it}', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'rose_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # The representation-length ratio approaches pi/2.
    assert abs(ratio - math.pi / 2) < 0.05, "trig/Chebyshev length ratio off"

    print("rose_curves: done")
    return True


if __name__ == "__main__":
    run()
