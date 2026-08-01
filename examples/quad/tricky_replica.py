"""Some tricky integrals.

Faithful replica of quad/Tricky.m by Fredrik Johansson and Nick
Trefethen: a gauntlet of integrals with spikes, high oscillation,
jumps, kinks, unbounded domains, and near-singular oscillation, each
checked against a high-precision reference value.

Original: https://www.chebfun.org/examples/quad/Tricky.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import numpy as np
from scipy.special import airy, erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def _report(name, Iexact, I):
    print(f"[{name}]")
    print("Iexact =")
    print("   NaN" if Iexact is None else f"   {Iexact:.15f}")
    print("I =")
    print(f"   {I:.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    W = warnings.catch_warnings

    # 1. Three spikes
    sech = lambda z: 1 / jnp.cosh(z)
    ff = lambda x: (sech(10 * (x - .2)) ** 2 + sech(100 * (x - .4)) ** 4
                    + sech(1000 * (x - .6)) ** 6)
    f = cj.chebfun(ff, domain=[0, 1])
    _report("spikes, global", 0.210802735500549277, float(f.sum()))

    # 2. Oscillation: sin(x + e^x)
    ff = lambda x: jnp.sin(x + jnp.exp(x))
    f = cj.chebfun(ff, domain=[0, 8])
    _report("sin(x+exp(x))", 0.34740017265724780787, float(f.sum()))

    # 3. Oscillation + jumps: (e^x - floor(e^x)) sin(x + e^x)
    with W():
        warnings.simplefilter("ignore")
        efloor = cj.chebfun(lambda x: jnp.exp(x),
                            domain=[0, 8]).floor()
        # pointwise floor(exp(x)) inside the handle: evaluating the
        # 2980-piece efloor chebfun per sample made construction
        # O(pieces^2) and astronomically slow
        f = cj.chebfun(lambda x: (jnp.exp(x) - jnp.floor(jnp.exp(x)))
                       * jnp.sin(x + jnp.exp(x)),
                       domain=[float(b) for b in
                               efloor.domain.breakpoints])
    _report("sawtooth-modulated", 0.098651704478365206119,
            float(f.sum()))

    # 4. Boundary layer: e^-x erf(sqrt(1250) x + 1.5)
    def ff4(x):
        xa = np.atleast_1d(np.asarray(x))
        v = np.exp(-xa) * erf(np.sqrt(1250.0) * xa + 1.5)
        return jnp.asarray(v.reshape(np.shape(x)) if np.shape(x)
                           else v[0])
    f = cj.chebfun(ff4)
    _report("erf layer", None, float(f.sum()))

    # 5. Unbounded domain: e^-x Ai(-x) on [0, inf)
    def ff5(x):
        xa = np.atleast_1d(np.asarray(x))
        v = np.exp(-xa) * airy(-xa)[0]
        return jnp.asarray(v.reshape(np.shape(x)) if np.shape(x)
                           else v[0])
    f = cj.chebfun(ff5, domain=[0.0, np.inf])
    _report("airy on [0,inf)", 0.378751605379086535, float(f.sum()))
    f = cj.chebfun(ff5, domain=[0.0, 40.0])
    _report("airy on [0,40]", 0.378751605379086535, float(f.sum()))

    # 6. |quartic| e^x with splitting
    with W():
        warnings.simplefilter("ignore")
        f = cj.chebfun(
            lambda x: jnp.abs(x**4 + 10*x**3 + 19*x**2 - 6*x - 6)
            * jnp.exp(x), domain=[0, 1], splitting=True)
    _report("|quartic| e^x", 11.1473105500571397339, float(f.sum()))

    # 7. ceil(x) on [0, 100]
    with W():
        warnings.simplefilter("ignore")
        f = cj.chebfun(lambda x: x, domain=[0, 100]).ceil()
    _report("ceil", 5050.0, float(f.sum()))

    # 8. Sawtooth times max(sin, cos)
    with W():
        warnings.simplefilter("ignore")
        xfl = cj.chebfun(lambda x: x, domain=[0, 10]).floor()
        f = cj.chebfun(
            lambda x: (x - xfl(x) - 0.5)
            * jnp.maximum(jnp.sin(x), jnp.cos(x)),
            domain=[float(b) for b in xfl.domain.breakpoints],
            splitting=True)
    _report("sawtooth*max", -0.14281864202632808376, float(f.sum()))

    # 9. Wild oscillation near x = 1
    ff = lambda x: jnp.sin((0.001 + (1 - x) ** 2) ** -1.5)
    with W():
        warnings.simplefilter("ignore")
        f = cj.chebfun(ff, domain=[0, 3], max_length=2**20)
    _report("wild oscillation", 0.74997436852719477011, float(f.sum()))
    return True


if __name__ == "__main__":
    run()
