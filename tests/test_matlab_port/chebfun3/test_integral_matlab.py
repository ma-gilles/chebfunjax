"""Literal Chebfun-curve source predicates from MATLAB test_integral.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def _source_curve(components, interval):
    return cj.chebfun(
        lambda t: jnp.stack(components(t), axis=-1), domain=interval
    )


def test_original_integral_source_curve_1():
    tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
    f = Chebfun3.from_function(lambda x, y, z: 2 * x * y + 3 * z)
    curve = _source_curve(lambda t: (2 * t, 5 * t, 4 * t), [0, 1])
    exact = 38 * np.sqrt(5)
    assert abs(f.integral(curve) - exact) / exact < tol


def test_original_integral_source_curve_2():
    tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
    f = Chebfun3.from_function(lambda x, y, z: x**2 * z)
    curve = _source_curve(lambda t: (-t, 6 + 2 * t, 2 + 5 * t), [0, 1])
    exact = 23 * np.sqrt(30) / 12
    assert abs(f.integral(curve) - exact) / exact < tol


def test_original_integral_source_curve_3():
    tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
    dom = (-1, 1, -1, 1, 0, 12 * np.pi)
    f = Chebfun3.from_function(lambda x, y, z: x * y * z, domain=dom)
    curve = _source_curve(
        lambda t: (jnp.cos(t), jnp.sin(t), 3 * t), [0, 4 * np.pi]
    )
    exact = -3 * np.sqrt(10) * np.pi
    value = f.integral(curve)
    # Literal source pass(3) divides by the negative exact value.
    assert abs(value - exact) / exact < tol
    # Retain the stronger existing supplemental error control.
    assert abs(value - exact) / abs(exact) < tol
