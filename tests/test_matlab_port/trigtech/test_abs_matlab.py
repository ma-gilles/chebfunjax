"""Port of MATLAB Chebfun tests/trigtech/test_abs.m (Opus 4.8).

abs(f) of a real function that does not change sign is that (or its
negation), and abs of a unit-modulus complex exponential is the constant
1.  ``normest`` here is the standard inf-norm proxy: max |.| on a fine grid.

Provenance
----------
MATLAB source : tests/trigtech/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
_FINE = jnp.asarray(np.linspace(-1.0, 1.0, 2011, endpoint=False))


def _tt(f):
    return Trigtech.from_function(f)


def _normest(f):
    return float(jnp.max(jnp.abs(f(_FINE))))


class TestTrigtechAbs:
    def test_positive_function(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x) + 2)
        h = abs(f)
        assert _normest(h - f) < 10 * EPS

    def test_negative_function(self):
        f2 = _tt(lambda x: -(jnp.sin(jnp.pi * x) + 2))
        h = abs(f2)
        assert _normest(h + f2) < 10 * EPS

    def test_complex_unit_modulus(self):
        f = _tt(lambda x: jnp.exp(1j * jnp.pi * x))
        h = abs(f)
        assert _normest(h - 1) < 10 * EPS

    def test_complex_array_valued(self):
        # pass(4): abs of a complex array-valued trigtech.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(
            lambda x: jnp.stack(
                [
                    (2 + jnp.sin(jnp.pi * x)) * jnp.exp(1j * jnp.pi * x),
                    -(2 + jnp.sin(jnp.pi * x)) * jnp.exp(1j * jnp.pi * x),
                    2 + jnp.sin(jnp.pi * x),
                ],
                axis=-1,
            )
        )
        g = _tt(lambda x: jnp.stack([2 + jnp.sin(jnp.pi * x)] * 3, axis=-1))
        h = abs(f)
        assert _normest(h - g) < 10 * EPS
