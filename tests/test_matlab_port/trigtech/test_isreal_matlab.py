"""Port of MATLAB Chebfun tests/trigtech/test_isreal.m (Opus 4.8).

isreal(f) reports whether the underlying periodic function is real-valued.
chebfunjax records this on the ``is_real`` flag inferred at construction
from the conjugate symmetry of the Fourier coefficients.

Provenance
----------
MATLAB source : tests/trigtech/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechIsreal:
    def test_complex_scalar(self):
        f = _tt(lambda x: jnp.sin(100 * jnp.pi * x) + 1j * jnp.sin(jnp.cos(10 * jnp.pi * x)))
        assert not f.is_real

    def test_pure_imaginary_scalar(self):
        f = _tt(lambda x: 1j * jnp.cos(jnp.pi * x))
        assert not f.is_real

    def test_real_scalar(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        assert f.is_real

    def test_complex_array(self):
        # pass(4): array with a complex column is not real.
        # FIXED (Fable 5, Big-Three array-valued epic): is_real == all(isReal).
        f = _tt(
            lambda x: jnp.stack(
                [jnp.sin(100 * jnp.pi * x) + 1j * jnp.sin(jnp.cos(10 * jnp.pi * x)), jnp.cos(10 * jnp.pi * x)],
                axis=-1,
            )
        )
        assert not f.is_real

    def test_mixed_array(self):
        # pass(5): array with a pure-imaginary column is not real.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([1j * jnp.cos(jnp.pi * x), jnp.cos(jnp.sin(jnp.pi * x))], axis=-1))
        assert not f.is_real

    def test_real_array(self):
        # pass(6): array with all real columns is real.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.sin(20 * jnp.pi * x), jnp.cos(jnp.sin(jnp.pi * x))], axis=-1))
        assert f.is_real
