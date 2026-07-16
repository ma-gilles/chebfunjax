"""Port of MATLAB Chebfun tests/trigtech/test_size.m (Opus 4.8).

size(f) == size(f.coeffs).  For a scalar-valued chebfunjax trigtech the
coefficient array is 1-D, so we check the single (length) dimension; for an
array-valued trigtech the coefficient array is (n, m) and size(f) is (n, m)
(FIXED, Fable 5, Big-Three array-valued epic).

Provenance
----------
MATLAB source : tests/trigtech/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _tt(f, n=None):
    return Trigtech.from_function(f, n=n)


class TestTrigtechSize:
    def test_scalar_size(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        assert f.coeffs.shape[0] == f.n

    def test_scalar_fixed_length(self):
        f = _tt(lambda x: jnp.sin(19 * jnp.pi * x), n=101)
        assert f.coeffs.shape == (101,)

    def test_array_size(self):
        # pass(2): size(f) == size(f.coeffs) for [sin19 cos22 exp(10i)].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = _tt(
            lambda x: jnp.stack(
                [
                    jnp.sin(19 * jnp.pi * x),
                    jnp.cos(22 * jnp.pi * x),
                    jnp.exp(10j * jnp.pi * x),
                ],
                axis=-1,
            )
        )
        assert f.coeffs.shape == (f.n, 3)

    def test_array_fixed_length_size(self):
        # pass(3): fixedLength=101 -> size(f) == [101, 3].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(
            lambda x: jnp.stack(
                [
                    jnp.sin(19 * jnp.pi * x),
                    jnp.cos(22 * jnp.pi * x),
                    jnp.exp(10j * jnp.pi * x),
                ],
                axis=-1,
            ),
            n=101,
        )
        assert f.coeffs.shape == (101, 3)
