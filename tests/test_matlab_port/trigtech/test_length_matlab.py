"""Port of MATLAB Chebfun tests/trigtech/test_length.m (Opus 4.8).

length(f) is the number of Fourier coefficients (rows of f.coeffs).

Provenance
----------
MATLAB source : tests/trigtech/test_length.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _tt(f, n=None):
    return Trigtech.from_function(f, n=n)


class TestTrigtechLength:
    def test_length_equals_ncoeffs(self):
        f = _tt(lambda x: jnp.tanh(jnp.sin(jnp.pi * x)))
        assert len(f) == f.coeffs.shape[0]

    def test_fixed_length(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x), n=101)
        assert len(f) == 101

    def test_array_valued_length(self):
        # pass(2): length(f) == size(f.coeffs, 1) for tanh([sin cos 1i*exp(x)]).
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Fourier coeffs;
        # len(f) is the row count regardless of column count.  The third column
        # is non-periodic, so construction is unhappy -- length is still defined.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = _tt(
                lambda x: jnp.tanh(
                    jnp.stack(
                        [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x), 1j * jnp.exp(x)],
                        axis=-1,
                    )
                )
            )
        assert len(f) == f.coeffs.shape[0]
