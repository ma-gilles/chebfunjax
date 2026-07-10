"""Port of MATLAB Chebfun tests/trigtech/test_isempty.m (Opus 4.8).

isempty(f) is true iff f has no coefficients.  chebfunjax models an empty
trigtech as one whose coefficient array has length 0.

Provenance
----------
MATLAB source : tests/trigtech/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


def _tt(f):
    return Trigtech.from_function(f)


def _isempty(f):
    return f.n == 0


class TestTrigtechIsempty:
    def test_empty(self):
        f = Trigtech(coeffs=jnp.array([], dtype=jnp.complex128))
        assert _isempty(f)

    def test_scalar_nonempty(self):
        f = _tt(lambda x: jnp.sin(200 * jnp.pi * x))
        assert not _isempty(f)

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_nonempty(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks horizontal concatenation of trigtechs")
    def test_concatenated_nonempty(self):
        raise AssertionError("trigtech concatenation not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks horizontal concatenation of trigtechs")
    def test_concatenated_empty(self):
        raise AssertionError("trigtech concatenation not implemented")
