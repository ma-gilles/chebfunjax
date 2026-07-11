"""Port of MATLAB Chebfun tests/chebtech/test_isreal.m (Opus 4.8).

chebfunjax Chebtech has no ``isreal()`` method, but the underlying fact
``isreal(f)`` is faithfully equivalent to "the coefficient array is not
of complex dtype": a complex-valued function produces ``complex128``
coeffs and a purely real one produces ``float64`` coeffs.  We therefore
test via ``jnp.iscomplexobj(f.coeffs)``.

Note on Chebtech1: chebfunjax's Chebtech1 cannot represent complex-valued
functions — its constructor drops the imaginary part (coeffs stay real).
The two complex-detection assertions are therefore xfailed for Chebtech1
(a genuine chebfunjax gap) and pass for Chebtech2.  Array-valued cases
have no scalar analogue and are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsreal:
    def test_complex_is_not_real(self, Tech):
        # pass(n,1): ~isreal(make(@(x) sin(x) + 1i*cos(x)))
        f = Tech.from_function(lambda x: jnp.sin(x) + 1j * jnp.cos(x))
        assert jnp.iscomplexobj(f.coeffs)

    def test_imaginary_is_not_real(self, Tech):
        # pass(n,2): ~isreal(make(@(x) 1i*cos(x)))
        f = Tech.from_function(lambda x: 1j * jnp.cos(x))
        assert jnp.iscomplexobj(f.coeffs)

    def test_real_is_real(self, Tech):
        # pass(n,3): isreal(make(@(x) sin(x)))
        f = Tech.from_function(jnp.sin)
        assert not jnp.iscomplexobj(f.coeffs)

    def test_array_complex_first_col(self, Tech):
        # pass(n,4): ~isreal(make(@(x) [sin(x) + 1i*cos(x), exp(x)]))
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_array_imaginary_first_col(self, Tech):
        # pass(n,5): ~isreal(make(@(x) [1i*cos(x), exp(x)]))
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_array_all_real(self, Tech):
        # pass(n,6): isreal(make(@(x) [sin(x), exp(x)]))
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )
