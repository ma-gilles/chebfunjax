"""Port of MATLAB Chebfun tests/chebtech/test_sign.m (Opus 4.8).

The MATLAB file loops ``for type = 1:2`` over ``{chebtech1(), chebtech2()}``
and checks ``sign(f)`` for positive/negative/complex/complex-array functions.
chebfunjax now implements ``sign`` on both Chebtech1 and Chebtech2 (real:
constant sign from the endpoint/interior mean; complex: ``f./|f|`` via
``compose``), so every assertion is exercised at the SAME tolerance MATLAB
uses.

Provenance
----------
MATLAB source : tests/chebtech/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechSign:
    def test_sign_positive(self, Tech):
        # pass(type, 1): sign(sin(x) + 2) == 1.
        f = Tech.from_function(lambda x: jnp.sin(x) + 2.0)
        h = f.sign()
        assert float((h - 1.0).norm(jnp.inf)) < 10 * EPS

    def test_sign_negative(self, Tech):
        # pass(type, 2): sign(-(sin(x) + 2)) == -1.
        f = Tech.from_function(lambda x: -(jnp.sin(x) + 2.0))
        h = f.sign()
        assert float((h + 1.0).norm(jnp.inf)) < 10 * EPS

    def test_sign_complex(self, Tech):
        # pass(type, 3): sign(exp(1i pi x)) == exp(1i pi x).
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        h = f.sign()
        assert float((h - f).norm(jnp.inf)) < 1e2 * EPS

    def test_sign_complex_array(self, Tech):
        # pass(type, 4): complex array-valued sign == f./|f| pointwise.
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))

        def F(x):
            return jnp.stack(
                [
                    (2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    -(2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    (2 + jnp.sin(x)).astype(jnp.complex128),
                ],
                axis=-1,
            )

        f = Tech.from_function(F)
        ff = F(xx)
        gg = ff / jnp.abs(ff)
        h = f.sign()
        hh = h(xx)
        assert _ninf(hh - gg) < 10 * EPS
