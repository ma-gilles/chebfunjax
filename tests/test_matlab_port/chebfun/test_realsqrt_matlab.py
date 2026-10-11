"""All8 literal tests/chebfun/test_realsqrt.m predicates, Chebfun7574c77.

Actual public x/pref, horzcat, abs, powers and complex multiplication routes;
exact native error identifiers and acceptance gates. Old positive/negative
sin probe remains supplemental below.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunRealsqrt:
    def test_positive_ok_negative_raises(self):
        import pytest

        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        g = f.realsqrt()
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 20))
        np.testing.assert_allclose(
            np.asarray(g(xs)), np.sqrt(2 + np.sin(np.asarray(xs))), atol=1e-12
        )
        with pytest.raises(ValueError):
            cj.chebfun(jnp.sin).realsqrt()


import jax
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(scope="module")
def native_realsqrt_inputs():
    pref = ChebfunPref()
    x = cj.chebfun("x", pref=pref)
    X = Chebfun.horzcat(x, x)
    return x, X


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("clause", range(1, 9))
def test_native_realsqrt_predicate(clause, native_realsqrt_inputs):
    x, X = native_realsqrt_inputs
    if clause == 1:
        operand = x
    elif clause == 2:
        operand = 1j * abs(x)
    elif clause == 3:
        operand = x**2
    elif clause == 4:
        operand = abs(x)
    elif clause == 5:
        operand = 1j * X
    elif clause == 6:
        operand = X
    elif clause == 7:
        operand = X**2
    else:
        operand = abs(X)
    if clause in (1, 2, 5, 6):
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:realsqrt:complexRes"):
            operand.realsqrt()
    else:
        # Native acceptance-only clause: do not invent or relax a norm gate.
        operand.realsqrt()
