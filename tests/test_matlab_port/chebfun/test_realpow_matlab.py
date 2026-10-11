"""All8 literal tests/chebfun/test_realpow.m predicates, Chebfun7574c77.

Actual public x/pref, horzcat and arithmetic routes; exact native error
identifier and acceptance gates. Prior sin/power probe remains supplemental.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunRealpow:
    def test_integer_ok_fractional_guarded(self):
        import pytest

        f = cj.chebfun(jnp.sin)
        g = f.realpow(2)
        assert abs(float(g(jnp.asarray(0.4))) - np.sin(0.4) ** 2) < 1e-13
        with pytest.raises(ValueError):
            f.realpow(0.5)


import jax
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(scope="module")
def native_realpow_inputs():
    pref = ChebfunPref()
    x = cj.chebfun("x", pref=pref)
    # MATLAB X=[x x] uses the public horizontal concatenation operation.
    X = Chebfun.horzcat(x, x)
    return x, X


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("clause", range(1, 9))
def test_native_realpow_predicate(clause, native_realpow_inputs):
    x, X = native_realpow_inputs
    if clause == 1:
        operand = x
        power = 1 / 3
    elif clause == 2:
        operand = 1j * abs(x)
        power = 3
    elif clause == 3:
        operand = x
        power = 1
    elif clause == 4:
        operand = 1j * x
        power = 0
    elif clause == 5:
        operand = 1j * X
        power = 3
    elif clause == 6:
        operand = X
        power = 0.42
    elif clause == 7:
        operand = X
        power = 1
    else:
        operand = 1j * X
        power = 0
    if clause in (1, 2, 5, 6):
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:realpow:complexRes"):
            operand.realpow(power)
    else:
        # Native pass=true after the call: no invented numerical gate.
        operand.realpow(power)
