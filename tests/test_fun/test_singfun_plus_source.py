"""Bounded public Case3 return types and warning semantics.

Provenance
----------
MATLAB source : @singfun/plus.m
Chebfun commit: 7574c77
"""
import warnings

import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("reverse", [False, True])
def test_public_case3_demotes_smooth_result_and_warns(reverse):
    # A high fractional power stays cheap to resolve while contributing a
    # full unit at the right endpoint; this exercises actual construction.
    f = Singfun(2.0 ** (-20.5), (20.5, 0.0))
    g = Singfun(1.0, (0.0, 0.0))
    with pytest.warns(UserWarning) as caught:
        h = g + f if reverse else f + g
    assert len(caught) == 1
    assert str(caught[0].message) == (
        "CHEBFUN:SINGFUN:plus:exponentDiff: "
        "Non-integer difference in the exponents of the two SINGFUN "
        "objects: The result may not be accurate."
    )
    assert isinstance(h, Chebtech2) and not isinstance(h, Singfun)
    assert h.ishappy and len(h) < 128
    x = jnp.array([-0.7, 0.2, 0.9, 1.0])
    exact = 1.0 + ((1.0 + x) / 2.0) ** 20.5
    assert float(jnp.max(jnp.abs(h(x) - exact))) < 10 * float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("survivor_exponents", [(0.0, 0.0), (0.5, 0.0)])
def test_case3_zero_returns_survivor_before_demotion(reverse, survivor_exponents):
    zero_exponents = (0.5, 0.0) if survivor_exponents == (0.0, 0.0) else (0.0, 0.0)
    zero = Singfun(0.0, zero_exponents)
    survivor = Singfun(2.0, survivor_exponents)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = survivor + zero if reverse else zero + survivor
    assert result is survivor
    assert not caught


def test_case3_zero_preserves_promoted_scalar_type():
    zero = Singfun(0.0, (0.5, 0.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = zero + 2.0
    assert isinstance(result, Singfun)
    assert result.exponents == (0.0, 0.0)
    assert result.isequal(Singfun(2.0, (0.0, 0.0)))
    assert not caught
