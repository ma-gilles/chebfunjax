"""Port of MATLAB Chebfun tests/chebfun2v/test_size.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

INF = float("inf")


class TestChebfun2vSize:
    def test_two_components(self):
        # pass(1)-(4): size(F) == [2 inf inf].
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y))
        assert F.shape == (2, INF, INF)
        assert tuple(F.size()) == (2, INF, INF)
        assert F.size(1) == 2
        assert F.size(2) == INF
        assert F.size(3) == INF

    def test_three_components(self):
        # pass(5)-(8): size(F) == [3 inf inf].
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y),
                                     lambda x, y: jnp.cos(x))
        assert F.shape == (3, INF, INF)
        assert tuple(F.size()) == (3, INF, INF)
        assert F.size(1) == 3
        assert F.size(2) == INF
        assert F.size(3) == INF


def test_transposed_source_dimensions():
    field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y).T
    assert field.shape == (INF, INF, 2)
    assert tuple(field.size()) == (INF, INF, 2)
    assert field.size(1) == INF
    assert field.size(2) == INF
    assert field.size(3) == 2


def test_empty_source_size_early_return():
    field = Chebfun2v.empty()
    assert field.size().shape == (0,)
    assert field.size(4).shape == (0,)


@pytest.mark.parametrize("dim", [-1, 0, 4, 1.5])
def test_source_dimension_errors(dim):
    field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:size:dim"):
        field.size(dim)
