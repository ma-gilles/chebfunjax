"""Core tests for tech-level empty representations (Fable 5).

MATLAB's ``chebtech1()`` / ``chebtech2()`` / ``trigtech()`` with no arguments
give an empty tech: ``isempty`` is True and arithmetic / restriction with an
empty operand propagates the empty.  These mirror the tech empty-case
assertions from the MATLAB port suite.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechEmpty:
    def test_empty_is_empty(self, Tech):
        assert Tech.empty().isempty()

    def test_constructed_not_empty(self, Tech):
        assert not Tech.from_function(jnp.sin).isempty()

    def test_add_propagates(self, Tech):
        e = Tech.empty()
        g = Tech.from_function(jnp.sin)
        assert (e + e).isempty()
        assert (e + g).isempty()
        assert (g + e).isempty()

    def test_sub_propagates(self, Tech):
        e = Tech.empty()
        g = Tech.from_function(jnp.sin)
        assert (e - g).isempty()
        assert (g - e).isempty()

    def test_mul_propagates(self, Tech):
        e = Tech.empty()
        g = Tech.from_function(jnp.sin)
        assert (e * g).isempty()
        assert (g * e).isempty()
        assert (2.0 * e).isempty()
        assert (e * 2.0).isempty()


class TestTrigtechEmpty:
    def test_empty_is_empty(self):
        assert Trigtech.empty().isempty()

    def test_constructed_not_empty(self):
        assert not Trigtech.from_function(
            lambda x: jnp.sin(jnp.pi * x)
        ).isempty()

    def test_restrict_propagates(self):
        assert Trigtech.empty().restrict(-0.5, 0.5).isempty()
