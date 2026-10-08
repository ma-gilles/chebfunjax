"""Original test_times.m: all five operand combinations and source bounds.

Provenance: Chebfun 7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_times.m and @adchebfun/*TestingBinary.m.
Fixed degree-seven inputs replace rand(8,1), without RNG-stream equivalence.
"""
import operator

import jax.numpy as jnp
import pytest

from ._binary_source import data as data
from ._binary_source import evaluate, taylor_errors, value_error

OPERATIONS = [operator.mul]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index", range(5))
def test_source_value_clause(data, operation, index):
    error = value_error(operation, data, index)
    assert error == 0


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index", range(5))
def test_source_taylor_clause(data, operation, index):
    order1, order2, remainder = taylor_errors(operation, data, index)
    if index == 0:
        assert float(jnp.max(jnp.abs(order1-1))) < 1e-2
        assert float(jnp.max(jnp.abs(order2-2))) < 1e-2
    else:
        # Preserve literal source max(abs(order1))-1 < tolOrder.
        assert float(jnp.max(jnp.abs(order1)))-1 < 1e-2
        assert float(jnp.max(remainder)) < 1e-14


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index", range(5))
def test_source_linearity_clause(data, operation, index):
    assert evaluate(operation, data, index, value_seeded=True).is_linear == (index != 0)
