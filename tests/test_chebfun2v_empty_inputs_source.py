"""Actual native empty-input Chebfun2v constructor observations.

Provenance
----------
MATLAB source: @chebfun2v/chebfun2v.m, @separableApprox/domainCheck.m,
    @chebfun2v/isempty.m, @chebfun2v/subsref.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Supplemental native controls, not additional original MATLAB test predicates.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


def _arguments(name):
    def empty():
        return jnp.empty((0,), dtype=jnp.float64)
    return {
        "noargs": lambda: ((), {}),
        "numeric_empty": lambda: ((empty(),), {}),
        "numeric_0by2": lambda: ((jnp.empty((0, 2)),), {}),
        "cell_empty_numeric": lambda: (([empty()],), {}),
        "empty_cell": lambda: (([],), {}),
        "existing_empty": lambda: ((Chebfun2.empty(),), {}),
        "empty_and_constant": lambda: ((empty(), 2), {}),
        "two_empty": lambda: ((empty(), empty()), {}),
        "empty_custom_domain": lambda: ((empty(),), {"domain": (-2, 3, 0, 1)}),
        "empty_constant_custom_domain": lambda: ((empty(), 2), {"domain": (-2, 3, 0, 1)}),
        "cell_empty_constant": lambda: (([empty(), 2],), {}),
        "empty_existing_custom": lambda: ((Chebfun2.empty(), Chebfun2.from_function(
            lambda x, y: x, domain=(-2, 3, 0, 1))), {}),
    }[name]()


@pytest.mark.parametrize("name", [
    "noargs", "numeric_empty", "numeric_0by2", "cell_empty_numeric",
    "existing_empty", "two_empty", "empty_custom_domain",
])
def test_native_all_empty_construction(name):
    args, kwargs = _arguments(name)
    field = Chebfun2v.from_functions(*args, **kwargs)
    assert field.isempty()
    assert field(jnp.asarray([0.1, 0.2]), jnp.asarray([0.3, 0.4])).size == 0
    assert field.norm().size == 0


@pytest.mark.parametrize("name", [
    "empty_and_constant", "empty_constant_custom_domain", "cell_empty_constant",
    "empty_existing_custom",
])
def test_native_mixed_empty_domain_error(name):
    args, kwargs = _arguments(name)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:chebfun2v:domainCheck"):
        Chebfun2v.from_functions(*args, **kwargs)


def test_native_empty_cell_is_distinct_from_numeric_empty():
    args, kwargs = _arguments("empty_cell")
    with pytest.raises(ValueError, match="MATLAB:UndefinedFunction"):
        Chebfun2v.from_functions(*args, **kwargs)
