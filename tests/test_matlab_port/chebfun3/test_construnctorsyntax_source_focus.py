"""Focused source predicates from Chebfun3 ``test_construnctorsyntax``.

Provenance
----------
MATLAB source: tests/chebfun3/test_construnctorsyntax.m
Chebfun commit: 7574c77

The source's split-domain positional spelling ``chebfun3(f, [-1, 0],
[2, 3])`` has no direct Python factory equivalent; Python's domain adapter is
the six-value ``domain=`` keyword. Likewise, Python represents constructor
preference selection with ``pref=`` rather than a positional character flag.
Those syntax differences are kept explicit instead of being hidden by a
different constructor test.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


def _syntax_function(x, y, z):
    return jnp.cos(x) + jnp.sin(x * y * z)


@pytest.mark.parametrize("route", ["callable", "string", "domain", "x_only", "x_z"])
def test_native_clause1_supported_constructor_syntax(route):
    """Separate source clause-1 calls so adaptive work is independently bounded."""
    if route == "callable":
        result = chebfun3(_syntax_function)
    elif route == "string":
        result = chebfun3("cos(x) + sin(x.*y.*z)")
    elif route == "domain":
        result = chebfun3(_syntax_function, domain=(-1, 1, 1, 2, 0, 2))
    elif route == "x_only":
        result = chebfun3(lambda x, y, z: x)
    else:
        result = chebfun3(lambda x, y, z: x + jnp.cos(z))
    assert isinstance(result, Chebfun3)


def _rank_callback(x, y, z):
    return jnp.sin(x * y * z)


def _rank_tuple(f):
    return tuple(int(v) for v in f.rank)


def test_native_clauses8_9_rank_keyword_adapter():
    """Native clauses 8-9: componentwise rank reduction and default domain."""
    baseline = chebfun3(_rank_callback)
    old_rank = _rank_tuple(baseline)
    result = chebfun3(_rank_callback, rank=tuple(v - 1 for v in old_rank))
    assert all(new < old for new, old in zip(_rank_tuple(result), old_rank))
    assert result.domain == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_native_clauses10_11_domain_rank_keyword_adapter():
    """Native clauses 10-11: rank reduction on an explicit domain."""
    domain = (-2.0, 1.0, -2.0, 1.0, -2.0, 1.0)
    baseline = chebfun3(_rank_callback, domain=domain)
    old_rank = _rank_tuple(baseline)
    result = chebfun3(
        _rank_callback,
        domain=domain,
        rank=tuple(v - 1 for v in old_rank),
    )
    assert all(new < old for new, old in zip(_rank_tuple(result), old_rank))
    assert result.domain == domain


def test_native_clauses12_13_reverse_order_rank_adapter():
    """Native clauses 12-13: explicit domain plus the literal rank vector."""
    domain = (-2.0, 1.0, -2.0, 1.0, -2.0, 1.0)
    baseline = chebfun3(_rank_callback, domain=domain)
    old_rank = _rank_tuple(baseline)
    # Native source uses [5, 6, 7] after the prior default rank construction.
    result = chebfun3(_rank_callback, domain=domain, rank=(5, 6, 7))
    assert all(new < old for new, old in zip(_rank_tuple(result), old_rank))
    assert result.domain == domain


def test_native_clause14_existing_object_constructor_selectors():
    """Map existing-object selector use to the Python preference adapter."""
    domain = (-2.0, 1.0, -2.0, 1.0, -2.0, 1.0)
    existing = chebfun3(_rank_callback, domain=domain)
    classic = ChebfunPref()
    classic.cheb3Prefs.constructor = "classic"
    source_default = ChebfunPref()
    source_default.cheb3Prefs.constructor = "chebfun3f"
    assert chebfun3(existing, pref=classic) is existing
    assert chebfun3(existing, pref=source_default) is existing
