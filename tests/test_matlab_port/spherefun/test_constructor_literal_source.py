"""Seven qualified predicates of the pinned scalar Spherefun constructor test.

Source clauses 1, 2, 8, 9, 16, 26 and 30 are included. The other 23 clauses
remain in the preserved full-source draft pending public API implementation.

Provenance
----------
MATLAB source : tests/spherefun/test_constructor.m, @spherefun/constructor.m,
    @spherefun/sphf2cartf.m, @separableApprox/rank.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.

Existing classmethod spellings adapt supported input forms. Unsupported source
constructor overloads require a public library factory; this adapter deliberately
fails rather than implementing the missing algorithms in the test. No skips,
xfails, sampled substitutes for continuous norms, or stored-rank substitutes.
"""

import inspect

import jax.numpy as jnp
import pytest

import chebfunjax.spherefun as sphere_api
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun import Spherefun
from chebfunjax.utils.quadrature import trigpts


def _construct(op, *source_options):
    """Syntax adapter; a future public factory owns unsupported semantics."""
    factory = getattr(sphere_api, "spherefun", None)
    if callable(factory):
        return factory(op, *source_options)
    if source_options == ("coeffs",):
        return Spherefun.coeffs2spherefun(op)
    if not source_options:
        if callable(op) and not isinstance(op, Spherefun):
            if len(inspect.signature(op).parameters) == 2:
                return Spherefun.from_function(op)
        elif not isinstance(op, (str, Spherefun)):
            return Spherefun.from_values(op)
    raise AssertionError(
        "Missing public spherefun source-input dispatch: Cartesian callback, "
        "vectorize, fixed rank/copy, eps option, or string input. Implement "
        "the library API; do not replace this call by a test-local algorithm."
    )


def _spherical(f):
    # Exactly the test's explicit redefine_function_handle/sphf2cartf(...,0).
    return lambda lam, th: f(jnp.cos(lam) * jnp.sin(th),
                            jnp.sin(lam) * jnp.sin(th), jnp.cos(th))


def _sample_error(f, g):
    x = trigpts(12, [-jnp.pi, jnp.pi])[0]
    y = jnp.linspace(0.0, jnp.pi, 6)
    lam, th = jnp.meshgrid(x, y)
    return jnp.max(jnp.abs(f(lam, th) - g.fevalm(x, y)))


def _tol():
    return 2e3 * ChebfunPref().cheb2Prefs.chebfun2eps




def _error_id(call, expected):
    # Python ValueError text carries the MATLAB identifier; type is not a
    # substitute for the exact, case-sensitive identifier required by source.
    with pytest.raises(Exception) as caught:
        call()
    exc = caught.value
    identifier = getattr(exc, "identifier", str(exc).split()[0])
    assert identifier == expected


@pytest.mark.parametrize("clause,f,wrap_first", [
    (1, lambda x, y, z: x**2 + y**2 + z**2, True),
    (2, lambda x, y, z: jnp.exp(-jnp.cos(jnp.pi * (x + y + z))), True),
    (8, lambda x, y, z: jnp.sin(x + y * z), True),
    (9, lambda x, y, z: jnp.sin(x + y * z) + 1, True),
], ids=[f"source_clause_{k:02d}" for k in (1, 2, 8, 9)])
def test_constructor_sample_error(clause, f, wrap_first):
    spherical = _spherical(f)
    g = _construct(spherical if wrap_first else f)
    assert _sample_error(spherical, g) < _tol(), clause










def test_source_clause_16():
    _error_id(lambda: _construct(jnp.ones((1, 2))),
              "CHEBFUN:SPHEREFUN:constructor:poleSamples")




















def test_source_clause_26():
    f = _construct(1)
    assert (f - 1).norm(jnp.inf) == 0








def test_source_clause_30():
    f = _construct(jnp.zeros((5, 4)))
    n, m = f.length()
    assert (m == 5) and (n == 4)
