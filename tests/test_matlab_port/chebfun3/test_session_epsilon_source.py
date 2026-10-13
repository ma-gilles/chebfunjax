"""Chebfun3 session epsilon and original epsilon controls.

Provenance
----------
MATLAB source: @chebfun3/constructor.m; @chebfun3/chebfun3f.m;
              tests/chebfun3/test_constructor2.m, clauses 13--14.
Chebfun commit: 7574c77

Session/keyword equivalence is supplemental; length/rank inequalities below
are the original native predicates with the unchanged function and epsilon.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def restore_preferences():
    original = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref.setDefaults(original)


def _source_function(x, y, z):
    return -x * jnp.sin(jnp.sqrt(x + 15 + 2*y + 3*z))


def _same_representation(a, b):
    assert a.length() == b.length()
    assert a.rank == b.rank
    np.testing.assert_array_equal(a.core, b.core)
    for group in ('cols', 'rows', 'tubes'):
        for left, right in zip(getattr(a, group), getattr(b, group)):
            np.testing.assert_array_equal(left.coeffs, right.coeffs)


def test_session_epsilon_matches_explicit_eps():
    reference = chebfun3(_source_function, eps=1e-8)
    ChebfunPref.setDefaults(['cheb3Prefs', 'chebfun3eps'], 1e-8)
    actual = chebfun3(_source_function)
    _same_representation(actual, reference)


def test_constructor2_clauses_13_14_and_override_precedence():
    tight = chebfun3(_source_function)
    loose = chebfun3(_source_function, eps=1e-8)
    assert all(a < b for a, b in zip(loose.length(), tight.length()))
    assert all(a < b for a, b in zip(loose.rank, tight.rank))
    ChebfunPref.setDefaults(['cheb3Prefs', 'chebfun3eps'], 1e-8)
    _same_representation(chebfun3(_source_function, tol=2**-52), tight)
    _same_representation(chebfun3(_source_function, tol=2**-52, eps=1e-8), loose)
    assert ChebfunPref().cheb3Prefs.chebfun3eps == 1e-8


@pytest.mark.parametrize("session,kwargs,expected", [
    (0.0, {}, 0.0),
    (1e-8, {"tol": None}, 1e-8),
    (1e-8, {"tol": 0.0}, 0.0),
    (1e-8, {"tol": 1e-6, "eps": 0.0}, 0.0),
])
def test_zero_and_none_dispatch(monkeypatch, session, kwargs, expected):
    seen = []
    sentinel = object()
    def record(cls, function, **options):
        seen.append(options["tol"])
        return sentinel
    monkeypatch.setattr(Chebfun3, "from_function", classmethod(record))
    ChebfunPref.setDefaults(['cheb3Prefs', 'chebfun3eps'], session)
    assert chebfun3(lambda x, y, z: x + y + z, **kwargs) is sentinel
    assert seen == [expected]
