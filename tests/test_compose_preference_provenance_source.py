"""Source periodic compose preference forwarding, pin7574c77.

@chebfun/compose.m269–283 sets source overrides then actual operand Tech.
Spies qualify public option routing; separate actual fixed-grid checks below.
"""
import copy

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.tech.trigtech import Trigtech


@pytest.fixture(autouse=True)
def isolated_session():
    saved = ChebfunPref._defaults
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref._defaults = saved


@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('route', [
    'omitted', 'object_omitted', 'object_explicit', 'flat_dict',
    'nested_dict', 'session_explicit', 'materialized',
])
def test_public_effective_tech_and_explicit_equal_defaults(monkeypatch, route, binary):
    f = chebfun(lambda x: jnp.sin(jnp.pi*x), trig=True, n=8)
    seen = []

    def spy(self, op, g=None, data=None, pref=None):
        seen.append(copy.deepcopy(pref))
        return self

    monkeypatch.setattr(Trigtech, 'compose', spy)
    pref = None
    expected = 65536
    if route == 'object_omitted':
        pref = ChebfunPref(tech='chebtech2')
    elif route == 'object_explicit':
        pref = ChebfunPref(maxLength=65537)
        expected = 65537
    elif route == 'flat_dict':
        pref = {'maxLength': 65537, 'splitting': False}
        expected = 65537
    elif route == 'nested_dict':
        pref = {'techPrefs': {'maxLength': 65537}, 'domain': (-2, 3)}
        expected = 65537
    elif route == 'session_explicit':
        ChebfunPref.setDefaults(maxLength=65537, sampleTest=False)
        expected = 65537
    elif route == 'materialized':
        pref = {'techPrefs': dict(ChebfunPref().techPrefs)}
        expected = 65537
    before = copy.deepcopy(pref)
    if binary:
        f.compose(jnp.add, f, pref=pref)
    else:
        f.compose(lambda x: x, pref=pref)
    assert len(seen) == 1 and seen[0]['maxLength'] == expected
    assert seen[0]['gridType'] == 2
    assert all(key not in seen[0] for key in ('domain', 'splitting', 'tech'))
    if route == 'session_explicit':
        assert seen[0]['sampleTest'] is False
        assert ChebfunPref().maxLength == 65537
    if isinstance(pref, ChebfunPref):
        assert pref._top == before._top and pref._tech_overrides == before._tech_overrides
    else:
        assert pref == before


def test_source_split_length_override_precedes_resolution(monkeypatch):
    f = chebfun(lambda x: jnp.cos(jnp.pi*x), trig=True, n=8)
    pref = ChebfunPref(maxLength=65537, splitting=True, splitPrefs={'splitLength': 31})
    seen = []

    def spy(self, op, g=None, data=None, pref=None):
        seen.append(pref)
        return self

    monkeypatch.setattr(Trigtech, 'compose', spy)
    f.compose(lambda x: x, pref=pref)
    assert seen[0]['maxLength'] == 31
    assert pref.maxLength == 65537 and pref.tech == 'chebtech2'


def test_array_binary_forwarding_preserves_columns(monkeypatch):
    f = chebfun(lambda x: jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x)), axis=-1), trig=True, n=8)
    seen = []

    def spy(self, op, g=None, data=None, pref=None):
        seen.append((self.coeffs.shape, g.coeffs.shape, pref))
        return self

    monkeypatch.setattr(Trigtech, 'compose', spy)
    result = f.compose(jnp.add, f, pref={'minSamples': 33, 'sampleTest': False})
    assert result.n_columns == 2
    assert seen[0][0][1] == seen[0][1][1] == 2
    assert seen[0][2]['minSamples'] == 33 and seen[0][2]['sampleTest'] is False


def test_source_multiinterval_override_in_existing_piece_adapter(monkeypatch):
    # Piecewise Trig storage adapter only; not a claim public periodic
    # constructor accepts multiple intervals (it intentionally rejects them).
    t = Trigtech.from_values(jnp.ones(4))
    f = Chebfun(funs=[_Piece(t, (-1., 0.)), _Piece(t, (0., 1.))], domain=Domain((-1., 0., 1.)))
    seen = []

    def spy(self, op, g=None, data=None, pref=None):
        seen.append(pref)
        return self

    monkeypatch.setattr(Trigtech, 'compose', spy)
    pref = ChebfunPref(extrapolate=False)
    f.compose(lambda x: x, pref=pref)
    assert len(seen) == 2 and all(p['extrapolate'] is True for p in seen)
    assert pref.extrapolate is False


@pytest.mark.parametrize('binary', [False, True])
def test_actual_fixed_length_session_unary_and_explicit_binary(binary):
    f = chebfun(lambda x: jnp.sin(jnp.pi*x), trig=True, n=8)
    if binary:
        result = f.compose(jnp.add, f, pref={'fixedLength': 14})
        n, scale = 14, 2
    else:
        ChebfunPref.setDefaults(fixedLength=12)
        result = f.compose(lambda x: x)
        n, scale = 12, 1
    assert result.funs[0].tech.n == n
    x = jnp.linspace(-.97, .97, 19)
    assert jnp.max(jnp.abs(result(x) - scale*jnp.sin(jnp.pi*x))) < 4e-14
