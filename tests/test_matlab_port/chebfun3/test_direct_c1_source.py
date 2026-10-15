"""Source C1 preference dispatch, supplementary to native test_techs 1–2."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1


@pytest.fixture(autouse=True)
def reset_preferences():
    previous = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref.setDefaults(previous)


def test_explicit_first_kind_keeps_native_c2_sampling(monkeypatch):
    from chebfunjax.chebfun3d import _chebfun3f_source as source
    seen = []
    original = source.cosine_points

    def points(n, a, b):
        values = original(n, a, b)
        seen.append(values)
        return values

    monkeypatch.setattr(source, "cosine_points", points)
    pref = ChebfunPref()
    pref.tech = "chebtech1"
    f = chebfun3(lambda x, y, z: 0*x, pref=pref)
    assert all(isinstance(factor, Chebtech1) for factors in
               (f.cols, f.rows, f.tubes) for factor in factors)
    assert seen[0].shape == (17,)
    assert seen[0][0] == -1 and seen[0][-1] == 1
    assert float(f(.17, -.29, .43)) == 0
    assert pref.tech == "chebtech1"


def test_session_first_kind_and_legacy_keyword_remain_distinct(monkeypatch):
    pref = ChebfunPref()
    pref.tech = "chebtech1"
    ChebfunPref.setDefaults(pref)
    # Session C1 reaches the native path without calling the legacy converter.
    original = Chebfun3.change_tech
    seen = []

    def conversion(self, tech):
        seen.append(tech)
        return original(self, tech)

    monkeypatch.setattr(Chebfun3, "change_tech", conversion)
    f = chebfun3(lambda x, y, z: 0*x)
    assert isinstance(f.cols[0], Chebtech1)
    assert seen == []
    ChebfunPref.setDefaults(ChebfunPref(tech="chebtech2"))
    legacy = chebfun3(lambda x, y, z: 0*x, tech="chebtech1")
    assert isinstance(legacy.cols[0], Chebtech1)
    assert seen == ["chebtech1"]


@pytest.mark.slow
def test_explicit_chebtech1_original_1_2(monkeypatch, record_property):
    """Literal original bounds; native warned-field observations are separate."""
    import importlib

    module = importlib.import_module('chebfunjax.chebfun3d.chebfun3')

    def forbidden(*args, **kwargs):
        raise AssertionError('Direct C1 used postconversion or repaired stored core')

    monkeypatch.setattr(Chebfun3, 'change_tech', forbidden)
    monkeypatch.setattr(module, '_stored_factor_core', forbidden)
    pref = ChebfunPref()
    pref.tech = 'chebtech1'
    op = lambda x, y, z: jnp.cos(x*y*z)
    with pytest.warns(RuntimeWarning, match='chebfun3f: max number of restarts reached'):
        f = chebfun3(op, (-1, 1, -1, 1, -1, 1), pref=pref)
    assert isinstance(f.cols[0], Chebtech1)
    assert abs(float(f(0, 0, 0))-float(op(0., 0., 0.))) < 100*pref.cheb3Prefs.chebfun3eps
    # Source test_techs has no off-grid acceptance bound. Preserve measurements
    # without inventing one from the native warned approximation.
    q = jnp.asarray([-.7, -.25, .13, .51, .89])
    values = f(q, q[::-1], q/3)
    record_property('offgrid_values', str(values.tolist()))
    record_property('factor_lengths', str(f.length()))
