"""Literal 12 clauses of native test_besselh.m, pin7574c77.

Tuple-to-complex observation preserves the existing Python public adapter.
Native builtin Hankel values are portable captured reference data.
RNG6178 first100 is bit-qualified by trreal1/RNG_COMPARISON.json.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(np.float64).eps
REFERENCE = np.load(Path(__file__).parent / "fixtures/besselh_source_native_20261011.npz")
XR = jnp.asarray(REFERENCE["xr"])
NU = 1.663

_MISSING = object()


def observed(f, k=_MISSING, scale=_MISSING, pref=_MISSING):
    args = [] if k is _MISSING else [k]
    kwargs = {}
    if scale is not _MISSING:
        kwargs["scale"] = scale
    if pref is not _MISSING:
        kwargs["pref"] = pref
    re, im = f.besselh(NU, *args, **kwargs)
    return re + 1j * im


def op(x):
    return jnp.stack((jnp.exp(-x), 1 / (1 + 25 * (x - 0.1) ** 2)), axis=-1)


@pytest.fixture(scope="module")
def scalar():
    return cj.chebfun(jnp.exp, domain=[-1, -0.5, 0, 0.5, 1], pref=ChebfunPref())


@pytest.fixture(scope="module")
def array():
    return cj.chebfun(op, domain=[-1, -0.5, 0, 0.5, 1], pref=ChebfunPref())


@pytest.mark.parametrize("clause", [1, 2, 3, 4, 5])
def test_scalar_source(clause, scalar):
    h = observed(scalar)
    if clause in (2, 3):
        h2 = observed(scalar, 1) if clause == 2 else observed(scalar, 1, 0)
        assert float((h - h2).normest()) < 10 * max(EPS * h.vscale, EPS * h2.vscale)
        return
    k = 2 if clause == 4 else 1
    scale = int(clause == 5)
    if clause != 1:
        h = observed(scalar, k, scale)
    exact = REFERENCE["scalar_h2" if k == 2 else "scalar_h1e" if scale else "scalar_h1"]
    assert float(jnp.max(jnp.abs(h(XR) - exact))) < 1e2 * EPS * h.vscale


@pytest.mark.parametrize("clause", [6, 7, 8, 9, 10])
def test_array_source(clause, array):
    h = observed(array)
    if clause in (7, 8):
        h2 = observed(array, 1) if clause == 7 else observed(array, 1, 0)
        assert float((h - h2).normest()) < 10 * max(EPS * h.vscale, EPS * h2.vscale)
        return
    k = 2 if clause == 9 else 1
    scale = int(clause == 10)
    if clause != 6:
        h = observed(array, k, scale)
    exact = REFERENCE["array_h2" if k == 2 else "array_h1e" if scale else "array_h1"]
    factor = {6: 1e3, 9: 1e2, 10: 10}[clause]
    assert float(jnp.max(jnp.abs(h(XR) - exact).reshape(-1))) < factor * EPS * h.vscale


def test_complex_source_11():
    pref = ChebfunPref()
    pref.splitting = True
    fn = lambda x: jnp.where(x <= 0, jnp.exp(4 * jnp.pi * 1j * x), jnp.exp(x))  # noqa: E731 - native source handle
    f = cj.chebfun(fn, domain=[-1, 0, 0.5, 1], pref=pref)
    h = observed(f, 1, 0, pref=pref)
    assert float(jnp.max(jnp.abs(h(XR) - REFERENCE["complex_h1"]))) < 1e2 * EPS * h.vscale


def test_zero_source_12():
    pref = ChebfunPref()
    pref.splitting = True
    f = cj.chebfun(lambda x: x, pref=pref)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:besselh:zero"):
        observed(f)


def test_public_pref_forwarding_supplement():
    """Source-body supplemental: module adapter forwards exact pref identity."""
    pref = ChebfunPref()
    calls = []
    token = object()

    class Input:
        def besselh(self, nu, k, *, scale, pref):
            calls.append((nu, k, scale, pref))
            return token

    assert cj.besselh(Input(), NU, 2, scale=1, pref=pref) is token
    assert calls == [(NU, 2, 1, pref)]
