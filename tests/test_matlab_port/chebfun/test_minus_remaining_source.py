"""Literal source predicates 1-14 and 17 from Chebfun test_minus.m.

Provenance
----------
MATLAB source : tests/chebfun/test_minus.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
RNG source: native seed6178 uniform sites are checked against captured x1
and continuation x3 using the qualified explicit-state MATLAB Twister helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform

jax.config.update("jax_enable_x64", True)
EPS = float(np.finfo(np.float64).eps)
ALPHA = -0.194758928283640 + 0.075474485412665j
DRAWS = json.loads((Path(__file__).parent / "fixtures" / "times_source_matlab.json").read_text())


def _native_uniform_sites():
    u, _ = matlab_rand_uniform(init_matlab_twister(6178), (200, 1))
    first = 2.0 * u[:100, 0] - 1.0
    continuation = 200.0 * u[100:, 0] - 100.0
    assert np.array_equal(np.asarray(first), np.asarray(DRAWS["x1"]))
    assert np.array_equal(np.asarray(continuation), np.asarray(DRAWS["x3"]))
    return u


def _native_norm_inf(a):
    """MATLAB matrix infinity norm: largest row sum of magnitudes."""
    values = jnp.asarray(a)
    # MATLAB norm(X, inf) uses max(abs(X)) for vectors, including row and
    # column vectors; only a genuine matrix uses the maximum absolute row sum.
    if values.ndim == 1 or 1 in values.shape:
        return float(jnp.max(jnp.abs(values)))
    return float(jnp.max(jnp.sum(jnp.abs(values), axis=1)))


def _vscale(f):
    return float(jnp.max(jnp.asarray(f.vscale)))


def _f1(t):
    return jnp.sin(t) * jnp.abs(t - 0.1)


def _g1(t):
    return jnp.cos(t) * jnp.sign(t + 0.2)


def _f2(t):
    return jnp.stack((_f1(t), jnp.exp(t)), axis=-1)


def _g2(t):
    return jnp.stack((_g1(t), jnp.tan(t)), axis=-1)


def test_source_minus_empty_predicates_01_02():
    # Native default preferences, before enabling splitting for remaining tests.
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun()
    assert (f - []).isempty()
    assert (f - g).isempty()


def test_source_minus_scalar_predicates_03_06():
    u = _native_uniform_sites()
    x = 2.0 * u[:100, :] - 1.0
    t = x[:, 0]
    fop, gop = lambda z: jnp.sin(z) * jnp.abs(z - 0.1), lambda z: jnp.cos(z) * jnp.sign(z + 0.2)
    f = cj.chebfun(fop, splitting=True)
    g = cj.chebfun(gop, splitting=True)

    # Source passes 3-4: f-alpha and alpha-f, source helper equality + bound.
    fma, amf = f - ALPHA, ALPHA - f
    assert fma.isequal(-amf)
    assert fma(x).shape == (100, 1)
    assert _native_norm_inf(fma(x) - (fop(t) - ALPHA)[:, None]) < 100.0 * _vscale(fma) * EPS

    # Source passes 5-6: f-g and g-f, source helper equality + bound.
    fmg, gmf = f - g, g - f
    assert fmg.isequal(-gmf)
    assert fmg(x).shape == (100, 1)
    assert _native_norm_inf(fmg(x) - (fop(t) - gop(t))[:, None]) < 100.0 * _vscale(fmg) * EPS


def test_source_minus_array_and_transposed_predicates_07_14():
    u = _native_uniform_sites()
    x = 2.0 * u[:100, :] - 1.0
    t = x[:, 0]
    f2 = cj.chebfun(_f2, splitting=True)
    g2 = cj.chebfun(_g2, splitting=True)

    # Passes 7-8: native err is 100x2; norm(err,inf) sums across each row.
    fma, amf = f2 - ALPHA, ALPHA - f2
    assert fma.isequal(-amf)
    assert fma(x).shape == (100, 2)
    exact_fma = _f2(t) - ALPHA
    assert _native_norm_inf(fma(x) - exact_fma) < 100.0 * _vscale(fma) * EPS

    # Passes 9-10: native two-column function difference and row-sum norm.
    fmg, gmf = f2 - g2, g2 - f2
    assert fmg.isequal(-gmf)
    assert fmg(x).shape == (100, 2)
    assert _native_norm_inf(fmg(x) - (_f2(t) - _g2(t))) < 100.0 * _vscale(fmg) * EPS

    f1 = cj.chebfun(lambda z: jnp.sin(z) * jnp.abs(z - 0.1), splitting=True)
    g1 = cj.chebfun(lambda z: jnp.cos(z) * jnp.sign(z + 0.2), splitting=True)
    fr, gr = f1.T, g1.T

    # Passes 11-12: native row result is 1x100; its inf norm is max(abs(row)).
    fma, amf = fr - ALPHA, ALPHA - fr
    assert fma.isequal(-amf)
    expected_fma = (_f1(t) - ALPHA)[None, :]
    assert fma(x).shape == (1, 100)
    assert _native_norm_inf(fma(x) - expected_fma) < 100.0 * _vscale(fma) * EPS

    # Passes 13-14: row-by-row Chebfun subtraction, native row norm.
    fmg, gmf = fr - gr, gr - fr
    assert fmg.isequal(-gmf)
    expected_fmg = (_f1(t) - _g1(t))[None, :]
    assert fmg(x).shape == (1, 100)
    assert _native_norm_inf(fmg(x) - expected_fmg) < 100.0 * _vscale(fmg) * EPS


def test_source_minus_unbounded_left_domain_predicate_17():
    u = _native_uniform_sites()
    dom = (-jnp.inf, 3.0 * jnp.pi)
    dom_check = (-1.0e6, 3.0 * jnp.pi)
    x = (dom_check[1] - dom_check[0]) * u[100:, :] + dom_check[0]
    t = x[:, 0]
    opf = lambda z: z * jnp.exp(z)
    opg = lambda z: (1.0 - jnp.exp(z)) / z
    f = cj.chebfun(opf, domain=dom)
    g = cj.chebfun(opg, domain=dom)
    h = f - g
    exact = (opf(t) - opg(t))[:, None]
    assert h(x).shape == (100, 1)
    assert _native_norm_inf(h(x) - exact) < 10.0 * EPS * _vscale(h)
