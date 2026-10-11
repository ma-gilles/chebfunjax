"""Port of MATLAB Chebfun tests/chebfun/test_sign.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunSign:
    def test_sign_of_positive_square(self):
        # sign(x^2) == 1 a.e. (test points avoid the root)
        f = cj.chebfun(lambda x: x**2 + 0.01)
        f1 = f.sign()
        assert bool(jnp.all(f1(X) == 1))

    def test_sign_of_shifted_cos(self):
        f = cj.chebfun(lambda x: jnp.cos(np.pi * x) + 2)
        f1 = f.sign()
        assert bool(jnp.all(f1(X) == 1))

    def test_point_values(self):
        # pass(1,2): sign propagates an explicit pointValues override
        # element-wise.  ``f = cos(pi x) + 2`` is strictly positive so sign
        # adds no breakpoints; the stored point values (all set to -pi) map to
        # sign(-pi) = -1.
        pts = np.sort(2 * RNG.uniform(size=8) - 1)
        f = cj.chebfun(lambda x: jnp.cos(np.pi * x) + 2, domain=tuple(float(v) for v in pts))
        n_ends = len(f.domain.breakpoints)
        f = f.set_point_values(-np.pi * jnp.ones(n_ends))
        f1 = f.sign()
        assert bool(jnp.all(f1.point_values == -1))

    def test_sign_with_jump(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: x - 0.3, splitting=True)
        f1 = f.sign()
        mask = jnp.abs(X - 0.3) > 1e-6
        exact = jnp.sign(X - 0.3)
        assert bool(jnp.all(f1(X)[mask] == exact[mask]))


# Full literal tests/chebfun/test_sign.m predicates, Chebfun7574c77.
# Uniform RandomState stream is a Python adapter, not a MATLAB draw identity claim.

from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)
_native_rng = np.random.RandomState(6178)
NATIVE_X = jnp.asarray(2 * _native_rng.uniform(size=100) - 1)
SING_REAL_X = jnp.asarray(9 * _native_rng.uniform(size=100) - 2)
SING_COMPLEX_X = jnp.asarray(2 * _native_rng.uniform(size=100) - 1)
UNBOUNDED_X = jnp.asarray(200 * _native_rng.uniform(size=100) - 100)


def native_pref():
    pref = ChebfunPref()
    pref.splitting = 0
    return pref


def native_tol():
    return EPS * float(cj.chebfun("x.^2", pref=native_pref()).get("hscale"))


def record(clause, **values):
    print({"clause": clause, **values}, flush=True)


def test_native_sign_simple_square():
    f = cj.chebfun("x.^2", pref=native_pref())
    h = f.sign()
    assert bool(jnp.all(h(NATIVE_X) == 1))
    record("1,1", all_one=True)


def test_native_sign_simple_stored100():
    pref = native_pref()
    f = cj.chebfun(
        lambda x: jnp.cos(jnp.pi * x) + 2, domain=np.sort(np.asarray(NATIVE_X)), pref=pref
    )
    assert len(f.domain.breakpoints) == 100
    f = f.set_point_values(-jnp.pi * jnp.ones(100))
    h = f.sign()
    assert bool(jnp.all(h.point_values == -1))
    record("1,2", original_endpoint_count=100, all_negative_one=True)


def test_native_sign_simple_long_interval():
    f = cj.chebfun(lambda x: x, domain=(-1, 100), pref=native_pref())
    h = cj.chebfun([-1, 1], domain=(-1, 0, 100))
    error = float((h - f.sign()).normest())
    bound = native_tol()
    record("1,3", normest=error, bound=bound)
    assert error < bound


def test_native_sign_simple_existing_break():
    f = cj.chebfun(lambda x: x, domain=(-1, 1e-10, 1), pref=native_pref())
    h = cj.chebfun([-1, 1], domain=(-1, 0, 100)).restrict((-1, 1))
    error = float((f.sign() - h).normest())
    bound = native_tol()
    record("1,4", normest=error, bound=bound)
    assert error < bound


def test_native_sign_real_four():
    pref = native_pref()
    dom = tuple(range(-3, 4))
    tol = native_tol()
    f = cj.chebfun([-1, 1, -1, 1, -1, 1], domain=dom)
    g = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=dom, pref=pref)
    g1 = g.sign()
    e = float((f - g1).normest())
    record("2,1/2", FUN_count=len(g1.funs), normest=e, bound=tol)
    assert len(g1.funs) == 6
    assert e < tol
    pref.splitting = 1
    h = cj.chebfun(lambda x: jnp.sign(jnp.sin(jnp.pi * x)), domain=dom, pref=pref)
    e = float((f - h).normest())
    record("2,3/4", FUN_count=len(h.funs), normest=e, bound=100 * tol)
    assert len(h.funs) == 6
    assert e < 100 * tol


def test_native_sign_array_four():
    pref = native_pref()
    pref.splitting = 1
    dom = tuple(range(-3, 4))
    tol = native_tol()
    f = cj.chebfun([-1, 1, -1, 1, -1, 1], domain=dom)
    f1 = cj.chebfun(lambda x: f(jnp.stack((x, x), axis=-1)), domain=dom, pref=pref)

    def op(x):
        return jnp.stack((jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * (x - 0.5))), axis=-1)

    g = cj.chebfun(op, domain=dom, pref=pref)
    g4 = g.sign()
    e = float((f1 - g4).normest())
    record("3,1/2", FUN_count=len(g4.funs), normest=e, bound=10 * tol)
    assert len(g4.funs) == 6
    assert e < 10 * tol
    h = cj.chebfun(lambda x: jnp.sign(op(x)), domain=dom, pref=pref)
    e = float((f1 - h).normest())
    record("3,3/4", FUN_count=len(h.funs), normest=e, bound=tol)
    assert len(h.funs) == 6
    assert e < tol


def test_native_sign_complex_complicated():
    def op(x):
        return jnp.sin(1j * x) * (1j * x + jnp.exp(5j * x))

    f = cj.chebfun(op)
    g = cj.chebfun(lambda x: jnp.sign(op(x)), domain=(-1, 0, 1), extrapolate=True)
    h = f.sign()
    e = float((g - h).normest())
    bound = 1e4 * EPS * len(h)
    record("4", normest=e, bound=bound, length=len(h))
    assert e < bound


def test_native_sign_complex_phase():
    f = cj.chebfun(lambda x: jnp.exp(2 * jnp.pi * 1j * x) / (1 + (x - 0.1) ** 2), domain=(-1, 1))
    h = f.sign()
    e = float(jnp.linalg.norm(h(NATIVE_X) - jnp.exp(2 * jnp.pi * 1j * NATIVE_X), ord=jnp.inf))
    bound = 100 * float(h.vscale) * EPS
    record("5", error=e, bound=bound)
    assert e < bound


def test_native_sign_singular_real():
    f = cj.chebfun(
        lambda x: (7 - x) ** (-0.5) * jnp.sin(10 * x) ** 2,
        domain=(-2, 7),
        exps=(0, -0.5),
        splitting=True,
    )
    s = f.sign()
    e = float(jnp.linalg.norm((s - 1)(SING_REAL_X), ord=jnp.inf))
    record("6", error=e, bound=EPS)
    assert e < EPS


def test_native_sign_singular_complex():
    f = cj.chebfun(
        lambda x: (1 - x) ** (-0.5) * jnp.exp(2 * jnp.pi * 1j * x),
        domain=(-1, 1),
        exps=(0, -0.5),
        splitting=True,
    )
    s = f.sign()
    e = float(
        jnp.linalg.norm(s(SING_COMPLEX_X) - jnp.exp(2 * jnp.pi * 1j * SING_COMPLEX_X), ord=jnp.inf)
    )
    bound = 10 * EPS * float(s.get("vscale"))
    record("7", error=e, bound=bound)
    assert e < bound


def test_native_sign_unbounded():
    f = cj.chebfun(lambda x: (1 - jnp.exp(-(x**2))) / x, domain=(-np.inf, np.inf))
    s = f.sign()
    exact = 2 * (0.5 * (jnp.sign(UNBOUNDED_X) + 1)) - 1
    e = float(jnp.linalg.norm(s(UNBOUNDED_X) - exact, ord=jnp.inf))
    record("8", error=e, required_zero=True)
    assert not e


def test_sign_source_complex_zero_point_and_limits():
    """Supplemental source control: stored zero differs from piece limits."""
    f = cj.chebfun(lambda x: (1 + 1j) * x, domain=(-1, 0, 1))
    h = f.sign()
    assert h.domain.breakpoints == (-1.0, 0.0, 1.0)
    assert bool(jnp.all(h.point_values == jnp.sign(f.point_values)))
    assert complex(h(jnp.asarray(0.0))) == 0
    phase = (1 + 1j) / jnp.sqrt(2.0)
    e = max(
        float(jnp.max(jnp.abs(h.funs[0].tech(jnp.asarray([-1.0, 1.0])) + phase))),
        float(jnp.max(jnp.abs(h.funs[1].tech(jnp.asarray([-1.0, 1.0])) - phase))),
    )
    record("supplemental_complex_zero", stored_zero=True, limit_error=e, bound=100 * EPS)
    assert e < 100 * EPS
