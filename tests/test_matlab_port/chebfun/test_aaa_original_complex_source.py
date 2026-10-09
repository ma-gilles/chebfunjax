"""Focused literal predicates from tests/chebfun/test_aaa.m.

MATLAB Chebfun7574c77680d7e82b79626300bf255498271a72df.
Original slots1–13,19,20,24,41,42; exact source formulas/grid sizes/options and
bounds. Random15/16 and derivative33–40 are outside this focused package.
"""

from functools import lru_cache

import jax.numpy as jnp
import pytest
from jax.scipy.special import gamma

from chebfunjax.utils.aaa import aaa

TOL = 1e4 * jnp.finfo(jnp.float64).eps


@lru_cache(None)
def exp_case():
    z = jnp.linspace(-1, 1, 1000)
    f = jnp.exp(z)
    return z, f, aaa(f, z)


@lru_cache(None)
def tan_case():
    z = jnp.linspace(-1, 1, 1000)

    def callback(x):
        return jnp.tan(jnp.pi * x)

    return z, callback(z), aaa(callback, z)


@lru_cache(None)
def scale_case():
    z = jnp.linspace(0.3, 1.5, 100)
    f = jnp.exp(z) / (1 + 1j)
    return aaa(f, z)[0], aaa(2.0**311 * f, z)[0], aaa(2.0**-311 * f, z)[0]


@pytest.mark.parametrize("slot", list(range(1, 14)) + [19, 20, 24, 41, 42])
def test_original(slot):
    if slot <= 5:
        z, f, result = exp_case()
        r = result[0]
        m1 = len(result[4])
        if slot == 1:
            assert jnp.max(jnp.abs(f - r(z))) < TOL
        elif slot == 2:
            assert jnp.isnan(r(jnp.asarray(jnp.nan)))
        elif slot == 3:
            assert not jnp.isinf(r(jnp.asarray(jnp.inf)))
        elif slot == 4:
            assert len(aaa(f, z, mmax=m1 - 1)[4]) == m1 - 1
        else:
            assert len(aaa(f, z, tol=1e-3)[4]) < m1
    elif slot <= 9:
        z, f, result = tan_case()
        r, p, res, zeros = result[:4]
        if slot == 6:
            assert jnp.max(jnp.abs(f - r(z))) < 10 * TOL
        elif slot == 7:
            assert jnp.min(jnp.abs(zeros)) < TOL
        elif slot == 8:
            assert jnp.min(jnp.abs(p - 0.5)) < TOL
        else:
            assert jnp.min(jnp.abs(res)) > 1e-13
    elif slot in (10, 11):
        z = jnp.array([0.0, 1.0]) if slot == 10 else jnp.array([0.0, 1.0, 2.0])
        f = jnp.array([1.0, 2.0]) if slot == 10 else jnp.array([1.0, 0.0, 0.0])
        r = aaa(f, z)[0]
        assert jnp.max(jnp.abs(f - r(z))) < TOL
    elif slot in (12, 13):
        r1, r2, r3 = scale_case()
        if slot == 12:
            assert r1(jnp.asarray(0.2j)) == 2.0**-311 * r2(jnp.asarray(0.2j))
        else:
            assert r1(jnp.asarray(1.4)) == 2.0**311 * r3(jnp.asarray(1.4))
    elif slot in (19, 20):
        x = jnp.linspace(-1.337, 2, 537)
        f = jnp.exp(x) / x if slot == 19 else (1 + 1j) * gamma(x)
        _, p, res, *_ = aaa(f, x)
        target = 0 if slot == 19 else -1
        selected = res[jnp.abs(p - target) < 1e-8]
        assert len(selected) > 0
        assert jnp.all(jnp.abs(selected - (1 if slot == 19 else -(1 + 1j))) < 1e-10)
    elif slot == 24:
        z = jnp.exp(2j * jnp.pi * jnp.arange(1, 501) / 500)
        f = jnp.log(2 - z**4)
        r1 = aaa(f, z, mmax=16, lawson=0)[0]
        r2 = aaa(f, z, mmax=16)[0]
        err1 = jnp.max(jnp.abs(f - r1(z)))
        err2 = jnp.max(jnp.abs(f - r2(z)))
        assert jnp.abs(err2 / err1 - 1) < 1.01
    elif slot == 41:
        z = jnp.linspace(-1, 1, 100)
        f = z + 1 / (z - 1.5)
        _, p, res, *_ = aaa(f, z)
        k = jnp.argmin(jnp.abs(p - 1.5))
        assert jnp.abs(res[k] - 1) < 1e-8
    else:
        z = jnp.logspace(-15, 0, 300)
        r = aaa(jnp.sqrt(z), z)[0]
        zz = jnp.logspace(-15, 0, 500)
        assert jnp.max(jnp.abs(r(zz) - jnp.sqrt(zz))) < 1e-8
