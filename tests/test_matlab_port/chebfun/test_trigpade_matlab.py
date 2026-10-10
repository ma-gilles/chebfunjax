"""Literal30 source slots of tests/chebfun/test_trigpade.m (7574c77).

Clauses5,10,15 use tt captured from the original native source order.
No seeded Python stream substitutes for those predicates.
"""
import json
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

TOL = 1e-12


@lru_cache(None)
def _case(group):
    if group == "sine":
        f = cj.chebfun(lambda t: jnp.sin(jnp.pi*t), trig=True)
        m, n = 1, 0
    elif group in ("defect", "asymmetric_defect"):
        h = 1/9
        k = 1/h if group == "defect" else 1/h+1
        coefficients = jnp.concatenate((k**jnp.arange(-16,0,dtype=jnp.float64),
            jnp.asarray([3.]), h**jnp.arange(1,17,dtype=jnp.float64)))
        f = cj.chebfun(coefficients, coeffs=True, trig=True)
        m, n = 1, 2
    elif group == "exp_sine":
        f = cj.chebfun(lambda t: jnp.exp(jnp.sin(jnp.pi*t)), trig=True)
        m, n = 8, 8
    elif group == "m_zero":
        f = cj.chebfun(lambda t: 1/(1+jnp.sin(jnp.pi*(t-.8)/2)**2), trig=True)
        m, n = 0, 1
    elif group == "exp_m_less_n":
        f = cj.chebfun(lambda t: jnp.exp(jnp.sin(3*jnp.pi*t)+jnp.cos(jnp.pi*t)), trig=True)
        m, n = 5, 15
    elif group in ("piecewise", "piecewise_m_less_n"):
        f = cj.chebfun(lambda x: (-x+.7)/(1+.7)*(x<.7)+(x-.7)/(1-.7)*(x>=.7),
                      n=1001, trig=True)
        m, n = (5,5) if group == "piecewise" else (2,7)
    elif group in ("complex", "complex_m_greater_n"):
        f = cj.chebfun(lambda t: 1j/(1+25*jnp.sin(jnp.pi*(t+.5)/2)**2)
                      + jnp.exp(jnp.sin(3*jnp.pi*t)+jnp.cos(jnp.pi*t)), trig=True)
        m, n = (1,3) if group == "complex" else (5,2)
    else:
        raise AssertionError(group)
    return f, m, n, cj.trigpade(f,m,n)


def _norm(f):
    return float(f.norm("inf"))


def _compare_coeffs(f, g, n, tolerance):
    # Source compare_coeffs: pad centered coefficient arrays to common length,
    # then compare exactly the central2n+1 entries (or the full shorter array).
    assert len(f.funs) == len(g.funs) == 1
    c1, c2 = jnp.ravel(f.funs[0].tech.coeffs), jnp.ravel(g.funs[0].tech.coeffs)
    # Source half-length indices must be integral; do not silently floor
    # even lengths into a different coefficient window.
    assert c1.size % 2 == c2.size % 2 == 1
    l1, l2 = (c1.size-1)//2, (c2.size-1)//2
    if l1 > l2:
        c2 = jnp.pad(c2,(l1-l2,l1-l2))
    else:
        c1 = jnp.pad(c1,(l2-l1,l2-l1))
    assert c1.size == c2.size, "Source compare_coeffs: must have same length"
    midpoint = (c1.size-1)//2
    delta = c1-c2
    if n <= midpoint:
        delta = delta[midpoint-n:midpoint+n+1]
    return float(jnp.max(jnp.abs(delta))) < tolerance


@pytest.mark.parametrize("clause", range(1,31))
def test_original_predicate(clause):
    if clause in (5,10,15):
        fixture = Path(__file__).parents[2]/"fixtures/trigpade_source_tt.json"
        captured = json.loads(fixture.read_text())
        assert captured["source_commit"] == "7574c77680d7e82b79626300bf255498271a72df"
        assert captured["source_sha256"] == "3f6ae01eb993a95b46ee04cbfc61558c2d5d539083d9a1ceafcfaf0372fc1c76"
        tt = jnp.asarray(captured["tt"], dtype=jnp.float64)
        assert tt.shape == (100,)

    if clause == 1:
        assert cj.trigpade(cj.chebfun()).isempty()
        return
    group = ("sine" if clause <= 10 else "defect" if clause <= 13 else
             "exp_sine" if clause <= 15 else "m_zero" if clause <= 18 else
             "exp_m_less_n" if clause == 19 else "piecewise" if clause <= 22 else
             "piecewise_m_less_n" if clause <= 25 else "asymmetric_defect" if clause <= 28 else
             "complex" if clause == 29 else "complex_m_greater_n")
    f,m,n,(p,q,r,s,t,u,v) = _case(group)
    if clause in (5,10,15):
        assert float(jnp.max(jnp.abs(f(tt)-r(tt)))) < TOL
    elif clause in (2,21,24):
        assert len(p) <= 2*m+1
    elif clause in (3,22,25):
        assert len(q) <= 2*n+1
    elif clause == 6:
        assert len(p) <= 2*m+1 and len(q) <= 2*n+1
    elif clause == 7:
        assert len(s) <= 2*m+1 and len(t) <= 2*n+1
    elif clause == 8:
        assert len(u) <= 2*m+1 and len(v) <= 2*n+1
    elif clause == 4:
        assert _norm(p/q-f) < TOL
    elif clause in (13,14,18,26):
        assert _norm(f-p/q) < TOL
    elif clause == 9:
        assert _norm(p/q-(s/t+u/v)) < TOL
    elif clause in (11,12):
        assert len(p if clause == 11 else q) == 3
    elif clause == 16:
        assert len(p) == 1
    elif clause == 17:
        assert len(q) == 3
    elif clause in (19,20,23,29,30):
        assert _compare_coeffs(f,p/q,m+n,100*TOL if clause in (20,23) else TOL)
    elif clause == 27:
        assert len(q) == 2*(n-1)+1
    elif clause == 28:
        assert len(p) == 2*m+1
    else:
        raise AssertionError(clause)
