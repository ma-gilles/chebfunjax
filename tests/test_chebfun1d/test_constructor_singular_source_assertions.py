"""Source-derived tests for MATLAB constructor_singfun passes 13,15–17.

NumPy
RandomState is a reproducible Python MT19937 sampling adapter, not MATLAB
seedRNG/random-stream parity.

Provenance
----------
MATLAB source: tests/chebfun/test_constructor_singfun.m, pass(13),(15)-(17)
Chebfun commit: 7574c77
APIs: @chebfun/constructor.m, @singfun/singfun.m, @fun/detectEdge.m
Original copyright: 2017 The University of Oxford and Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


def _python_mt_samples(domain: tuple[float, float], n: int = 100) -> np.ndarray:
    """Literal affine source sampling formula with explicitly adapted stream."""
    rng = np.random.RandomState(6178)
    a, b = domain
    return (b - a) * rng.rand(n) + a


def _source_inf_error(actual, exact) -> tuple[float, float]:
    """Vector norm(..., inf) and source exact-value scale, without rescaling."""
    actual = np.asarray(actual)
    exact = np.asarray(exact)
    return (float(np.linalg.norm(actual - exact, ord=np.inf)),
            float(np.linalg.norm(exact, ord=np.inf)))


def test_source_pass13_explicit_exps_split_double_pole():
    # MATLAB: dom=[-2,3], op=sin(300*x)/((x-dom(1))*(x-dom(2))),
    # exps=[-1,-1], splitting='on', norm(error,inf)<1e4*eps*norm(exact,inf).
    dom = (-2.0, 3.0)

    def op(x):
        return jnp.sin(300.0 * x) / ((x - dom[0]) * (x - dom[1]))

    x = _python_mt_samples(dom)
    f = cj.chebfun(op, domain=dom, exps=(-1.0, -1.0), splitting=True)
    exact = np.sin(300.0 * x) / ((x - dom[0]) * (x - dom[1]))
    error, scale = _source_inf_error(f(jnp.asarray(x)), exact)
    assert error < 1e4 * EPS * scale


def test_source_pass15_nan_exponent_autodetection_original_bound():
    # MATLAB: default domain [-1,1], exps=[.5,NaN], x=2*rand(100,1)-1,
    # op=exp(x)*sqrt(1+x)/(1-x)^2, original bound 1e1*eps*norm(exact,inf).

    def op(x):
        return jnp.exp(x) * jnp.sqrt(1.0 + x) / (1.0 - x) ** 2

    x = 2.0 * _python_mt_samples((0.0, 1.0)) - 1.0
    f = cj.chebfun(op, exps=(0.5, float("nan")))
    exact = np.exp(x) * np.sqrt(1.0 + x) / (1.0 - x) ** 2
    error, scale = _source_inf_error(f(jnp.asarray(x)), exact)
    assert error < 1e1 * EPS * scale


def test_source_pass16_tan_blowup_splitting_nonzero_norm():
    # Keep separate: construction can be expensive and exercises pole splitting.
    # MATLAB assertion is norm(f,inf)>0, not just a nonzero point sample.
    f = cj.chebfun(jnp.tan, domain=(0.0, np.pi), splitting=True, blowup=True)
    assert float(f.norm(np.inf)) > 0.0


def test_source_pass17_tan_has_three_domain_points():
    # MATLAB assertion: numel(f.domain)==3 for [5*pi,6*pi], splitting on,
    # blowup=1. Domain.breakpoints is the Python representation of that list.
    a, b = 5.0 * np.pi, 6.0 * np.pi
    f = cj.chebfun(jnp.tan, domain=(a, b), splitting=True, blowup=1)
    assert len(f.domain.breakpoints) == 3
