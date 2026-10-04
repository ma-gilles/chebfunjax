"""Port of the complete MATLAB Chebfun tests/chebfun/test_inv.m battery.

Provenance
----------
MATLAB source : tests/chebfun/test_inv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import tweak_domain
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps
ALGORITHMS = ['roots', 'newton', 'bisection', 'regulafalsi', 'illinois', 'brent']


def _sausagemap(x):
    # MATLAB's degree-nine polynomial from test_inv.m, in monomial form.
    weights = jnp.asarray([1.0, 1 / 6, 3 / 40, 5 / 112, 35 / 1152])
    coeffs = jnp.zeros(10).at[jnp.asarray([8, 6, 4, 2, 0])].set(weights)
    coeffs = coeffs / jnp.sum(coeffs)
    return jnp.polyval(coeffs, x)


@pytest.mark.parametrize('algorithm', ALGORITHMS)
class TestChebfunInvAlgorithms:
    def test_sine_and_inverse_roundtrip(self, algorithm):
        # MATLAB pass(k,1:2): asin and inv(inv(sin)).
        f = cj.chebfun(jnp.sin)
        expected = cj.chebfun(jnp.arcsin, domain=(-np.sin(1.0), np.sin(1.0)))
        inverse = f.inv(algorithm=algorithm)
        tol = 100 * EPS * inverse.vscale
        assert float((expected - inverse).norm(jnp.inf)) < tol
        restored = inverse.inv(algorithm=algorithm)
        # MATLAB [f,g] = tweakDomain(f,g,2e-15) aligns near-common breaks
        # before evaluating the unchanged full-domain infinity norm.
        f_aligned, restored_aligned, _, _ = tweak_domain(
            f, restored, tol=2e-15)
        assert float((f_aligned - restored_aligned).norm(jnp.inf)) < tol

    def test_sausagemap(self, algorithm):
        # MATLAB pass(k,3): both compositions of the degree-nine map.
        x = cj.chebfun('x')
        f = cj.chebfun(_sausagemap)
        inverse = f.inv(ChebfunPref(), algorithm=algorithm)
        tol = 100 * EPS * inverse.vscale
        assert float((f(inverse) - x).norm(jnp.inf)
                     + (inverse(f) - x).norm(jnp.inf)) < tol

    def test_monocheck_and_rangecheck(self, algorithm):
        # MATLAB pass(k,4:5): exp inversion, range correction, and domain.
        f = cj.chebfun(jnp.exp)
        inverse = f.inv(algorithm=algorithm, monocheck='on', rangecheck='on')
        tol = 100 * EPS * inverse.vscale
        values = jnp.linspace(float(inverse.domain.a), float(inverse.domain.b), 20)
        assert float(jnp.max(jnp.abs(f(inverse(values)) - values))) < tol
        endpoints = np.asarray(inverse.domain.breakpoints)[[0, -1]]
        assert bool(np.all(np.abs(endpoints - np.exp([-1.0, 1.0])) < 10 * EPS))


class TestChebfunInv:
    def test_reject_nonmonotonic(self):
        # MATLAB pass(:,6).
        f = cj.chebfun(lambda x: x**2)
        with pytest.raises(ValueError) as exc_info:
            f.inv(splitting='on', monocheck='on')
        source_id = 'CHEBFUN:CHEBFUN:inv:doMonoCheck:notMonotonic'
        actual_id = str(exc_info.value).split(': ', maxsplit=1)[0]
        assert actual_id.lower() == source_id.lower()

    def test_piecewise_inverse_with_jump(self):
        # MATLAB pass(:,7): ten-point numeric roundtrip residual.
        x = cj.chebfun('x')
        f = x + 0.5 * abs(x) + 0.6 * (x - 0.5).sign()
        inverse = f.inv()
        points = jnp.linspace(-1.0, 1.0, 10)
        assert float(jnp.max(jnp.abs(inverse(f(points)) - points))) < 10 * EPS * inverse.vscale

    def test_piecewise_jump_domain_extra_source_derived(self):
        # Additional domain characterization; MATLAB pass(:,7) checks only
        # the roundtrip residual, not this five-breakpoint vector.
        x = cj.chebfun('x')
        f = x + 0.5 * abs(x) + 0.6 * (x - 0.5).sign()
        inverse = f.inv()
        np.testing.assert_allclose(np.asarray(inverse.domain.breakpoints),
                                   [-1.1, -0.6, 0.15, 1.35, 2.1],
                                   atol=1e-14, rtol=0)

    def test_decreasing_inverse(self):
        # MATLAB pass(:,8), issue #1098.
        f = cj.chebfun(lambda x: -jnp.sin(x))
        inverse = f.inv()
        points = jnp.linspace(-1.0, 1.0, 10)
        assert float(jnp.max(jnp.abs(inverse(f(points)) - points))) < 10 * EPS * inverse.vscale

    def test_reject_multiple_columns(self):
        f = cj.chebfun(lambda x: jnp.stack([x, 2 * x], axis=-1))
        with pytest.raises(ValueError, match='noquasi'):
            f.inv()

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match='badAlgo'):
            cj.chebfun('x').inv(algorithm='unknown')
