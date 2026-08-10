"""Port of MATLAB Chebfun tests/chebfun/test_feval.m (Fable 5).

String endpoint syntaxes ('left','start','-') are skipped; numeric
evaluation incl. endpoint one-sided values ported.

Provenance
----------
MATLAB source : tests/chebfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunFeval:
    def test_empty_input_shape(self):
        f = cj.chebfun(lambda x: x)
        fx = f(jnp.zeros((0,)))
        assert np.asarray(fx).shape == (0,)

    def test_string_endpoint_syntax(self):
        # MATLAB pass(2,3): feval(f, 'left'/'start'/'-') and
        # feval(f, 'right'/'end'/'+').
        f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))))
        lvals = [float(f(s)) for s in ("left", "start", "-")]
        rvals = [float(f(s)) for s in ("right", "end", "+")]
        assert abs(lvals[0] - erf(-1.0)) < 10 * EPS * f.vscale
        assert lvals[0] == lvals[1] == lvals[2]
        assert abs(rvals[0] - erf(1.0)) < 10 * EPS * f.vscale
        assert rvals[0] == rvals[1] == rvals[2]
        with pytest.raises(ValueError):
            f("middle")

    def test_side_string_one_sided_limits(self):
        # MATLAB pass(6,7): feval(f, x, 'left'/'-') and 'right'/'+' at a
        # jump.  f = sign under splitting; x = [-0.5, 0, 0.5].
        f = cj.chebfun(jnp.sign, domain=[-1.0, 1.0], splitting=True)
        x = jnp.asarray([-0.5, 0.0, 0.5])
        for s in ("left", "-"):
            got = np.asarray(f(x, side=s))
            assert np.allclose(got, [-1.0, -1.0, 1.0])
        for s in ("right", "+"):
            got = np.asarray(f(x, side=s))
            assert np.allclose(got, [-1.0, 1.0, 1.0])

    def test_endpoint_values(self):
        f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))))
        lval = float(f(jnp.asarray(-1.0)))
        rval = float(f(jnp.asarray(1.0)))
        assert abs(lval - erf(-1.0)) < 10 * EPS * f.vscale
        assert abs(rval - erf(1.0)) < 10 * EPS * f.vscale

    def test_vectorized_evaluation(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        err = jnp.abs(f(XR) - jnp.exp(XR) * jnp.sin(3 * XR))
        assert float(jnp.max(err)) < 100 * f.vscale * EPS

    def test_matrix_shaped_input(self):
        f = cj.chebfun(lambda x: x ** 2)
        xm = jnp.asarray(np.linspace(-1, 1, 12).reshape(3, 4))
        out = f(xm)
        assert np.asarray(out).shape == (3, 4)
        assert float(jnp.max(jnp.abs(out - xm ** 2))) < 100 * EPS

    def test_complex_points_just_outside_domain(self):
        # MATLAB test_feval.m pass(20): evaluate a real chebfun at complex
        # points (and points just outside the domain), comparing to the
        # analytic function.  f_exact = cos(x - 0.2) on [-1, 1].
        def f_exact(x):
            return np.cos(x - 0.2)

        f = cj.chebfun(lambda x: jnp.cos(x - 0.2), domain=[-1, 1])
        x = np.array([-1 - 1e-6, 1e-6 + 1e-6j, 1 + 1e-6])
        err = np.asarray(f(jnp.asarray(x))) - f_exact(x)
        assert float(np.max(np.abs(err))) < 10 * EPS * f.vscale

    def test_complex_valued_chebfun_at_real_points(self):
        # MATLAB test_feval.m pass(13): a complex-VALUED chebfun (complex
        # coefficients) evaluated at real points.  f_exact = sinh(t*z),
        # z = exp(2*pi*1i/6).
        z = np.exp(2 * np.pi * 1j / 6)

        def f_exact(t):
            return np.sinh(t * z)

        f = cj.chebfun(lambda t: jnp.sinh(t * z), domain=[-1, 1])
        x = np.asarray(XR)
        err = np.asarray(f(jnp.asarray(x))) - f_exact(x)
        assert float(np.max(np.abs(err))) < 10 * EPS * f.vscale
