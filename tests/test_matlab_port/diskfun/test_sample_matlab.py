"""Port of MATLAB Chebfun tests/diskfun/test_sample.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_sample.m
Chebfun commit: 7574c77

The MATLAB three-output low-rank form ``[U, D, V] = sample(f, ...)``
(pass 5/7/9/11) is not exposed by chebfunjax's value-only ``sample`` and is
omitted; the value-matrix assertions (pass 1-4, 6, 8, 10, 12-14) are ported.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 100 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


def _ref_grid(m, n):
    """MATLAB (trigpts(m,[-pi,pi]), upper-half chebpts(2n-1)) tensor grid."""
    theta = -np.pi + 2.0 * np.pi * np.arange(m) / m
    if n == 1:
        r = np.zeros(1)
    else:
        r = np.cos(np.pi * (n - 1 - np.arange(n)) / (2 * n - 2))
    return theta, r


class TestDiskfunSample:
    def test_default_size(self):
        # pass(1): [m,n]=length(f); size(sample(f)) == [n, m]
        f = _df(lambda t, r: jnp.sin(jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t))))
        m, n = f.length()
        S = np.asarray(f.sample())
        assert S.shape == (n, m)

    def test_fixed_sizes(self):
        # pass(2)/(3): fixed m, n give the right-sized output.
        f = _df(lambda t, r: jnp.sin(jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t))))
        assert np.asarray(f.sample(120, 121)).shape == (121, 120)
        assert np.asarray(f.sample(121, 120)).shape == (120, 121)

    @pytest.mark.parametrize("m, ncols", [(30, 21), (31, 21), (31, 20), (30, 20)])
    def test_values_match_evaluation(self, m, ncols):
        # pass(4)/(6)/(8)/(10): sampled values equal direct evaluation on the
        # trig x upper-half-chebyshev tensor grid.
        f = _df(lambda t, r: jnp.sin(jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t))))
        theta, r = _ref_grid(m, ncols)
        tt, rr = np.meshgrid(theta, r)
        F = np.asarray(f(jnp.asarray(tt), jnp.asarray(rr)))
        G = np.asarray(f.sample(m, ncols))
        assert np.linalg.norm(F.ravel() - G.ravel(), np.inf) < _TOL

    def test_constant_all_ones(self):
        # pass(12): sample of the constant 1 is all ones.
        f = _df(lambda t, r: 1.0 + 0.0 * r)
        F = np.asarray(f.sample(128, 128))
        assert np.linalg.norm(F.ravel() - 1.0, np.inf) < _TOL

    def test_bad_inputs_raise(self):
        # pass(13)/(14): non-positive sample counts raise.
        f = _df(lambda t, r: 1.0 + 0.0 * r)
        with pytest.raises(ValueError):
            f.sample(0, 20)
        with pytest.raises(ValueError):
            f.sample(20, 0)
