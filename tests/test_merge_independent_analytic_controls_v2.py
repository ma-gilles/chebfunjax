"""Independent analytic merge controls; pinned MATLAB Chebfun 7574c77.

These supplement the unchanged original source assertions12/13.


Provenance
----------
MATLAB source : tests/chebfun/test_merge.m, @fun/merge.m,
    @singfun/singfun.m, @unbndfun/unbndfun.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np  # uses-numpy: independent test assertions only

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _merge_fun_source, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


def test_analytic_singular_union_actually_removes_break():
    """Accuracy alone permits a no-op merge; require actual break removal."""
    f = cj.chebfun(lambda x: 1 / (x + 2), domain=(-2., 1.), exps=[-1., 0.])
    split = f.addBreaks([0.])
    assert len(split.funs) == 2  # The one deliberately inserted breakpoint.
    merged = split.merge(max_length=129)
    assert len(merged.funs) < len(split.funs)
    assert tuple(merged.domain.breakpoints) == (-2., 1.)
    x = jnp.array([-1.75, -1.25, -.5, 0., .25, .75])
    # Values are at most4; absolute allowance64 binary64 ulps at that scale.
    np.testing.assert_allclose(merged(x), 1 / (x + 2), rtol=0,
                               atol=256 * np.finfo(float).eps)
    assert jnp.isposinf(merged(jnp.asarray(-2.)))


def test_unbounded_exponent_transition_compensates_physical_values(monkeypatch):
    """Inspect the fit input against hand-derived physical functions."""
    left = _Piece(Chebtech2.from_coeffs(jnp.array([5., 5.])), (0., 10.))
    right = Unbndfun.from_chebtech(
        Singfun(Chebtech2.from_coeffs(jnp.array([3.])), (0., -2.)),
        Domain((10., float('inf'))))
    observed = []

    def fit(cls, op, **kwargs):
        # Union [0,Inf] maps these t to physical x=5,15,45. On the left,
        # f=x. On the right, f=(x+5)^2/300, independently of the union map.
        # Source supplied-exponent inversion yields eb=+2, hence dividing
        # full values by (1-t)^2 gives the three rational numbers below.
        t = jnp.array([-.5, 0., .5])
        observed.append(op(t))
        np.testing.assert_allclose(observed[-1], [20/9, 4/3, 100/3],
                                   rtol=0, atol=256 * np.finfo(float).eps)
        return cls.from_coeffs(jnp.array([1.]))

    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(fit))
    result = _merge_fun_source(
        left, right, maxpow2=6, max_length=65, tol=jnp.finfo(jnp.float64).eps,
        splitting=False, vscale=8., hscale=1., sample_test=False,
        min_samples=17, refinement_function='resampling', turbo=False,
        check='classic')
    assert len(observed) == 1
    assert result.tech.exponents == (0., 2.)


def test_actual_inf_endpoint_discovers_pole_and_constructs_function():
    """Real detector/fit dependency gate: x+15 maps exactly to30/(1-t)."""
    # Fixed33 samples bound fit cost; no supplied exponents and no mocks.
    f = Unbndfun.from_function(lambda x: x + 15., Domain((0., float('inf'))), n=33)
    assert isinstance(f.tech, Singfun)
    np.testing.assert_array_equal(f.tech.exponents, [0., -1.])
    x = jnp.array([0., 3., 15., 45.])
    np.testing.assert_allclose(f(x), x + 15., rtol=0,
                               atol=128 * 60 * np.finfo(float).eps)
    assert jnp.isposinf(f(jnp.asarray(float('inf'))))


def test_actual_nan_endpoint_retains_smooth_finite_limit():
    """NaN at infinity alone must not invent a singular representation."""
    # x/x+2 is identically3 at finite x>=1 and NaN at infinity. It tests
    # endpoint extrapolation without replacing the raw callback by a limit.
    f = Unbndfun.from_function(lambda x: x / x + 2.,
                              Domain((1., float('inf'))), n=33)
    assert isinstance(f.tech, Chebtech2)
    x = jnp.array([1., 3., 15., 45., float('inf')])
    np.testing.assert_allclose(f(x), 3., rtol=0,
                               atol=96 * np.finfo(float).eps)
