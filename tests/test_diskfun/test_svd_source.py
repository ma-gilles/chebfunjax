"""Source disk SVD controls, independent of singular-vector signs.

Provenance
----------
MATLAB source : @diskfun/svd.m, @diskfun/norm.m
Chebfun commit: 7574c77
Analytic fixtures supplement original tests; they are not MATLAB captures.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun._svd import _angular_qr, _disk_qr
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech, _trig_coeffs2vals_impl
from chebfunjax.utils.quadrature import legpts

TOL = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps


def _field():
    # Integral_0^1 r*(r^4-2*r^2/3)dr=0, squared norm=1/135.
    cols = [Chebtech2.from_coeffs(jnp.asarray([1.])),
            Chebtech2.from_coeffs(jnp.asarray([1/24, 0., 1/6, 0., 1/8]))]
    rows = [Trigtech.from_coeffs(jnp.asarray([1.]), is_real=True),
            Trigtech.from_coeffs(jnp.asarray([.5, 0., 0., 0., .5]), is_real=True)]
    return Diskfun(cols=cols, rows=rows, pivots=jnp.ones(2),
                   idx_plus=(0, 1), idx_minus=())


@pytest.mark.parametrize('disable', [False, True])
def test_analytic_singular_functions_and_norm(disable):
    f = _field()
    with jax.disable_jit(disable):
        u, s, v = f.svd(return_uv=True)
        values = f.svd()
        assert s.shape == (2, 2)
        assert jnp.max(jnp.abs(values-jnp.sqrt(jnp.asarray([jnp.pi, jnp.pi/135])))) < TOL
        assert jnp.max(jnp.abs(jnp.diag(s)-values)) < TOL
        assert (u.domain.a, u.domain.b) == (0., 1.)
        assert (v.domain.a, v.domain.b) == (-jnp.pi, jnp.pi)
        r, w = legpts(24, interval=(0., 1.))
        theta = -jnp.pi + 2*jnp.pi*jnp.arange(32)/32
        ur, vt = u(r), v(theta)
        expected = 1 + (r[:, None]**4-2*r[:, None]**2/3)*jnp.cos(2*theta)
        assert jnp.max(jnp.abs(ur@s@vt.T-expected)) < TOL
        assert jnp.max(jnp.abs(ur.T@((w*r)[:, None]*ur)-jnp.eye(2))) < TOL
        assert jnp.max(jnp.abs(vt.T@vt*(2*jnp.pi/32)-jnp.eye(2))) < TOL
        for order in (2, 'fro'):
            assert abs(f.norm(order)-jnp.sqrt(jnp.pi+jnp.pi/135)) < TOL


@pytest.mark.parametrize('disable', [False, True])
def test_radial_qr_positive_diagonal_and_common_length(disable):
    f = _field()
    cols = [-f.cols[0], f.cols[1]]
    with jax.disable_jit(disable):
        q, r = _disk_qr(cols)
        assert bool(jnp.all(jnp.diag(r) >= 0))
        assert q.funs[0].tech.n == 6  # restricted common length5, then n+1
        x, w = legpts(20, interval=(0., 1.))
        actual = q(x)
        expected = jnp.column_stack([-jnp.ones_like(x), x**4-2*x**2/3])
        assert jnp.max(jnp.abs(actual@r-expected)) < TOL
        assert jnp.max(jnp.abs(actual.T@((w*x)[:, None]*actual)-jnp.eye(2))) < TOL


@pytest.mark.parametrize('disable', [False, True])
def test_angular_even_nyquist_continuous_qr(disable):
    # Even-N highest mode is a cosine. Coefficient QR has the wrong norm.
    row = Trigtech.from_coeffs(jnp.asarray([1., 0., 0., 0.]), is_real=True)
    with jax.disable_jit(disable):
        q, r = _angular_qr([row])
        x = jnp.asarray([-.7, .21, .88])
        assert abs(r[0, 0]-jnp.sqrt(jnp.pi)) < TOL
        assert jnp.max(jnp.abs(q(x).reshape(-1)*r[0, 0]-jnp.cos(2*x))) < TOL


@pytest.mark.parametrize('return_uv', [False, True])
def test_empty_source_single_output(return_uv):
    actual = Diskfun.empty().svd(return_uv=return_uv)
    assert actual.shape == (0,)
    assert Diskfun.empty().norm().shape == (0,)


def test_zero_pivot_source_length_and_angular_zero_branch():
    f = _field()
    zero = Diskfun(cols=f.cols, rows=f.rows, pivots=jnp.zeros(2),
                   idx_plus=(0, 1), idx_minus=())
    assert jnp.array_equal(zero.svd(), jnp.zeros(2))
    row = Trigtech.from_coeffs(jnp.zeros(1), is_real=True)
    q, r = _angular_qr([row])
    assert r[0, 0] == 0
    assert abs(q(jnp.asarray(.2)).reshape(())-1/jnp.sqrt(2*jnp.pi)) < TOL


def test_angular_multicolumn_authoritative_cache():
    # The native values cache is authoritative at an unchanged length.
    # This seam control deliberately distinguishes it from coefficient QR.
    one = jnp.asarray([0., 1., 0.], dtype=jnp.complex128)
    cosine = jnp.asarray([.5, 0., .5], dtype=jnp.complex128)
    rows = [Trigtech(coeffs=one, real_columns=(True,),
                    _values=2*jnp.real(_trig_coeffs2vals_impl(one))),
            Trigtech(coeffs=cosine, real_columns=(True,),
                     _values=jnp.real(_trig_coeffs2vals_impl(cosine)))]
    _, r = _angular_qr(rows)
    assert abs(r[0, 0]-2*jnp.sqrt(2*jnp.pi)) < TOL
    assert abs(r[1, 1]-jnp.sqrt(jnp.pi)) < TOL


@pytest.mark.parametrize('order', [None, 'fro', 4, 'unknown'])
def test_actual_empty_norm_precedes_dispatch(order):
    f = Diskfun.empty()
    result = f.norm() if order is None else f.norm(order)
    assert result.shape == (0,)
    assert result.dtype == jnp.float64
