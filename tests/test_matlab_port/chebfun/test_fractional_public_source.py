"""Public dispatch contracts from @chebfun/fracInt.m and fracDiff.m7574c77."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma

from chebfunjax import chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix


@pytest.mark.parametrize("mu", [0., 1., .5])
def test_integral_rejects_breakpoints_before_integer_dispatch(mu):
    f = chebfun(lambda x: x, domain=[0., .5, 1.])
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:fracInt:breakpoints"):
        f.fracInt(mu)


def test_derivative_native_breakpoint_error():
    f = chebfun(lambda x: x, domain=[0., .5, 1.])
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:fracDiff:breakpoints"):
        f.fracDiff(.5)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("storage", ["array", "quasimatrix"])
def test_fractional_columns_and_orientation(storage, transpose):
    f = chebfun(lambda x: jnp.stack([jnp.ones_like(x), x], axis=-1), domain=[0., 1.])
    if transpose:
        f = f.T
    if storage == "quasimatrix":
        f = Quasimatrix([f[0], f[1]], f.domain)
    x = jnp.array([.1, .4, .9])
    for name, expected in [
        ("fracInt", jnp.stack([x**.5/gamma(1.5), x**1.5/gamma(2.5)], axis=-1)),
        # Native row RL dispatch takes bndfun.diff(n, 2): it differentiates
        # the fractional-integral canonical factor without physical rescaling.
        ("fracDiff", .5*jnp.stack([x**(-.5)/gamma(.5), x**.5/gamma(1.5)], axis=-1)
         if transpose else jnp.stack([x**(-.5)/gamma(.5), x**.5/gamma(1.5)], axis=-1)),
    ]:
        g = getattr(f, name)(.5)
        assert isinstance(g, Quasimatrix)
        assert g.is_transposed == transpose
        np.testing.assert_allclose(g(x), expected.T if transpose else expected,
                                   rtol=0., atol=100*np.finfo(float).eps)


def test_non_caputo_kind_uses_native_rl_fallback():
    f = chebfun(lambda x: jnp.ones_like(x), domain=[0., 1.])
    x = jnp.array([.1, .4, .9])
    np.testing.assert_allclose(f.fracDiff(.5, "anything")(x), x**(-.5)/gamma(.5),
                               rtol=0., atol=100*np.finfo(float).eps)


@pytest.mark.parametrize("storage", ["array", "quasimatrix"])
def test_caputo_columns_use_derivative_then_integral(storage):
    f = chebfun(lambda x: jnp.stack([x, x*x], axis=-1), domain=[0., 1.])
    if storage == "quasimatrix":
        f = Quasimatrix([f[0], f[1]], f.domain)
    x = jnp.array([.1, .4, .9])
    expected = jnp.stack([x**.5/gamma(1.5), 2*x**1.5/gamma(2.5)], axis=-1)
    np.testing.assert_allclose(f.fracDiff(.5, "cApUtO")(x), expected,
                               rtol=0., atol=100*np.finfo(float).eps)


def test_diff_fractional_kind_and_numeric_dim_dispatch():
    f = chebfun(lambda x: x*x, domain=(0., 1.))
    x = jnp.array([.1, .4, .9])
    np.testing.assert_allclose(f.diff(.5, "cApUtO")(x),
                               f.fracDiff(.5, "cApUtO")(x), rtol=0., atol=0.)
    np.testing.assert_allclose(f.diff(.5, 2)(x),
                               f.fracDiff(.5, "RL")(x), rtol=0., atol=0.)


def test_row_rl_fractional_source_scaling_warning_and_endpoint_metadata():
    row = chebfun(lambda x: jnp.ones_like(x), domain=(0., 4.)).T
    with pytest.warns(RuntimeWarning, match="CHEBFUN:SINGFUN:diff:noSupport"):
        got = row.diff(.5)
    x = jnp.array([.4, 1., 2., 3.6])
    expected = 2.0 / (jnp.sqrt(jnp.pi) * jnp.sqrt(x))
    np.testing.assert_allclose(got(x), expected, rtol=0., atol=2e-12)
    # diffFiniteDim leaves the fractional-integral endpoint overrides intact.
    np.testing.assert_allclose(got.point_values,
                               jnp.array([0., 4. / jnp.sqrt(jnp.pi)]),
                               rtol=0., atol=2e-12)
