"""Exact coefficient and exponent-boundary controls for Chebfun7574c77 isequal."""
import jax.numpy as jnp
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


def sf(coeffs, exps=(0., 0.)):
    return Singfun(Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64)), exps)


@pytest.mark.parametrize('delta,expected', [(0., True), (1.05e-11, True), (1.1e-11, False), (1.2e-11, False)])
def test_strict_source_exponent_tolerance(delta, expected):
    f, g = sf([1.]), sf([1.], (delta, 0.))
    assert f.isequal(g) is expected
    assert (f == g) is expected


@pytest.mark.parametrize('coeffs', [[1.+1e-13], [1., 0.], [jnp.nan]])
def test_coefficients_are_exact_and_same_length(coeffs):
    assert not sf([1.]).isequal(sf(coeffs))


def test_smooth_promotion():
    f = sf([1., 2.])
    assert f.isequal(f.smoothPart)
    assert f.smoothPart.isequal(f)
    assert not sf([1., 2.], (-1., 0.)).isequal(f.smoothPart)


def test_distinct_equal_bounded_singular_objects():
    dom = Domain((-1., 1.))
    f, g = Bndfun(sf([1.], (-1., 0.)), dom), Bndfun(sf([1.], (-1., 0.)), dom)
    assert f.onefun is not g.onefun
    assert f.isequal(g) and g.isequal(f)


@pytest.mark.parametrize('kind', [1, 2])
def test_source_sum_zero_rule_and_column_scales(kind):
    from chebfunjax.tech.chebtech import Chebtech1
    cls = Chebtech1 if kind == 1 else Chebtech2
    a = cls.from_coeffs(jnp.array([1., 1e-17]))
    b = cls.from_coeffs(jnp.array([-1., 0.]))
    assert bool(jnp.all((a+b).coeffs == 0)) and (a+b).n == 1
    a = cls.from_coeffs(jnp.array([[1., 1e-20], [1e-17, 1e-36]]))
    b = cls.from_coeffs(jnp.array([[-1., -1e-20], [0., 0.]]))
    assert (a+b).n == 2
    assert not bool(jnp.all((a+b).coeffs == 0))


@pytest.mark.parametrize('n', [8, 17, 33])
def test_complex_transform_preserves_conjugacy_and_jvp(n):
    import jax

    from chebfunjax.utils.transforms import vals2coeffs
    x = jnp.linspace(-1., 1., n)
    values = jnp.stack([jnp.sin(x)+1j*jnp.cos(x), x+1j*x**2], axis=-1)
    coeffs = vals2coeffs(values)
    assert bool(jnp.array_equal(jnp.conj(coeffs), vals2coeffs(jnp.conj(values))))
    tangent = values*.13
    _, action = jax.jvp(vals2coeffs, (values,), (tangent,))
    assert float(jnp.max(jnp.abs(action-vals2coeffs(tangent)))) < 2e-15
