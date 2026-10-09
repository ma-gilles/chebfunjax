"""Summed-length quadrature controls; native Chebfun7574c77 innerProduct.m."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech, _trig_inner_product_jax

EPS = jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("phase", [1.0, 1.0j])
def test_even_nyquist_exact_integral_and_scalar_qr(phase):
    # n=2 values represent phase*cos(pi*(x+1)); integral |f|^2 is 1.
    f = Trigtech.from_values(phase * jnp.array([1.0, -1.0]))
    assert abs(f.innerProduct(f) - 1) < 10 * EPS
    q, r = f.qr()
    assert abs(r[0, 0] - 1) < 10 * EPS
    assert abs(q.innerProduct(q) - 1) < 10 * EPS
    assert not jnp.iscomplexobj(f.innerProduct(f))


def test_array_orientation_and_real_pairs():
    # Independent integrals of [1, i*cos(pi*(x+1))] against [cos, 1].
    f = Trigtech.from_values(jnp.array([[1, 1j], [1, -1j]]))
    g = Trigtech.from_values(jnp.array([[1.0, 1.0], [-1.0, 1.0]]))
    expected = jnp.array([[0, 2], [-1j, 0]])
    actual = f.innerProduct(g)
    assert actual.shape == (2, 2)
    assert jnp.max(jnp.abs(actual - expected)) < 10 * EPS
    assert jnp.all(jnp.imag(actual[0]) == 0)


def test_prolong_regenerates_stale_caches_before_equality():
    f = Trigtech(coeffs=jnp.array([1.0, 0.0]), is_real=True,
                 _values=jnp.array([99.0, 99.0]))
    g = Trigtech(coeffs=jnp.array([1.0, 0.0]), is_real=True,
                 _values=jnp.array([-99.0, -99.0]))
    out, equal = _trig_inner_product_jax(f, g)
    assert bool(equal)
    assert abs(out[0, 0] - 1) < 10 * EPS


@pytest.mark.parametrize("other", [2, [], jnp.empty((0, 3))])
def test_empty_first_and_invalid_input(other):
    assert Trigtech.empty().innerProduct(other).shape == (0, 0)
    f = Trigtech.from_values(jnp.array([1.0]))
    if isinstance(other, int):
        with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:innerProduct:input"):
            f.innerProduct(other)
    else:
        assert f.innerProduct(other).shape == (0, 0)


def test_jit_ad_nyquist_energy():
    def energy(a):
        f = Trigtech(coeffs=jnp.array([a, 0.0]), is_real=True)
        return jnp.real(f.innerProduct(f))
    assert abs(jax.jit(energy)(2.0) - 4) < 10 * EPS
    assert abs(jax.grad(energy)(2.0) - 4) < 10 * EPS


def test_nan_equality_is_false():
    f = Trigtech(coeffs=jnp.array([jnp.nan + 0j]), is_real=False)
    out, equal = _trig_inner_product_jax(f, f)
    assert not bool(equal)
    assert jnp.isnan(out[0, 0])


def test_scalar_zero_inner_and_native_qr_division():
    f = Trigtech.from_values(jnp.array([0.0]))
    assert f.innerProduct(f) == 0
    q, r = f.qr()
    assert r[0, 0] == 0
    # Native scalar QR divides by its zero norm; no replacement basis here.
    assert jnp.all(jnp.isnan(q.coeffs))
