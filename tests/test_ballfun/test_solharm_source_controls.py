"""Independent short-form controls; do not replace14 source predicates."""
import jax.numpy as jnp
import numpy as np  # uses-numpy: independent polynomial and coefficient reference controls.
import pytest

from chebfunjax.ballfun._solharm import radial_clenshaw
from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun.spherefun import Spherefun

EPS = np.finfo(float).eps


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_complex_degree_one_analytic_value_and_mode(m):
    f = Ballfun.solharm(1, m, "complex")
    r, lam, th = .7, .3, .4
    if m == 0:
        expected = np.sqrt(15/(4*np.pi))*r*np.cos(th)
    else:
        expected = (-1 if m > 0 else 1)*np.sqrt(15/(8*np.pi))*r*np.exp(1j*m*lam)*np.sin(th)
    assert abs(complex(f(r, lam, th))-expected) < 64*EPS
    coeffs = np.asarray(f.coeffs)
    occupied = abs(m)+m
    assert np.count_nonzero(np.delete(coeffs, occupied, axis=1)) == 0
    assert f.is_real == (m == 0)


@pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
def test_real_complex_source_combination(m):
    real = Ballfun.solharm(2, m)
    complex_value = Ballfun.solharm(2, abs(m), "complex")(.6, .7, .8)
    expected = (jnp.real(complex_value) if m >= 0 else jnp.imag(complex_value))
    if m:
        expected *= jnp.sqrt(2.)
    assert abs(float(real(.6, .7, .8))-float(expected)) < 128*EPS
    assert real.is_real


@pytest.mark.parametrize("radius", [-.3, 1., 1.2])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_paired_clenshaw_independent_polynomial(degree, radius):
    coefficients = (np.arange((degree+1)*6).reshape(degree+1, 2, 3)-4)/32
    coefficients = coefficients + 1j*coefficients[::-1]/8
    # Independent closed-form T0..T3, no recurrence shared with implementation.
    basis = np.array([1, radius, 2*radius**2-1, 4*radius**3-3*radius])[:degree+1]
    expected = np.sum(coefficients*basis[:, None, None], axis=0)
    actual = np.asarray(radial_clenshaw(jnp.asarray(coefficients), radius))
    assert np.max(np.abs(actual-expected)) < 32*EPS*max(1, np.max(np.abs(expected)))


def test_shell_uses_nonconjugate_coefficients_constructor(monkeypatch):
    coefficients = jnp.asarray(np.arange(18).reshape(3, 2, 3)/32*(1+.25j))
    f = Ballfun.from_coeffs(coefficients, is_real=False)
    captured = []
    sentinel = object()

    def constructor(value):
        captured.append(np.asarray(value))
        return sentinel

    monkeypatch.setattr(Spherefun, "coeffs2spherefun", staticmethod(constructor))
    assert f.to_spherefun(1) is sentinel
    np.testing.assert_array_equal(captured[0], np.sum(np.asarray(coefficients), axis=0).T)
    assert Ballfun.empty().to_spherefun().isempty()


@pytest.mark.parametrize("option", ["COMPLEX", "real", None])
def test_native_unrecognized_option(option):
    with pytest.raises(ValueError, match="CHEBFUN:BALLFUN:solharm:input"):
        Ballfun.solharm(1, 0, option)


def test_native_degree_error_and_ignored_extra_option():
    with pytest.raises(ValueError, match="CHEBFUN:BALLFUN:solHarm"):
        Ballfun.solharm(0, 1)
    a = Ballfun.solharm(1, 1, "complex")
    b = Ballfun.solharm(1, 1, "complex", "ignored")
    np.testing.assert_array_equal(a.coeffs, b.coeffs)
