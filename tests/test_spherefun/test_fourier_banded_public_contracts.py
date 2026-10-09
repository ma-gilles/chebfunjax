"""Independent small caller controls; these are not original MATLAB test slots.

Pinned Chebfun 7574c77680d7e82b79626300bf255498271a72df:
@spherefun/poisson.m (explicit sizes, trigpts, constructor then constant),
@spherefun/helmholtz.m (K=0, eigenvalue predicate, scalar mean, even m,
linspace). MATLAB gt compares real parts and round rounds complex components
independently, with ties away from zero:
https://www.mathworks.com/help/matlab/ref/double.gt.html
https://www.mathworks.com/help/matlab/ref/double.round.html

The linspace check binds the Python runtime API and endpoint convention; it
DOES NOT establish bitwise equality with a particular MATLAB release's
linspace implementation. No native numerical capture is available here.
Coefficient capture leaves the actual band/QR solver active, and avoids the
native real projection (@spherefun/coeffs2spherefun.m32) and odd-size
padding in coeffs2spherefun; neither is changed by these controls. Scalar-mean
controls isolate final constant arithmetic; genuine output controls below
exercise the unmodified public constructor and evaluator.
"""
import cmath

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun, _sphere_fourier_operators
from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs
from chebfunjax.utils.quadrature import trigpts

EPS = np.finfo(float).eps


def _capture_coefficients(monkeypatch):
    seen = []

    def capture(coefficients):
        seen.append(np.asarray(coefficients))
        return coefficients

    monkeypatch.setattr(Spherefun, 'coeffs2spherefun', staticmethod(capture))
    return seen


def _constant(value):
    # Exact rank-one CDR; complex value is in the pivot, not a real-only
    # constructor. mean2 still executes its genuine surface contraction.
    one = Trigtech.from_coeffs(jnp.asarray([1.0]))
    return Spherefun(cols=[one], rows=[one],
                     pivots=jnp.asarray([1 / value]),
                     idx_plus=(0,), idx_minus=())


def test_poisson_preserves_explicit_odd_size_and_global_trigpts(monkeypatch):
    seen = _capture_coefficients(monkeypatch)
    grids = []

    def rhs(lam, theta):
        grids.append((np.asarray(lam), np.asarray(theta)))
        return -2 * jnp.cos(theta)

    result = Spherefun.poisson(rhs, 0, 5, 7)
    assert result.shape == (5, 7)
    assert len(seen) == len(grids) == 1
    lam, _ = trigpts(7, (-jnp.pi, jnp.pi))
    theta, _ = trigpts(5, (-jnp.pi, jnp.pi))
    np.testing.assert_array_equal(grids[0][0], np.broadcast_to(lam, (5, 7)))
    np.testing.assert_array_equal(grids[0][1], np.broadcast_to(theta[:, None], (5, 7)))
    expected = np.zeros((5, 7), dtype=complex)
    expected[[1, 3], 3] = .5
    assert np.max(np.abs(seen[0] - expected)) <= 64 * 5 * EPS


@pytest.mark.parametrize('k', [2 + 4e-13j, .7 + 4e-13j])
def test_helmholtz_rounds_only_latitude_and_preserves_tiny_complex_k(monkeypatch, k):
    seen = _capture_coefficients(monkeypatch)
    grids = []

    def rhs(lam, theta):
        grids.append((np.asarray(lam), np.asarray(theta)))
        return jnp.cos(theta)

    result = Spherefun.helmholtz(rhs, k, 5, 7)
    assert result.shape == (6, 7)
    assert len(seen) == len(grids) == 1
    lam = jnp.linspace(-jnp.pi, jnp.pi, 8)[:-1]
    theta = jnp.linspace(-jnp.pi, jnp.pi, 7)[:-1]
    np.testing.assert_array_equal(grids[0][0], np.broadcast_to(lam, (6, 7)))
    np.testing.assert_array_equal(grids[0][1], np.broadcast_to(theta[:, None], (6, 7)))
    # Independent retained dense source assembly, not band reconstruction.
    # helmholtz.m99-103,109-116,123-133; retain tiny Trigtech bands/weights.
    d1, d2, dn, cs, sin2, weights, zero = _sphere_fourier_operators(6, 7)
    values = jnp.asarray(jnp.cos(jnp.asarray(grids[0][1])), dtype=jnp.complex128)
    forcing = np.asarray(trig_vals2coeffs(trig_vals2coeffs(values).T).T)
    matrix = (sin2 @ d2 + cs @ d1) / (k * k) + sin2
    weighted = sin2 @ forcing / (k * k)
    integral = weights @ forcing[:, 3] / (k * k)
    indices = np.delete(np.arange(6), zero)
    expected = np.empty_like(forcing)
    for j in range(7):
        if j == 3:
            operator = np.vstack((weights, matrix[indices]))
            rhs = np.concatenate(([integral], weighted[indices, j]))
        else:
            operator = matrix + dn[j, j] / (k * k) * np.eye(6)
            rhs = weighted[:, j]
        expected[:, j] = np.linalg.solve(operator, rhs)
    assert np.max(np.abs(seen[0] - expected)) <= 64 * 6 * EPS
    assert np.max(np.abs(seen[0].imag - expected.imag)) <= 8 * EPS
    # The original K near2 has border cond2=6.95e13: both independent dense
    # and compact source-rounded solves differ from ideal cos(theta)/(K^2-2)
    # by2.003e-5, while agreeing to1.11e-16. condition_v1 preserves that
    # diagnostic and callers_v2 the original failed analytic oracle.
    # The alternative cond2=125 additionally supports the analytic check.
    if k.real == .7:
        analytic = np.zeros((6, 7), dtype=complex)
        analytic[[2, 4], 3] = .5 / (k * k - 2)
        assert np.max(np.abs(seen[0] - analytic)) <= 64 * 6 * EPS


@pytest.mark.parametrize('k', [2 + .5j, cmath.sqrt(2) + 4e-13j])
def test_scalar_complex_mean_and_allowed_near_eigenvalue(monkeypatch, k):
    f = _constant(2 + 3j)
    assert abs(complex(f.mean2()) - (2 + 3j)) <= 8 * EPS
    calls = []

    def zero_product(self, factor):
        assert self is f and factor == 0
        calls.append(factor)
        return 0.0

    # Capture only final zero-function arithmetic. The genuine mean2 and
    # public K predicate/scalar branch remain active; no complex output
    # construction parity is claimed by this isolated caller check.
    monkeypatch.setattr(Spherefun, '__mul__', zero_product)
    result = Spherefun.helmholtz(f, k, 1, 1)
    expected = (2 + 3j) / (k * k)
    assert calls == [0.0]
    assert abs(complex(result) - expected) <= 16 * EPS * max(1, abs(expected))


@pytest.mark.parametrize('sizes', [(5, 7), (-2, 0)])
def test_k_zero_delegates_before_size_validation_or_rounding(monkeypatch, sizes):
    marker, answer = object(), object()
    calls = []

    def poisson(f, const, m, n):
        calls.append((f, const, m, n))
        return answer

    monkeypatch.setattr(Spherefun, 'poisson', staticmethod(poisson))
    assert Spherefun.helmholtz(marker, 0j, *sizes) is answer
    assert calls == [(marker, 0, *sizes)]


@pytest.mark.parametrize('k', [cmath.sqrt(2), cmath.sqrt(2) + 1e-14j,
                               cmath.sqrt(1 + 3j), cmath.sqrt(1 - 3j)])
def test_literal_native_complex_eigenvalue_rejection(k):
    # For K^2=1+/-3j the selected eigenvalue is exactly1+/-1j in
    # mathematics: real(e)>0, round(real(e))+i*round(imag(e)) == e.
    # Rejection precedes RHS evaluation and dimension validation.
    def forbidden_rhs(lam, theta):
        raise AssertionError('eigenvalue rejection must precede RHS evaluation')

    with pytest.raises(ValueError, match='SPHEREFUN:HELMHOLTZ:EIGENVALUE'):
        Spherefun.helmholtz(forbidden_rhs, k, -2, 0)


@pytest.mark.parametrize('solver', ['poisson', 'helmholtz'])
def test_genuine_small_public_degree_one_harmonic(solver):
    # Independent analytic degree-one harmonic, with actual reconstruction.
    # Fixed off-grid probes qualify this caller control, not a continuous
    # norm or the original larger MATLAB harmonic test suite.
    if solver == 'poisson':
        result = Spherefun.poisson(lambda lam, theta: -2 * jnp.cos(theta), 0, 8, 8)
    else:
        result = Spherefun.helmholtz(lambda lam, theta: 2 * jnp.cos(theta), 2, 8, 8)
    assert isinstance(result, Spherefun)
    lam = jnp.asarray([-2.7, -1.1, .2, 1.4, 2.9])
    theta = jnp.asarray([.13, .71, 1.37, 2.21, 3.01])
    error = np.max(np.abs(np.asarray(result(lam, theta) - jnp.cos(theta))))
    assert error <= 128 * 8 * EPS


def test_gaussfilt_anisotropic_dimensions_parameter_and_rhs(monkeypatch):
    # @spherefun/gaussfilt.m21,24,36,39: dt=.5*sig^2; [n,m]=length(f);
    # K=sqrt(1/dt)*1i; helmholtz(-1/dt*f,K,m,n).
    # @separableApprox/length.m25-26: out1=rows, out2=cols.
    col = Trigtech.from_coeffs(jnp.asarray([0., .5, 0., .5, 0.]))
    row = Trigtech.from_coeffs(jnp.asarray([0., 0., 0., 1., 0., 0., 0.]))
    f = Spherefun(cols=[col], rows=[row], pivots=jnp.asarray([1.]),
                  idx_plus=(0,), idx_minus=())
    assert f.length() == (7, 5)
    captured = []
    answer = object()

    def helmholtz(rhs, k, m, n):
        captured.append((rhs, k, m, n))
        return answer

    monkeypatch.setattr(Spherefun, 'helmholtz', staticmethod(helmholtz))
    sig = .25
    assert f.gaussfilt(sig) is answer
    assert len(captured) == 1
    rhs, k, m, n = captured[0]
    assert (m, n) == (5, 7)
    dt = .5 * sig**2
    assert abs(complex(k) - 1j * np.sqrt(1 / dt)) <= 8 * EPS * abs(k)
    lam = jnp.asarray([-.7, .2, 1.3])
    theta = jnp.asarray([.1, .8, 2.2])
    expected = -np.asarray(f(lam, theta)) / dt
    assert np.max(np.abs(np.asarray(rhs(lam, theta)) - expected)) <= 16 * EPS * max(1, np.max(np.abs(expected)))
