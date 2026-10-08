"""Independent polynomial, continuity and migration controls for JAX splines."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.utils._spline import spline_coefficients, spline_evaluate

EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("disabled", [True, False])
@pytest.mark.parametrize("n", [2, 3, 4, 9])
@pytest.mark.parametrize("complex_data", [False, True])
def test_exact_polynomial_nonuniform_extrapolation(disabled, n, complex_data):
    x = jnp.linspace(-1.0, 1.0, n) ** 3
    t = jnp.linspace(-1.2, 1.2, 47)
    degree = min(n - 1, 3)

    def polynomial(z):
        return sum((k + 1) * z**k for k in range(degree + 1))

    factor = 1 + 2j if complex_data else 1.0
    with jax.disable_jit(disabled):
        y = jnp.stack((polynomial(x), factor * polynomial(x)), axis=1)
        coeffs = spline_coefficients(x, y)
        actual = spline_evaluate(x, coeffs, t)
    expected = jnp.stack((polynomial(t), factor * polynomial(t)), axis=1)
    assert float(jnp.max(jnp.abs(actual - expected))) < 100 * EPS * float(jnp.max(jnp.abs(expected)))
    assert actual.shape == (47, 2)


@pytest.mark.parametrize("disabled", [True, False])
@pytest.mark.parametrize("n", [2, 3, 7])
def test_clamped_cubic_small_and_large(disabled, n):
    x = jnp.linspace(-0.7, 1.0, n) ** 3
    t = jnp.linspace(-0.5, 1.1, 31)
    def polynomial(z):
        return (2 - 1j) * z**3 - 0.2 * z + 1

    def derivative(z):
        return 3 * (2 - 1j) * z**2 - 0.2
    with jax.disable_jit(disabled):
        c = spline_coefficients(x, polynomial(x), derivative(x[jnp.asarray([0, -1])]))
        actual = spline_evaluate(x, c, t)
    assert float(jnp.max(jnp.abs(actual - polynomial(t)))) < 100 * EPS * float(jnp.max(jnp.abs(polynomial(t))))


@pytest.mark.parametrize("disabled", [True, False])
def test_interpolation_c2_and_not_a_knot(disabled):
    x = jnp.asarray([-1.0, -0.8, -0.35, 0.2, 0.7, 1.0])
    y = jnp.exp(2j * x)
    with jax.disable_jit(disabled):
        c = spline_coefficients(x, y)
        actual = spline_evaluate(x, c, x)
    h = jnp.diff(x)
    right_d1 = c[:-1, 1] + h[:-1] * (2 * c[:-1, 2] + 3 * h[:-1] * c[:-1, 3])
    right_d2 = 2 * c[:-1, 2] + 6 * h[:-1] * c[:-1, 3]
    assert float(jnp.max(jnp.abs(actual - y))) < 10 * EPS
    assert float(jnp.max(jnp.abs(right_d1 - c[1:, 1]))) < 100 * EPS
    assert float(jnp.max(jnp.abs(right_d2 - 2 * c[1:, 2]))) < 100 * EPS
    assert float(jnp.abs(c[0, 3] - c[1, 3])) < 100 * EPS
    assert float(jnp.abs(c[-1, 3] - c[-2, 3])) < 100 * EPS


@pytest.mark.parametrize("n", [2, 3, 4, 17])
def test_legacy_interp1d_complex_nonuniform(n):
    from scipy.interpolate import interp1d

    x = np.linspace(-1, 1, n) ** 3
    y = np.sin(2 * x) + 1j * np.exp(x)
    t = np.linspace(-1.1, 1.1, 53)
    kind = "linear" if n == 2 else "quadratic" if n == 3 else "cubic"
    expected = interp1d(x, y, kind=kind, fill_value="extrapolate")(t)
    actual = spline_evaluate(jnp.asarray(x), spline_coefficients(x, y), t)
    assert float(jnp.max(jnp.abs(actual - expected))) < 100 * EPS * float(np.max(np.abs(expected)))


@pytest.mark.parametrize("transpose", [False, True])
def test_public_complex_array_orientation_and_slopes(transpose):
    x = jnp.asarray([-1.0, -0.6, 0.0, 0.2, 1.0])
    y = jnp.stack((x**3 + 1j * x, (1 + 2j) * x**2), axis=1)
    slopes = jnp.stack((3 * x[jnp.asarray([0, -1])] ** 2 + 1j,
                        2 * (1 + 2j) * x[jnp.asarray([0, -1])]), axis=1)
    values = jnp.concatenate((slopes[:1], y, slopes[1:]))
    f = Chebfun.spline(x, values.T if transpose else values)
    t = jnp.linspace(-1, 1, 21)
    expected = jnp.stack((t**3 + 1j * t, (1 + 2j) * t**2), axis=1)
    assert float(jnp.max(jnp.abs(f(t) - expected))) < 100 * EPS
    assert float(jnp.max(jnp.abs(f.diff()(x[jnp.asarray([0, -1])]) - slopes))) < 100 * EPS


@pytest.mark.parametrize("n", [2, 3])
def test_public_small_node_conventions(n):
    x = jnp.linspace(-1, 1, n)
    y = (1 + 2j) * x ** (n - 1)
    f = Chebfun.spline(x, y)
    t = jnp.linspace(-1, 1, 19)
    assert float(jnp.max(jnp.abs(f(t) - (1 + 2j) * t ** (n - 1)))) < 100 * EPS
    assert len(f) == 4 * (n - 1)


def test_public_domain_internal_breaks_and_extrapolation():
    x = jnp.asarray([-1.0, -0.2, 0.3, 1.0])
    f = Chebfun.spline(x, x**3 + 2j * x, (-0.7, 0.1, 1.3))
    assert f.domain.breakpoints == (-0.7, -0.2, 0.1, 0.3, 1.0, 1.3)
    t = jnp.linspace(-0.7, 1.3, 41)
    assert float(jnp.max(jnp.abs(f(t) - (t**3 + 2j * t)))) < 100 * EPS
    assert len(f) == 20


def test_array_piecewise_second_derivative():
    f = Chebfun.from_function(
        lambda x: jnp.stack((x**2 + 1j * x, x**3), axis=1),
        Domain((-1.0, 0.0, 1.0)), n=4)
    t = jnp.linspace(-1, 1, 17)
    expected = jnp.stack((jnp.full_like(t, 2.0), 6 * t), axis=1)
    assert float(jnp.max(jnp.abs(f.diff(2)(t) - expected))) < 100 * EPS
    assert f.diff().deltas == ()


def test_array_jump_warns_and_omits_source_unsupported_delta():
    left = Chebfun.from_values(jnp.asarray([[1.0, 2.0], [1.0, 2.0]]), Domain((-1.0, 0.0)))
    right = Chebfun.from_values(jnp.asarray([[3.0, 4.0], [3.0, 4.0]]), Domain((0.0, 1.0)))
    f = Chebfun(funs=[left.funs[0], right.funs[0]], domain=Domain((-1.0, 0.0, 1.0)))
    with pytest.warns(RuntimeWarning, match="makeDeltaFun:array"):
        derivative = f.diff()
    assert derivative.deltas == ()
    assert float(jnp.max(jnp.abs(derivative(jnp.asarray([-0.5, 0.5]))))) == 0
