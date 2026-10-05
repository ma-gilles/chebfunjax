"""Independent accuracy and autodiff checks for the scratch complex extension.

These are proposed diagnostic bounds, not MATLAB assertions. Mathematical
references are mpmath at 100 decimal digits with each binary64 input promoted
exactly. The formulas follow DLMF 10.2.2 and 10.16.1; the RH motivation is
Chebfun ``lagpts.m`` ``asyBessel`` at source commit 7574c776.
"""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from chebfunjax.utils.bessel_half_complex import (
    _bessel_j_half_complex,
    _bessel_j_half_complex_array,
)

ORDERS = (-1.5, -0.5, 0.5, 1.5)
EPS = np.finfo(np.float64).eps
BOUND = 256.0 * EPS


def _mp_value(order, value):
    with mp.workdps(100):
        z = mp.mpc(float(value.real), float(value.imag))
        return complex(mp.besselj(mp.mpf(order), z))


@pytest.mark.parametrize("order", ORDERS)
def test_axis_values_match_exact_binary_inputs_on_real_and_imaginary_axes(order):
    magnitudes = np.array([0.01, 0.1, 0.9, 1.0, 1.1, 3.0, 20.0, 100.0])
    points = np.concatenate((magnitudes.astype(np.complex128), 1j * magnitudes))
    expected = np.array([_mp_value(order, z) for z in points])
    actual = np.asarray(_bessel_j_half_complex_array(order, jnp.asarray(points)))
    scale = np.maximum(1.0, np.abs(expected))
    # Same proposed elementwise envelope; avoid NumPy ndarray-atol formatter.
    bound = BOUND * scale + BOUND * np.abs(expected)
    error = np.abs(actual-expected)
    assert np.all(error <= bound), (error, bound)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("magnitude", (0.2, 0.99, 1.0, 1.01, 4.0, 30.0))
def test_imaginary_axis_jvp_matches_mpmath_derivative(order, magnitude):
    _, tangent = jax.jvp(
        lambda t: _bessel_j_half_complex(order, 1j * t),
        (jnp.float64(magnitude),),
        (jnp.float64(1.0),),
    )
    with mp.workdps(100):
        z = mp.mpc(0.0, float(magnitude))
        reference = 1j * complex(mp.diff(lambda zz: mp.besselj(mp.mpf(order), zz), z))
    scale = max(1.0, abs(reference))
    assert abs(complex(tangent) - reference) <= BOUND * scale


@pytest.mark.parametrize("order", ORDERS)
def test_axis_extension_jit_vmap_and_shape_preservation(order):
    points = jnp.asarray([[0.1 + 0.0j, 0.0 + 0.3j], [2.0 + 0.0j, 0.0 + 5.0j]])
    eager = _bessel_j_half_complex_array(order, points)
    compiled = jax.jit(lambda z: _bessel_j_half_complex_array(order, z))(points)
    vmapped = jax.vmap(lambda z: _bessel_j_half_complex(order, z))(points.ravel())
    assert eager.shape == points.shape
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=BOUND, atol=BOUND)
    np.testing.assert_allclose(np.asarray(vmapped).reshape(points.shape), np.asarray(eager),
                               rtol=BOUND, atol=BOUND)


def test_axis_zero_limits_and_out_of_domain_values_are_explicit():
    for order, expected in ((-1.5, -np.inf), (-0.5, np.inf), (0.5, 0.0), (1.5, 0.0)):
        value = complex(_bessel_j_half_complex(order, jnp.asarray(0.0 + 0.0j)))
        if np.isinf(expected):
            assert np.isinf(value.real) and np.sign(value.real) == np.sign(expected)
        else:
            assert value == 0.0j

        invalid = _bessel_j_half_complex_array(
            order,
            jnp.asarray([-1.0 + 0.0j, 0.2 + 0.1j, 0.0 - 0.2j, np.inf + 0.0j]),
        )
        assert np.isnan(np.asarray(invalid)).all()
