"""Captured MATLAB degree-tangent behavior for dense Chebfun3T."""

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
from chebfunjax.utils.degree_tangent import degree_tangent


def test_real_degree_tangent_exact_angles_poles_and_reduction():
    angles = jnp.asarray(
        [0.0, 45.0, -45.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0],
        dtype=jnp.float64,
    )
    expected = jnp.asarray(
        [0.0, 1.0, -1.0, jnp.inf, -jnp.inf, -0.0, 0.0, -jnp.inf, jnp.inf],
        dtype=jnp.float64,
    )
    got = jax.jit(degree_tangent)(angles)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint64), np.asarray(expected).view(np.uint64)
    )

    near = jnp.asarray(
        [
            np.nextafter(90.0, 0.0),
            np.nextafter(90.0, np.inf),
            np.nextafter(-90.0, 0.0),
            np.nextafter(-90.0, -np.inf),
        ],
        dtype=jnp.float64,
    )
    native = jnp.asarray(
        [
            4.031832051015932e15,
            -4.031832051015932e15,
            -4.031832051015932e15,
            4.031832051015932e15,
        ],
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(degree_tangent(near), native, rtol=2e-15, atol=0)

    large = jnp.asarray(
        [360.0 * 2**40, -360.0 * 2**40, 90.0 * 2**45, 360.0 * 2**45 + 45.0],
        dtype=jnp.float64,
    )
    large_expected = jnp.asarray([0.0, -0.0, 0.0, 0.96568877480707394])
    large_got = degree_tangent(large)
    np.testing.assert_array_equal(large_got[:3], large_expected[:3])
    np.testing.assert_array_equal(
        np.asarray(large_got[:3]).view(np.uint64),
        np.asarray(large_expected[:3]).view(np.uint64),
    )
    np.testing.assert_allclose(large_got[3], large_expected[3], rtol=2e-15, atol=0)

    # Native odd symmetry near zero and across the signed degree reduction.
    # These negative values used to pass through mod(-x, 180), losing x when
    # the result was shifted back by 180.
    reduced_angles = jnp.asarray(
        [1e-6, -1e-6, 1e-3, -1e-3, 30.0, -30.0,
         np.nextafter(-180.0, np.inf), np.nextafter(-180.0, -np.inf)],
        dtype=jnp.float64,
    )
    reduced_expected = jnp.asarray(
        [1.7453292519943298e-8, -1.7453292519943298e-8,
         1.7453292521715489e-5, -1.7453292521715489e-5,
         0.57735026918962573, -0.57735026918962573,
         4.9605240860567208e-16, -4.9605240860567208e-16],
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        jax.jit(degree_tangent)(reduced_angles), reduced_expected,
        rtol=2e-15, atol=0,
    )


def test_degree_tangent_preserves_tiny_components_and_complex_source_cases():
    eps = jnp.finfo(jnp.float64).eps
    tiny = jnp.finfo(jnp.float64).tiny
    z = jnp.asarray(
        [eps + 1j * tiny, 30.0 + 20.0j, -30.0 + 20.0j],
        dtype=jnp.complex128,
    )
    got = jax.jit(degree_tangent)(z)
    expected_tiny = complex(3.8754094422318132e-18, 3.88348649310052e-310)
    np.testing.assert_array_equal(got[0], expected_tiny)
    np.testing.assert_allclose(
        got[1], complex(0.49381237980388487, 0.43121214353788617),
        rtol=2e-15, atol=0,
    )
    np.testing.assert_allclose(
        got[2], complex(-0.49381237980388487, 0.43121214353788617),
        rtol=2e-15, atol=0,
    )

    near_complex = jnp.asarray(
        [
            np.nextafter(90.0, 0.0) + 1e-8j,
            90.0 + 1e-8j,
            np.nextafter(90.0, np.inf) + 1e-8j,
            90.0 + 1e-12j,
        ],
        dtype=jnp.complex128,
    )
    near_got = degree_tangent(near_complex)
    assert bool(jnp.isfinite(near_got[0]))
    assert bool(jnp.isnan(jnp.real(near_got[1])) & jnp.isnan(jnp.imag(near_got[1])))
    assert bool(jnp.isfinite(near_got[2]))
    assert bool(jnp.isnan(jnp.real(near_got[3])) & jnp.isnan(jnp.imag(near_got[3])))
    np.testing.assert_allclose(
        near_got[jnp.asarray([0, 2])],
        jnp.asarray(
            [
                8142.2199845301593 + 5.7295779512966614e9j,
                -8142.2199845301593 + 5.7295779512966614e9j,
            ]
        ),
        rtol=2e-14,
        atol=0,
    )


def test_degree_tangent_exact_value_branches_keep_analytic_derivative():
    scale = jnp.pi / 180.0
    for angle, multiplier in [(0.0, 1.0), (45.0, 2.0), (180.0, 1.0)]:
        derivative = jax.jit(jax.grad(degree_tangent))(angle)
        np.testing.assert_allclose(derivative, scale * multiplier, rtol=2e-15, atol=0)


def test_chebfun3t_tand_preserves_empty_and_constant_domain():
    empty = Chebfun3T()
    assert empty.tand() is empty

    domain = (-2.0, 3.0, 1.0, 4.0, -5.0, 2.0)
    constant = Chebfun3T.from_coeffs(jnp.asarray([[[45.0]]]), domain)
    result = constant.tand()
    assert isinstance(result, Chebfun3T)
    assert result.domain == domain
    np.testing.assert_allclose(result(0.2, 2.0, -1.0), 1.0, rtol=0, atol=2e-14)
