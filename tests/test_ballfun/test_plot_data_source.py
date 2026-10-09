"""Source Ballfun plot grids/transforms; no graphics lighting parity claim."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np  # uses-numpy: independent sparse Fourier and FFT references.
import pytest

from chebfunjax._ball_plot_data import (
    ball_plot_data,
    plot_grid_shape,
    prolong_plot_coefficients,
    source_coefficients_to_values,
    source_slice_data,
)


@pytest.mark.parametrize("shape,expected", [((1, 1, 1), (25, 28, 28)),
                                             ((25, 28, 28), (25, 28, 28)),
                                             ((26, 29, 30), (31, 32, 32))])
def test_native_grid_rules(shape, expected):
    assert plot_grid_shape(shape) == expected


def sparse_coefficients(kind):
    if kind == "constant":
        return np.asarray([[[.25]]], dtype=complex)
    if kind == "odd":
        c = np.zeros((4, 5, 7), complex)
        c[0, 2, 3] = .125
        c[1, 1, 5] = .0625 + .03125j
        c[3, 4, 2] = -.125j
        return c
    c = np.zeros((3, 4, 6), complex)
    c[0, 2, 3] = .125
    c[1, 0, 0] = .0625 + .03125j
    c[2, 1, 4] = -.125j
    return c


def independent_prolong(c, shape):
    """Expand source modes directly, splitting each input Nyquist cosine."""
    result = np.zeros(shape, complex)
    assigned = set()
    for i, j, k in zip(*np.nonzero(c)):
        lambda_modes = [(j - c.shape[1] // 2, 1.)]
        theta_modes = [(k - c.shape[2] // 2, 1.)]
        if c.shape[1] % 2 == 0 and j == 0:
            lambda_modes = [(-c.shape[1] // 2, .5), (c.shape[1] // 2, .5)]
        if c.shape[2] % 2 == 0 and k == 0:
            theta_modes = [(-c.shape[2] // 2, .5), (c.shape[2] // 2, .5)]
        for kl, wl in lambda_modes:
            for kt, wt in theta_modes:
                target = (i, kl + shape[1] // 2, kt + shape[2] // 2)
                assert target not in assigned
                assigned.add(target)
                value = c[i, j, k]
                if wl == .5:
                    value = value / 2
                if wt == .5:
                    value = value / 2
                result[target] = value
    return result


@pytest.mark.parametrize("kind", ["odd", "even"])
def test_source_aliases_and_nyquist_exact(kind):
    c = sparse_coefficients(kind)
    c[0, 0, 0] += complex(2.**-90, -2.**-91)
    expected = independent_prolong(c, (31, 32, 32))
    actual = np.asarray(prolong_plot_coefficients(jnp.asarray(c), (31, 32, 32)))
    assert actual.tobytes() == expected.tobytes()


def numpy_source_values(c):
    """Independent literal @ballfun/coeffs2vals radial-first reference."""
    m, n, p = c.shape
    a = c.copy()
    if m > 1:
        a[1:-1] /= 2
        a = np.fft.fft(np.concatenate((a, a[-2:0:-1])), axis=0)[m - 1::-1]
    sign_n = (-1.) ** (np.arange(n) - n // 2)
    sign_p = (-1.) ** (np.arange(p) - p // 2)
    a *= (sign_n[:, None] * ((n * p) * sign_p)[None, :])[None]
    return np.fft.ifft(np.fft.ifft(np.fft.ifftshift(a, axes=(1, 2)), axis=1), axis=2)


def qualification_bound(reference, shape):
    return 64 * max(shape) * np.finfo(float).eps * max(1., float(np.max(np.abs(reference))))


@pytest.mark.parametrize("kind", ["constant", "odd", "even"])
def test_radial_first_transform_against_independent_fft(kind):
    c = sparse_coefficients(kind)
    assert np.sum(np.abs(c)) <= 1
    shape = (25, 28, 28)
    expected_coefficients = independent_prolong(c, shape)
    expected = numpy_source_values(expected_coefficients)
    actual = np.asarray(source_coefficients_to_values(
        prolong_plot_coefficients(jnp.asarray(c), shape)))
    assert np.max(np.abs(actual - expected)) <= qualification_bound(expected, shape)


def test_native_theta_reorder_closure_and_all_five_slice_values():
    values = np.arange(25 * 28 * 28, dtype=float).reshape(25, 28, 28)
    data = source_slice_data(jnp.asarray(values + .5j))
    expected = values[12:, :, :].transpose(0, 2, 1)[:, [0] + list(range(27, 13, -1)), :]
    expected = np.concatenate((expected, expected[:, :, :1]), axis=2)
    np.testing.assert_array_equal(data.values, expected)
    assert [s.kind for s in data.surfaces] == ["elevation", "elevation", "radius", "longitude", "longitude"]
    slices = [expected[:, 0, :], expected[:, 7, :], expected[data.radius_index],
              expected[:, :, 0], expected[:, :, 7]]
    for surface, cdata in zip(data.surfaces, slices):
        np.testing.assert_array_equal(surface.values, cdata)
        assert surface.x.shape == surface.y.shape == surface.z.shape == cdata.shape
    assert data.radius_index == int(np.argmin(np.abs(np.asarray(data.radius) - .5)))
    assert data.surfaces[0].values.shape == (13, 29)
    assert data.surfaces[2].values.shape == (15, 29)
    assert data.surfaces[3].values.shape == (13, 15)


def test_analytic_cartesian_polynomial_on_every_surface():
    # f=1/8+x/8+y/16+z/8+(x*x+y*y+z*z)/16, derived in CFF basis.
    c = np.zeros((3, 3, 3), complex)
    c[0, 1, 1] = .125 + 1 / 32
    c[2, 1, 1] = 1 / 32
    for kl in (-1, 1):
        for kt in (-1, 1):
            c[1, kl + 1, kt + 1] += -1j * kt / 32 - kl * kt / 64
    c[1, 1, 0] += 1 / 16
    c[1, 1, 2] += 1 / 16
    assert np.sum(np.abs(c)) <= 1
    ball = SimpleNamespace(coeffs=jnp.asarray(c), shape=c.shape, is_real=True,
                           isempty=lambda: False)
    data = ball_plot_data(ball)
    for surface in data.surfaces:
        x, y, z = map(np.asarray, (surface.x, surface.y, surface.z))
        expected = .125 + x / 8 + y / 16 + z / 8 + (x*x + y*y + z*z) / 16
        assert np.max(np.abs(np.asarray(surface.values) - expected)) <= qualification_bound(expected, data.shape)


def test_complex_source_warning_and_unconditional_real_projection():
    c = jnp.asarray([[[.25 + .5j]]])
    ball = SimpleNamespace(coeffs=c, shape=c.shape, is_real=False, isempty=lambda: False)
    with pytest.warns(UserWarning, match="CHEBFUN:BALLFUN:plot:isReal"):
        data = ball_plot_data(ball)
    for surface in data.surfaces:
        assert not np.iscomplexobj(np.asarray(surface.values))
        np.testing.assert_array_equal(surface.values, .25)


def test_empty_source_error_before_coefficient_access():
    with pytest.raises(ValueError, match="CHEBFUN:BALLFUN:plot:isempty: Function is empty"):
        ball_plot_data(SimpleNamespace(isempty=lambda: True))
