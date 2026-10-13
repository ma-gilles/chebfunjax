"""Independent source-contract checks for the MATLAB R2024b HSV palette."""

from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


def _decimal_hsv_rgb(hue):
    """Independent six-sector construction for fully saturated HSV."""
    hue = Decimal(hue) % Decimal(1)
    sector_position = hue * Decimal(6)
    sector = int(sector_position)
    fraction = sector_position - sector
    falling = Decimal(1) - fraction
    return (
        (Decimal(1), falling, Decimal(0), Decimal(0), fraction, Decimal(1))[sector],
        (fraction, Decimal(1), Decimal(1), falling, Decimal(0), Decimal(0))[sector],
        (Decimal(0), Decimal(0), fraction, Decimal(1), Decimal(1), falling)[sector],
    )


def test_r2024b_source_formula_and_cardinal_rgb():
    # hsv.m: h=(0:m-1)'/max(m,1); hsv2rgb applies the six HSV sectors.
    from chebfunjax._hsv_colormap import hsv_colors

    colors = np.asarray(hsv_colors(6))
    expected = np.asarray([
        (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 1, 1), (0, 0, 1), (1, 0, 1),
    ], dtype=float)
    np.testing.assert_array_equal(colors, expected)


def test_decimal_reference_for_non_cardinal_explicit_lengths():
    from chebfunjax._hsv_colormap import hsv_colors

    with localcontext() as context:
        context.prec = 60
        for m in (2, 7, 17, 64):
            expected = np.asarray([
                [float(channel) for channel in _decimal_hsv_rgb(Decimal(k) / m)]
                for k in range(m)
            ])
            actual = np.asarray(hsv_colors(m))
            assert actual.shape == (m, 3)
            assert np.max(np.abs(actual - expected)) <= 8 * np.finfo(float).eps


def test_zero_and_singleton_match_hsv_m_source_behavior():
    from chebfunjax._hsv_colormap import hsv_colors

    assert np.asarray(hsv_colors(0)).shape == (0, 3)
    np.testing.assert_array_equal(np.asarray(hsv_colors(1)), [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize('length, error', [(-1, ValueError), (2.5, TypeError)])
def test_invalid_lengths_follow_existing_palette_convention(length, error):
    from chebfunjax._hsv_colormap import hsv_colors

    with pytest.raises(error):
        hsv_colors(length)


def test_renderer_index_mapping_and_positive_length():
    from chebfunjax._hsv_colormap import hsv_colormap, hsv_colors

    m = 7
    palette = hsv_colormap(m)
    samples = np.array([-1.0, 0.0, 0.1, 1.0 / m, 0.5,
                        np.nextafter(1.0, 0.0), 1.0, 2.0])
    indices = np.array([0, 0, 0, 1, 3, m - 1, m - 1, m - 1])
    np.testing.assert_array_equal(palette(samples)[:, :3],
                                  np.asarray(hsv_colors(m))[indices])
    assert palette.N == m
    with pytest.raises(ValueError):
        hsv_colormap(0)


def test_native_r2024b_capture_at_renderer_length_600():
    from chebfunjax._hsv_colormap import hsv_colors

    fixture = Path(__file__).parent / 'fixtures' / 'hsv_r2024b_m600_native.npy'
    native = np.load(fixture, allow_pickle=False)
    actual = np.asarray(hsv_colors(600))
    assert native.shape == actual.shape == (600, 3)
    # hsv.m divides each integer hue by m, hsv2rgb multiplies by six and
    # subtracts the integer sector before forming the affine channel value.
    # Division and multiplication contribute at most about 6*eps absolute
    # in h*6; fraction subtraction is exact by Sterbenz within each nonzero
    # sector, while 1-fraction may round once (<=0.5*eps). Masking by 0/1 and
    # adding zero in the six source branches is exact. 8*eps therefore leaves
    # a conservative margin over this operation-derived <=6.5*eps bound.
    assert np.max(np.abs(actual - native)) <= 8 * np.finfo(np.float64).eps


def test_jit_static_length_agrees_with_eager_source_palette():
    import jax
    import jax.numpy as jnp

    from chebfunjax._hsv_colormap import hsv_colors

    eager = hsv_colors(600)
    compiled = jax.jit(hsv_colors, static_argnums=(0,))(600)
    assert float(jnp.max(jnp.abs(eager - compiled))) <= 8 * jnp.finfo(jnp.float64).eps
