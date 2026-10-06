"""Short real CDR source default-SUM/channel contracts, Chebfun7574c77."""
import jax.numpy as jnp
import numpy as np  # uses-numpy: independent analytic assertions

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _tech(c):
    return Trigtech.from_coeffs(jnp.asarray(c, dtype=jnp.complex128), is_real=True)


def _stored_sphere(columns, rows, plus=(0,), minus=(1,), pivots=(1., 1.)):
    return Spherefun(cols=columns, rows=rows, pivots=jnp.asarray(pivots),
                     idx_plus=plus, idx_minus=minus, nonzero_poles=True,
                     pivot_locations=())


def test_short_real_rank_two_retains_source_two_channels():
    # 3 + sin(theta)*cos(lambda): canonical real CDR, no complex input.
    f = _stored_sphere([_tech([3]), _tech([.5j, 0, -.5j])],
                       [_tech([1]), _tech([.5, 0, .5])])
    for x in [-1., 0., .4]:
        result = np.asarray(f.sum(1)(x)).reshape(-1)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [6., 6*np.cos(x)], rtol=0, atol=2e-14)


def test_longer_real_rank_two_uses_frequency_reduction():
    # 3 + sin(2 theta)*cos(lambda), five Fourier modes rather than three.
    f = _stored_sphere([_tech([3]), _tech([.5j, 0, 0, 0, -.5j])],
                       [_tech([1]), _tech([.5, 0, .5])])
    result = np.asarray(f.sum(1)(.4)).reshape(-1)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [6.], rtol=0, atol=2e-14)


def test_sum2_preserves_source_single_frequency_row_broadcast():
    # Deliberately redundant stored CDR. Source accepts numeric factor fields;
    # its default SUM dimension broadcasts 4 across both diagonal entries.
    f = _stored_sphere([_tech([2]), _tech([.5, 0, .5])],
                       [_tech([1]), _tech([1])], plus=(0, 1), minus=())
    np.testing.assert_allclose(f.sum2(), 16*np.pi, rtol=0, atol=5e-14)



def test_short_channels_keep_independent_diagonal_pivot_weights():
    f = _stored_sphere([_tech([3]), _tech([.5j, 0, -.5j])],
                       [_tech([1]), _tech([.5, 0, .5])], pivots=(2., 4.))
    result = np.asarray(f.sum(1)(.37)).reshape(-1)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [3., 1.5*np.cos(.37)], rtol=0, atol=2e-14)
    assert f.sum(1).is_transposed


def test_zero_output_channel_does_not_contaminate_other_channel():
    f = _stored_sphere([_tech([3]), _tech([.5j, 0, -.5j])],
                       [_tech([1]), _tech([0])])
    result = np.asarray(f.sum(1)(.37)).reshape(-1)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [6., 0.], rtol=0, atol=2e-14)


def test_global_channel_scale_allows_tiny_nonconstant_channel_to_chop():
    row = jnp.zeros(17, dtype=jnp.complex128).at[1].set(5e-21).at[-2].set(5e-21)
    f = _stored_sphere([_tech([3]), _tech([.5j, 0, -.5j])],
                       [_tech([1]), Trigtech.from_coeffs(row, is_real=True)])
    output = f.sum(1)
    result = np.asarray(output(.37)).reshape(-1)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [6., 0.], rtol=0, atol=2e-14)
    # A value tolerance alone would pass even without global simplification.
    # With globaltol the tiny nonconstant column has tolerance>1; its zero
    # DC term is retained and every nonconstant coefficient must disappear.
    tiny = np.asarray(output.funs[0].tech.coeffs)[:, 1]
    np.testing.assert_array_equal(tiny, np.zeros_like(tiny))


def test_all_zero_output_channels_preserve_source_empty_scale_reduction():
    # Both local scales and the global scale vanish; default SUM still
    # yields two channels. Source simplify's zero-coefficient branch must
    # return zeros even though the globaltol scale quotient is NaN.
    f = _stored_sphere([_tech([0]), _tech([.5j, 0, -.5j])],
                       [_tech([1]), _tech([0])])
    output = f.sum(1)
    value = np.asarray(output(.37)).reshape(-1)
    assert value.shape == (2,)
    np.testing.assert_array_equal(value, [0., 0.])
