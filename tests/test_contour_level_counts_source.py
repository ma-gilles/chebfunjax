"""Scalar level-count predicates from R2017a contourobjHelper localParseargs."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.plotting import contour


class Ramp:
    domain = (2., 9., -1., 1.)

    def __call__(self, x, y):
        return x + 0*y


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('count', [3, 3.9, [3], np.array(3)])
def test_exact_scalar_count_and_interior_levels(count):
    _, ax = contour(Ramp(), levels=count, n_pts=5)
    np.testing.assert_array_equal(ax.collections[0].levels, [3.75, 5.5, 7.25])


def test_one_level_is_midpoint():
    _, ax = contour(Ramp(), levels=1, n_pts=5)
    np.testing.assert_array_equal(ax.collections[0].levels, [5.5])


def test_zero_count_has_no_lines():
    _, ax = contour(Ramp(), levels=0, n_pts=5)
    assert ax.collections[0].levels.size == 0
    assert not ax.collections[0].get_paths()


def test_filled_count_includes_minimum():
    _, ax = contour(Ramp(), levels=3, filled=True, n_pts=5)
    np.testing.assert_array_equal(ax.collections[0].levels, [2., 3.75, 5.5, 7.25])


def test_explicit_vector_and_repeated_level_remain_values():
    _, ax = contour(Ramp(), levels=[3., 4., 8.], n_pts=5)
    np.testing.assert_array_equal(ax.collections[0].levels, [3., 4., 8.])
    _, other = contour(Ramp(), levels=[4., 4.], n_pts=5)
    np.testing.assert_array_equal(other.collections[0].levels, [4.])


@pytest.mark.parametrize('count', [-1, -.1, np.nan, np.inf])
def test_invalid_scalar_counts(count):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        contour(Ramp(), levels=count, n_pts=5)


def test_extrema_ignore_nonfinite_samples():
    class NonfiniteRamp(Ramp):
        def __call__(self, x, y):
            values = np.array(x+0*y)
            values[2, 2] = np.nan
            values[1, 2] = np.inf
            values[3, 2] = -np.inf
            return values

    _, ax = contour(NonfiniteRamp(), levels=3, n_pts=5)
    np.testing.assert_array_equal(ax.collections[0].levels, [3.75, 5.5, 7.25])
