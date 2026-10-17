"""Source Chebtech2 composition sampling and independent analytic controls.

Provenance: @chebtech2/compose.m composeResample1/2 and composeNested1/2,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech2


def _data(array, offset):
    a = np.array([2., 3.]) if array else np.array(2.)
    b = np.array([.25, -.5]) if array else np.array(.25)
    return np.stack((a+offset, b)), a+offset, b


def _forbid_point_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Callable composition sampled an operand by pointwise evaluation')
    monkeypatch.setattr(Chebtech2, '__call__', forbidden)


@pytest.mark.parametrize('array', [False, True])
def test_unary_analytic_square_without_point_sampling(monkeypatch, array):
    coefficients, a, b = _data(array, 0.)
    f = Chebtech2.from_coeffs(jnp.asarray(coefficients))
    _forbid_point_sampling(monkeypatch)
    result = f.compose(lambda u: u*u)
    expected = np.stack((a*a+b*b/2, 2*a*b, b*b/2))
    assert result.coeffs.shape == expected.shape
    assert np.max(np.abs(np.asarray(result.coeffs)-expected)) < 20*np.finfo(float).eps*np.max(np.abs(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), coefficients)


@pytest.mark.parametrize('f_array,g_array', [(False, False), (True, False), (False, True), (True, True)])
def test_binary_analytic_columns_without_point_sampling(monkeypatch, f_array, g_array):
    fc, a, b = _data(f_array, 0.)
    gc, c, d = _data(g_array, 1.)
    f = Chebtech2.from_coeffs(jnp.asarray(fc))
    g = Chebtech2.from_coeffs(jnp.asarray(gc))
    _forbid_point_sampling(monkeypatch)
    result = f.compose(lambda u, v: u+2*v, g)
    expected = np.stack((a+2*c, b+2*d))
    assert result.coeffs.shape == expected.shape
    assert np.max(np.abs(np.asarray(result.coeffs)-expected)) < 20*np.finfo(float).eps*np.max(np.abs(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), fc)
    np.testing.assert_array_equal(np.asarray(g.coeffs), gc)


@pytest.mark.parametrize('binary', [False, True])
def test_nested_refinement_reuses_samples_and_skips_extrapolated_endpoints(monkeypatch, binary):
    f = Chebtech2.from_coeffs(jnp.asarray([0., 1.]))
    g = Chebtech2.from_coeffs(jnp.asarray([2., .25]))
    observed = []

    def operation(u, v=None):
        observed.append(np.asarray(u))
        assert not np.any(np.abs(np.asarray(u)) == 1.)
        return u*u if v is None else u+v

    def construct(cls, function, **kwargs):
        refine = kwargs['refinement_function']
        pref = dict(minSamples=17, maxLength=65, extrapolate=True)
        first, gave_up = refine(function, None, pref)
        assert not gave_up and first.shape == (17,)
        assert np.isnan(np.asarray(first)[[0, -1]]).all()
        changed = first.at[3].add(.125)
        second, gave_up = refine(function, changed, pref)
        assert not gave_up and second.shape == (33,)
        # Stored points (including NaN sentinels) are retained exactly.
        np.testing.assert_array_equal(np.asarray(second)[::2], np.asarray(changed))
        assert [len(v) for v in observed] == [15, 16]
        expected_grid = np.asarray(Chebtech2.coeffs2vals(jnp.pad(f.coeffs, (0, 31))))
        np.testing.assert_array_equal(observed[1], expected_grid[1:-1:2])
        return f

    monkeypatch.setattr(Chebtech2, '_adaptive_construct', classmethod(construct))
    _forbid_point_sampling(monkeypatch)
    f.compose(operation, g if binary else None, extrapolate=True)


def test_initial_max_length_cap_and_later_give_up(monkeypatch):
    f = Chebtech2.from_coeffs(jnp.asarray([0., 1.]))

    def construct(cls, function, **kwargs):
        refine = kwargs['refinement_function']
        pref = dict(minSamples=33, maxLength=17, extrapolate=False)
        first, gave_up = refine(function, None, pref)
        assert first.shape == (17,) and not gave_up
        np.testing.assert_allclose(first, np.cos(np.pi*np.arange(16, -1, -1)/16),
                                   rtol=0, atol=4*np.finfo(float).eps)
        second, gave_up = refine(function, first, pref)
        assert gave_up and second is first
        return f

    monkeypatch.setattr(Chebtech2, '_adaptive_construct', classmethod(construct))
    _forbid_point_sampling(monkeypatch)
    f.compose(lambda u: u)


@pytest.mark.parametrize('binary', [False, True])
def test_actual_complex_extrapolation_matches_polynomial(monkeypatch, binary):
    f = Chebtech2.from_coeffs(jnp.asarray([2j, .25j]))
    g = Chebtech2.from_coeffs(jnp.asarray([1., .5]))
    _forbid_point_sampling(monkeypatch)
    result = (f.compose(lambda u, v: u*u+v, g, extrapolate=True) if binary
              else f.compose(lambda u: u**3, extrapolate=True))
    expected = (np.asarray([-3.03125, -.5, -.03125]) if binary
                else -1j*np.asarray([8.1875, 3.01171875, .1875, .00390625]))
    assert result.coeffs.shape == expected.shape
    error = np.max(np.abs(np.asarray(result.coeffs)-expected))
    assert error < 40*np.finfo(float).eps*np.max(np.abs(expected))
