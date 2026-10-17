"""New source-route and mathematical controls; no original MATLAB bound claim."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

SCENARIOS = ['basis_t32', 'arrval', 'inner_column', 'outer_column', 'both_column',
             'outer_array', 'outer_array_inner_column', 'inner_array',
             'inner_array_outer_column']


@pytest.mark.parametrize('Inner,Outer', [(Chebtech1, Chebtech2), (Chebtech2, Chebtech1)],
                         ids=['c1_inner', 'c2_inner'])
@pytest.mark.parametrize('route', ['default', 'pref', 'data'])
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_crosskind_source(Inner, Outer, route, scenario, monkeypatch, record_property):
    inner_coeffs = jnp.array([0., 1.])
    outer_coeffs = jnp.array([.5, 0., .5])
    if scenario in ('arrval', 'inner_array', 'inner_array_outer_column'):
        inner_coeffs = jnp.stack((inner_coeffs, .5*inner_coeffs), axis=1)
    elif scenario in ('inner_column', 'both_column', 'outer_array_inner_column'):
        inner_coeffs = inner_coeffs[:, None]
    if scenario in ('arrval', 'outer_array', 'outer_array_inner_column'):
        outer_coeffs = jnp.stack((outer_coeffs, jnp.array([0., 1., 0.])), axis=1)
    elif scenario in ('outer_column', 'both_column', 'inner_array_outer_column'):
        outer_coeffs = outer_coeffs[:, None]
    elif scenario == 'basis_t32':
        outer_coeffs = jnp.zeros(33).at[32].set(1.)
    inner = Inner.from_coeffs(inner_coeffs)
    outer = Outer.from_coeffs(outer_coeffs)
    calls = []
    original = Inner._adaptive_construct

    def observe(cls, op, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(op, *args, **kwargs)

    monkeypatch.setattr(Inner, '_adaptive_construct', classmethod(observe))
    kwargs = {} if route == 'default' else {'pref': {}} if route == 'pref' else {
        'pref': {}, 'data': {'vscale': 1., 'hscale': 1.}}
    if scenario == 'arrval':
        with pytest.raises(ValueError) as caught:
            inner.compose(outer, **kwargs)
        assert getattr(caught.value, 'identifier', None) == 'CHEBFUN:CHEBTECH:compose:arrval'
        assert not calls
        return
    result = inner.compose(outer, **kwargs)
    x = np.linspace(-.97, .96, 61)
    values = np.asarray(result(jnp.asarray(x)))
    if scenario == 'basis_t32':
        expected = np.polynomial.chebyshev.chebval(x, np.asarray(outer_coeffs))
    elif scenario in ('inner_array', 'inner_array_outer_column'):
        expected = np.stack((x*x, .25*x*x), axis=1)
    elif scenario in ('outer_array', 'outer_array_inner_column'):
        expected = np.stack((x*x, x), axis=1)
    else:
        expected = x*x
    record_property('inner_coeff_shape', str(inner_coeffs.shape))
    record_property('outer_coeff_shape', str(outer_coeffs.shape))
    record_property('result_coeff_shape', str(result.coeffs.shape))
    record_property('value_shape', str(values.shape))
    record_property('expected_shape', str(expected.shape))
    assert values.shape == expected.shape
    error = np.linalg.norm((values-expected).reshape(-1), np.inf)
    bound = 100*np.finfo(np.float64).eps*result.vscale
    effective_min = calls[0].get('min_samples')
    if effective_min is None:
        effective_min = 2**calls[0].get('start_pow2', 4)+1
    record_property('effective_min_samples', int(effective_min))
    record_property('error', float(error))
    record_property('new_analytic_bound', float(bound))
    assert type(result) is Inner
    assert effective_min >= max(inner.n, outer.n)
    assert error < bound
