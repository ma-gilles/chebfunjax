"""All44 literal predicates from native tests/chebtech/test_compose.m.

Four preference rows persist resampling into row4; only clauses1--6 pass
those preferences. Clauses7--11 omit pref and use default compose settings.
Clauses3/4 retain h and its vscale from clause2. Numeric matrix infinity
norms are row-sum norms. Old adapted22 controls remain in the legacy companion.
Chebfun source7574c77680d7e82b79626300bf255498271a72df.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

EPS = np.finfo(float).eps


def _columns(*values):
    return jnp.column_stack(values)


def _scale(h):
    return float(jnp.max(h.vscale_columns))


@pytest.fixture(scope='module', params=[1, 2, 3, 4], ids=lambda n: f'native_row{n}')
def source_row(request):
    row = request.param
    Tech = Chebtech1 if row == 1 else Chebtech2
    kind = 1 if row == 1 else 2
    prefs = {} if row < 3 else {'refinement_function': 'resampling'}
    if row == 4:
        prefs['extrapolate'] = True
    out = {}

    def compare(clause, actual, expected, bound):
        out[clause] = dict(actual=actual, expected=expected, bound=bound)

    f = Tech.from_function(lambda x: x)
    g = f.compose(jnp.sin, **prefs)
    h = Tech.from_function(jnp.sin)
    compare(1, h.coeffs, g.coeffs, 10 * _scale(h) * EPS)

    f = Tech.from_function(lambda x: _columns(x, x))
    g = f.compose(jnp.sin, **prefs)
    h = Tech.from_function(lambda x: _columns(jnp.sin(x), jnp.sin(x)))
    compare(2, h.coeffs, g.coeffs, 10 * _scale(h) * EPS)
    retained_h = h

    f = Tech.from_function(lambda x: _columns(x, x**2))
    g = f.compose(jnp.sin, **prefs)
    x = chebpts(g.n, kind)
    compare(3, jnp.sin(_columns(x, x**2)), g.coeffs2vals(g.coeffs),
            1e2 * _scale(retained_h) * EPS)

    f = Tech.from_function(lambda x: _columns(x, x, x**2))
    g = f.compose(jnp.sin, **prefs)
    x = chebpts(g.n, kind)
    compare(4, jnp.sin(_columns(x, x, x**2)), g.coeffs2vals(g.coeffs),
            1e2 * _scale(retained_h) * EPS)

    f1, f2 = Tech.from_function(jnp.sin), Tech.from_function(jnp.cos)
    g = f1.compose(jnp.add, f2, **prefs)
    x = chebpts(g.n, kind)
    h = Tech.from_values(jnp.sin(x) + jnp.cos(x))
    compare(5, h.coeffs2vals(h.coeffs), g.coeffs2vals(g.coeffs), 10 * _scale(h) * EPS)

    f1 = Tech.from_function(lambda x: _columns(jnp.sin(x), jnp.cos(x)))
    f2 = Tech.from_function(lambda x: _columns(jnp.cos(x), jnp.exp(x)))
    g = f1.compose(jnp.multiply, f2, **prefs)
    h = Tech.from_function(lambda x: _columns(jnp.sin(x)*jnp.cos(x), jnp.cos(x)*jnp.exp(x)))
    compare(6, h.coeffs2vals(h.coeffs), g.coeffs2vals(g.coeffs),
            10 * float(jnp.max(10 * h.vscale_columns * EPS)))

    # Native omits PREF for all remaining calls, regardless of outer row.
    f, g = Tech.from_function(lambda x: x**2), Tech.from_function(jnp.sin)
    h = f.compose(g)
    x = chebpts(h.n, kind)
    compare(7, h.coeffs2vals(h.coeffs), jnp.sin(x**2), 10 * _scale(h) * EPS)

    f = Tech.from_function(lambda x: x**2)
    g = Tech.from_function(lambda x: _columns(jnp.sin(x), jnp.cos(x)))
    h = f.compose(g)
    x = chebpts(h.n, kind)
    compare(8, h.coeffs2vals(h.coeffs), _columns(jnp.sin(x**2), jnp.cos(x**2)),
            10 * _scale(h) * EPS)

    f = Tech.from_function(lambda x: _columns(x, x**2))
    g = Tech.from_function(jnp.sin)
    h = f.compose(g)
    x = chebpts(h.n, kind)
    compare(9, h.coeffs2vals(h.coeffs), _columns(jnp.sin(x), jnp.sin(x**2)),
            10 * _scale(h) * EPS)

    f = Tech.from_function(lambda x: _columns(x, x**2))
    g = Tech.from_function(lambda x: _columns(jnp.sin(x), jnp.cos(x)))
    try:
        f.compose(g)
        out[10] = dict(identifier=None)
    except Exception as error:
        out[10] = dict(identifier=getattr(error, 'identifier', None), exception=repr(error))

    f = Tech.from_function(lambda x: _columns(x, x**2))
    g = Tech.from_function(jnp.sin)
    try:
        f.compose(jnp.add, g)
        out[11] = dict(completed=True)
    except Exception as error:
        out[11] = dict(completed=False, exception=repr(error))
    return out


@pytest.mark.parametrize('clause', range(1, 12), ids=lambda n: f'clause{n:02d}')
def test_matlab_clause(source_row, clause, record_property):
    result = source_row[clause]
    if clause == 10:
        record_property('source_identifier', str(result['identifier']))
        assert result['identifier'] == 'CHEBFUN:CHEBTECH:compose:arrval'
    elif clause == 11:
        assert result['completed'], result
    else:
        actual, expected = np.asarray(result['actual']), np.asarray(result['expected'])
        record_property('actual_shape', str(actual.shape))
        record_property('expected_shape', str(expected.shape))
        assert actual.shape == expected.shape
        error = float(np.linalg.norm(actual - expected, ord=np.inf))
        record_property('source_error', error)
        record_property('source_bound', result['bound'])
        assert error < result['bound']
