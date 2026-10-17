"""Nine original public Bndfun compose predicates, Chebfun 7574c77.

Source tests/bndfun/test_compose.m uses a row linspace. Native array-valued
feval concatenates the column results into one row; all its infinity norms
are therefore vector max norms, while clause6 uses the default vector2 norm.
The earlier manual-onefun adapted cases remain in test_compose_legacy_controls.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = np.finfo(np.float64).eps
DOM = Domain((-2., 7.))
X = jnp.linspace(-2., 7., 1000)


def _bf(op, domain=DOM):
    return Bndfun.from_function(op, domain)


def _columns(*values):
    return jnp.stack(values, axis=-1)


@pytest.mark.parametrize('clause', range(1, 10))
def test_original_compose(clause, record_property):
    pref = ChebfunPref()
    if clause <= 4:
        if clause == 1:
            op = lambda x: x
        elif clause == 2:
            op = lambda x: _columns(x, x)
        elif clause == 3:
            op = lambda x: _columns(x, x*x)
        else:
            op = lambda x: _columns(x, x, x*x)
        f = _bf(op)
        result = f.compose(jnp.sin, None, None, pref)
        error = result(X) - jnp.sin(op(X))
    elif clause == 5:
        result = _bf(jnp.sin).compose(jnp.add, _bf(jnp.cos), None, pref)
        error = result(X) - (jnp.sin(X) + jnp.cos(X))
    elif clause == 6:
        f1 = _bf(lambda x: _columns(jnp.sin(x), jnp.cos(x)))
        f2 = _bf(lambda x: _columns(jnp.cos(x), jnp.exp(x)))
        result = f1.compose(jnp.multiply, f2, None, pref)
        h = _bf(lambda x: _columns(jnp.sin(x)*jnp.cos(x),
                                   jnp.cos(x)*jnp.exp(x)))
        error = h(X) - result(X)
    else:
        f = _bf((lambda x: _columns(x, x*x)) if clause == 9 else (lambda x: x*x))
        outer_domain = Domain((-2., 49.)) if clause == 9 else Domain((0., 49.))
        outer = _bf((lambda x: _columns(jnp.sin(x), jnp.cos(x)))
                    if clause == 8 else jnp.sin, outer_domain)
        result = f.compose(outer)
        exact = (_columns(jnp.sin(X*X), jnp.cos(X*X)) if clause == 8 else
                 _columns(jnp.sin(X), jnp.sin(X*X)) if clause == 9 else jnp.sin(X*X))
        error = result(X) - exact
    # Column-major flatten is the native row-valued array feval convention.
    vector = np.asarray(error).reshape(-1, order='F')
    measured = np.linalg.norm(vector) if clause == 6 else np.linalg.norm(vector, np.inf)
    bound = (10 if clause in (1, 2, 5) else 100)*result.vscale*EPS
    record_property('native_clause', clause)
    record_property('error', float(measured))
    record_property('original_bound', float(bound))
    assert result.domain == DOM
    assert measured < bound
