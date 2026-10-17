"""Public Chebfun N-D query adaptation over strict raw tech evaluation.

Provenance
----------
MATLAB source: @chebfun/feval.m, @chebfun/subsref.m.
Chebfun commit: 7574c77.
"""

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.tech.chebtech import Chebtech2


def _multi():
    coeffs = jnp.asarray([
        [1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [0.25, 0.75, -0.5],
    ])
    return Chebfun.from_coeffs(coeffs)


def test_public_nd_queries_flatten_array_valued_funs_but_raw_tech_stays_strict():
    f = _multi()
    x = jnp.arange(24.0, dtype=jnp.float64).reshape((2, 3, 4)) / 24.0
    full = f(x)
    expected = jnp.concatenate([f[:, i](x) for i in range(f.n_columns)], axis=1)
    assert full.shape == (2, 9, 4)
    assert jnp.array_equal(full, expected)

    row = f.T(x)
    assert row.shape == (9, 2, 4)
    assert jnp.array_equal(row, jnp.swapaxes(full, 0, 1))

    compiled = jax.jit(lambda q: f(q))(x)
    assert compiled.shape == full.shape
    assert jnp.array_equal(compiled, full)

    raw = Chebtech2.from_coeffs(f.funs[0].tech.coeffs)
    try:
        raw(x)
    except ValueError as exc:
        assert "more than two dimensions" in str(exc)
    else:
        raise AssertionError("raw multi-column CHEBTECH rank guard was bypassed")
