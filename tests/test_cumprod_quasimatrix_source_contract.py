"""Source-derived CUMPROD contracts; no native chebfun test_cumprod exists.

100eps is a supplemental accuracy control, borrowed from native PROD18;
it is not a claimed native CUMPROD predicate or native runtime capture.
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebpref import ChebfunPref


def test_quasimatrix_column_loop_and_orientation():
    f = cj.chebfun(lambda x: jnp.exp(x), pref=ChebfunPref())
    g = cj.chebfun(lambda x: jnp.exp(2 * x), domain=[-1, 0, 1], pref=ChebfunPref())
    q = Quasimatrix([f, g], f.domain)
    for source in [q, q.T]:
        result = source.cumprod()
        assert result.is_transposed == source.is_transposed
        assert len(result.cols) == 2
        for actual, column in zip(result.cols, source.cols):
            expected = column.cumprod()
            assert actual.domain == expected.domain
            for a, b in zip(actual.funs, expected.funs):
                np.testing.assert_array_equal(a.tech.coeffs, b.tech.coeffs)


def test_independent_continuous_oracle_scalar_array_rows():
    for array in [False, True]:
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.exp(x), jnp.exp(2 * x)], axis=-1) if array else jnp.exp(x),
            pref=ChebfunPref(),
        )
        exact = cj.chebfun(
            lambda x: (
                jnp.stack([jnp.exp((x * x - 1) / 2), jnp.exp(x * x - 1)], axis=-1)
                if array
                else jnp.exp((x * x - 1) / 2)
            ),
            pref=ChebfunPref(),
        )
        for row in [False, True]:
            source = f.T if row else f
            oracle = exact.T if row else exact
            result = source.cumprod()
            assert result.is_transposed == row
            assert float((result - oracle).norm(jnp.inf)) < 100 * np.finfo(float).eps
