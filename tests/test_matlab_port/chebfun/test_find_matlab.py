"""Port of MATLAB Chebfun tests/chebfun/test_find.m (Fable 5).

MATLAB's two-output ``[x, col] = find(F)`` for an array-valued logical
chebfun maps to ``F.find(return_cols=True)``; ``~logical(f - 0.5)`` of an
array-valued chebfun is formed column by column (``logical_eq`` rejects
array-valued input, as MATLAB's ``==`` does) and the columns are searched
in turn.

Provenance
----------
MATLAB source : tests/chebfun/test_find.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _find_cols(f):
    xs, cols = [], []
    for j in range(f.n_columns):
        xj = (f.extract_columns(j) - 0.5).logical_not().find()
        xs.extend(float(v) for v in np.asarray(xj))
        cols.extend([j] * len(np.asarray(xj)))
    if f.is_transposed:  # MATLAB returns {idx.', x.'} for a row chebfun
        return np.asarray(cols), np.asarray(xs)
    return np.asarray(xs), np.asarray(cols)


class TestChebfunFind:
    def test_all_matlab_assertions(self):
        assert Chebfun.empty().find().size == 0                     # pass(1)

        f = chebfun(jnp.sin, domain=(0.0, 2 * np.pi))
        x = np.asarray(f.logical_eq(0.5).find()) / np.pi
        assert np.max(np.abs(x - [1 / 6, 5 / 6])) < 10 * f.vscale * EPS  # pass(2)

        f = chebfun(jnp.exp, domain=(-1.0, -0.5, 0.0, 0.5, 1.0))
        x = np.asarray(f.logical_eq(1.0).find())
        assert x.size == 1 and abs(x[0]) < 10 * f.vscale * EPS      # pass(3)
        x = np.asarray((f <= 0.0).find())
        assert x.size == 0                                          # pass(4)

        f = chebfun(lambda t: jnp.stack([jnp.sin(t), jnp.exp(t)], axis=-1),
                    domain=(-np.pi, np.pi))
        x, col = _find_cols(f)                                      # [x, col] = find(~logical(f - 0.5))
        assert len(x) == 3 and len(col) == 3                        # pass(5)
        fx = np.asarray(f(jnp.asarray(x)))
        err = np.asarray([fx[k, col[k]] for k in range(3)])
        assert np.max(np.abs(err - 0.5)) < 10 * f.vscale * EPS      # pass(6)
        row, y = _find_cols(f.T)                                    # transposed: (row, y)
        assert np.array_equal(row, col) and np.array_equal(x, y)    # pass(7)

        with pytest.raises(ValueError, match="find:arrout"):
            f.find()                                                # pass(8)
        with pytest.raises(ValueError, match="find:infset"):
            chebfun(jnp.sin).find()                                 # pass(9)
