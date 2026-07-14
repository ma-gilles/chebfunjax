"""Port of MATLAB Chebfun tests/chebtech/test_assignColumns.m (Opus 4.8).

MATLAB ``assignColumns(f, cols, g)`` overwrites selected columns of an
array-valued (quasimatrix) chebtech.  FIXED (Fable 5, Big-Three
array-valued epic): techs now carry (n, m) coefficient matrices and
``assign_columns`` (0-based column indices) ports all three assertions
at the same tolerances.

Provenance
----------
MATLAB source : tests/chebtech/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))


def _mk3(Tech):
    return Tech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))


class TestChebtechAssignColumns:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_assign_two_columns(self, Tech, kind):
        # pass(1): assignColumns(f, [1 3], g) == [x cos(x) x^2].
        f = _mk3(Tech)
        g = Tech.from_function(
            lambda x: jnp.stack([x, x ** 2], axis=-1))
        h = f.assign_columns([0, 2], g)   # MATLAB [1 3] is 0-based [0 2]
        h_exact = jnp.stack([X, jnp.cos(X), X ** 2], axis=-1)
        assert h.ishappy
        assert float(jnp.max(jnp.abs(h(X) - h_exact))) \
            < 10 * h.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_assign_unhappy_column(self, Tech, kind):
        # pass(2): assigning sqrt(x) -> ~ishappy.
        f = _mk3(Tech)
        # MATLAB uses sqrt(x); the equivalent unresolvable kink here is
        # sqrt(|x|), which never converges and leaves ishappy False.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Tech.from_function(lambda x: jnp.sqrt(jnp.abs(x)),
                                   maxpow2=10)
        h = f.assign_columns(0, g)
        assert not h.ishappy

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_delete_column(self, Tech, kind):
        # pass(3): assignColumns(f, 1, []) -> 2 columns remain.
        f = _mk3(Tech)
        h = f.assign_columns(0, None)
        assert h.coeffs.shape[1] == 2
