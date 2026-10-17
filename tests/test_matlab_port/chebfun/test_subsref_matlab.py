"""Port of MATLAB Chebfun tests/chebfun/test_subsref.m (Fable 5).

Native clauses1–3,7–11,13–16 compare public feval and call dispatch exactly,
using captured seed7681 Nx1 queries and MATLAB column-major reshapes. Clause12
is retained in test_subsref_numeric_columns_source.py; clause23 is below.

Original composition equality4, colon identity5, exact error identifier6,
brace/index-struct clauses17–22 and multivariate compositions24–32 remain open
in this file. The analytic composition, error-rejection and column-extraction
controls below are supplemental and do not replace those source predicates.

Provenance
----------
MATLAB source : tests/chebfun/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

import chebfunjax as cj

jax.config.update("jax_enable_x64", True)


class TestChebfunSubsref:
    def test_native_empty_column_slice(self):
        # Native pass(23) uses the default [-1, 1] domain. The second domain
        # is a supplemental adapter check for domain preservation.
        for domain in ((-1.0, 1.0), (-2.0, 3.0)):
            f = cj.chebfun(lambda x: jnp.cos(x), domain=domain)

            empty_column = f[:, 1:]
            assert empty_column.isempty()
            assert empty_column.domain == f.domain
            assert not empty_column.is_transposed

            # Empty explicit selectors use the same empty-column result.
            empty_list_column = f[:, []]
            assert empty_list_column.isempty()
            assert empty_list_column.domain == f.domain
            assert not empty_list_column.is_transposed

            # Row subscripts are transposed back after source column selection.
            empty_row = f.T[1:, :]
            assert empty_row.isempty()
            assert empty_row.domain == f.domain
            assert empty_row.is_transposed

            empty_list_row = f.T[[], :]
            assert empty_list_row.isempty()
            assert empty_list_row.domain == f.domain
            assert empty_list_row.is_transposed

            # Empty slices and out-of-range integer indices have distinct results.
            with pytest.raises(IndexError, match="out of bounds"):
                _ = f[:, 1]

    @pytest.mark.parametrize("clause", [1, 2, 3, 7, 8, 9, 10,
                                        11, 13, 14, 15, 16])
    def test_native_exact_evaluation_dispatch(self, clause):
        # Native seedRNG(7681); xr = 2*rand(1000,1)-1. The fixture retains
        # the literal MATLAB Nx1 rank, values and incoming RNG sequence.
        xr = jnp.asarray(loadmat(Path(__file__).with_name("data")
                                / "feval_seedRNG_7681_xr.mat")["xr"])
        if clause <= 10:
            f = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        else:
            f = cj.chebfun(lambda x: jnp.stack(
                [jnp.sin(x - 0.1), jnp.cos(x - 0.2)], axis=-1))
        shapes = {7: (100, 10), 8: (5, 20, 10), 9: (5, 4, 5, 10),
                  13: (100, 10), 14: (5, 20, 10), 15: (5, 4, 5, 10)}
        if clause in (10, 16):
            f = f.T
            queries = (xr, xr.reshape((5, 4, 5, 10), order="F"))
        else:
            queries = (xr.reshape(shapes[clause], order="F"),) if clause in shapes else (xr,)
        for x in queries:
            if clause in (2, 3):
                side = "left" if clause == 2 else "right"
                actual, expected = f(x, side), cj.feval(f, x, side)
            else:
                actual, expected = f(x), cj.feval(f, x)
            assert np.array_equal(np.asarray(actual), np.asarray(expected))

    def test_supplemental_analytic_composition(self):
        f = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        g = cj.chebfun(lambda x: jnp.cos(x + 0.2))
        h = f(g)
        xs = jnp.linspace(-0.9, 0.9, 21)
        assert float(jnp.max(jnp.abs(
            jnp.asarray(h(xs))
            - jnp.sin(jnp.cos(xs + 0.2) - 0.1)))) < 1e-13

    def test_supplemental_nonnumeric_rejection(self):
        # Source6 additionally requires its exact native error identifier.
        f = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        with pytest.raises((TypeError, ValueError)):
            f("X")

    def test_supplemental_analytic_column_extraction(self):
        # Source12's exact public numeric-column comparison is in the companion.
        rng = np.random.RandomState(7681)
        xr = jnp.asarray(2 * rng.rand(100) - 1)
        f = cj.chebfun(lambda x: jnp.stack(
            [jnp.sin(x - 0.1), jnp.cos(x - 0.2)], axis=-1))
        g = f.extract_columns(1)
        assert np.allclose(np.asarray(g(xr)),
                           np.cos(np.asarray(xr) - 0.2))
