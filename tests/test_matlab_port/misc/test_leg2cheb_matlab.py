"""Port of all fourteen assertions in MATLAB tests/misc/test_leg2cheb.m.

Source norms, stored constants and bounds are retained. MATLAB row inputs
remain two-dimensional rows. Unexported MATLAB rand streams are replaced
with deterministic Python data; matching the original random inputs remains
open. Scalar CHEBFUN L2 norms are evaluated through the equivalent Tech
coefficient inner product. The last test is an additional independent control.

Provenance
----------
MATLAB source : tests/misc/test_leg2cheb.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C
from scipy.special import eval_legendre

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.transforms import cheb2leg, leg2cheb

TOL = 1e-13


def _alternating_input(n):
    coefficients = 1.0 / jnp.arange(n, 0, -1, dtype=jnp.float64) ** 2
    return coefficients.at[1::2].multiply(-1.0)


class TestLeg2cheb:
    @pytest.mark.parametrize("n,scale", [(20, 1), (1000, 10)])
    def test_row_identity(self, n, scale):
        # Source passes 1/4: the trailing transpose makes this a row.
        coefficients = jnp.zeros((1, n + 1), dtype=jnp.float64).at[0, -1].set(1)
        actual = leg2cheb(coefficients)
        assert float(jnp.linalg.norm(actual - coefficients, ord=jnp.inf)) < scale * TOL

    @pytest.mark.parametrize(
        "n,index,golden,scale",
        [(20, 1, -0.087275909551917, 1),
         (1000, 558, 6.37950860067600201345500683285806679e-04, 10)],
    )
    def test_stored_coefficients(self, n, index, golden, scale):
        # Source passes 2/5 retain the source's relative error bound.
        actual = leg2cheb(_alternating_input(n))
        assert abs(float(actual[index]) - golden) / abs(golden) < scale * TOL

    @pytest.mark.parametrize("n,scale", [(20, 1), (1000, 10)])
    def test_alternating_inverse(self, n, scale):
        # Source passes 3/6.
        coefficients = _alternating_input(n)
        actual = cheb2leg(leg2cheb(coefficients))
        assert float(jnp.linalg.norm(actual - coefficients, ord=jnp.inf)) < scale * TOL

    @pytest.mark.parametrize("n", [3, 514])
    def test_vectorization(self, n):
        # Source passes 7/8: numeric MATLAB norm(matrix) is spectral norm.
        coefficients = jnp.asarray(np.random.default_rng(0).random((n, 10)))
        batched = leg2cheb(coefficients)
        separate = jnp.stack([leg2cheb(coefficients[:, j]) for j in range(10)], axis=1)
        assert float(jnp.linalg.norm(batched - separate, ord=2)) < TOL

    @pytest.mark.parametrize("n", [10, 1000])
    def test_normalized_constant_norm(self, n):
        # Source passes 9/10 are one-sided norm(f)-1 < tol, unchanged.
        coefficients = jnp.zeros(n, dtype=jnp.float64).at[0].set(1)
        f = Chebtech2(coeffs=leg2cheb(coefficients, normalize=True))
        assert float(f.norm()) - 1.0 < TOL

    @pytest.mark.parametrize("n,scale", [(10, 1), (1000, 10)])
    def test_normalized_constant_inverse(self, n, scale):
        # Source passes 11/12: vector default norm is Euclidean.
        coefficients = jnp.zeros(n, dtype=jnp.float64).at[0].set(1)
        actual = cheb2leg(leg2cheb(coefficients, normalize=True), normalize=True)
        assert float(jnp.linalg.norm(actual - coefficients)) < scale * TOL

    @pytest.mark.parametrize("n", [10, 514])
    def test_transpose(self, n):
        # Source passes 13/14. The MATLAB seedRNG(0) stream is not claimed.
        vector = jnp.asarray(np.random.default_rng(0).random(n))
        matrix = leg2cheb(jnp.eye(n, dtype=jnp.float64))
        actual = leg2cheb(vector, trans=True)
        assert float(jnp.linalg.norm(matrix.T @ vector - actual, ord=jnp.inf)) < 10 * TOL

    def test_single_legendre_mode_values(self):
        xs = np.linspace(-0.95, 0.95, 40)
        for k in [0, 1, 5, 10]:
            e = jnp.zeros(12, dtype=jnp.float64).at[k].set(1.0)
            cc = np.asarray(leg2cheb(e))
            got = C.chebval(xs, cc)
            exact = eval_legendre(k, xs)
            # Historical independent value-control bound, not a source pass.
            assert float(np.max(np.abs(got - exact))) < 1e-11, f"k={k}"
