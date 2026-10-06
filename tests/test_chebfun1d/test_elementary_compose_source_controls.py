"""Independent public endpoint contracts from source columnCompose.

Provenance
----------
MATLAB source : @chebfun/compose.m (columnCompose endpoint policy/state)
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("row", [False, True])
@pytest.mark.parametrize("columns", [1, 2])
def test_piecewise_composition_avoids_one_sided_end_values_and_uses_stored_points(row, columns):
    pieces = []
    for a, b in [(-1.0, 0.0), (0.0, 1.0)]:
        coeffs = jnp.array([(a + b) / 2, (b - a) / 2])
        if columns == 2:
            coeffs = jnp.stack([coeffs, 2 * coeffs], axis=-1)
        pieces.append(_Piece(Chebtech2.from_coeffs(coeffs), (a, b)))
    f = Chebfun(pieces, Domain((-1.0, 0.0, 1.0)))
    pv = jnp.array([5.0, 6.0, 7.0])
    if columns == 2:
        pv = jnp.stack([pv, 2 * pv], axis=-1)
    f = f.set_point_values(pv)
    if row:
        f = f.T

    def square_without_endpoint_limits(values):
        # Source composes smooth pieces from INTERIOR samples, and separately
        # transforms pointValues. Calling this on smooth endpoint limits is
        # undefined, while the stored values 5,6,7 are in its valid domain.
        # Only first-column limits identify sampled rows: the second column
        # equals 1 at the legitimate interior point x=.5.
        first_column = values[..., 0] if values.ndim == 2 else values
        if bool(jnp.any((first_column == -1) | (first_column == 0) | (first_column == 1))):
            raise ValueError("one-sided endpoint limits must not be sampled")
        return values**2

    g = f._apply_fun(square_without_endpoint_limits)
    assert g.is_transposed == row
    np.testing.assert_array_equal(np.asarray(g.point_values), np.asarray(pv**2))
    # Direct piece values avoid row/evaluation-shape conventions while checking
    # the continuous interpolants against independent analytic polynomials.
    for piece in g.funs:
        a, b = piece.interval
        x = jnp.array([a + 0.2 * (b - a), a + 0.7 * (b - a)])
        expected = x * x if columns == 1 else jnp.stack([x * x, 4 * x * x], axis=-1)
        np.testing.assert_allclose(
            np.asarray(piece(x)), np.asarray(expected), rtol=0, atol=32 * np.finfo(float).eps
        )


def test_empty_elementary_composition_keeps_source_domain_and_orientation():
    f = Chebfun.empty().T
    g = f.sin()
    assert g.isempty() and g.is_transposed
    assert g.domain == f.domain
