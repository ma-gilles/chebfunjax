"""Bounded public roots obey mapping.For, Chebfun commit 7574c77."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("a,b", [(1.5, 2.0 - 1 / 3), (-2.0, 7.0)])
@pytest.mark.parametrize("root", [-1.0, -0.7, 0.0, 0.4, 1.0])
def test_piece_and_public_root_source_forward_map(a, b, root):
    piece = _Piece(Chebtech2(jnp.asarray([-root, 1.0])), (a, b))
    # Literal scalar mapping.For expression; do not use the library helper
    # as this oracle. Source order matters on fractional physical intervals.
    expected = b * (root + 1) / 2 + a * (1 - root) / 2
    np.testing.assert_array_equal(piece.roots(), [expected])
    f = Chebfun([piece], Domain((a, b)))
    np.testing.assert_array_equal(f.roots(), [expected])
