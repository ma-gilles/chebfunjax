"""Scalar-column storage adapter for native Chebfun3 plane products.

Provenance
----------
MATLAB source: @chebfun3/feval.m and @chebfun3/restrict.m, commit 7574c77.
The adapter preserves the source left-to-right matrix multiplication and
nonconjugating transpose. A singleton coefficient column is exposed as a
scalar tech before the existing Chebfun2 outer product stores its factors.
"""
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.mtimes import _columns


def scalar_plane_product(left, matrix, right):
    """Compute ``(left @ matrix) @ right.T`` with scalar factor columns."""
    product = left @ matrix
    scalar_columns = [column.extract_columns(0) for column in _columns(product)]
    return Chebfun.horzcat(*scalar_columns) @ right.T
