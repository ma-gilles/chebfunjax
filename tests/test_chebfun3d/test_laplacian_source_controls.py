"""Independent controls for the native Laplacian dispatch.

Provenance
----------
MATLAB source : @chebfun3/{laplacian,lap}.m
Chebfun commit: 7574c77
"""

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_zero_laplacian_uses_native_default_domain():
    f = Chebfun3.from_function(lambda x, y, z: 1+0*x,
                              domain=(0., 1., 2., 3., 4., 6.))
    result = f.lap()
    assert tuple(result.domain) == (-1., 1., -1., 1., -1., 1.)
    assert float(result(0., 0., 0.)) == 0.


def test_empty_laplacian_is_zero_function():
    result = Chebfun3.empty().laplacian()
    assert not result.isempty()
    assert tuple(result.domain) == (-1., 1., -1., 1., -1., 1.)
    assert float(result(0., 0., 0.)) == 0.
