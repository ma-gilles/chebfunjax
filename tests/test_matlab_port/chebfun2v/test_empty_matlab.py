"""All19 empty operations from tests/chebfun2v/test_empty.m, pin7574c77.

Provenance
----------
MATLAB source: tests/chebfun2v/test_empty.m.
MATLAB elementwise multiplication maps to *, matrix multiplication to @.
The original single predicate requires all19 calls to complete without error.
"""
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


def test_all_native_empty_operations():
    f = Chebfun2v.empty()
    f.conj()
    f.cross(f)
    f.transpose()
    f.curl()
    f.divergence()
    f.dot(f)
    f(1, 1)
    f.imag()
    f.isempty()
    f.laplacian()
    f - f
    f * f
    f.norm()
    f + f
    f**2
    f.real()
    f.roots()
    f @ f
    f.ctranspose()
