"""Default/orientation contracts from native separableApprox mean/std.

MATLAB sources: @separableApprox/mean.m, @separableApprox/std.m, sum.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
This supplemental source contract does not replace any literal native predicate.
"""
from chebfunjax.chebfun2d.chebfun2 import Chebfun2


def test_native_defaults_and_return_orientation():
    f = Chebfun2.from_function(lambda x, y: y**2 + x)
    assert f.mean().is_transposed
    assert f.mean(1).is_transposed
    assert not f.mean(2).is_transposed
    assert float((f.mean() - f.mean(1)).norm()) == 0
    assert not f.std().is_transposed
    assert not f.std(None, 1).is_transposed
    assert not f.std(None, 2).is_transposed
    # Native ignores the flag; [] is represented by None in the Python API.
    assert float((f.std(1, 1) - f.std(None, 1)).norm()) == 0
