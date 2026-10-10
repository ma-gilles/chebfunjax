"""Complete original eig assertions: native gallery and continuous residual.

MATLAB source: tests/chebfun2/test_eig.m, +cheb/gallery2.m, @chebfun2/eig.m
Chebfun commit: 7574c77
Python full=True adapts MATLAB two-output eig; @ is native operator product.
The native domain-test function's unused third argument is omitted in Python.
"""
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.gallery2 import gallery2


class TestChebfun2Eig:
    def test_pass1_eigen_residual(self):
        f = gallery2('challenge')
        functions, diagonal = f.eig(full=True)
        tolerance = 1e4*ChebfunPref().cheb2Prefs.chebfun2eps
        assert float((f @ functions-functions @ diagonal).norm()) < tolerance

    def test_pass2_domain_error(self):
        f = chebfun2(lambda x, y: x+y, domain=(-1., 1., -2., 2.))
        with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN2:eig:domainerr:'):
            f.eig()
