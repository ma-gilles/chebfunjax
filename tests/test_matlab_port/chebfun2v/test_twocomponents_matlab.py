"""All 19 predicates from the original two-component vector-field test.

Python ``*``/``@`` distinguish MATLAB TIMES/MTIMES; nested numeric lists
preserve the source row/column dimensions. Scalar SeparableApprox returns
are wrapped without recomputing their representation.

Provenance
----------
MATLAB source: tests/chebfun2v/test_twocomponents.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

TOL = 1e5 * ChebfunPref().cheb2Prefs.chebfun2eps


def _scalar(f):
    return f if isinstance(f, Chebfun2) else Chebfun2(approx=f)


def _ev(f, x, y):
    return np.asarray(f(jnp.asarray(x), jnp.asarray(y))).reshape(-1)


class TestChebfun2vTwocomponents:
    def test_construction_equivalence(self):
        for domain in (None, (-1., 1., -1., 1.)):
            kwargs = {} if domain is None else {"domain": domain}
            f = chebfun2(lambda x, y: x, **kwargs)
            f1 = Chebfun2v.from_functions(f, f, **kwargs)
            f2 = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: x, **kwargs)
            assert float((f1-f2).norm()) < TOL  # predicates 1,2

    def test_original_arithmetic(self):
        f = chebfun2(lambda x, y: x, domain=(-1., 1., -1., 1.))
        f1 = Chebfun2v.from_functions(f, f)
        g = Chebfun2v.from_functions(f, 2*f)
        p = np.pi/6
        v = np.array([1., 2.])
        assert np.linalg.norm(_ev(f1+g, 1., 1.)-[2., 3.]) < TOL  # 3
        assert np.linalg.norm(_ev(g+1, p, 1.)-p*v-1) < TOL  # 4
        assert np.linalg.norm(_ev(g+[1, 2], p, 1.)-p*v-v) < TOL  # 5
        assert np.linalg.norm(_ev(1+g, p, 1.)-p*v-1) < TOL  # 6
        assert np.linalg.norm(_ev([1, 2]+g, p, 1.)-p*v-v) < TOL  # 7
        assert np.linalg.norm(_ev(g*1, p, 1.)-_ev(g, p, 1.)) < TOL  # 8
        assert np.linalg.norm(_ev(g*[[1], [2]], p, 1.)-p*np.array([1., 4.])) < TOL  # 9
        assert np.linalg.norm(_ev(1*g, p, 1.)-_ev(g, p, 1.)) < TOL  # 10
        assert np.linalg.norm(_ev([[1], [2]]*g, p, 1.)-p*np.array([1., 4.])) < TOL  # 11
        assert np.linalg.norm(_ev(g @ 1, p, 1.)-_ev(g, p, 1.)) < TOL  # 12
        assert float(((g.ctranspose() @ [[1], [2]])-(f+4*f)).norm()) < TOL  # 13
        assert np.linalg.norm(_ev(1 @ g, p, 1.)-_ev(g, p, 1.)) < TOL  # 14
        with pytest.raises(ValueError):  # 15: source deliberately invalid MTIMES
            [[1], [2]] @ g

    def test_original_vector_calculus(self):
        coordinate = chebfun2(lambda x, y: x)
        f1 = Chebfun2v.from_functions(coordinate, coordinate)
        g = Chebfun2v.from_functions(coordinate, 2*coordinate)
        f = chebfun2(lambda x, y: jnp.sin(x*y))
        assert float((_scalar((f1+g).curl())
                      - (_scalar(f1.curl())+_scalar(g.curl()))).norm()) < 100*TOL  # 16
        assert float((_scalar((f*g).divergence())
                      - (_scalar(g.dot(f.grad()))+f*_scalar(g.divergence()))).norm()) < 100*TOL  # 17
        curl = _scalar(g.curl())
        assert float((curl.diff(dim=1, k=1)+curl.diff(dim=2, k=1)).norm()) < 10*TOL  # 18
        assert float((_scalar(f.grad().divergence())-f.laplacian()).norm()) < 10*TOL  # 19
