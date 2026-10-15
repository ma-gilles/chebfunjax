"""All 22 predicates from the original three-component vector-field test.

Python ``*``/``@`` distinguish MATLAB TIMES/MTIMES; nested numeric lists
preserve the source row/column dimensions. Scalar SeparableApprox returns
are wrapped without recomputing their representation.

Provenance
----------
MATLAB source: tests/chebfun2v/test_threecomponents.m
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


class TestChebfun2vThreeComponents:
    def test_construction_equivalence(self):
        for domain in (None, (-1., 1., -1., 1.)):
            kwargs = {} if domain is None else {"domain": domain}
            f = chebfun2(lambda x, y: x, **kwargs)
            f1 = Chebfun2v.from_functions(f, f, f, **kwargs)
            f2 = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: x, lambda x, y: x, **kwargs)
            assert float((f1-f2).norm()) < TOL  # predicates 1,2

    def test_original_arithmetic(self):
        f = chebfun2(lambda x, y: x, domain=(-1., 1., -1., 1.))
        f1 = Chebfun2v.from_functions(f, f, f)
        g = Chebfun2v.from_functions(f, 2*f, 3*f)
        p = np.pi/6
        v = np.array([1., 2., 3.])
        assert np.linalg.norm(_ev(f1+g, 1., 1.)-[2., 3., 4.]) < TOL  # 3
        assert np.linalg.norm(_ev(g+1, p, 1.)-p*v-1) < TOL  # 4
        assert np.linalg.norm(_ev(g+[1, 2, 3], p, 1.)-p*v-v) < TOL  # 5
        assert np.linalg.norm(_ev(1+g, p, 1.)-p*v-1) < TOL  # 6
        assert np.linalg.norm(_ev([1, 2, 3]+g, p, 1.)-p*v-v) < TOL  # 7
        assert np.linalg.norm(_ev(g*1, p, 1.)-_ev(g, p, 1.)) < TOL  # 8
        assert np.linalg.norm(_ev(g*[[1], [2], [3]], p, 1.)-p*np.array([1., 4., 9.])) < TOL  # 9
        assert np.linalg.norm(_ev(1*g, p, 1.)-_ev(g, p, 1.)) < TOL  # 10
        assert np.linalg.norm(_ev([[1], [2], [3]]*g, p, 1.)-p*np.array([1., 4., 9.])) < TOL  # 11
        assert np.linalg.norm(_ev(g @ 1, p, 1.)-_ev(g, p, 1.)) < TOL  # 12
        assert float(((g.ctranspose() @ [[1], [2], [3]])-(f+4*f+9*f)).norm()) < TOL  # 13
        assert np.linalg.norm(_ev(1 @ g, p, 1.)-_ev(g, p, 1.)) < TOL  # 14
        with pytest.raises(ValueError):  # 15: source deliberately invalid MTIMES
            [[1], [2], [3]] @ g

    def test_original_vector_calculus(self):
        coordinate = chebfun2(lambda x, y: x)
        f1 = Chebfun2v.from_functions(coordinate, coordinate, coordinate)
        g = Chebfun2v.from_functions(coordinate, 2*coordinate, 3*coordinate)
        f = chebfun2(lambda x, y: jnp.sin(x*y))
        assert float(((f1+g).curl()-(f1.curl()+g.curl())).norm()) < 100*TOL  # 16
        assert float((_scalar((f*g).divergence())
                      - (_scalar(g.dot(f.grad().vertcat(0)))
                         + f*_scalar(g.divergence()))).norm()) < 100*TOL  # 17
        assert float(_scalar(g.curl().divergence()).norm()) < 10*TOL  # 18
        assert float((_scalar(f.grad().divergence())-f.laplacian()).norm()) < 10*TOL  # 19
        expected = _scalar(g.divergence()).grad().vertcat(0)-g.laplacian()
        assert float((g.curl().curl()-expected).norm()) < 10*TOL  # 20

    def test_original_torus_boundaries(self):
        domain = (0., 2*np.pi, 0., 2*np.pi)
        u = chebfun2(lambda u, v: u, domain=domain)
        v = chebfun2(lambda u, v: v, domain=domain)
        x = (3+v.cos())*u.cos()
        y = (3+v.cos())*u.sin()
        z = v.sin()
        r = Chebfun2v.from_functions(x, y, z)
        n = r.normal()
        r1, r2 = r.diff(1, 1), r.diff(1, 2)
        g = x.diff(dim=1, k=1)*r1 + x.diff(dim=2, k=1)*r2
        velocity = n.cross(g)
        # Python components[1] adapts native V(2), without resampling.
        f1 = _scalar(velocity.components[1]).diff(dim=2, k=1)
        product = f1*r.diff(1, 2)
        fx = _scalar(product.components[0])
        assert float(fx(2*np.pi, ":").norm()) < 10*TOL  # 21
        assert float(fx(0., ":").norm()) < 10*TOL  # 22
