"""All six original minandmax2est predicates, Chebfun commit7574c77.

Provenance
----------
MATLAB source: tests/chebfun2v/test_minandmax2est.m
Chebfun commit:7574c77
The native empty array maps to the public Python empty tuple.
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

TOL = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
DOMAIN = (-2., -1., 3., 6.)


class TestChebfun2vMinandmax2est:
    def test_empty_box(self):
        assert len(Chebfun2v.empty().minandmax2est()) == 0

    def test_one_component_box(self):
        field = Chebfun2v.from_functions(lambda x, y: x, domain=DOMAIN)
        box = field.minandmax2est()
        assert len(box) == 2
        assert jnp.linalg.norm(jnp.asarray(box)-jnp.array([-2., -1.])) < TOL

    def test_two_components_box(self):
        field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                         domain=DOMAIN)
        box = field.minandmax2est()
        assert len(box) == 4
        assert jnp.linalg.norm(jnp.asarray(box)-jnp.asarray(DOMAIN)) < TOL

    def test_three_components_box(self):
        field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                         lambda x, y: x+y, domain=DOMAIN)
        assert len(field.minandmax2est()) == 6
