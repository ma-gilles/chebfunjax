"""All eight clauses of the pinned MATLAB CF test, without altered bounds.

Provenance
----------
MATLAB source : tests/chebfun/test_cf.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Quasimatrix conversion uses existing extract_columns/cell2quasi and scalar
transpose because public cheb2quasi/Quasimatrix.T aliases are absent.
MATLAB cell r maps to nested tuples; s retains literal row/column shape.
"""
import jax.numpy as jnp

from chebfunjax import cell2quasi, cf
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain


def make(function, domain=(-1.,1.)):
    return Chebfun.from_function(function, domain=Domain(domain))


def test_clause1_exp_polynomial_error():
    f = make(jnp.exp)
    p,q,r,lam = cf(f,2)
    assert abs(lam-.045017)<1e-4


def test_clause2_mapped_exp_polynomial_error():
    f = make(lambda x:jnp.exp(-1+x/2),(0.,4.))
    p,q,r,lam = cf(f,2)
    assert abs(lam-.045017)<1e-4


def test_clause3_cosine_degree_adjustment():
    x = make(lambda x:x)
    f = x.cos()
    p,q,r,lam = cf(f,1,1)
    assert abs(p(.3)-.77015046914)<1e-4


def test_clause4_abs_continuous_l2():
    x = make(lambda x:x)
    f = abs(x)
    p,q,r,lam = cf(f,4,4,100)
    assert (f-p/q).norm()<.05


def test_clause5_exp_exp_vector_two_norm():
    x = make(lambda x:x)
    f = x.exp().exp()
    p,q,r,lam = cf(f,0,10)
    xx = jnp.linspace(-1.,1.,17)
    assert jnp.linalg.norm(f(xx)-r(xx),ord=2)<1e-4


def test_clause6_noncanonical_domain_infinity_norm():
    f = make(jnp.exp,(2.,6.))
    p,q,r,lam = cf(f,5,5)
    xx = jnp.linspace(2.,6.,100)
    assert jnp.linalg.norm(f(xx)-r(xx),ord=jnp.inf)<1e-6


def array_input():
    return make(lambda x:jnp.stack((jnp.sin(x),jnp.cos(x),jnp.exp(x)),axis=-1))


def test_clause7_array_output_counts():
    p,q,r,s = cf(array_input(),2,2)
    assert p.n_cols==3 and q.n_cols==3
    assert sum(len(row) for row in r)==3 and s.size==3


def test_clause8_transposed_quasimatrix_output_orientation():
    f = array_input()
    f = cell2quasi([f.extract_columns(i).transpose() for i in range(3)])
    p,q,r,s = cf(f,2,2)
    assert p.n_cols==3 and q.n_cols==3
    assert sum(len(row) for row in r)==3 and s.size==3
    assert p[0].is_transposed and q[0].is_transposed
    assert len(r)==3 and all(len(row)==1 for row in r) and s.shape==(3,1)
