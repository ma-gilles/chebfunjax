"""Original disk Helmholtz coefficient-input assertion (source case 8).

Provenance: tests/diskfun/test_helmholtz.m, Chebfun commit 7574c77.
Retains source m=100 and actual disk L2 bound 2000 eps.
Other source clauses, including the unresolved k=7 failure, are separate.
"""
import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 2e3*jnp.finfo(jnp.float64).eps


def test_source_8_coefficient_rhs():
    k=jnp.sqrt(2.)
    exact=Diskfun.from_function(lambda t,r:jnp.cos(r**5*jnp.sin(5*t))-r**2)
    rhs=(exact.laplacian()+k*k*exact).coeffs2()
    actual=Diskfun.helmholtz(rhs,k,lambda t:exact(t,jnp.ones_like(t)),m=100)
    assert float((actual-exact).norm()) < TOL
