"""Literal Carrier source inputs and predicates; no fixed-grid substitute.

Provenance
----------
MATLAB source: tests/chebop/test_carrier_C1.m, test_carrier_C2.m,
    test_carrier_US.m, @cheboppref/cheboppref.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


def run_original_carrier(backend, record_property):
    dom = (-1., 1.)
    # Source C1/C2 explicitly set bvpTol1e-10. US retains factory5e-13.
    bvp_tol = 5e-13 if backend == 'ultraS' else 1e-10
    tolerance = 100*bvp_tol if backend == 'ultraS' else 1e-10
    op = Chebop(lambda x, u: .01*u.diff(2)+2*(1-x**2)*u+u**2-1, dom)
    op.bc = lambda x, u: [u(-1.), u(1.)]
    x = chebfun(lambda x: x, domain=dom)
    op.init = 2*(x**2-1)*(1-2/(1+20*x**2))
    u, _ = op.solvebvp(0., tol=bvp_tol, discretization=backend)
    xx = jnp.arange(-4, 5, dtype=jnp.float64)/4
    reference = jnp.asarray([
        0., -1.487429807540814, -1.785617248281071, 1.572366197526305,
        -1.539652044363185, 1.572366197526230, -1.785617248281089,
        -1.487429807540795, 0.])
    errors = [float(jnp.linalg.norm(u(xx)-reference)),
              float(jnp.linalg.norm(u(jnp.asarray([-1., 1.]))))]
    record_property('source_backend', backend)
    record_property('source_bvpTol', bvp_tol)
    record_property('source_bound', tolerance)
    for slot, error in enumerate(errors, 1):
        record_property(f'source_carrier_error_{slot}', error)
    assert all(error < tolerance for error in errors), errors
