"""Original tests/chebfun/test_roots.m clauses11–16.

Provenance
----------
MATLAB source: tests/chebfun/test_roots.m
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("clause", [11, 12, 13])
def test_original_unbounded_roots(clause):
    if clause == 11:
        f = cj.chebfun(lambda x: (1-jnp.exp(-x**2))/x,
                       domain=(-jnp.inf, jnp.inf))
        exact = 0.
    elif clause == 12:
        f = cj.chebfun(lambda x: x**2*(1-jnp.exp(-x**2))-2,
                       domain=(-jnp.inf, jnp.inf), exps=(2., 2.))
        exact = np.array([-1.4962104914103104707, 1.4962104914103104707])
    else:
        f = cj.chebfun(lambda x: .15+jnp.sin(10*x)/jnp.exp(x),
                       domain=(0., jnp.inf))
        exact = np.array([.33529141416564289113, .60061694515161002799,
                          .98375750309184861332, 1.2042605667187311146,
                          1.6619482204330474390, 1.7760894757659030239])
    r = np.asarray(f.roots())
    if clause == 11:
        r = r[np.isfinite(r)]
    err = r-exact
    error = np.max(np.abs(err)) if err.size else 0.
    tol = 100*np.finfo(float).eps*float(f.vscale)
    print("clause", clause, "roots", r.tolist(), "error", error, "tol", tol)
    assert error < tol


@pytest.mark.parametrize("scale", [1., 1e50, 1e-50])
def test_original_scale_invariance_14_16(scale):
    multiplier = {1.: 1., 1e50: 1e-50, 1e-50: 1e50}[scale]
    f = cj.chebfun(lambda x: jnp.exp(multiplier*x), domain=(-scale, scale))
    r = np.asarray((f-1).roots())
    eps = np.finfo(float).eps
    assert r.shape == (1,)
    assert np.all(np.abs(r-scale*eps) < 10*scale*eps)
