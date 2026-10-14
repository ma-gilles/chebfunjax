"""Port of the six literal predicates in MATLAB tests/chebfun3/test_root.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_root.m; @chebfun3/root.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_root_guide_example_source_predicates(record_property):
    pref = ChebfunPref()
    tol = 10 * pref.cheb3Prefs.chebfun3eps
    f = Chebfun3.from_function(lambda x, y, z: y - x ** 2)
    g = Chebfun3.from_function(lambda x, y, z: z - x ** 3)
    h = Chebfun3.from_function(
        lambda x, y, z: jnp.cos(jnp.exp(x * jnp.sin(-2 + y + z))))
    r = Chebfun3.root(f, g, h)
    assert float(f(r[0], r[1], r[2])) < tol  # pass(1)
    assert float(g(r[0], r[1], r[2])) < tol  # pass(2)
    assert float(h(r[0], r[1], r[2])) < tol  # pass(3)
    # Supplemental prior-Python bound, not a native predicate. Record actual
    # errors independently of the threshold for later review.
    residuals = tuple(float(q(r[0], r[1], r[2])) for q in (f, g, h))
    for name, residual in zip(("f", "g", "h"), residuals):
        record_property(f"guide_{name}_absolute_residual", abs(residual))
        assert abs(residual) < 1e3 * tol


def test_root_trivial_example_source_predicates(record_property):
    tol = 1e-2
    f = Chebfun3.from_function(lambda x, y, z: x ** 2 - tol)
    g = Chebfun3.from_function(lambda x, y, z: y ** 2 - tol)
    h = Chebfun3.from_function(lambda x, y, z: z ** 2 - tol)
    r = Chebfun3.root(f, g, h)
    assert float(f(r[0], r[1], r[2])) < tol  # pass(4)
    assert float(g(r[0], r[1], r[2])) < tol  # pass(5)
    assert float(h(r[0], r[1], r[2])) < tol  # pass(6)
    # Supplemental prior-Python location check, not a native predicate.
    distance = float(jnp.max(jnp.abs(jnp.abs(r) - 0.1)))
    record_property("trivial_root_max_distance_from_signed_0p1", distance)
    assert distance < 1e-8
