"""Original six predicates from MATLAB test_roots07, without substitutions.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots07.m
Chebfun commit: 7574c77
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL


def test_circle_hyperbola_algorithm_agreement():
    f = chebfun2(lambda x, y: (x**2 + y**2 - 1)*(x - 1.1))
    g = chebfun2(lambda x, y: (25*x*y - 12)*(x - 1.1))
    r1 = np.asarray(f.roots(g, method="ms"))
    r2 = np.asarray(f.roots(g, method="resultant"))
    assert r1.shape == r2.shape
    assert np.linalg.norm(np.sort(r1[:, 0]) - np.sort(r2[:, 0])) < TOL
    assert np.linalg.norm(np.sort(r1[:, 1]) - np.sort(r2[:, 1])) < TOL


def test_polynomial_resultant_residuals(record_property):
    f = chebfun2(lambda x, y: y**4 - y**3 + 2*x**2*y**2 + 3*x**2*y + x**4)

    def gfun(x, y):
        u, v = 2*x, 2*(y + .5)
        return v**10 - 2*u**8*v**2 + 4*u**4*v - 2

    g = chebfun2(gfun)
    r = jnp.asarray(f.roots(g, method="resultant"))
    record_property("returned_root_count", int(r.shape[0]))
    assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < TOL
    assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < 1000*TOL
    # Additional native set check at the algorithm's own collapse radius.
    # The nearly repeated roots are ill conditioned: this is a root-set check,
    # not a claim of machine-precision coordinate agreement.
    reference = json.loads((Path(__file__).parents[2] / "references/roots07_native.json").read_text())
    native = jnp.asarray(reference["roots"][1])
    assert r.shape == native.shape
    distances = jnp.linalg.norm(r[:, None, :] - native[None, :, :], axis=2)
    closest = jnp.argmin(distances, axis=1)
    assert jnp.unique(closest).size == native.shape[0]
    error = float(jnp.max(jnp.min(distances, axis=1)))
    record_property("native_set_max_coordinate_error", error)
    assert error < 10*jnp.sqrt(jnp.finfo(jnp.float64).eps)


def test_tiny_domain_resultant_residuals(record_property):
    a = 1e-9
    rect = a*np.array([-1., 1., -1., 1.])
    f = chebfun2(lambda x, y: jnp.cos(x*y/a**2) + jnp.sin(3*x*y/a**2), rect)
    g = chebfun2(lambda x, y: jnp.cos(y/a) - jnp.cos(2*x*y/a**2), rect)
    r = jnp.asarray(f.roots(g, method="resultant"))
    record_property("returned_root_count", int(r.shape[0]))
    assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < 1000*TOL
    assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < 1000*TOL

    # Additional completeness/coordinate regression: native R2025b returns these
    # two analytic roots. Original residual predicates above allow empty output.
    assert r.shape == (2, 2)
    expected = jnp.array([[-.5, jnp.pi/4], [.5, -jnp.pi/4]])
    normalized = r[jnp.argsort(r[:, 0])] / a
    record_property("normalized_coordinate_error", float(jnp.linalg.norm(normalized - expected)))
    assert float(jnp.linalg.norm(normalized - expected)) < TOL
