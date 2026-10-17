"""Even-function extrema controls with a scoped pinned-native oracle.

Chebfun 7574c77 @spherefun/minandmax2.m: rank-one four-product selection,
column-major seed, active-set refinement and final BMC mapping. R2025b
captures in sphere_extrema_tie_20261017 show the same maximum antipode.
These are authored controls, not additional original MATLAB predicates.
No canonical antipode rule is imposed for arbitrary even functions.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d._extrema_fallback import original_objective
from chebfunjax.spherefun._extrema_source import _reconstruct, _seed
from chebfunjax.spherefun.spherefun import Spherefun

# Unchanged original test_optimization.m default value bound.
TOL = 5e4 * np.finfo(float).eps


def _xyz(location):
    lam, theta = location
    return jnp.array([jnp.cos(lam)*jnp.sin(theta),
                      jnp.sin(lam)*jnp.sin(theta), jnp.cos(theta)])


def test_rank_one_even_maximum_native_location():
    f = Spherefun.from_function(lambda lam, theta:
                               (jnp.cos(lam)*jnp.sin(theta))**2)
    value, location = f.max2()
    assert abs(float(value)-1) < TOL
    assert abs(float(f(*location))-float(value)) < TOL
    # Captured public native max2: lambda=0, theta=pi/2.
    assert float(jnp.max(jnp.abs(location-jnp.array([0., jnp.pi/2])))) < TOL


def test_rotated_even_maximum_native_seed_and_antipode():
    f = Spherefun.from_function(lambda lam, theta:
        (jnp.cos(lam)*jnp.sin(theta)
         + .3*jnp.sin(lam)*jnp.sin(theta)+.7*jnp.cos(theta))**2)
    g, rows, cols = _reconstruct(f)
    _, seeds = _seed(rows, cols, original_objective(g.approx))
    # Exact retained native discrete seed, before optimizer or BMC mapping.
    np.testing.assert_array_equal(np.asarray(seeds), [
        [0.6283185307179586, -0.6283185307179586],
        [-3.141592653589793, 1.8849555921538759]])
    value, location = f.max2()
    xyz = _xyz(location)
    axis = jnp.array([1., .3, .7])
    assert abs(float(value)-1.58) < TOL
    assert abs(float(f(*location))-float(value)) < TOL
    assert abs(float(jnp.dot(axis, xyz)**2)-1.58) < TOL
    # Native selects the negative-axis maximizer. The sign separates the
    # antipodes without claiming equal optimizer coordinates to all digits.
    assert float(jnp.dot(axis, xyz)) < 0
