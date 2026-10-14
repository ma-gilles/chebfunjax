"""The nine original assertions in MATLAB tests/spherefun/test_feval.m.

The source seeds MATLAB Twister with rng(9), then checks angular scalar,
column, row, matrix, and meshgrid evaluation, Cartesian evaluation, and the
source off-sphere error identifier.

Provenance
----------
MATLAB source : tests/spherefun/test_feval.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils._matlab_twister import init_matlab_twister, matlab_rand_uniform

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


def _source_rand_points():
    state = init_matlab_twister(9)
    out = {}
    for name, shape in (
        ("lambda_scalar", ()), ("theta_scalar", ()),
        ("lambda_column", (10, 1)), ("theta_column", (10, 1)),
        ("lambda_row", (1, 10)), ("theta_row", (1, 10)),
        ("lambda_matrix", (2, 10)), ("theta_matrix", (2, 10)),
        ("mesh_axis", (3, 1)), ("cart_scalar", (1, 3)),
        ("cart_rows", (10, 3)), ("cart_x", (10, 3)),
        ("cart_y", (10, 3)), ("cart_z", (10, 3)),
    ):
        out[name], state = matlab_rand_uniform(state, shape)
    return out


def _make_source_function():
    return sph_xyz(lambda x, y, z: jnp.sin(x + y*z) + 1)


def _source_value(lam_arg, theta_arg):
    x = jnp.cos(lam_arg)*jnp.sin(theta_arg)
    y = jnp.sin(lam_arg)*jnp.sin(theta_arg)
    z = jnp.cos(theta_arg)
    return jnp.sin(x + y*z) + 1


def test_source_angular_feval_assertions_one_through_five():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    g = _make_source_function()
    p = _source_rand_points()

    # pass(1); the MATLAB source calls g(theta, lambda).
    lam, theta = p["theta_scalar"], p["lambda_scalar"]
    assert abs(float(g(lam, theta) - _source_value(lam, theta))) < tol

    # pass(2): column-vector infinity norm.
    lam, theta = p["theta_column"], p["lambda_column"]
    err = g(lam, theta) - _source_value(lam, theta)
    assert float(jnp.linalg.norm(err, ord=jnp.inf)) < tol

    # pass(3): MATLAB norm(row) is the spectral/Euclidean 2-norm.
    lam, theta = p["theta_row"], p["lambda_row"]
    err = g(lam, theta) - _source_value(lam, theta)
    assert float(jnp.linalg.norm(err, ord=2)) < tol

    # pass(4): matrix infinity norm.
    lam, theta = p["theta_matrix"], p["lambda_matrix"]
    err = g(lam, theta) - _source_value(lam, theta)
    assert float(jnp.linalg.norm(err, ord=jnp.inf)) < tol

    # pass(5): meshgrid from the same rand(3,1) vector in both dimensions.
    axis = p["mesh_axis"].reshape((3,))
    theta, lam = jnp.meshgrid(axis, axis, indexing="xy")
    err = g(lam, theta) - _source_value(lam, theta)
    assert float(jnp.linalg.norm(err, ord=jnp.inf)) < tol


def test_source_cartesian_scalar_feval_assertion_six():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    g = _make_source_function()
    xyz = _source_rand_points()["cart_scalar"].reshape((3,))
    xyz = xyz / jnp.sqrt(jnp.sum(xyz**2))
    x, y, z = xyz
    expected = jnp.sin(x + y*z) + 1
    assert abs(float(g(x, y, z) - expected)) < tol


def test_source_cartesian_vector_feval_assertions_seven_and_eight():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    g = _make_source_function()
    p = _source_rand_points()

    # pass(7): normalized rows of rand(10,3).
    xyz = p["cart_rows"]
    xyz = xyz / jnp.sqrt(jnp.sum(xyz**2, axis=1, keepdims=True))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    expected = jnp.sin(x + y*z) + 1
    assert float(jnp.linalg.norm(g(x, y, z) - expected, ord=jnp.inf)) < tol

    # pass(8): independently drawn x/y/z matrices, normalized elementwise.
    x, y, z = p["cart_x"], p["cart_y"], p["cart_z"]
    radius = jnp.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/radius, y/radius, z/radius
    expected = jnp.sin(x + y*z) + 1
    assert float(jnp.linalg.norm(g(x, y, z) - expected, ord=jnp.inf)) < tol


def test_source_off_sphere_error_assertion_nine():
    g = _make_source_function()
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:FEVAL:pointsNotOnSphere"):
        g(1.0, 1.0, 1.0)


def test_supplemental_matrix_radius_matches_source_any_if_semantics():
    """MATLAB `any(rad > limit)` plus `if` has columnwise matrix semantics."""
    g = _make_source_function()
    # any(A) reduces down columns; a nonscalar if condition is true only if
    # every remaining entry is nonzero. Thus one offending column is
    # insufficient to throw when another column has no offending radius.
    x = jnp.array([[1.0001, 0.5], [0.5, 0.5]])
    y = jnp.zeros_like(x)
    z = jnp.zeros_like(x)
    assert jnp.asarray(g(x, y, z)).shape == x.shape

    # Each column now contains an offending radius, making the source if
    # condition true and raising the source identifier.
    x = jnp.array([[1.0001, 0.5], [0.5, 1.0001]])
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:FEVAL:pointsNotOnSphere"):
        g(x, y, z)
