"""Analytic contracts for source Cartesian surface lighting.

Provenance
----------
MATLAB R2025b graph3d/surfnorm.m, material.m and camlight.m.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax._surface_lighting import dull_headlight_rgb, surface_vertex_normals


def grid():
    return jnp.meshgrid(jnp.linspace(-1., 1., 5), jnp.linspace(-1., 1., 7))


@pytest.mark.parametrize('compiled', [False, True])
def test_plane_orientation(compiled):
    x, y = grid()
    normal = jnp.array([-2., 3., 1.])/jnp.sqrt(14.)
    fun = jax.jit(surface_vertex_normals) if compiled else surface_vertex_normals
    actual = fun(x, y, 2*x-3*y+1)
    assert float(jnp.max(jnp.abs(actual-normal))) < 2e-15
    reversed_normal = fun(x[:, ::-1], y[:, ::-1], (2*x-3*y+1)[:, ::-1])
    assert float(jnp.max(jnp.abs(reversed_normal+normal))) < 2e-15


def test_quadratic_boundary_normals():
    x, y = grid()
    z = x*x+2*y*y
    exact = jnp.stack((-2*x, -4*y, jnp.ones_like(x)), axis=-1)
    exact = exact/jnp.linalg.norm(exact, axis=-1, keepdims=True)
    assert float(jnp.max(jnp.abs(surface_vertex_normals(x, y, z)-exact))) < 2e-15


def test_degenerate_surface():
    z = jnp.zeros((3, 4))
    assert bool(jnp.all(surface_vertex_normals(z, z, z) == 0))


def test_shape_contracts():
    with pytest.raises(ValueError, match='3 by 3'):
        surface_vertex_normals(jnp.zeros((2, 3)), jnp.zeros((2, 3)), jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match='matching'):
        surface_vertex_normals(jnp.zeros((3, 3)), jnp.zeros((3, 4)), jnp.zeros((3, 3)))


@pytest.mark.parametrize('compiled', [False, True])
def test_local_headlight_material_and_backface(compiled):
    x = jnp.array([[0., 3.]])
    y, z = jnp.zeros_like(x), jnp.zeros_like(x)
    normals = jnp.array([[[0., 0., 1.], [0., 0., -1.]]])
    rgb = jnp.full((1, 2, 3), .5)
    camera = jnp.array([0., 0., 4.])
    fun = jax.jit(dull_headlight_rgb, static_argnames=('backface',)) if compiled else dull_headlight_rgb
    lit = fun(rgb, normals, x, y, z, camera, backface='lit')
    reversed_lit = fun(rgb, normals, x, y, z, camera, backface='reverselit')
    assert float(jnp.max(jnp.abs(lit-jnp.array([[[.55]*3, [.15]*3]])))) < 2e-16
    assert float(jnp.max(jnp.abs(reversed_lit-jnp.array([[[.55]*3, [.47]*3]])))) < 2e-16


def test_no_specular_and_bounded_output():
    z = jnp.zeros((1, 1))
    normals = jnp.array([[[0., 0., 1.]]])
    black = dull_headlight_rgb(jnp.zeros((1, 1, 3)), normals, z, z, z, jnp.array([0., 0., 1.]))
    white = dull_headlight_rgb(jnp.ones((1, 1, 3)), normals, z, z, z, jnp.array([0., 0., 1.]))
    assert bool(jnp.all(black == 0))
    assert bool(jnp.all(white == 1))
