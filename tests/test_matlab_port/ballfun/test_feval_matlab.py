"""Evaluation predicates from tests/ballfun/test_feval.m, Chebfun 7574c77.

Native clauses 1--9 use their original operands and 1e4*eps threshold.
Cartesian calls use feval, and radial slices use to_spherefun, explicit Python
adapters. Clause 10 has no tolerance in MATLAB and tests a nonzero residual;
we retain that literal predicate separately from a supplemental accuracy check.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils.quadrature import chebpts, trigpts

from ._helpers import EPS

TOL = 1e4 * EPS


@pytest.fixture(scope="module")
def spherical_x():
    return Ballfun.from_function(lambda r, lam, th: r*jnp.cos(lam)*jnp.sin(th), spherical=True)


@pytest.mark.parametrize("r,lam,th,exact", [
    ([.5, .7], [0, 0], [np.pi/2, np.pi/2], [.5, .7]),
    ([1, 1], [np.pi/4, np.pi/3], [np.pi/2, np.pi/2], [np.cos(np.pi/4), np.cos(np.pi/3)]),
    ([1, 1], [0, 0], [np.pi/5, np.pi/7], [np.sin(np.pi/5), np.sin(np.pi/7)]),
])
def test_spherical_pointwise(spherical_x, r, lam, th, exact):
    got = spherical_x.feval(r, lam, th, "spherical")
    assert got.shape == (2,)
    assert np.linalg.norm(np.asarray(got).ravel() - np.asarray(exact).ravel()) < TOL


def test_tensor_grid_axis_order():
    f = Ballfun.from_function(lambda x, y, z: x*y)
    sizes = (22, 23, 25)
    r = chebpts(sizes[0])
    lam = np.pi * trigpts(sizes[1])[0]
    th = np.pi * trigpts(sizes[2])[0]
    exact = Ballfun.coeffs2vals(f.coeffs3(*sizes))
    got = f.fevalm(r, lam, th)
    assert got.shape == exact.shape == sizes
    assert np.linalg.norm(np.asarray(got - exact).ravel()) < TOL


@pytest.mark.parametrize("axis,point,want", [
    (0, (1, 0, 0), 1), (1, (0, .5, 0), .5), (2, (0, 0, -.3), -.3),
])
def test_cartesian(axis, point, want):
    f = Ballfun.from_function(lambda x, y, z: (x, y, z)[axis])
    assert abs(want - f.feval(*point)) < TOL
    assert abs(want - f(*point, coord="cartesian")) < TOL


def test_surface_slice(spherical_x):
    h = Spherefun.from_function(lambda lam, th: jnp.cos(lam)*jnp.sin(th))
    assert (spherical_x.to_spherefun(1) - h).norm() < TOL


def test_interior_slice():
    f = Ballfun.from_function(lambda r, lam, th: r*jnp.cos(th), spherical=True)
    h = Spherefun.from_function(lambda lam, th: .5*jnp.cos(th))
    assert (f.to_spherefun(.5) - h).norm() < TOL


def test_complex_literal_predicate():
    f = Ballfun.from_function(lambda x, y, z: jnp.cos(x)*1j)
    residual = abs(f.feval(0, 0, 0) - 1j)
    print("native_clause_10_residual", float(residual))
    assert bool(residual)  # Literal native pass(10), not an accuracy assertion.
    assert residual < TOL  # Additional correctness control.


def test_feval_contract_and_jit(spherical_x):
    with pytest.raises(ValueError, match="same dimension"):
        spherical_x.feval([.5, .7], [0], [0])
    with pytest.raises(ValueError, match="unit ball"):
        spherical_x.feval(1.1, 0, 0)
    assert Ballfun.empty().feval([], [], []).size == 0
    got = jax.jit(lambda r: spherical_x.feval(r, jnp.zeros_like(r), jnp.full_like(r, jnp.pi/2), "spherical"))(jnp.array([.5, .7]))
    assert np.linalg.norm(np.asarray(got) - [.5, .7]) < TOL
