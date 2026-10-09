"""Independent source-policy controls for both polynomial technologies.

MATLAB @chebtech/minandmax.m, pin7574c77680d7e82b79626300bf255498271a72df.
Exact binary polynomials distinguish tie order and the source grid safeguard.
The two omitted-root controls isolate the policy; they are not rootfinder tests.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_constant_midpoint_scalar_array_complex(tech):
    f = tech.from_coeffs(jnp.asarray([3.]))
    assert f.minandmax() == ((3., 0.), (3., 0.))
    a = tech.from_coeffs(jnp.asarray([[3., -2.]]))
    (mn, xmin), (mx, xmax) = a.minandmax()
    assert bool(jnp.array_equal(mn, jnp.asarray([3., -2.])))
    assert bool(jnp.array_equal(mx, mn))
    assert bool(jnp.array_equal(xmin, jnp.zeros(2)))
    assert bool(jnp.array_equal(xmax, xmin))
    c = tech.from_coeffs(jnp.asarray([2.+3.j]))
    assert c.minandmax() == ((2.+3.j, 0.), (2.+3.j, 0.))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_roots_precede_right_endpoint_in_tie(tech, monkeypatch):
    # -x^2*(x-1)^2: maxima0 at x=0 and1. Exact derivative roots isolate
    # the candidate ordering without asserting floating eigensolver bit ties.
    f = tech.from_coeffs(jnp.asarray([-7/8, 3/2, -1., 1/2, -1/8]))
    monkeypatch.setattr(tech, 'roots', lambda self: jnp.asarray([0., .5, 1.]))
    (minimum, xmin), (maximum, xmax) = f.minandmax()
    assert (minimum, xmin) == (-4., -1.)
    assert (maximum, xmax) == (0., 0.)


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_native_grid_minimum_safeguard(tech, monkeypatch):
    # x^2-1 has a grid minimum at0. Supply an omitted derivative root to
    # exercise the literal native grid-value safeguard independently.
    f = tech.from_coeffs(jnp.asarray([-.5, 0., .5]))
    monkeypatch.setattr(tech, 'roots', lambda self: jnp.empty((0,)))
    (minimum, xmin), _ = f.minandmax()
    kind = 1 if tech is Chebtech1 else 2
    grid = chebpts(3, kind=kind)
    assert jnp.abs(minimum + 1.) <= 64*jnp.finfo(jnp.float64).eps
    assert xmin == grid[1]


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_pinned_maximum_grid_min_quirk(tech, monkeypatch):
    # Native line104 uses MIN, not MAX: missing turning point means the
    # sampled maximum1 does not replace the candidate maximum0.
    f = tech.from_coeffs(jnp.asarray([.5, 0., -.5]))
    monkeypatch.setattr(tech, 'roots', lambda self: jnp.empty((0,)))
    _, (maximum, xmax) = f.minandmax()
    assert maximum == 0.
    assert xmax == -1.


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_complex_array_actual_values_and_diagonal_positions(tech):
    # [1+i*x, 2-i*x]; |.|^2 has unique min0 and tied endpoints (left first).
    f = tech.from_coeffs(jnp.asarray([[1.+0j, 2.+0j], [1.j, -1.j]]))
    (minimum, xmin), (maximum, xmax) = f.minandmax()
    bound = 64*jnp.finfo(jnp.float64).eps
    assert bool(jnp.all(jnp.abs(minimum-jnp.asarray([1., 2.])) <= bound))
    assert bool(jnp.all(jnp.abs(maximum-jnp.asarray([1.-1.j, 2.+1.j])) <= bound))
    assert bool(jnp.all(jnp.abs(xmin) <= bound))
    assert bool(jnp.array_equal(xmax, jnp.asarray([-1., -1.])))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_zero_column_outputs(tech):
    # Stored 2D coefficient shapes map directly to native size(f,2)==0.
    # The Python sentinel Tech.empty() is a distinct unresolved representation.
    for n in (0, 1, 3):
        f = tech.from_coeffs(jnp.empty((n, 0), dtype=jnp.float64))
        (minimum, xmin), (maximum, xmax) = f.minandmax()
        assert minimum.shape == xmin.shape == maximum.shape == xmax.shape == (0,)
