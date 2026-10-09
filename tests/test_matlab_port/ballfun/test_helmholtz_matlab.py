"""All 38 slots (79 scalar comparisons) of the literal Helmholtz source.

MATLAB source: tests/ballfun/test_helmholtz.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Copyright 2019 The University of Oxford and The Chebfun Developers.
No RNG, sampled norm replacements, or tolerance changes.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun import spherefun

TOL = 1e7 * ChebfunPref().techPrefs.chebfuneps


def _ball(op, spherical=True):
    return Ballfun.from_function(op, spherical=spherical)


def _cos_forcing(r, l, t):
    return -2 * (2*r**2*jnp.cos(l)**2*jnp.cos(r**2*jnp.cos(l)**2*jnp.sin(t)**2)
                 * jnp.sin(t)**2 + jnp.sin(r**2*jnp.cos(l)**2*jnp.sin(t)**2))


def _harmonic(r, l, t):
    return r**5*jnp.exp(3j*l)*jnp.sin(t)**3*(9*jnp.cos(t)**2-1)


def _boundary_coeffs(exact):
    # Source reshape(sum(diff(exact,1,'spherical').coeffs,1),S(2),S(3)).
    return jnp.sum(exact.diff(1, coord='spherical').coeffs, axis=0)


@pytest.mark.parametrize('clause', range(1, 39),
                         ids=[f'source_clause_{k:02d}' for k in range(1, 39)])
def test_literal_helmholtz(clause):
    k = clause
    if k <= 12 or k in (33, 37, 38):
        index = {8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 33: 7, 37: 5, 38: 3}.get(k, k)
        exact_op = {
            1: lambda r, l, t: 1,
            2: lambda r, l, t: r**2,
            3: lambda r, l, t: r**2*jnp.sin(t)**2,
            4: lambda r, l, t: r*jnp.sin(l)*jnp.sin(t),
            5: lambda r, l, t: jnp.cos(r**2*jnp.sin(t)**2*jnp.cos(l)**2),
            6: lambda r, l, t: r**2*jnp.sin(t)**2*jnp.exp(2j*l),
            7: _harmonic,
        }[index]
        forcing_op = {
            1: lambda r, l, t: 0,
            2: lambda r, l, t: 6,
            3: lambda r, l, t: 4,
            4: lambda r, l, t: 0,
            5: _cos_forcing,
            6: lambda r, l, t: 0,
            7: lambda r, l, t: 0,
        }[index]
        frequency = 2 if 8 <= k <= 12 else 0
        if frequency:
            forcing_op = {
                8: lambda r, l, t: 4+0*r,
                9: lambda r, l, t: 6+4*r**2,
                10: lambda r, l, t: 4+4*r**2*jnp.sin(t)**2,
                11: lambda r, l, t: 4*r*jnp.sin(l)*jnp.sin(t),
                12: lambda r, l, t: _cos_forcing(r, l, t)
                    + 4*jnp.cos(r**2*jnp.sin(t)**2*jnp.cos(l)**2),
            }[k]
            if k == 8:
                exact_op = lambda r, l, t: 1+0*r
        # Constants in source cases 1-4/38 use Cartesian constructor dispatch.
        if k in (6, 7, 33):
            f = Ballfun.from_values(0)  # literal source zero = ballfun(0)
        else:
            f = _ball(forcing_op, spherical=k not in (1, 2, 3, 4, 38))
        exact = _ball(exact_op)
        if index == 4:
            bc = lambda l, t: exact_op(1, t, l)  # literal source argument swap
        else:
            bc = lambda l, t: exact_op(1, l, t)
        if k in (37, 38):
            bc = spherefun(bc)
        sizes = (50,) if k == 33 else ((39, 40, 41) if k <= 5 or k >= 37 else (50, 50, 50))
        u = Ballfun.helmholtz(f, frequency, bc, *sizes)
        if k in (6, 7, 33):
            assert u.laplacian().norm() < TOL
        assert (u-exact).norm() < TOL
        return

    if k in (13, 30):
        exact = _ball(lambda r, l, t: 1)
        bc = lambda l, t: 0
    elif k in (14, 31):
        exact = _ball(lambda r, l, t: r**2*jnp.sin(t)**2)
        bc = lambda l, t: 2*jnp.sin(t)**2
    elif k == 15:
        exact = _ball(lambda r, l, t: r**3*jnp.sin(t)**3*jnp.cos(l))
        bc = lambda l, t: 3*jnp.sin(t)**3*jnp.cos(l)
    elif k == 16:
        exact = _ball(lambda r, l, t: r**2*jnp.sin(t)**2*jnp.cos(l)**2)
        bc = lambda l, t: 2*jnp.sin(t)**2*jnp.cos(l)**2
    elif k in (17, 18, 19, 23, 24, 25, 26, 36):
        exact = _ball(lambda x, y, z: y**2, spherical=False)
        bc = lambda l, t: 2*(jnp.sin(t)*jnp.sin(l))**2
        if k == 36:
            bc = spherefun(bc)
    elif k in (20, 27):
        exact = _ball(lambda r, l, t: (r*jnp.sin(t)*jnp.cos(l)*r*jnp.cos(t))**2)
        bc = _boundary_coeffs(exact)
    elif k in (21, 28, 34):
        exact = _ball(lambda r, l, t: jnp.sin((r*jnp.sin(t)*jnp.sin(l))**2))
        bc = _boundary_coeffs(exact)
    elif k in (22, 29):
        exact = _ball(lambda r, l, t: jnp.cos((r*jnp.sin(t)*jnp.cos(l))**3))
        bc = _boundary_coeffs(exact)
    elif k == 32:
        exact = _ball(lambda r, l, t: r**4*jnp.sin(t)**2)
        bc = lambda l, t: 4*jnp.sin(t)**2
    else:
        assert k == 35
        exact = _ball(lambda x, y, z: (x**2+y**2+z**2)*(x**2+y**2), spherical=False)
        bc = spherefun(lambda x, y, z: 4*(x**2+y**2))
    frequency = 2 if k in (30, 31, 32, 35) else 0
    f = exact.laplacian()
    if frequency:
        f = f + 4*exact
    sizes = {
        17: (40, 54, 41), 18: (41, 54, 41), 19: (42, 54, 41),
        20: (42, 54, 41), 21: (42, 54, 41), 22: (42, 54, 41),
        23: (43, 54, 42), 24: (43, 54, 43), 25: (43, 54, 44),
        26: (42, 42, 42), 27: (42, 42, 42), 28: (42, 42, 42), 29: (42, 42, 42),
        30: (38, 17, 24), 31: (38, 17, 24), 32: (38, 17, 24),
        34: (42,), 35: (40,), 36: (43, 54, 44),
    }.get(k, (39, 40, 41))
    u = Ballfun.helmholtz(f, frequency, bc, *sizes, bc_type='neumann')
    if frequency:
        assert (u-exact).norm() < TOL
    else:
        coord = 'spherical' if k == 13 else 'cartesian'
        for dim in (1, 2, 3):
            assert (u.diff(dim, coord=coord)-exact.diff(dim, coord=coord)).norm() < TOL
