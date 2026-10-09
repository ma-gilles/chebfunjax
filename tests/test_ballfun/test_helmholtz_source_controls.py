"""Independent dispatch controls for @ballfun/helmholtz.m, commit 7574c77."""
import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun import Spherefun


@pytest.mark.parametrize('sizes,kind,expected', [
    ((), 'dirichlet', (39, 40, 41)),
    ((10,), 'dirichlet', (10, 10, 10)),
    ((10,), 'neumann', (11, 10, 10)),
    ((10, 9, 7), 'dirichlet', (10, 9, 7)),
    ((10, 9, 7), 'neumann', (11, 9, 7)),
])
def test_source_dimension_dispatch(monkeypatch, sizes, kind, expected):
    module = importlib.import_module('chebfunjax.ballfun.ballfun')
    def solver(cfs, frequency, boundary, neumann):
        assert cfs.shape == expected
        return jnp.zeros(expected, dtype=jnp.complex128)
    monkeypatch.setattr(module, '_ballfun_helmholtz_spectral', solver)
    f = Ballfun.from_coeffs(jnp.zeros((1, 1, 1)))
    assert Ballfun.helmholtz(f, 0, None, *sizes, bc_type=kind).shape == expected


def test_complex_boundary_retains_imaginary_solution(monkeypatch):
    module = importlib.import_module('chebfunjax.ballfun.ballfun')
    cfs = jnp.zeros((5, 5, 5), dtype=jnp.complex128).at[0, 2, 2].set(1j)
    monkeypatch.setattr(module, '_ballfun_helmholtz_spectral', lambda *args: cfs)
    f = Ballfun.from_coeffs(jnp.zeros((1, 1, 1)))
    u = Ballfun.helmholtz(f, 0, lambda l, t: 1j, 5)
    assert not u.is_real
    assert jnp.abs(u(.2, .3, .4)-1j) < 4*jnp.finfo(jnp.float64).eps


def test_sphere_boundary_uses_coefficient_dispatch(monkeypatch):
    module = importlib.import_module('chebfunjax.ballfun.ballfun')
    expected = jnp.arange(35).reshape(7, 5)
    calls = []
    def coefficients(self, n, p):
        calls.append((n, p))
        return expected
    monkeypatch.setattr(Spherefun, 'coeffs2', coefficients)
    def solver(cfs, frequency, boundary, neumann):
        assert jnp.array_equal(boundary, expected.T)
        return jnp.zeros((5, 5, 7), dtype=jnp.complex128)
    monkeypatch.setattr(module, '_ballfun_helmholtz_spectral', solver)
    f = Ballfun.from_coeffs(jnp.zeros((1, 1, 1)))
    Ballfun.helmholtz(f, 0, Spherefun.empty(), 5, 5, 7)
    assert calls == [(5, 7)]


def test_empty_forcing_precedes_all_other_argument_inspection():
    class Forbidden:
        def __call__(self, *args):
            raise AssertionError("Empty solve must not evaluate boundary data")

        def __str__(self):
            raise AssertionError("Empty solve must not inspect boundary type")

        def __int__(self):
            raise AssertionError("Empty solve must not validate grid sizes")

        def __complex__(self):
            raise AssertionError("Empty solve must not inspect frequency")

    f = Ballfun.empty()
    invalid = Forbidden()
    assert Ballfun.helmholtz(f, invalid, invalid, invalid, invalid, invalid,
                            bc_type=invalid) is f
