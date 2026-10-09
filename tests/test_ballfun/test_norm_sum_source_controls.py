"""Independent small CFF controls for native Ballfun norm/sum3.

Provenance
----------
MATLAB source: @ballfun/norm.m, sum3.m, constructor.m, vals2coeffs.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax._ball_plot_data import source_coefficients_to_values
from chebfunjax.ballfun._integrals import coefficient_is_real, values_to_coefficients
from chebfunjax.ballfun.ballfun import Ballfun

EPS = np.finfo(float).eps


def explicit_values(c):
    m, n, p = c.shape
    r = np.cos(np.pi * np.arange(m - 1, -1, -1) / (m - 1)) if m > 1 else np.array([0.])
    lam = -np.pi + 2 * np.pi * np.arange(n) / n
    theta = -np.pi + 2 * np.pi * np.arange(p) / p
    t = np.polynomial.chebyshev.chebvander(r, m - 1)
    e = np.exp(1j * lam[:, None] * (np.arange(n) - n // 2))
    g = np.exp(1j * theta[:, None] * (np.arange(p) - p // 2))
    return np.einsum('ia,jb,kc,abc->ijk', t, e, g, c)


@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 3, 3), (4, 4, 4), (3, 4, 5)])
def test_sparse_transform_oracle(shape):
    c = np.zeros(shape, complex)
    c[0, shape[1] // 2, shape[2] // 2] = .5 + .125j
    c[-1, 0, -1] += .25 - .0625j
    v = explicit_values(c)
    bound = 64 * max(shape) * EPS * max(1., np.abs(c).sum())
    assert np.max(np.abs(np.asarray(source_coefficients_to_values(c)) - v)) < bound
    assert np.max(np.abs(np.asarray(values_to_coefficients(v)) - c)) < bound


@pytest.mark.parametrize('shape', [(2, 3, 3), (2, 4, 3), (2, 3, 4), (2, 4, 4)])
def test_constructor_realness_even_rows(shape):
    c = jnp.zeros(shape, dtype=jnp.complex128).at[0, shape[1] // 2, shape[2] // 2].set(1)
    assert bool(coefficient_is_real(c))
    assert not bool(coefficient_is_real(c.at[0, shape[1] // 2, shape[2] // 2].set(1 + 1e-5j)))
    if shape[1] % 2 == 0:
        assert not bool(coefficient_is_real(c.at[0, 0, shape[2] // 2].set(1e-5)))
    if shape[2] % 2 == 0:
        assert not bool(coefficient_is_real(c.at[0, shape[1] // 2, 0].set(1e-5)))


@pytest.mark.parametrize('which', ['sentinel', 'radial', 'longitude', 'theta'])
def test_empty_source_outputs(which):
    shapes = {'radial': (0, 2, 2), 'longitude': (2, 0, 2), 'theta': (2, 2, 0)}
    f = Ballfun.empty() if which == 'sentinel' else Ballfun.from_coeffs(jnp.zeros(shapes[which]))
    assert f.norm().shape == (0,)
    assert f.sum().shape == (0,)


@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)])
def test_constant_norm_and_complex_integral(shape):
    c = jnp.zeros(shape, dtype=jnp.complex128).at[0, shape[1] // 2, shape[2] // 2].set(1 + .5j)
    f = Ballfun.from_coeffs(c, is_real=False)
    bound = 128 * max(shape) * EPS
    assert abs(complex(f.sum()) - (1 + .5j) * 4 * np.pi / 3) < bound
    assert abs(f.norm() - np.sqrt(1.25 * 4 * np.pi / 3)) < bound


def test_exact_doubled_dimensions(monkeypatch):
    import chebfunjax.ballfun._integrals as module

    original = module.prolong_plot_coefficients
    shapes = []

    def observe(c, shape):
        shapes.append(shape)
        return original(c, shape)

    monkeypatch.setattr(module, 'prolong_plot_coefficients', observe)
    c = jnp.ones((1, 1, 1), dtype=jnp.complex128)
    Ballfun.from_coeffs(c).norm()
    assert shapes == [(2, 2, 2), (4, 2, 4)]


@pytest.mark.parametrize('axis', [1, 2])
def test_even_nyquist_norm(axis):
    # Native even Nyquist coefficient prolongs to equal +/- modes: cos(angle).
    shape = (1, 2, 1) if axis == 1 else (1, 1, 2)
    c = jnp.zeros(shape, dtype=jnp.complex128).at[0, 0, 0].set(1)
    f = Ballfun.from_coeffs(c, is_real=True)
    squared_integral = 2 * np.pi / 3 if axis == 1 else 4 * np.pi / 9
    assert abs(f.norm() - np.sqrt(squared_integral)) < 256 * EPS
    assert abs(f.sum()) < 256 * EPS
