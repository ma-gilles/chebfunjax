"""Explicit source palette table, interpolation and renderer indexing.

Provenance
----------
MATLAB R2018a graph3d/parula.m; Chebfun commit: 7574c77.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax._colormaps import parula_colormap, parula_colors

TABLE = json.loads((Path(__file__).parent/'fixtures/parula_r2018a.json').read_text())['source_table']


@pytest.mark.parametrize('m', [0, 1, 2, 256])
def test_source_table_and_endpoints(m):
    result = np.asarray(parula_colors(m))
    expected = {0: np.empty((0, 3)), 1: np.array(TABLE[-1:]),
                2: np.array([TABLE[0], TABLE[-1]]), 256: np.array(TABLE)}[m]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('m', [64, 255])
def test_linear_interpolation(m):
    # Independent decimal arithmetic verifies the source table interpolation;
    # the tolerance admits double rounding, not a captured pixel fit.
    with localcontext() as ctx:
        ctx.prec = 50
        expected = []
        for k in range(m):
            p = Decimal(255)*k/Decimal(m-1)
            left = min(int(p), 254)
            a = p-left
            expected.append([float((1-a)*Decimal(str(TABLE[left][j]))
                                   + a*Decimal(str(TABLE[left+1][j]))) for j in range(3)])
    assert np.max(np.abs(np.asarray(parula_colors(m))-expected)) < 1e-15


def test_jit_explicit_size():
    eager = parula_colors(64)
    compiled = jax.jit(parula_colors, static_argnums=(0,))(64)
    assert float(jnp.max(jnp.abs(eager-compiled))) < 1e-15


def test_scaled_listed_indices_and_clamping():
    palette = parula_colormap(64)
    t = np.array([-1., 0., np.nextafter(1/64, 0), 1/64,
                  np.nextafter(1/64, 1), .5, np.nextafter(1., 0), 1., 2.])
    indices = np.array([0, 0, 0, 1, 1, 32, 63, 63, 63])
    np.testing.assert_array_equal(palette(t)[:, :3], np.asarray(parula_colors(64))[indices])
    assert palette.N == 64


def test_invalid_lengths():
    with pytest.raises(ValueError):
        parula_colors(-1)
    with pytest.raises(TypeError):
        parula_colors(2.5)
    with pytest.raises(ValueError):
        parula_colormap(0)
