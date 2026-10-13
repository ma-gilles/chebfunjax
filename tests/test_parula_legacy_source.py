"""Official pre-R2017a palette; no colors inferred from reference pixels."""
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from chebfunjax._colormaps import parula_colormap
from chebfunjax._parula_2016a import parula_2016a_colors

TABLE = json.loads((Path(__file__).parent/'fixtures/parula_r2016a.json').read_text())['source_table']


def test_exact_full_table_and_source_endpoints():
    np.testing.assert_array_equal(parula_2016a_colors(256), TABLE)
    np.testing.assert_array_equal(parula_2016a_colors(1), TABLE[-1:])
    assert parula_2016a_colors(0).shape == (0, 3)


def test_source_linear_interpolation_at_historical_64_length():
    with localcontext() as ctx:
        ctx.prec = 50
        expected = []
        for k in range(64):
            p = Decimal(255)*k/63
            left = min(int(p), 254)
            fraction = p-left
            expected.append([float((1-fraction)*Decimal(str(TABLE[left][j]))
                                   + fraction*Decimal(str(TABLE[left+1][j]))) for j in range(3)])
    assert np.max(np.abs(np.asarray(parula_2016a_colors(64))-expected)) < 1e-15


def test_explicit_version_keeps_modern_default_and_listed_mapping():
    modern = parula_colormap(64)
    old = parula_colormap(64, version="R2016a")
    assert modern.name == "parula_r2018a_64"
    assert old.name == "parula_r2016a_64"
    assert old.N == modern.N == 64
    np.testing.assert_array_equal(old([0., 1.])[:, :3], [TABLE[0], TABLE[-1]])
    assert not np.array_equal(old.colors, modern.colors)
    with pytest.raises(ValueError):
        parula_colormap(64, version="unknown")
