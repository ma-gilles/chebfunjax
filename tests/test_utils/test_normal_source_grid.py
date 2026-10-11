"""Exact licensed R2025b linspace arithmetic for Chebfun locator consumers.

Provenance: @fun/detectEdge.m, Chebfun commit: 7574c77; licensed R2025b
linspace.m. Independent binary64 host arithmetic, no tolerance envelope.
"""

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from chebfunjax.utils._binary64_grid import _normal_source_grid, locator_source_grid


def reference(a, b, n):
    n1 = n - 1
    if a == -b and n > 2:
        factor = b / n1
        out = [(2 * k - n1) * factor for k in range(n)]
        if n % 2:
            out[n // 2] = 0.0
    else:
        width = b - a
        c = width * (n1 - 1)
        if math.isinf(c):
            if math.isinf(width):
                out = [(a + (b / n1) * k) - (a / n1) * k for k in range(n)]
            else:
                out = [a + k * (width / n1) for k in range(n)]
        else:
            out = [a + (k * width) / n1 for k in range(n)]
        if a == b:
            out = [a] * n
    out[0], out[-1] = a, b
    return out


@pytest.mark.parametrize("n", [4, 15, 50])
@pytest.mark.parametrize(
    "a,b",
    [
        (-2.0, 7.0),
        (-1e-200, 2e-200),
        (0.0, 1.0),
        (-3.0, 3.0),
        (-1e308, 1e308),
        (-1.7e308, 1.6e308),
        (1e307, 1.7e308),
        (2.0, 2.0),
    ],
)
def test_source_normal_branches(a, b, n):
    expected = [struct.pack(">d", v).hex() for v in reference(a, b, n)]
    for fn in (locator_source_grid, _normal_source_grid):
        actual = np.asarray(fn(a, b, n))
        assert [struct.pack(">d", v).hex() for v in actual] == expected


def test_captured_first_native_zoom():
    with np.load(Path(__file__).parent / "fixtures/locator_first_native_zoom.npz") as tape:
        a, b = tape["endpoints"]
        actual = locator_source_grid(float(a), float(b), 50)
        assert np.array_equal(np.asarray(actual)[1:-1], tape["interior"])
