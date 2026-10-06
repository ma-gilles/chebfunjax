"""Independent rational IEEE controls for scoped tiny locator grids.

Provenance
----------
MATLAB source : @fun/detectEdge.m; installed MATLAB R2025b linspace.m
Chebfun commit: 7574c77
Independent source per-operation rational controls and actual captured MATLAB
stall-grid assertions; candidate numerical qualification remains required.
"""
import json
import math
import struct
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


def reference(a, b, n):
    def f(x):
        return Fraction.from_float(x)
    if a == -b:
        factor = float(f(b)/(n-1))
        # Fraction has no signed zero: use IEEE sign rule when factor rounded0.
        out = [(-0.0 if 2*i-(n-1) < 0 else 0.0) if factor == 0.0
               else float((2*i-(n-1))*f(factor)) for i in range(n)]
    else:
        width = float(f(b)-f(a))
        products = [float(i*f(width)) for i in range(n)]
        quotients = [float(f(v)/(n-1)) for v in products]
        out = [float(f(a)+f(v)) for v in quotients]
    out[0], out[-1] = a, b
    return out


@pytest.mark.parametrize("n", [4, 15, 50])
@pytest.mark.parametrize("hex_a,hex_b", [
    ("-0x1.01681a3385b72p-1022", "0x0p+0"),
    ("-0x1p-1020", "0x1p-1020"),
    ("-0x0.fffffffffffffp-1022", "0x0.0000000000001p-1022"),
    ("-0x0p+0", "0x1.0000000000001p-1022"),
    ("0x0.0000000000001p-1022", "0x0.0000000000002p-1022"),
])
def test_source_separate_rounding(hex_a, hex_b, n):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    a, b = float.fromhex(hex_a), float.fromhex(hex_b)
    expected = [struct.pack(">d", v).hex() for v in reference(a, b, n)]
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        actual = fn(a, b, n)
        assert [struct.pack(">d", v).hex() for v in actual.tolist()] == expected


def test_normal_dispatch_preserves_original_grid():
    from chebfunjax.utils._binary64_grid import locator_source_grid
    for a, b in [(-2., 7.), (-1e-200, 2e-200), (0., 1.)]:
        for n in (4, 15, 50):
            actual = locator_source_grid(a, b, n)
            expected = jnp.linspace(a, b, n)
            assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("n", [4, 15, 50])
def test_symmetric_minimum_subnormal_zero_signs(n):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    minimum = float.fromhex("0x0.0000000000001p-1022")
    expected = ["8000000000000000" if 2*i < n-1 else "0000000000000000"
                for i in range(n)]
    expected[0], expected[-1] = "8000000000000001", "0000000000000001"
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        assert [struct.pack(">d", v).hex() for v in fn(-minimum, minimum, n).tolist()] == expected


@pytest.mark.parametrize("units,rounded_factor_units", [(7, 0), (21, 2)])
def test_symmetric_division_halfway_ties(units, rounded_factor_units):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    b = struct.unpack(">d", units.to_bytes(8, "big"))[0]
    expected = []
    for i in range(15):
        multiplier = 2*i-14
        magnitude = abs(multiplier)*rounded_factor_units
        bits = magnitude | ((1 << 63) if multiplier < 0 else 0)
        expected.append(bits.to_bytes(8, "big").hex())
    expected[0] = ((1 << 63) | units).to_bytes(8, "big").hex()
    expected[-1] = units.to_bytes(8, "big").hex()
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        assert [struct.pack(">d", v).hex() for v in fn(-b, b, 15).tolist()] == expected


@pytest.mark.parametrize("n", [4, 15, 50])
@pytest.mark.parametrize("case", ["asymmetric_bound", "negative_zero_right"])
def test_range_and_endpoint_payloads(n, case):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    a = -float.fromhex("0x1p-1020")
    b = math.nextafter(-a, 0.) if case == "asymmetric_bound" else -0.0
    expected = [struct.pack(">d", v).hex() for v in reference(a, b, n)]
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        assert [struct.pack(">d", v).hex() for v in fn(a, b, n).tolist()] == expected


def test_host_dispatch_stall_and_both_cutoff_sides():
    from chebfunjax.utils._binary64_grid import locator_source_grid
    bound = float.fromhex("0x1p-1020")
    inside = [(-float.fromhex("0x1.01681a3385b72p-1022"), 0.),
              (-bound, 0.), (0., bound),
              (math.nextafter(-bound, 0.), 0.), (0., math.nextafter(bound, 0.))]
    outside = [(math.nextafter(-bound, -math.inf), 0.),
               (0., math.nextafter(bound, math.inf))]
    for n in (4, 15, 50):
        for a, b in inside:
            expected = [struct.pack(">d", v).hex() for v in reference(a, b, n)]
            assert [struct.pack(">d", v).hex() for v in locator_source_grid(a, b, n).tolist()] == expected
        for a, b in outside:
            expected = [struct.pack(">d", v).hex() for v in jnp.linspace(a, b, n).tolist()]
            assert [struct.pack(">d", v).hex() for v in locator_source_grid(a, b, n).tolist()] == expected


@pytest.mark.parametrize("n", [4, 15, 50])
def test_actual_matlab_stall_grid_oracle(n):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    fixture = json.loads(Path(__file__).with_name("tiny_source_grid_matlab_fixture.json").read_text())
    # Literal captured INPUT coordinates only; expected grids never enter library.
    a, b = -float.fromhex("0x1.01681a3385b72p-1022"), 0.0
    assert struct.pack(">d", a).hex() == fixture["a_hex"]
    assert struct.pack(">d", b).hex() == fixture["b_hex"]
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        assert [struct.pack(">d", v).hex() for v in fn(a, b, n).tolist()] == fixture["grids"][str(n)]
