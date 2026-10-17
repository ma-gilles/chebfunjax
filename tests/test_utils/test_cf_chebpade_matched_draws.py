"""Expected-only public MATLAB padding data with matched primitive normal draws.

Provenance
----------
MATLAB source : @chebfun/chebpade.m, Clenshaw-Lord branch
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Capture: cf_chebpade_padding_v1_matlab_cpu_20261007 (public function untouched).
These are additional rounding diagnostics, not original MATLAB test predicates.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils import _cf_kernel as kernel
from chebfunjax.utils import _cf_padding as padding


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("case", [1, 2, 3])
def test_public_source_matched_normal_draws(monkeypatch, disabled, case):
    record = json.loads((Path(__file__).parent / "data" / "cf_padding" /
                         f"case{case}.json").read_text())

    def decode(name):
        packed = record[name]
        assert all(struct.unpack(">d", bytes.fromhex(word))[0] == 0
                   for word in packed["imag_hex"])
        return jnp.asarray([struct.unpack(">d", bytes.fromhex(word))[0]
                            for word in packed["real_hex"]], dtype=jnp.float64)

    calls = []

    def draw(count):
        calls.append(count)
        assert count == record["missing"]
        return decode("normals")

    monkeypatch.setattr(padding, "_normal_draw", draw)
    with jax.disable_jit(disabled):
        p, q = kernel._chebpade_clenshaw_lord_jax(
            decode("c"), record["m"], record["n"])
    assert record["rng_state_consumption_matches"] is True
    assert calls == [record["missing"]]
    for actual, name in [(p, "pk"), (q, "qk")]:
        expected = decode(name)
        assert actual.shape == expected.shape
        if case in (1, 3):
            # Independent one-pole and n0 reductions have no general matrix solve.
            assert jnp.array_equal(actual, expected)
        else:
            # New, explicit componentwise rounding diagnostic for this tiny,
            # nonsingular 2x2 solve. No unit floor hides near-zero q coefficients.
            assert bool(jnp.all(jnp.abs(actual - expected) <=
                                16 * jnp.finfo(jnp.float64).eps * jnp.abs(expected)))


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("case", [1, 2, 3])
def test_public_frontdoor_matches_native_padding_fixture(monkeypatch, disabled, case):
    """Exercise source normal padding through the public adapter, including n>0."""
    import chebfunjax as cj

    record = json.loads((Path(__file__).parent / "data" / "cf_padding" /
                         f"case{case}.json").read_text())

    def decode(name):
        packed = record[name]
        return jnp.asarray([complex(struct.unpack(">d", bytes.fromhex(r))[0],
                                    struct.unpack(">d", bytes.fromhex(i))[0])
                            for r, i in zip(packed["real_hex"], packed["imag_hex"])])

    calls = []

    def draw(count):
        calls.append(count)
        assert count == record["missing"]
        return decode("normals").real

    monkeypatch.setattr(padding, "_normal_draw", draw)
    f = cj.chebfun(decode("c").real, coeffs=True)
    with jax.disable_jit(disabled):
        p, q, _ = cj.chebpade(f, record["m"], record["n"])
    assert calls == [record["missing"]]
    for f_out, name in [(p, "pk"), (q, "qk")]:
        expected = decode(name).real
        actual = f_out.chebcoeffs(len(expected))
        assert actual.shape == expected.shape
        if case in (1, 3):
            assert jnp.array_equal(actual, expected)
        else:
            assert bool(jnp.all(jnp.abs(actual-expected) <=
                                16*jnp.finfo(jnp.float64).eps*jnp.abs(expected)))
