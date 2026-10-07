"""Deterministic source output contracts; no MATLAB RNG equivalence claim.

Provenance
----------
MATLAB source : spin2.m, @spinoperator/solvepde.m, @spinop2/discretize.m
Chebfun commit: 7574c77
Expected-only JSON is the untouched R2025b source capture; input expressions,
domain, steps and queries below are independent literals. New capture bound
1e-11 does not replace or modify any existing source-test bound.
"""

import json
import math
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.operators._spin2_public import (
    Spin2SolutionMatrix,
    _dealias_indexes,
    _preferences,
    _record_states,
)
from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinpref import SpinPref2

_CASES = [
    ("final_public_one", (0, .04), False, False, "etdrk4", False),
    ("final_public_two", (0, .04), False, False, "etdrk4", False),
    ("aligned_scalar_pairs", (0, .02, .04), False, True, "etdrk4", True),
    ("aligned_scalar_object", (0, .02, .04), False, True, "etdrk4", False),
    ("aligned_system", (0, .02, .04), True, False, "etdrk4", False),
    ("unaligned_scalar", (0, .015, .04), False, False, "etdrk4", False),
    ("multistep_aligned", (0, .04, .06), False, False, "abnorsett4", False),
    ("multistep_early", (0, .01, .06), False, False, "abnorsett4", False),
]


def _decode(packed, part):
    return jnp.asarray([struct.unpack(">d", bytes.fromhex(h))[0]
                        for h in packed[part + "_hex"]])


def _problem(times, system=False, complex_input=False):
    domain = (-1., 2 * math.pi - 1, 2., 4 * math.pi + 2)
    op = Spinop2(domain, times)
    if system:
        op.lin = "@(u,v) [lap(u);2*lap(v)]"
        op.nonlin = [lambda u, v: 0 * u, lambda u, v: 0 * v]
        op.init = [lambda x, y: jnp.cos(x + 1) * jnp.sin((y - 2) / 2),
                   lambda x, y: jnp.sin(2 * (x + 1)) * jnp.cos((y - 2) / 2)]
    else:
        op.lin = "@(u) lap(u)"
        op.nonlin = lambda u: 0 * u
        if complex_input:
            op.init = lambda x, y: jnp.exp(1j * (x + 1)) * jnp.cos((y - 2) / 2)
        else:
            op.init = lambda x, y: jnp.cos(x + 1) * jnp.sin((y - 2) / 2)
    return op


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_captured_public_output(case, disabled):
    name, requested, system, complex_input, scheme, pairs = case
    expected = json.loads((Path(__file__).parent / "data/spin2_source_outputs"
                           / (name + ".json")).read_text())
    op = _problem(requested, system, complex_input)
    pref = SpinPref2(plot="off", M=16, dealias="off", scheme=scheme)
    args = ("plot", "off", "M", 16, "dealias", "off", "scheme", scheme) if pairs else (pref,)
    with jax.disable_jit(disabled):
        if name == "final_public_one":
            result = spin2(op, 8, .01, *args)
        else:
            result, actual_times = spin2(op, 8, .01, *args, return_times=True)
            assert jnp.array_equal(actual_times, _decode(expected["returned_times"], "real"))
        if expected["output_class"] == "chebfun2":
            assert isinstance(result, Chebfun2)
            entries = [result]
        else:
            assert isinstance(result, Spin2SolutionMatrix)
            assert result.shape == tuple(expected["output_size"])
            entries = list(result)
            for col in range(result.shape[1]):
                for row in range(result.shape[0]):
                    assert result[row, col] is entries[col * result.shape[0] + row]
        assert len(entries) == len(expected["entries"])
        x = jnp.asarray([-.7, .13, 2.19])
        y = jnp.asarray([2.2, 4.1, 7.3])
        for entry, source in zip(entries, expected["entries"]):
            assert isinstance(entry, Chebfun2)
            assert tuple(entry.domain) == op.domain
            values = entry(x, y)
            assert jnp.max(jnp.abs(jnp.real(values) - _decode(source["values"], "real"))) < 1e-11
            assert jnp.max(jnp.abs(jnp.imag(values) - _decode(source["values"], "imag"))) < 1e-11


def test_preferences_default_and_forwarding():
    default = _preferences((), {})
    assert (default.dealias, default.M, default.scheme) == ("off", 32, "etdrk4")
    original = SpinPref2(M=18, scheme="abnorsett4", dealias="on")
    selected = _preferences((original,), {"M": 20, "dealias": "off"})
    assert (selected.M, selected.dealias, selected.scheme) == (20, "off", "abnorsett4")
    assert original.M == 18 and original.dealias == "on"
    with pytest.raises(ValueError):
        _preferences((), {"unknown": 1})


@pytest.mark.parametrize("disabled", [False, True])
def test_source_dealias_is_output_only(disabled):
    # For N8, source square mask covers Fourier indices2:6 in each axis.
    # Mode(3,3) decays through both steps even when its saved output is masked.
    op = _problem((0, .01, .02))
    op.lin = "@(u) 0*lap(u)"
    op.init = lambda x, y: jnp.exp(3j * (x + 1) + 3j * (y - 2) / 2)
    op.nonlin = lambda u: jnp.abs(u) ** 2
    with jax.disable_jit(disabled):
        off, t, _, _ = _record_states(op, 8, .01, SpinPref2(plot="off", dealias="off"))
        on, ton, _, _ = _record_states(op, 8, .01, SpinPref2(plot="off", dealias="on"))
        assert jnp.array_equal(t, ton)
        assert jnp.max(jnp.abs(on[0][0] - off[0][0])) == 0
        assert jnp.max(jnp.abs(off[-1][0])) > .99
        # Dealias must not erase the high mode from future nonlinear history.
        # The unmasked Fourier coefficients must agree with the off arm.
        mask = _dealias_indexes(8)
        difference = jnp.fft.fft2(on[-1][0]) - jnp.fft.fft2(off[-1][0])
        assert jnp.max(jnp.abs(jnp.where(mask, 0, difference))) < 1e-11
        assert jnp.abs(jnp.mean(on[-1][0])) > .019
        mask = _dealias_indexes(8)
        assert bool(mask[3, 3]) and not bool(mask[0, 3]) and not bool(mask[3, 0])


@pytest.mark.parametrize("disabled", [False, True])
def test_source_rectangular_symbol_and_linear_analytic_evolution(disabled):
    # Source intentionally uses x-width for BOTH directional wave numbers.
    op = _problem((0, .02, .04), complex_input=True)
    with jax.disable_jit(disabled):
        values, times, _, _ = _record_states(op, 8, .01, SpinPref2(plot="off", M=16))
        for state, time in zip(values, times):
            assert jnp.max(jnp.abs(state[0] - values[0][0] * jnp.exp(-2 * time))) < 1e-12
