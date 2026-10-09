"""Literal linear rejection boundaries from the pinned native roots source.

Provenance
----------
MATLAB source : @chebtech/roots.m, n == 2 branch
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module

EPS = float(jnp.finfo(jnp.float64).eps)
HTOL = 100*EPS
TECHS = [module.Chebtech1, module.Chebtech2]
SCALES = [("positive", 1.), ("negative", -3.), ("imaginary", 2j),
          ("complex", 1+2j), ("tiny", 1e-200), ("huge", 1e200)]


def _adjacent(value, location):
    if location == "equal":
        return value
    toward = -jnp.inf if location == "below" else jnp.inf
    return float(jnp.nextafter(jnp.asarray(value), toward))


def _coeffs(root, scale=1.):
    return scale*jnp.stack([-jnp.asarray(root), jnp.asarray(1.)])


def _native_rejection(raw):
    # Native linear source rejects with >, <, >; its eigenvalue leaf uses a
    # different, strict imaginary acceptance predicate and is unchanged.
    return ((jnp.abs(jnp.imag(raw)) > HTOL)
            | (jnp.real(raw) < -(1+HTOL))
            | (jnp.real(raw) > 1+HTOL))


def _native_filtered(raw):
    return jnp.clip(jnp.real(raw[~_native_rejection(raw)]), -1., 1.)


def _write(name, rows):
    destination = os.environ.get("CHEBFUN_RUNTIME_REPORT")
    if destination:
        (Path(destination).parent / (name+".json")).write_text(
            json.dumps(rows, indent=2)+"\n")


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("location", ["below", "equal", "above"])
def test_exact_linear_imaginary_boundary(sign, location):
    root = complex(.25, sign*_adjacent(HTOL, location))
    coeffs = jax.device_get(_coeffs(root))
    raw = jnp.asarray(module._roots_main(coeffs, HTOL, all_roots=True))
    actual = jnp.asarray(module._roots_main(coeffs, HTOL))
    assert bool(jnp.array_equal(raw, jnp.asarray([root])))
    expected = jnp.asarray([.25]) if location != "above" else jnp.empty(0)
    assert bool(jnp.array_equal(actual, expected))
    _write(f"imaginary_{sign}_{location}", {
        "raw_real": float(jnp.real(raw[0])), "raw_imag": float(jnp.imag(raw[0])),
        "htol": HTOL, "accepted_count": actual.size})


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("location", ["below", "equal", "above"])
@pytest.mark.parametrize("representation", ["real", "complex_real", "imaginary_equal"])
def test_exact_linear_expanded_domain_boundary(sign, location, representation):
    real = sign*_adjacent(1+HTOL, location)
    root = (real if representation == "real" else
            complex(real, HTOL if representation == "imaginary_equal" else 0.))
    coeffs = jax.device_get(_coeffs(root))
    raw = jnp.asarray(module._roots_main(coeffs, HTOL, all_roots=True))
    actual = jnp.asarray(module._roots_main(coeffs, HTOL))
    assert bool(jnp.array_equal(raw, jnp.asarray([root])))
    expected = jnp.asarray([sign]) if location != "above" else jnp.empty(0)
    assert bool(jnp.array_equal(actual, expected))
    _write(f"endpoint_{sign}_{location}_{representation}", {
        "raw_real": float(jnp.real(raw[0])), "raw_imag": float(jnp.imag(raw[0])),
        "expanded_bound": 1+HTOL, "accepted_count": actual.size})


@pytest.mark.parametrize("Tech", TECHS)
@pytest.mark.parametrize("label,scale", SCALES)
def test_public_scaled_linear_filter_uses_actual_unfiltered_ratio(Tech, label, scale):
    cases = [complex(.25, sign*_adjacent(HTOL, location))
             for sign in (-1., 1.) for location in ("below", "equal", "above")]
    cases += [sign*_adjacent(1+HTOL, location)
              for sign in (-1., 1.) for location in ("below", "equal", "above")]
    rows = []
    for root in cases:
        tech = Tech(coeffs=_coeffs(root, scale))
        # Both calls execute the actual engine. The unfiltered ratio avoids
        # assuming that complex scaling leaves a boundary value bit-identical.
        raw = tech.roots(all_roots=True)
        actual = tech.roots()
        expected = _native_filtered(raw)
        assert bool(jnp.array_equal(actual, expected))
        rows.append({"intended_real": float(jnp.real(root)),
                     "intended_imag": float(jnp.imag(root)),
                     "computed_real": float(jnp.real(raw[0])),
                     "computed_imag": float(jnp.imag(raw[0])),
                     "accepted_count": actual.size})
    _write(f"scaled_{Tech.__name__}_{label}", rows)


@pytest.mark.parametrize("Tech", TECHS)
def test_array_columns_apply_literal_linear_filter(Tech):
    roots = [.25+1j*HTOL, .25+1j*_adjacent(HTOL, "above"),
             .25+1j*_adjacent(HTOL, "below")]
    coeffs = jnp.stack([_coeffs(root, scale)
                        for root, scale in zip(roots, [1., -3., 1+2j])], axis=1)
    tech = Tech(coeffs=coeffs)
    raw = tech.roots(all_roots=True)
    actual = tech.roots()
    keep = ~_native_rejection(raw[0])
    expected = (jnp.where(keep, jnp.clip(jnp.real(raw[0]), -1., 1.), jnp.nan)[None, :]
                if bool(jnp.any(keep)) else jnp.empty((0, 3)))
    assert bool(jnp.array_equal(actual, expected, equal_nan=True))


@pytest.mark.parametrize("qz", [False, True])
def test_linear_equality_shortcut_precedes_eigensolver_choice(qz):
    coeffs = jax.device_get(_coeffs(.25+1j*HTOL))
    roots = jnp.asarray(module._roots_main(coeffs, HTOL, qz=qz))
    assert bool(jnp.array_equal(roots, jnp.asarray([.25])))
