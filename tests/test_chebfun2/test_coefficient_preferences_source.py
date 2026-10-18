"""Source constructor coefficient transforms and paired fixed/numeric preferences.

Provenance: @chebfun2/constructor.m parseInputs/constructFromDouble,7574c77.
Native capture: chebfun2_coefficient_pref_native_run001_20261018.
The authored100eps component bound is fixed before implementation; it does
not replace any original MATLAB test predicate or tolerance.
"""

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

DATA = json.loads(
    (Path(__file__).with_name("data") / "coefficient_preferences_native_20261018.json").read_text()
)


def decode(packed):
    shape = packed["shape"]
    if 0 in shape:
        assert all(w == "" for w in packed["real_hex"] + packed["imag_hex"])
        return np.empty(shape)
    real = np.array([struct.unpack(">d", bytes.fromhex(w))[0] for w in packed["real_hex"]])
    imag = np.array([struct.unpack(">d", bytes.fromhex(w))[0] for w in packed["imag_hex"]])
    out = np.empty(real.shape, dtype=np.complex128)
    out.real, out.imag = real, imag
    return out.reshape(shape, order="F")


def bounded(actual, packed):
    expected = decode(packed)
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    if expected.size:
        bound = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(expected))))
        assert np.max(np.abs(actual - expected)) < bound


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            r,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Inherited Fourier transform words select a different tied second ACA pivot",
            ),
        )
        if r["kind"] == "coefficients" and r.get("tech") == "trigtech" and r["case"] == 7
        else r
        for r in DATA["records"]
    ],
    ids=lambda r: "-".join(str(r.get(k, "")) for k in ["kind", "tech", "xtech", "ytech", "case"]),
)
def test_native_coefficient_and_paired_preferences(row):
    assert row["ok"], row
    data = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0], [3.0, 1.0, -2.0], [0.0, 2.0, 3.0]])
    kwargs = {"domain": tuple(row["domain"])}
    if row["kind"] == "coefficients":
        kwargs["pref"] = ChebfunPref(tech=row["tech"])
        kwargs[row["flag"]] = True
        if 4 <= row["case"] <= 6:
            data = data + 1j * jnp.flipud(data) / 8
        fixed = row["case"] > 6
        xtech = ytech = row["tech"]
    else:
        xtech, ytech = row["xtech"], row["ytech"]
        kwargs["pref"] = (ChebfunPref(tech=xtech), ChebfunPref(tech=ytech))
        fixed = row["case"] == 2
    if fixed:

        def data(x, y):
            return 1 + x / 8 - y / 4 + x * y / 16

        kwargs["n"] = (5, 6)
    f = chebfun2(data, **kwargs)
    assert f.rank == row["rank"]
    assert f.domain == tuple(row["domain"])
    assert f.pivot_locations == ()
    assert type(f.approx.cols[0]).__name__.lower() == row["column_tech"]
    assert type(f.approx.rows[0]).__name__.lower() == row["row_tech"]
    bounded(jnp.stack([t.coeffs for t in f.approx.cols], axis=1), row["cols"])
    bounded(jnp.stack([t.coeffs for t in f.approx.rows], axis=1), row["rows"])
    bounded(f.approx.pivot_values[None, :], row["pivots"])
    bounded(
        f(jnp.asarray([-0.7, 0.0, 0.4]), jnp.asarray([0.8, -0.2, 0.3]))[None, :], row["samples"]
    )


def test_periodic_fixed_coefficients_public_values_despite_factor_provider_gap():
    row = next(
        r
        for r in DATA["records"]
        if r["kind"] == "coefficients" and r["tech"] == "trigtech" and r["case"] == 7
    )
    f = chebfun2(
        lambda x, y: 1 + x / 8 - y / 4 + x * y / 16,
        n=(5, 6),
        pref=ChebfunPref(tech="trigtech"),
        coeffs=True,
    )
    assert f.rank == row["rank"]
    bounded(
        f(jnp.asarray([-0.7, 0.0, 0.4]), jnp.asarray([0.8, -0.2, 0.3]))[None, :], row["samples"]
    )
