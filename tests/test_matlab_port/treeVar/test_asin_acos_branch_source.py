"""Native ASIN/ACOS branch capture, exact classifications and bounded residuals.

Provenance
----------
MATLAB source: @treeVar/treeVar.m numeric ASIN/ACOS, Chebfun7574c77.
R2025b primitive fixture uses exact input/output IEEE words. Supplemental
finite-component guard is 8eps*max(1,abs(component)), fixed before this gate;
it is not an original source predicate or a universal libm error theorem.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS, to_first_order

DATA = json.loads((Path(__file__).with_name("data")
                   / "asin_acos_boundary_native.json").read_text())["records"]


def _decode(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _input(row):
    x = row["input"]
    re, im = _decode(x["real_hex"]), _decode(x["imag_hex"])
    return jnp.asarray(re if x["isreal"] else complex(re, im))


@pytest.mark.parametrize("row", DATA,
                         ids=[r["method"]+"-"+r["input"]["label"] for r in DATA])
def test_native_primitive_branch(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    for part in ["real", "imag"]:
        target = _decode(expected[part+"_hex"])
        value = float(getattr(got, part))
        assert np.isnan(value) == np.isnan(target)
        assert np.isposinf(value) == np.isposinf(target)
        assert np.isneginf(value) == np.isneginf(target)
        if np.isfinite(target):
            assert abs(value-target) <= 8*np.finfo(float).eps*max(1., abs(target))
        if target == 0:
            assert value == 0
            assert np.signbit(value) == np.signbit(target)


@pytest.mark.parametrize("name", ["asin", "acos"])
@pytest.mark.parametrize("x", [-2., 2.])
def test_public_rhs_outside_real_domain(name, x):
    rhs = to_first_order(lambda u: u.diff(2)+getattr(u, name)(), 0., (-3., 3.))[0]
    row = next(r for r in DATA if r["method"] == name
               and r["input"]["label"] == ("real_n2" if x<0 else "real_p2"))
    expected = complex(_decode(row["expected"]["real_hex"]),
                       _decode(row["expected"]["imag_hex"]))
    got = np.asarray(rhs(0., jnp.asarray([x, .3])))
    assert np.linalg.norm(got.ravel()-np.asarray([.3, -expected])) < 10*np.finfo(float).eps
