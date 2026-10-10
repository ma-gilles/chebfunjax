"""Native polynomial cumsum endpoint metadata, Chebfun7574c77/R2025b."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, _source_breakpoint_values, chebfun
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


def decode(x):
    if isinstance(x, dict) and "array_shape" in x:
        a = np.asarray(x["real"], dtype=np.float64).reshape(x["array_shape"])
        return (
            a
            if x["imag"] is None
            else a + 1j * np.asarray(x["imag"], dtype=np.float64).reshape(x["array_shape"])
        )
    if isinstance(x, dict):
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


CASES = decode(
    json.loads((Path(__file__).with_name("native_cumsum_endpoints_20261010.json")).read_text())[
        "cases"
    ]
)


def numeric(x):
    return np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)


def pieces(record):
    n = len(np.atleast_1d(record["domain"])) - 1
    raw = [record["coeffs"]] if n == 1 else list(record["coeffs"])
    return [np.atleast_1d(numeric(c)) for c in raw]


def restore(record):
    domain = numeric(record["domain"]).ravel()
    funs = [
        _Piece(
            tech=Chebtech2(coeffs=jnp.asarray(c), ishappy=True), interval=tuple(domain[k : k + 2])
        )
        for k, c in enumerate(pieces(record))
    ]
    f = Chebfun(funs=funs, domain=Domain(tuple(domain)))
    pv = numeric(record["pointValues"])
    columns = 1 if funs[0].tech.coeffs.ndim == 1 else funs[0].tech.coeffs.shape[1]
    f = f.set_point_values(jnp.asarray(pv.reshape(len(domain), columns)))
    return f.T if bool(record["isTransposed"]) else f


def error(a, b):
    a, b = np.squeeze(np.asarray(a)), np.squeeze(numeric(b))
    assert a.shape == b.shape
    return float(np.max(np.abs(a - b)))


def construct(case):
    if case["name"] == "callback":
        return chebfun(lambda x: (x**3 - x) / 8, domain=(-1.0, 1.0))
    return restore(case["initial"])


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_public_cumsum_native_endpoints(case):
    g = construct(case).cumsum(int(case["order"]))
    values = np.asarray(g(jnp.asarray(numeric(case["actual"]["nodes"]))))
    ends = np.asarray(g(jnp.asarray(g.domain.breakpoints)))
    expected = _source_breakpoint_values(g.funs, g.domain.breakpoints)
    assert np.asarray(g._point_values).tobytes() == np.asarray(expected).tobytes()
    public_expected = np.asarray(expected).T if g.is_transposed else np.asarray(expected)
    assert np.squeeze(ends).shape == np.squeeze(public_expected).shape
    assert np.squeeze(ends).tobytes() == np.squeeze(public_expected).tobytes()
    bound = 10 * float(g.vscale) * np.finfo(float).eps
    native_ends = numeric(case["actual"]["pointValues"])
    if g.is_transposed:
        native_ends = native_ends.T
    assert error(values, case["actual"]["samples"]) < bound
    assert error(ends, native_ends) < bound
    for k, x in enumerate(g.domain.breakpoints):
        assert error(g(jnp.asarray(x)), np.asarray(expected)[k]) == 0
