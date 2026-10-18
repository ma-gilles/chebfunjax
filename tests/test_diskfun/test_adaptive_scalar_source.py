"""Actual native scalar/adaptive zero routing; source pin 7574c77."""
import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun, diskfun

DATA = json.loads((Path(__file__).parents[1] / "fixtures/disk_adaptive_scalar_native_20261018.json").read_text())
REQUESTS = {c["name"]: c for c in DATA["requests"]["cases"]}


def words(value):
    a = np.asarray(value)
    return [struct.pack(">d", float(x)).hex() for x in a.reshape(-1, order="F")]


def check_words(value, expected):
    a = np.asarray(value)
    assert a.size == math.prod(expected["shape"])
    assert words(a.real) == expected["re_hex"]
    assert words(a.imag) == expected["im_hex"]


def indices(rec):
    return tuple(int(struct.unpack(">d", bytes.fromhex(w))[0]) - 1
                 for w in rec["re_hex"] if w)


def scalar(case):
    c = REQUESTS[case["name"]]
    re = struct.unpack(">d", bytes.fromhex(c["real_hex"]))[0]
    im = struct.unpack(">d", bytes.fromhex(c["imag_hex"]))[0]
    v = jnp.asarray(re if c["is_real"] else complex(re, im)).reshape(1, 1)
    option = c["option"]
    if option == "eps_loose":
        return diskfun(v, "eps", 1e-4)
    if option == "eps_zero":
        return diskfun(v, "eps", 0.)
    if option == "fixed_rank_zero":
        return diskfun(v, fixed_rank=0)
    if option == "fixed_rank_two":
        return diskfun(v, fixed_rank=2)
    if option == "fixed_length":
        return diskfun(v, fixed_length=(3, 4))
    # Native numeric scalar re-entry discards the outer preference object.
    return Diskfun.from_values(v, preserve_lengths=True)


def callback(case):
    name = case["name"]
    funcs = {
        "zero_scalar_polar": lambda t, r: 0.,
        "zero_theta_polar": lambda t, r: t-t,
        "zero_radius_polar": lambda t, r: r-r,
        "zero_cartesian": lambda x, y: x-x,
        "zero_vectorized": lambda t, r: 0.,
        "zero_fixed_length": lambda t, r: 0*t,
        "zero_fixed_rank_two": lambda t, r: 0*t,
        "nonzero_callback": lambda t, r: .7+0*t,
        "inf_callback": lambda t, r: jnp.inf+0*t,
        "nan_callback": lambda t, r: jnp.nan+0*t,
        "mixed_inf_nan": lambda t, r: jnp.where(t < 0, jnp.inf, jnp.nan),
    }
    opts = {} if name == "zero_cartesian" else {"coordinates": "polar"}
    if name == "zero_vectorized":
        opts["vectorize"] = True
    if name == "zero_fixed_length":
        opts["fixed_length"] = (3, 4)
    if name == "zero_fixed_rank_two":
        opts["fixed_rank"] = 2
    return diskfun(funcs[name], **opts)


@pytest.mark.parametrize("route,case", [("scalar", c) for c in DATA["scalar"]["cases"]]
                         + [("callback", c) for c in DATA["callbacks"]["cases"]],
                         ids=lambda c: c["name"] if isinstance(c, dict) else c)
def test_actual_native_adaptive_constructor(route, case):
    make = scalar if route == "scalar" else callback
    if "error" in case:
        kind = TypeError if case["error"]["identifier"].startswith("MATLAB:feval") else ValueError
        with pytest.raises(kind) as caught:
            make(case)
        assert caught.value.identifier == case["error"]["identifier"]
        assert str(caught.value).removeprefix(caught.value.identifier + ": ") == case["error"]["message"]
        return
    g = make(case)
    expected = case["actual"]
    assert len(g) == expected["rank"]
    assert g.idx_plus == indices(expected["idx_plus"])
    assert g.idx_minus == indices(expected["idx_minus"])
    assert g.nonzero_poles == expected["nonzero_poles"]
    cols = jnp.stack([t.coeffs for t in g.cols], axis=1)
    rows = jnp.stack([t.coeffs for t in g.rows], axis=1)
    assert list(cols.shape) == expected["cols"]["shape"]
    assert list(rows.shape) == expected["rows"]["shape"]
    for value, key in [(cols, "cols"), (rows, "rows"), (g.pivots, "pivots"),
                       (jnp.asarray(g.pivot_locations), "locations"),
                       (g(jnp.asarray([-math.pi, -.7, 0., math.pi]),
                          jnp.asarray([0., .25, .5, 1.])), "values")]:
        check_words(value, expected[key])


def test_actual_scalar_keeps_adaptive_route(monkeypatch):
    calls = []
    original = Diskfun.from_function.__func__

    def observed(cls, f, **kwargs):
        calls.append(f)
        return original(cls, f, **kwargs)

    monkeypatch.setattr(Diskfun, "from_function", classmethod(observed))
    Diskfun.from_values(jnp.asarray([[.7]]), preserve_lengths=True)
    assert len(calls) == 1
