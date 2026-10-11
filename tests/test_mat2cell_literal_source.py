"""All literal native MAT2CELL36 slots; slot(2,1) is source constant true.

Native seed7681 first1000/next100 uniform arrays are captured at assignments.
Quasimatrix constructor maps to an explicit collection of scalar Chebfuns.
CellList is flat; shape records native cell layout without nested indexing.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebpref import ChebfunPref

INPUTS = np.load(Path(__file__).with_name("fixtures") / "mat2cell_native_inputs.npz")
XR = jnp.asarray(INPUTS["xr"])
X = jnp.asarray(INPUTS["x"])
EPS = float(jnp.finfo(jnp.float64).eps)
DOM = [-1, -0.5, 0, 0.5, 1]


def scale(f):
    return float(jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale))


def infnorm(values):
    return float(jnp.max(jnp.abs(values)))


def constructor(quasi, operators, domain, pref=None):
    if quasi:
        columns = [cj.chebfun(op, domain=domain, pref=pref) for op in operators]
        return Quasimatrix(columns, columns[0].domain)
    return cj.chebfun(
        lambda x: jnp.stack([op(x) for op in operators], axis=-1), domain=domain, pref=pref
    )


def test_native_empty_and_constant_slots():
    C = cj.chebfun().mat2cell()
    assert C.shape == (1, 1) and C[0].isempty()  # native(1,1)
    assert True  # literal native(2,1), constant control not algorithm assertion


@pytest.mark.parametrize("quasi", [False, True])
def test_native_finite_slots_2_through_17(quasi):
    pref = ChebfunPref()
    F = constructor(quasi, [jnp.sin, jnp.cos, jnp.exp], DOM, pref)
    refs = [cj.chebfun(op, domain=[-1, 1], pref=pref) for op in [jnp.sin, jnp.cos, jnp.exp]]
    fg = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
    C = F.mat2cell()
    assert C.shape == (1, 3) and all(c.size(2) == 1 for c in C)
    for c, ref in zip(C, refs):
        assert infnorm(c(XR) - ref(XR)) < 10 * scale(c) * EPS
    for args in [([2, 1],), (1, [2, 1])]:
        C = F.mat2cell(*args)
        assert C.shape == (1, 2) and C[0].size(2) == 2 and C[1].size(2) == 1
        assert (
            infnorm((C[0](XR) - fg(XR)).reshape(-1)) < 10 * scale(C[0]) * EPS
            and infnorm(C[1](XR) - refs[2](XR)) < 10 * scale(C[1]) * EPS
        )
    Ft = F.T
    C = Ft.mat2cell()
    assert C.shape == (3, 1) and all(c.size(1) == 1 for c in C)
    for args in [([2, 1],), ([2, 1], 1)]:
        C = Ft.mat2cell(*args)
        assert C.shape == (2, 1) and C[0].size(1) == 2 and C[1].size(1) == 1
    for f, args in [
        (F, ("bad", [2, 1])),
        (F, (2, [2, 1])),
        (F, (1, [2, 2])),
        (Ft, ([2, 1], 2)),
        (Ft, ([2, 2], 1)),
    ]:
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:mat2cell:size"):
            f.mat2cell(*args)


@pytest.mark.parametrize("quasi", [False, True])
def test_native_unbounded_slot18(quasi):
    dom = [-jnp.inf, -3 * jnp.pi]
    ops = [lambda x: jnp.exp(x), lambda x: x * jnp.exp(x), lambda x: (1 - jnp.exp(x)) / x]
    f = constructor(quasi, ops, dom)
    C = f.mat2cell(1, [1, 2])
    err1 = C[0](X) - jnp.exp(X)
    exact2 = jnp.stack([X * jnp.exp(X), (1 - jnp.exp(X)) / X], axis=-1)
    err2 = C[1](X) - exact2
    assert infnorm(jnp.concatenate([err1.reshape(-1), err2.reshape(-1)])) < 1e2 * EPS * scale(f)
