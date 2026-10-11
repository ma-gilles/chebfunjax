"""Eleven native FLIPLR predicates, original operands/prefs/norms/bounds.

Source pin7574c77. Native seeded uniform prefixes qualify retained inputs;
principal complex powers explicitly promote MATLAB negative-real bases.
Python singleton scalar values use1-D arrays; native row matrix infinity
norm for singular clause11 remains the absolute row sum, not max.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

DATA = np.load(Path(__file__).with_name("fixtures") / "fliplr_native_inputs.npz")
XR = jnp.asarray(DATA["xr"])
X = jnp.asarray(DATA["x"])
EPS = float(jnp.finfo(jnp.float64).eps)
PREF = ChebfunPref()
PREF.splitting = True


def scalarop(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


def arrayop(x):
    return jnp.stack([scalarop(x), jnp.cos(x) * jnp.sign(x + 0.2)], axis=-1)


def test_native_empty_clause1():
    assert cj.chebfun().fliplr().isempty()


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("row", [True, False])
def test_native_smooth_clauses2_through9(array, row):
    op = arrayop if array else scalarop
    f = cj.chebfun(op, domain=[-1, -0.2, 0.1, 1] if array else [-1, 0.1, 1], pref=PREF)
    if row:
        f = f.T
    out = f.fliplr()
    expected = op(-XR) if row else (arrayop(XR)[:, ::-1] if array else scalarop(XR))
    if row and array:
        expected = expected.T
    error = float(jnp.max(jnp.abs(out(XR) - expected).reshape(-1)))
    bound = (100 if array else 10) * float(out.vscale) * EPS
    clauses = (4, 5) if array and row else (2, 3) if row else (8, 9) if array else (6, 7)
    print({"source_clauses": clauses, "error": error, "bound": bound}, flush=True)
    assert error < bound
    assert out.fliplr().isequal(f)


@pytest.mark.parametrize("row", [False, True])
def test_native_singular_clauses10_11(row):
    dom = (-2.0, 7.0)
    power = -0.5
    op = lambda x: (x - dom[0] + 0j) ** power * jnp.sin(100 * x) * (x - dom[1] + 0j) ** power  # noqa: E731
    f = cj.chebfun(op, domain=dom, exps=(power, power), splitting=not row)
    if row:
        f = f.T
    out = f.fliplr()
    if row:
        expected = (
            (dom[0] - X + 0j) ** power * jnp.sin(100 * (sum(dom) - X)) * (dom[1] - X + 0j) ** power
        )
        error = float(jnp.sum(jnp.abs(out(X) - expected)))
        bound = 1e4 * EPS * float(jnp.sum(jnp.abs(expected)))
    else:
        expected = f(X)
        error = float(jnp.max(jnp.abs(expected - out(X))))
        bound = EPS * float(jnp.max(jnp.abs(expected)))
    print({"source_clause": 11 if row else 10, "error": error, "bound": bound}, flush=True)
    assert error < bound
