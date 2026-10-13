"""Literal predicates from tests/chebfun/test_sqrt.m, Chebfun 7574c77.

NumPy RandomState(6178) is an explicit uniform-stream adapter; MATLAB RNG
identity is not claimed. Original root-touching controls are retained below.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)
_rng = np.random.RandomState(6178)
X = jnp.asarray(np.sort(9 * _rng.uniform(size=100) - 2))
XS = jnp.asarray(8.8 * _rng.uniform(size=100) - 1.9)
XF = jnp.asarray(200 * _rng.uniform(size=100) - 100)
XH = jnp.asarray(99 * _rng.uniform(size=100) + 1)


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("clause", range(1, 12))
def test_native_sqrt(clause):
    started = time.monotonic()
    pref = ChebfunPref()
    domain = (-2.0, 7.0)
    x = X
    kwargs = {}
    factor = 100.0
    scale_exact = True
    if clause == 1:
        op = lambda x: 1 / (1 + 25 * x**2)
        exact = lambda x: 1 / jnp.sqrt(1 + 25 * x**2)
        kwargs = {"pref": pref}
    elif clause == 2:
        op = lambda x: jnp.sin(50 * x) ** 2 + 1
        exact = lambda x: jnp.sqrt(jnp.sin(50 * x) ** 2 + 1)
        kwargs = {"splitting": True}
    elif clause == 3:
        op = lambda x: jnp.sin(50 * x)
        exact = lambda x: jnp.sqrt(jnp.sin(50 * x).astype(jnp.complex128))
        pref.splitting = True
        kwargs = {"pref": pref}
        factor = 1e4
    elif clause == 4:
        op = lambda x: jnp.sin(50 * x) + 1j * jnp.cos(30 * x)
        exact = lambda x: jnp.sqrt(op(x))
        pref.splitting = True
        kwargs = {"pref": pref}
        factor = 1e4
    elif clause == 5:
        op = lambda x: jnp.stack((jnp.sin(x), jnp.sin(x) - 0.5), axis=-1)
        exact = lambda x: jnp.sqrt(op(x).astype(jnp.complex128))
        factor = 1e3
    elif clause == 6:
        x = XS
        op = lambda x: (jnp.sin(50 * x) ** 2 + 1) * (x + 2) ** (-1.5)
        exact = lambda x: jnp.sqrt(jnp.sin(50 * x) ** 2 + 1) * (x + 2) ** (-0.75)
        kwargs = {"exps": [-1.5, 0.0], "splitting": True}
    elif clause in (7, 8):
        domain = (-jnp.inf, jnp.inf)
        x = XF
        scale_exact = False
        if clause == 7:
            op = lambda x: x**2 * jnp.exp(-(x**2)) + 2
        else:
            op = lambda x: x**2 * (1 - jnp.exp(-(x**2))) + 2
            kwargs = {"exps": [2.0, 2.0]}
            factor = 1e4
        exact = lambda x: jnp.sqrt(op(x))
    else:
        domain = (1.0, jnp.inf)
        x = XH
        scale_exact = False
        if clause == 9:
            op = lambda x: x * jnp.exp(-x) + 3
        elif clause == 10:
            op = lambda x: 0.1 + jnp.sin(10 * x) / jnp.exp(x)
            kwargs = {"splitting": True}
            factor = 1e5
        else:
            op = lambda x: x * (5 + jnp.exp(-(x**3)))
            kwargs = {"exps": [0.0, 1.0]}
        exact = lambda x: jnp.sqrt(op(x).astype(jnp.complex128))
    f = cj.chebfun(op, domain=domain, **kwargs)
    g = f.sqrt()
    expected = exact(x)
    actual = g(x)
    assert actual.shape == expected.shape == ((100, 2) if clause == 5 else (100,))
    # MATLAB numeric matrix infinity norm is a maximum row sum for clause 5.
    norm = lambda value: float(jnp.linalg.norm(value, ord=jnp.inf))
    error = norm(actual - expected)
    scale = norm(expected) if scale_exact else float(g.vscale)
    bound = factor * EPS * scale
    evidence = {
        "clause": clause,
        "error": error,
        "bound": bound,
        "scale": scale,
        "shape": list(actual.shape),
        "seconds": time.monotonic() - started,
    }
    print(evidence, flush=True)
    capture = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if capture:
        Path(capture).mkdir(exist_ok=True)
        (Path(capture) / f"sqrt_{clause}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert error < bound


def test_sqrt_keeps_unbounded_tail_mapping_after_root_split():
    """Fractional root powers preserve the native unbounded FUN mapping."""
    from chebfunjax.fun.unbndfun import Unbndfun

    op = lambda x: 0.1 + jnp.sin(10 * x) / jnp.exp(x)
    f = cj.chebfun(op, domain=(1.0, jnp.inf), splitting=True)
    g = f.sqrt()
    assert isinstance(g.funs[-1], Unbndfun)
    assert g.funs[-1].mapping_type == "right_inf"
    assert g.funs[-1].domain == f.funs[-1].domain
    x = jnp.asarray([31.0, 42.0, 64.0, 98.0])
    # The native clause immediately above owns the numerical error bound;
    # this control isolates the unbounded-map failure mode.
    assert np.all(np.isfinite(np.asarray(g(x))))


def test_sqrt_negative_unbounded_root_split_uses_principal_branch():
    """Exercise a negative left tail and interior root on an unbounded FUN."""
    from chebfunjax.fun.unbndfun import Unbndfun

    op = lambda x: x - 1.0
    f = cj.chebfun(op, domain=(-jnp.inf, 3.0), splitting=True)
    g = f.sqrt()
    assert isinstance(g.funs[0], Unbndfun)
    assert g.funs[0].mapping_type == "left_inf"
    x = jnp.asarray([-100.0, -10.0, -1.0, 0.0, 0.5, 1.5, 2.0, 3.0])
    actual = g(x)
    expected = jnp.sqrt(op(x).astype(jnp.complex128))
    # This is a branch/mapping control; the MATLAB sqrt suite has no negative
    # whole-line tail predicate, so it deliberately adds no new error bound.
    assert np.all(np.isfinite(np.asarray(actual)))
    assert bool(jnp.all(jnp.imag(actual[:5]) > 0))
    assert bool(jnp.all(jnp.real(actual[5:]) > 0))
    print({"negative_unbounded_root_max_error": float(jnp.max(jnp.abs(actual - expected)))})


@pytest.mark.parametrize("transposed", [False, True])
def test_root_split_power_maps_overridden_point_values(transposed):
    """Native power maps stored values even when it inserts a root break."""
    f = cj.chebfun(lambda x: x - 0.25, domain=(-1.0, 1.0))
    f = f.define_point(-0.5, 4.0)
    source = f.T if transposed else f

    result = source.sqrt()

    assert result.is_transposed is transposed
    assert tuple(result.domain.breakpoints) == (-1.0, -0.5, 0.25, 1.0)
    values = result.point_values
    assert values.shape == (4,)
    # MATLAB @chebfun/power.m:215 applies .^b to pointValues. The assigned
    # non-root value 4 therefore remains 2, despite the smooth value -0.75.
    # A negative real stored value follows the principal complex branch.
    assert bool(jnp.isfinite(values[0]))
    assert bool(jnp.imag(values[0]) > 0)
    assert complex(values[1]) == 2.0 + 0.0j
    # The root inserted by addBreaksAtRoots keeps its source point value zero.
    assert complex(values[2]) == 0.0 + 0.0j
    evaluated = jnp.ravel(result(-0.5))[0]
    assert complex(evaluated) == 2.0 + 0.0j


class TestChebfunSqrt:
    def test_runge_reciprocal(self):
        f = cj.chebfun(lambda x: 1.0 / (1 + 25 * x**2), domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = 1.0 / jnp.sqrt(1 + 25 * X**2)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e2 * EPS * float(jnp.max(exact))

    def test_oscillatory_positive(self):
        f = cj.chebfun(lambda x: jnp.sin(50 * x) ** 2 + 1, domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = jnp.sqrt(jnp.sin(50 * X) ** 2 + 1)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * float(jnp.max(exact))

    def test_root_touching(self):
        # sqrt of a non-negative function that touches zero at its endpoints
        # carries a branch point into a fractional (0.5) SingFun exponent
        # (MATLAB @chebfun/power.m -> @singfun/power.m).  1 - x^2 vanishes at
        # x = +/-1, so sqrt(1 - x^2) has exponents (0.5, 0.5).
        f = cj.chebfun(lambda x: 1.0 - x**2, domain=(-1.0, 1.0))
        g = f.sqrt()
        from chebfunjax.fun.singfun import Singfun

        assert isinstance(g.funs[0].tech, Singfun)
        assert g.funs[0].tech.exponents == (0.5, 0.5)
        assert not bool(g.isinf())
        xt = jnp.asarray(np.linspace(-0.98, 0.98, 50))
        exact = jnp.sqrt(1.0 - xt**2)
        err = float(jnp.max(jnp.abs(g(xt) - exact)))
        assert err < 1e2 * EPS

    def test_root_touching_interior(self):
        # A non-negative function with an INTERIOR double root: sqrt gets a
        # breakpoint at the root and stays finite (|x| on either side).
        f = cj.chebfun(lambda x: x**2, domain=(-1.0, 1.0))
        g = f.sqrt()
        xt = jnp.asarray(np.linspace(-0.97, 0.97, 41))
        err = float(jnp.max(jnp.abs(g(xt) - jnp.abs(xt))))
        assert err < 1e2 * EPS
        assert not bool(g.isinf())
