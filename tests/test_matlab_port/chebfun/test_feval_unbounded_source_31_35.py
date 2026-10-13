"""Native source clauses 31--35 from tests/chebfun/test_feval.m.

Inputs and native constructor outputs were captured by executing the exact
source tail sequentially after seedRNG(6178), with no intervening reseeds.
"""
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import chebfunjax as cj

_CAPTURE = loadmat(
    Path(__file__).with_name("data") / "feval_tail31_35_native.mat",
    squeeze_me=False,
)
_EPS = np.finfo(np.float64).eps


def _column(name):
    """Keep native MATLAB 100-by-1 query shape (do not squeeze)."""
    return np.asarray(_CAPTURE[name], dtype=np.float64)


def _vscale(f):
    return np.asarray(f.vscale, dtype=np.float64)


def _native_vscale(clause):
    return np.asarray(_CAPTURE[f"vscale{clause}"], dtype=np.float64)


def _capped_bound(source_bound, native_bound):
    """Keep the candidate bound no looser than the captured source bound."""
    return min(float(np.asarray(source_bound).max()),
               float(np.asarray(native_bound).max()))


def _diagnostic(clause, got, exact, python_scale, native_scale, bound, extra=""):
    if os.environ.get("CJ_FEVAL_SOURCE_DIAGNOSTICS") == "1":
        error_norm = np.linalg.norm(np.asarray(got) - np.asarray(exact), ord=np.inf)
        print(
            f"FEVAL_SOURCE clause={clause} error_norm={error_norm:.17g} "
            f"python_vscale={np.asarray(python_scale).tolist()} "
            f"native_vscale={np.asarray(native_scale).tolist()} "
            f"bound={float(bound):.17g} {extra}",
            flush=True,
        )


def test_native_capture_is_complete_and_original_clauses_pass():
    assert np.array_equal(_CAPTURE["pass"].reshape(-1), np.ones(5))
    for clause in (31, 32, 34, 35):
        assert _column(f"x{clause}").shape == (100, 1)
    # Native clause 33 reuses clause 32's query vector without another draw.
    assert _column("x32").shape == (100, 1)
    assert np.asarray(_CAPTURE["fval33"]).shape == (100, 1)
    assert np.asarray(_CAPTURE["exact33"]).shape == (100, 1)
    assert np.asarray(_CAPTURE["fval35"]).shape == (100, 3)
    assert np.asarray(_CAPTURE["exact35"]).shape == (100, 3)


def test_pass_31_singular_finite_endpoint():
    dom = np.asarray(_CAPTURE["dom31"]).reshape(-1)
    power = float(np.asarray(_CAPTURE["pow31"]).item())
    op = lambda x: (x - dom[0]) ** power * jnp.sin(200 * x)
    f = cj.chebfun(op, domain=tuple(dom), exps=(power, 0), splitting=True)
    x = jnp.asarray(_column("x31"))
    exact = np.asarray(_CAPTURE["exact31"])
    got = np.asarray(f(x))
    assert got.shape == (100, 1)
    bound = _capped_bound(1e5 * _EPS * np.linalg.norm(exact, ord=np.inf),
                          _CAPTURE["bound31"])
    _diagnostic(31, got, exact, _vscale(f), _native_vscale(31), bound)
    assert np.linalg.norm(got - exact, ord=np.inf) < bound


def test_pass_32_entire_line_removable_singularity_and_limits():
    dom = tuple(np.asarray(_CAPTURE["dom32"]).reshape(-1))
    op = lambda x: (1 - jnp.exp(-x**2)) / x
    f = cj.chebfun(op, domain=dom)
    # Native isempty.m pass(11) uses this exact operator and domain.
    assert not f.isempty()
    x = jnp.asarray(_column("x32"))
    exact = np.asarray(_CAPTURE["exact32"])
    got = np.asarray(f(x))
    assert got.shape == (100, 1)
    vscale = _vscale(f)
    native_vscale = _native_vscale(32)
    bound = _capped_bound(1e1 * _EPS * vscale, _CAPTURE["bound32"])
    endpoint_bound = _capped_bound(_EPS * vscale, _EPS * native_vscale)
    _diagnostic(32, got, exact, vscale, native_vscale, bound,
                f"endpoint_bound={endpoint_bound:.17g}")
    assert np.linalg.norm(got - exact, ord=np.inf) < bound
    assert np.asarray(f(jnp.inf)) < endpoint_bound
    assert np.asarray(f(-jnp.inf)) < endpoint_bound


def test_pass_33_entire_line_blowup_exponents_and_limits():
    dom = tuple(np.asarray(_CAPTURE["dom33"]).reshape(-1))
    op = lambda x: x**2 * (1 - jnp.exp(-x**2))
    f = cj.chebfun(op, domain=dom, exps=(2, 2))
    # Native isempty.m pass(12) uses this exact operator/domain/exponents.
    assert not f.isempty()
    x = jnp.asarray(_column("x32"))
    exact = np.asarray(_CAPTURE["exact33"])
    got = np.asarray(f(x))
    assert got.shape == (100, 1)
    vscale = _vscale(f)
    native_vscale = _native_vscale(33)
    bound = _capped_bound(1e5 * _EPS * vscale, _CAPTURE["bound33"])
    _diagnostic(33, got, exact, vscale, native_vscale, bound)
    assert np.linalg.norm(got - exact, ord=np.inf) < bound
    assert np.asarray(f(jnp.inf)) == np.inf
    assert np.asarray(f(-jnp.inf)) == np.inf


def test_pass_34_right_half_line():
    dom = tuple(np.asarray(_CAPTURE["dom34"]).reshape(-1))
    op = lambda x: x * jnp.exp(-x)
    f = cj.chebfun(op, domain=dom)
    x = jnp.asarray(_column("x34"))
    exact = np.asarray(_CAPTURE["exact34"])
    got = np.asarray(f(x))
    assert got.shape == (100, 1)
    vscale = _vscale(f)
    native_vscale = _native_vscale(34)
    bound = _capped_bound(1e1 * _EPS * vscale, _CAPTURE["bound34"])
    endpoint_bound = _capped_bound(1e2 * _EPS * vscale,
                                   1e2 * _EPS * native_vscale)
    _diagnostic(34, got, exact, vscale, native_vscale, bound,
                f"endpoint_bound={endpoint_bound:.17g}")
    assert np.linalg.norm(got - exact, ord=np.inf) < bound
    assert np.asarray(f(jnp.inf)) < endpoint_bound


def test_pass_35_left_half_line_array_norm_and_limits():
    dom = tuple(np.asarray(_CAPTURE["dom35"]).reshape(-1))
    op = lambda x: jnp.stack(
        (jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x), axis=-1
    )
    f = cj.chebfun(op, domain=dom)
    # Native isempty.m pass(13) uses this exact three-column operator/domain.
    assert not f.isempty()
    x = jnp.asarray(_column("x35"))
    exact = np.asarray(_CAPTURE["exact35"])
    got = np.asarray(f(x))
    vscale = _vscale(f)
    native_vscale = _native_vscale(35)
    bound = _capped_bound(1e2 * np.max(_EPS * vscale),
                          _CAPTURE["bound35"])
    endpoint_bound = _capped_bound(1e2 * _EPS * vscale,
                                   1e2 * _EPS * native_vscale)
    assert got.shape == (100, 3)
    # Native norm(matrix, inf) is maximum row sum, not maximum entry.
    _diagnostic(35, got, exact, vscale, native_vscale, bound,
                f"endpoint_bound={endpoint_bound:.17g}")
    assert np.linalg.norm(got - exact, ord=np.inf) < bound
    assert np.all(np.asarray(f(-jnp.inf)) < endpoint_bound)
