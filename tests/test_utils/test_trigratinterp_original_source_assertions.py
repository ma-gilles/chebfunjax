"""Independent source-assertion tests for MATLAB trigratinterp.

Each source assertion is one pytest case (24 assignments, 23 MATLAB pass
slots because pass(6) is assigned twice). Deterministic source-order random
arrays are shared; fit blocks are lazily cached so a failure in one block does
not prevent unrelated assertion cases from being collected/executed.

Provenance
----------
MATLAB source: tests/chebfun/test_trigratinterp.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import inspect
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import trigpts
from chebfunjax.utils.ratapprox import trigratinterp

jax.config.update("jax_enable_x64", True)

# NumPy MT6178 is a Python RNG adapter, not a claim of MATLAB stream identity.
TOL = 1e-14


def _tp(n):
    pts = trigpts(n)
    return np.asarray(pts[0] if isinstance(pts, tuple) else pts)


def _fit(f, m, n, NN=None, xi=None, tol=TOL):
    domain = None
    f_domain = getattr(f, "domain", None)
    if f_domain is not None and hasattr(f_domain, "breakpoints"):
        domain = (f_domain.breakpoints[0], f_domain.breakpoints[-1])
    effective_domain = domain if domain is not None else (-1.0, 1.0)

    # The newer source adapter exposes MATLAB's first three outputs directly.
    # The older Python API exposes (r, ac, bc, ...); reconstruct p and q from
    # those same Fourier coefficient vectors on the already-resolved domain.
    if "outputs" in inspect.signature(trigratinterp).parameters:
        return trigratinterp(
            f, m, n, NN, xi, tol=tol, domain=domain, outputs=3,
        )
    r, ac, bc, *_ = trigratinterp(f, m, n, NN, xi, tol=tol, domain=effective_domain)
    p = chebfun(ac, coeffs=True, trig=True, domain=effective_domain)
    q = chebfun(bc, coeffs=True, trig=True, domain=effective_domain)
    return p, q, r


def _eval_ratio(p, q, x):
    x = jnp.asarray(x)
    return np.asarray(p(x)) / np.asarray(q(x))


@pytest.fixture(scope="module")
def source_rng_inputs():
    rng = np.random.RandomState(6178)
    # Preserve the order of all random draws in the MATLAB file.
    th1 = np.sort(-1.0 + 2.0 * rng.rand(5))
    tt1 = 2.0 * rng.rand(100) - 1.0
    tt2 = 2.0 * rng.rand(100) - 1.0
    tt3 = 2.0 * rng.rand(100) - 1.0
    th4 = -1.0 + 2.0 * rng.rand(21)
    tt5 = 2.0 * rng.rand(100) - 1.0
    return {"th1": th1, "tt1": tt1, "tt2": tt2, "tt3": tt3,
            "th4": th4, "tt5": tt5}


@pytest.fixture(scope="module")
def source_blocks(source_rng_inputs):
    inp = source_rng_inputs

    @lru_cache(None)
    def block1():
        fh = lambda x: np.tan(np.pi * x)  # noqa: E731
        p, q, _ = _fit(jnp.asarray(fh(inp["th1"])), 1, 1, xi=inp["th1"], tol=0.0)
        return p, q, fh

    @lru_cache(None)
    def block2():
        fh = lambda x: np.tan(np.pi * x)  # noqa: E731
        p, q, _ = _fit(lambda x: jnp.tan(jnp.pi * x), 1, 1,
                       NN=101, tol=0.0)
        return p, q, fh

    @lru_cache(None)
    def block3():
        fh = lambda x: np.tan(np.pi * x)  # noqa: E731
        th = _tp(51)
        t = chebfun("t")
        f = (np.pi * t).sin() / (np.pi * t).cos()
        p, q, _ = _fit(jnp.asarray(f(jnp.asarray(th))), 10, 15)
        return p, q, fh, th

    @lru_cache(None)
    def block4():
        fh = lambda x: np.exp(-4.0 * np.sin(np.pi * x / 2.0) ** 2)  # noqa: E731
        f = chebfun(lambda x: jnp.exp(-4.0 * jnp.sin(jnp.pi * x / 2.0) ** 2),
                    trig=True)
        p, q, _ = _fit(f, 5, 5, xi=inp["th4"], tol=0.0)
        return p, q, fh

    @lru_cache(None)
    def block5():
        fh = lambda x: np.cos(np.pi * x) / np.cos(2.0 * np.pi * x)  # noqa: E731
        th = _tp(51)
        p, q, _ = _fit(jnp.asarray(fh(th)), 10, 15)
        return p, q, fh, th

    @lru_cache(None)
    def block6():
        fh = lambda x: np.exp(np.sin(x))  # noqa: E731
        a, b = 2.0, 2.0 + 2.0 * np.pi
        f = chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(a, b), trig=True)
        xi = 2.0 + 2.0 * np.pi / 51.0 * np.arange(51)
        p, q, r = _fit(f, 6, 6, xi=xi)
        return fh, a, b, r

    @lru_cache(None)
    def block7():
        fi = np.asarray([2.0, 3.0, 1.0])
        xi = np.asarray([-1.0, 0.5, 0.8])
        p, q, r = _fit(jnp.asarray(fi), 1, 0, NN=3, xi=xi)
        return p, q, r, fi, xi

    return {1: block1, 2: block2, 3: block3, 4: block4, 5: block5,
            6: block6, 7: block7}


def _checks(blocks, inputs):
    # Keep each source block lazy. A fit error in one block should not block
    # the other independently collected pass assertions.
    checks = {}
    checks["pass01_p_length"] = lambda: _assert(len(blocks[1]()[0]) == 3)
    checks["pass02_q_length"] = lambda: _assert(len(blocks[1]()[1]) == 3)
    checks["pass03_samples"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[1]()[:2], inputs["th1"])
                      - blocks[1]()[2](inputs["th1"]))) < TOL)
    checks["pass04_offgrid"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[1]()[:2], inputs["tt1"])
                      - blocks[1]()[2](inputs["tt1"]))) < 1e3*TOL)

    checks["pass05_p_length"] = lambda: _assert(len(blocks[2]()[0]) == 3)
    checks["pass06a_q_length"] = lambda: _assert(len(blocks[2]()[1]) == 3)
    checks["pass07_samples"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[2]()[:2], inputs["th1"])
                      - blocks[2]()[2](inputs["th1"]))) < TOL)
    checks["pass08_offgrid"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[2]()[:2], inputs["tt2"])
                      - blocks[2]()[2](inputs["tt2"]))) < 1e3*TOL)

    checks["pass09_p_length"] = lambda: _assert(len(blocks[3]()[0]) == 3)
    checks["pass10_q_length"] = lambda: _assert(len(blocks[3]()[1]) == 3)
    checks["pass11_samples_literal_1e3tol"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[3]()[:2], blocks[3]()[3])
                      - blocks[3]()[2](blocks[3]()[3]))) < 1e3*TOL)
    checks["pass12_offgrid"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[3]()[:2], inputs["tt3"])
                      - blocks[3]()[2](inputs["tt3"]))) < 1e3*TOL)

    checks["pass13_p_length"] = lambda: _assert(len(blocks[4]()[0]) == 11)
    checks["pass14_q_length"] = lambda: _assert(len(blocks[4]()[1]) == 11)
    checks["pass15_grid"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[4]()[:2], _tp(21))
                      - blocks[4]()[2](_tp(21)))) < 5e-10)

    checks["pass16_p_length"] = lambda: _assert(len(blocks[5]()[0]) == 3)
    checks["pass06b_q_length"] = lambda: _assert(len(blocks[5]()[1]) == 5)
    checks["pass17_samples"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[5]()[:2], blocks[5]()[3])
                      - blocks[5]()[2](blocks[5]()[3]))) < 1e2*TOL)
    checks["pass18_offgrid"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[5]()[:2], inputs["tt5"])
                      - blocks[5]()[2](inputs["tt5"]))) < 1e3*TOL)

    def check_pass19():
        fh, a, b, r = blocks[6]()
        xx = np.linspace(a, b, 10001)
        assert np.max(np.abs(fh(xx) - np.asarray(r(jnp.asarray(xx))))) < 10*TOL
    checks["pass19_shifted_domain"] = check_pass19

    checks["pass20_data_ratio"] = lambda: _assert(
        np.max(np.abs(_eval_ratio(*blocks[7]()[:2], blocks[7]()[4])
                      - blocks[7]()[3])) < 10*TOL)
    checks["pass21_data_handle"] = lambda: _assert(
        np.max(np.abs(np.asarray(blocks[7]()[2](jnp.asarray(blocks[7]()[4])))
                      - blocks[7]()[3])) < 10*TOL)
    checks["pass22_p_length"] = lambda: _assert(len(blocks[7]()[0]) == 3)
    checks["pass23_q_length"] = lambda: _assert(len(blocks[7]()[1]) == 1)
    return checks


def _assert(condition):
    assert bool(condition)


@pytest.mark.parametrize(
    "assertion",
    ["pass01_p_length", "pass02_q_length", "pass03_samples", "pass04_offgrid",
     "pass05_p_length", "pass06a_q_length", "pass07_samples", "pass08_offgrid",
     "pass09_p_length", "pass10_q_length", "pass11_samples_literal_1e3tol",
     "pass12_offgrid", "pass13_p_length", "pass14_q_length", "pass15_grid",
     "pass16_p_length", "pass06b_q_length", "pass17_samples", "pass18_offgrid",
     "pass19_shifted_domain", "pass20_data_ratio", "pass21_data_handle",
     "pass22_p_length", "pass23_q_length"],
)
def test_each_matlab_assertion(assertion, source_blocks, source_rng_inputs):
    # Each assertion is an independent pytest item. Block constructors are
    # lazy and cached, so unrelated groups still execute if one fit fails.
    _checks(source_blocks, source_rng_inputs)[assertion]()
