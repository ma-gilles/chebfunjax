"""Supplemental source/API controls for +cheb/gallerydisk.m at 7574c77.

The pinned source has no standalone native gallerydisk test. Literal source
expressions below are evaluated with NumPy as an independent host adapter;
these controls do not claim MATLAB numerical captures or RNG equivalence.
"""

import importlib
import json
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import cheb

EPS = np.finfo(float).eps

ENTRIES = {
    "challenge": (
        False,
        "exp(sin(50*x/3)) + sin(60*exp(y/3)) + sin(70*sin(x/3)) + sin(sin(80*y/3)) - sin(10*((x+y)/3)) + (x**2+y**2)/(4*3**2)",
    ),
    "ellipsepeg": (True, "1/(1+(4*((r*cos(t))**2+.5*(r*sin(t))**2))**20)"),
    "flower": (True, "sin(21*pi*(1+cos(pi*r))* (r**2-2*r**5*cos(5*(t-0.11))))"),
    "roundpeg": (True, "1/(1+(2*r)**100)"),
    "translatepeg": (False, "1/(1+(2*((x-.2)**2+x**2)**20))"),
    "squarepeg": (False, "1/((1+(2*x)**20)*(1+(2*y)**20))"),
    "tiltedpeg": (False, "1/((1+(2*x+.4*y)**20)*(1+(2*y-.4*x)**20))"),
    "wave": (False, "cos(20*x+5*y)**2*(1-(x**2+y**2))"),
    "yinyang": (True, "-cos(((sin(pi*r)*cos(t) + sin(2*pi*r)*sin(t)))/4)"),
}

RANDOM_NAMES = (
    "challenge",
    "flower",
    "poisson",
    "ellipsepeg",
    "roundpeg",
    "squarepeg",
    "tiltedpeg",
    "wave",
    "yinyang",
)


def source_value(name, a, b):
    expression = ENTRIES[name][1]
    scope = {
        "x": a,
        "y": b,
        "t": a,
        "r": b,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "pi": np.pi,
    }
    return np.asarray(eval(expression, {"__builtins__": {}}, scope))


@pytest.mark.parametrize("name", ENTRIES)
def test_source_callback_and_constructor_route(name, monkeypatch):
    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    calls = []
    sentinel = object()

    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(module, "diskfun", constructor)
    f, fa = cheb.gallerydisk(name.upper(), return_handle=True)
    assert f is sentinel
    assert calls == [((fa, "polar") if ENTRIES[name][0] else (fa,), {})]
    for a, b in [(0.23, 0.61), (np.array([[-0.8], [0.17], [0.9]]), np.array([[0.1, 0.7]]))]:
        expected = source_value(name, a, b)
        bound = 64 * EPS * max(1.0, float(np.max(np.abs(expected))))
        for callback in (fa, jax.jit(fa)):
            actual = np.asarray(callback(jnp.asarray(a), jnp.asarray(b)))
            assert actual.shape == expected.shape
            assert np.max(np.abs(actual - expected)) < bound


def test_poisson_source_route_and_second_output(monkeypatch):
    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    sentinel = object()
    calls = []

    def poisson(rhs, bc, n):
        calls.append((rhs, bc, n))
        return sentinel

    monkeypatch.setattr(module.Diskfun, "poisson", staticmethod(poisson))
    f, fa = cheb.gallerydisk("POISSON", return_handle=True)
    assert f is fa is sentinel
    assert len(calls) == 1
    rhs, bc, n = calls[0]
    assert n == 512
    for t, r in [(0.23, 0.61), (np.array([[-0.8], [0.17], [0.9]]), np.array([[0.1, 0.7]]))]:
        expected = -np.exp(-40 * (r**2 - 1) ** 4) * np.sinh(
            5 * (1 - r**11 * np.cos(11 * (t - 1 / np.sqrt(2))))
        )
        bound = 64 * EPS * max(1.0, float(np.max(np.abs(expected))))
        for callback in (rhs, jax.jit(rhs)):
            actual = np.asarray(callback(jnp.asarray(t), jnp.asarray(r)))
            assert actual.shape == np.asarray(expected).shape
            assert np.max(np.abs(actual - expected)) < bound
        np.testing.assert_array_equal(bc(jnp.asarray(t)), 0 * t + 1)


def test_single_output_and_namespace_export(monkeypatch):
    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    sentinel = object()
    monkeypatch.setattr(module, "diskfun", lambda *args, **kwargs: sentinel)
    assert "gallerydisk" in cheb.__all__
    assert cheb.gallerydisk("wave") is sentinel


def test_random_selection_native_nine_order(monkeypatch):
    import random

    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    sentinel = object()

    def choice(names):
        assert tuple(names) == RANDOM_NAMES
        assert "translatepeg" not in names
        return "wave"

    monkeypatch.setattr(random, "choice", choice)
    monkeypatch.setattr(module, "diskfun", lambda *args, **kwargs: sentinel)
    assert cheb.gallerydisk() is sentinel


def test_unknown_source_identifier():
    with pytest.raises(ValueError, match="CHEB:GALLERYDISK:unknown:unknownFunction"):
        cheb.gallerydisk("peaks")


@pytest.mark.parametrize("name", ("yinyang", "roundpeg"))
def test_actual_default_source_samples(name):
    started = time.monotonic()
    caught = []
    previous = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        caught.append({"category": category.__name__, "message": str(message)})
        return previous(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    try:
        f, fa = cheb.gallerydisk(name, return_handle=True)
        theta = np.array([-np.pi, -2.1, -0.3, 0.0, 0.7, 2.2, np.pi])
        radius = np.array([0.0, 0.11, 0.37, 0.63, 0.87, 1.0])
        t, r = np.meshgrid(theta, radius)
        actual = np.asarray(f(jnp.asarray(t), jnp.asarray(r)))
        a, b = (t, r) if ENTRIES[name][0] else (r * np.cos(t), r * np.sin(t))
        expected = source_value(name, a, b)
        error = float(np.max(np.abs(actual - expected)))
        bound = float(1000 * EPS * max(1.0, float(np.max(np.abs(expected)))))
        output = os.environ.get("GALLERYDISK_RESULT_DIR")
        if output:
            Path(output).mkdir(exist_ok=True)
            (Path(output) / f"{name}_samples.json").write_text(
                json.dumps(
                    {
                        "theta": t.tolist(),
                        "r": r.tolist(),
                        "actual": actual.tolist(),
                        "expected": expected.tolist(),
                        "error": error,
                        "bound": bound,
                    },
                    indent=2,
                )
                + "\n"
            )
        report = {
            "name": name,
            "error": error,
            "bound": bound,
            "success": bool(error < bound),
            "rank": int(f.rank),
            "length": list(f.length()),
            "elapsed_seconds": time.monotonic() - started,
            "warnings": caught,
            "theta": t.tolist(),
            "r": r.tolist(),
            "actual": actual.tolist(),
            "expected": expected.tolist(),
        }
        if output:
            Path(output).mkdir(exist_ok=True)
            (Path(output) / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        assert actual.shape == expected.shape
        assert np.all(np.isfinite(actual))
        assert error < bound
    finally:
        warnings.showwarning = previous
