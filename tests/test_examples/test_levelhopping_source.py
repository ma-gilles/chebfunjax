"""Literal native LevelHopping forcing, order and renderer controls."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from chebfunjax import chebfun
from chebfunjax.operators.chebop import _IVPProxy, _TrigX


def page():
    spec = importlib.util.spec_from_file_location(
        "levelhopping_source_controls",
        Path(__file__).resolve().parents[2] / "examples/ode-random/levelhopping.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_actual_nonperiodic_source_draws(monkeypatch):
    mod = page()
    rf = importlib.import_module("chebfunjax.utils._randnfun")
    original = rf._periodic_coefficients
    calls = []

    def observed(draws, length, big, cmplx):
        result = original(draws, length, big, cmplx)
        calls.append((np.asarray(draws), length, big, cmplx, np.asarray(result)))
        return result

    monkeypatch.setattr(rf, "_periodic_coefficients", observed)
    key = jax.random.PRNGKey(0)
    objects = []
    for lam in (0.4, 0.2):
        f, key = mod._draw_forcing(lam, key)
        objects.append(f)
    assert len(calls) == 2
    for (draws, length, big, cmplx, result), m in zip(calls, (300, 600)):
        assert draws.shape == (2, 2 * m + 1) and length == 120 and big is True and cmplx is False
        order = np.r_[np.arange(2 * m, -1, -2), np.arange(1, 2 * m + 1, 2)]
        c = draws[:, order].T
        c = (c[:, :1] + 1j * c[:, 1:]) / np.sqrt(2.0)
        c = (c + c[::-1].conj()) / np.sqrt(2.0)
        np.testing.assert_allclose(result, c / np.sqrt(120.0), rtol=0, atol=1e-15)
    for f in objects:
        assert tuple(f.domain.breakpoints) == (0.0, 100.0)
        assert all(type(p.tech).__name__ == "Chebtech2" for p in f.funs)
        assert bool(jnp.all(jnp.isfinite(f(jnp.linspace(0, 100, 81)))))
    np.testing.assert_array_equal(
        key, jax.random.split(jax.random.split(jax.random.PRNGKey(0))[0])[0]
    )


def test_literal_operator():
    mod = page()
    y, dy = map(jnp.asarray, (0.37, -0.8))
    actual = mod._operator(_IVPProxy([y, dy], x=jnp.asarray(7.0)))
    if isinstance(actual, _TrigX):
        actual = actual.v
    np.testing.assert_array_equal(actual, dy + 2 * jnp.sin(2 * jnp.pi * y))


def test_source_order(monkeypatch, capsys, tmp_path):
    mod = page()
    events = []
    operators = []
    solutions = []

    class Operator:
        def __init__(self, op, domain):
            assert domain == (0.0, 100.0)
            operators.append(self)

        def solve(self, rhs):
            assert self.lbc == 0.0
            events.append(("solve", rhs))
            value = object()
            solutions.append(value)
            return value

    def draw(lam, key):
        events.append(("draw", lam))
        return lam, key

    def plot(y, lw, name):
        events.append(("plot", solutions.index(y), lw, name))

    times = iter((100.0, 105.0))
    monkeypatch.setattr(mod, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    for name, value in [
        ("Chebop", Operator),
        ("_draw_forcing", draw),
        ("_plot", plot),
        ("_IMG", tmp_path),
    ]:
        monkeypatch.setattr(mod, name, value)
    result = mod.run()
    assert len(operators) == 1
    assert events == [
        ("draw", 0.4),
        ("solve", 0.4),
        ("plot", 0, 2, "LevelHopping_01.png"),
        ("draw", 0.2),
        ("solve", 0.2),
        ("plot", 1, 1, "LevelHopping_02.png"),
    ]
    assert result["forcing"] == (0.4, 0.2) and result["solutions"] == tuple(solutions)
    assert capsys.readouterr().out == "total_time_in_seconds =\n  5.000000\n"


@pytest.mark.parametrize("linewidth", [2, 1])
def test_actual_saved_artists(monkeypatch, tmp_path, linewidth):
    mod = page()
    save = mod.save_chebfun_figure
    seen = []

    def observe(fig, path, **kwargs):
        ax = fig.axes[0]
        assert len(ax.lines) == 1 and ax.lines[0].get_linewidth() == linewidth
        assert ax.get_xlabel() == "t" and ax.get_ylabel() == "y" and ax.get_title() == ""
        assert all(t.get_fontsize() == 32 for t in (ax.xaxis.label, ax.yaxis.label))
        assert any(l.get_visible() for l in ax.get_xgridlines())
        result = save(fig, path, **kwargs)
        fig.canvas.draw()
        for t in (ax.xaxis.label, ax.yaxis.label):
            b = t.get_window_extent(fig.canvas.get_renderer())
            assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height
        assert Image.open(path).size == (600, 269)
        seen.append(path)
        return result

    monkeypatch.setattr(mod, "save_chebfun_figure", observe)
    monkeypatch.setattr(mod, "_IMG", tmp_path)
    mod._plot(chebfun(lambda x: jnp.sin(x / 8), domain=mod.DOM), linewidth, "actual.png")
    assert len(seen) == 1
