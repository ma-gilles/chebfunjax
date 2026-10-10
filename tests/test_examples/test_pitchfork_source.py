"""Pitchfork source forcing, operators, five-call signs and saved artists."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from PIL import Image

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import _IVPProxy, _TrigX


def page():
    spec = importlib.util.spec_from_file_location(
        "qualified_pitchfork_controls",
        Path(__file__).resolve().parents[2] / "examples/ode-random/pitchfork.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_actual_nonperiodic_draws(monkeypatch):
    mod = page()
    rf = importlib.import_module("chebfunjax.utils._randnfun")
    original = rf._periodic_coefficients
    calls = []

    def observe(draws, length, big, cmplx):
        result = original(draws, length, big, cmplx)
        calls.append((np.asarray(draws), length, big, cmplx, np.asarray(result)))
        return result

    monkeypatch.setattr(rf, "_periodic_coefficients", observe)
    key = jax.random.PRNGKey(0)
    f, key = mod._draw_forcing(key)
    g, key = mod._draw_forcing(key)
    assert len(calls) == 2
    for draws, length, big, cmplx, result in calls:
        assert draws.shape == (2, 721) and length == 720 and big is True and cmplx is False
        order = np.r_[np.arange(720, -1, -2), np.arange(1, 721, 2)]
        c = draws[:, order].T
        c = (c[:, :1] + 1j * c[:, 1:]) / np.sqrt(2.0)
        c = (c + c[::-1].conj()) / np.sqrt(2.0)
        np.testing.assert_allclose(result, c / np.sqrt(720.0), rtol=0, atol=1e-15)
    assert not np.array_equal(calls[0][0], calls[1][0])
    for obj in (f, g):
        assert tuple(obj.domain.breakpoints) == (0.0, 600.0)
        assert all(type(piece.tech).__name__ == "Chebtech2" for piece in obj.funs)
        assert bool(jnp.all(jnp.isfinite(obj(jnp.linspace(0, 600, 81)))))
    expected = jax.random.split(jax.random.split(jax.random.PRNGKey(0))[0])[0]
    np.testing.assert_array_equal(key, expected)


@pytest.mark.parametrize("damped", [False, True])
def test_literal_operator(damped):
    mod = page()
    t = jnp.asarray(321.0)
    y, dy, ddy = map(jnp.asarray, (0.25, -0.8, 1.5))
    actual = mod._operator(damped)(t, _IVPProxy([y, dy, ddy], x=t))
    if isinstance(actual, _TrigX):
        actual = actual.v
    expected = ddy - 2 * (-1 + t / 300) * y + 4 * y**3
    if damped:
        expected = expected + 0.2 * dy
    np.testing.assert_array_equal(actual, expected)


def test_full_source_order_and_signs_without_integration(monkeypatch, capsys, tmp_path):
    mod = page()
    events = []
    draws = iter((2.0, 3.0))
    solutions = []
    plots = []

    class Operator:
        def __init__(self, op, domain):
            self.op = op
            assert domain == (0.0, 600.0)

        def solve(self, rhs):
            assert self.lbc == [0.0, 0.0]
            events.append(("solve", rhs))
            solution = object()
            solutions.append(solution)
            return solution

    def draw(key):
        value = next(draws)
        events.append(("draw", value))
        return value, key

    def plot(sol, title, name):
        plots.append((sol, title, name))

    times = iter((100.0, 104.0))
    monkeypatch.setattr(mod, "Chebop", Operator)
    monkeypatch.setattr(mod, "_draw_forcing", draw)
    monkeypatch.setattr(mod, "_plot", plot)
    monkeypatch.setattr(mod, "_IMG", tmp_path)
    monkeypatch.setattr(mod, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    result = mod.run()
    assert events == [
        ("solve", 0.0),
        ("draw", 2.0),
        ("solve", 2.0),
        ("draw", 3.0),
        ("solve", 3.0),
        ("solve", -2.0),
        ("solve", 3.0),
    ]
    assert plots[0][0] == tuple(solutions[:3])
    assert plots[1][0] == (solutions[0], solutions[3], solutions[4])
    assert result["forcing"] == (2.0, 3.0)
    assert capsys.readouterr().out == "total_time_in_seconds =\n  4.000000\n"


@pytest.mark.parametrize("title", ["Pitchfork", "Pitchfork with damping"])
def test_saved_source_artists(monkeypatch, tmp_path, title):
    mod = page()
    original = mod.save_chebfun_figure
    observed = []

    def save(fig, path, **kwargs):
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        assert [line.get_linestyle() for line in ax.lines] == ["--", "-", "-"]
        assert [to_rgba(line.get_color()) for line in ax.lines] == [
            to_rgba(c) for c in ("k", "b", "r")
        ]
        assert all(line.get_linewidth() == 2.5 for line in ax.lines)
        assert ax.get_xlim() == (0.0, 600.0) and ax.get_ylim() == (-0.8, 0.8)
        assert ax.get_title() == title
        assert all(t.get_fontsize() == 32 for t in (ax.title, ax.xaxis.label, ax.yaxis.label))
        assert ax.get_xlabel() == "t" and ax.get_ylabel() == "y"
        result = original(fig, path, **kwargs)
        fig.canvas.draw()
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            b = t.get_window_extent(fig.canvas.get_renderer())
            assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height
        assert Image.open(path).size == (600, 269)
        observed.append(path)
        return result

    monkeypatch.setattr(mod, "_IMG", tmp_path)
    monkeypatch.setattr(mod, "save_chebfun_figure", save)
    sol = [
        chebfun(lambda x: 0 * x, domain=mod.DOM),
        chebfun(lambda x: x / 1000, domain=mod.DOM),
        chebfun(lambda x: -x / 1000, domain=mod.DOM),
    ]
    mod._plot(sol, title, "actual.png")
    assert len(observed) == 1 and not plt.get_fignums()
