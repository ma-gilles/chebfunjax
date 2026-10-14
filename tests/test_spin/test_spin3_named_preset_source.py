"""Named spin3 parser controls from the pinned MATLAB wrapper."""

from __future__ import annotations

import jax
import pytest

import chebfunjax.operators.spinop3 as spinop3_module
from chebfunjax.operators import _spin3_public
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.operators.spinpref import SpinPref3
from chebfunjax.utils import _randnfun

PRESETS = {
    "GL": (32, 0.1, (-1.0, 1.0), 2, 64),
    "GS": (32, 5.0, (0.25, 0.8, 0.0, 0.35), 10, 64),
    "SCHNAK": (32, 0.5, (0.7, 1.7, 0.65, 1.05), 10, 64),
    "SH": (32, 1.0, (-0.4, 0.5), 4, 64),
}


@pytest.fixture(autouse=True)
def _fixed_test_random_stream(monkeypatch):
    # Named preset construction must not perturb the process-wide RNG stream.
    monkeypatch.setattr(_randnfun, "_next_key", lambda: jax.random.key(0))


def _capture_solver(monkeypatch):
    seen = {}

    def capture(S, N, dt, args, kwargs, *, return_times=False):
        seen.update(S=S, N=N, dt=dt, args=args, kwargs=kwargs,
                    return_times=return_times)
        return "captured"

    monkeypatch.setattr(_spin3_public, "solve_public", capture)
    return seen


@pytest.mark.parametrize("name", PRESETS)
def test_named_preset_defaults_reach_solver_without_running_trajectory(monkeypatch, name):
    seen = _capture_solver(monkeypatch)
    assert spin3(name, return_times=True) == "captured"
    n, dt, clim, iterplot, nplot = PRESETS[name]
    assert isinstance(seen["S"], Spinop3)
    assert seen["N"] == n
    assert seen["dt"] == dt
    assert seen["return_times"] is True
    assert len(seen["args"]) == 1
    pref = seen["args"][0]
    assert isinstance(pref, SpinPref3)
    assert (pref.Clim, pref.iterplot, pref.Nplot) == (clim, iterplot, nplot)
    assert (pref.dealias, pref.M, pref.scheme, pref.plot) == (
        "off", 32, "etdrk4", "movie"
    )


def test_named_preference_pairs_and_keyword_options(monkeypatch):
    seen = _capture_solver(monkeypatch)
    original_factory = spinop3_module.Spinop3
    constructions = []

    def capture_factory(name):
        constructions.append(name)
        return original_factory(name)

    monkeypatch.setattr(spinop3_module, "Spinop3", capture_factory)
    spin3("GS", "plot", "off", "scheme", "pecec736", M=8)
    # Python keyword preferences map to additional native name/value pairs,
    # so parseInputs constructs the named operator once.
    assert constructions == ["GS"]
    pref = seen["args"][0]
    assert (seen["N"], seen["dt"]) == (32, 5.0)
    assert (pref.plot, pref.scheme, pref.M) == ("off", "pecec736", 8)
    assert pref.Clim == PRESETS["GS"][2]


def test_explicit_operator_forms_are_forwarded_unchanged(monkeypatch):
    seen = _capture_solver(monkeypatch)
    S = Spinop3((0, 1, 0, 1, 0, 1), (0, 0.2))
    pref = SpinPref3(M=7, plot="off")

    spin3(S, 12, 0.025, pref, return_times=True)
    assert seen == {
        "S": S, "N": 12, "dt": 0.025, "args": (pref,), "kwargs": {},
        "return_times": True,
    }

    spin3(S, 9, 0.05, "scheme", "krogstad")
    assert seen["S"] is S
    assert (seen["N"], seen["dt"]) == (9, 0.05)
    assert seen["args"] == ("scheme", "krogstad")


def test_source_supported_named_arity_and_invalid_options(monkeypatch):
    seen = _capture_solver(monkeypatch)
    spin3("SCHNAK", "plot", "off")
    assert (seen["N"], seen["dt"]) == (32, 0.5)
    assert seen["args"][0].plot == "off"

    spin3("SH", "plot", "off", "dealias", "on")
    assert seen["args"][0].dealias == "on"

    with pytest.raises(ValueError, match="HELP/SPINSPHERE"):
        spin3("not-a-preset")
    with pytest.raises(TypeError, match="preference pairs"):
        spin3("GL", "plot")
    with pytest.raises(ValueError, match="preference"):
        spin3("GL", "not-a-pref", 1)
    with pytest.raises(TypeError, match="require S, N, and dt"):
        spin3(Spinop3())


def test_bare_preset_matches_native_two_constructor_path(monkeypatch):
    seen = _capture_solver(monkeypatch)
    original_factory = spinop3_module.Spinop3
    constructions = []

    def capture_factory(name):
        constructions.append(name)
        return original_factory(name)

    monkeypatch.setattr(spinop3_module, "Spinop3", capture_factory)
    assert spin3("GL", return_times=True) == "captured"
    assert constructions == ["GL", "GL"]
    assert seen["return_times"] is True
    assert seen["S"].domain == (0.0, 50.0, 0.0, 50.0, 0.0, 50.0)
