"""Chebfun3 happiness preference dispatch from pinned native predicates.

MATLAB provenance: @chebfun3/chebfun3f.m:513-542,
@chebtech/happinessCheck.m:60-94, @trigtech/happinessCheck.m:32-67,
tests/chebtech/test_happinessCheck.m:67-108.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d._chebfun3classic import happiness_btd
from chebfunjax.chebfun3d._chebfun3f_tech import envelope_happiness
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech._trig_constructor import happiness as trig_happiness
from chebfunjax.tech._trig_constructor import resolve_pref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech, trigpts
from chebfunjax.utils.quadrature import chebpts

_NATIVE_HAPPINESS = json.loads(
    (Path(__file__).parent / "fixtures" / "native_happiness_c1c2_7574c77.json")
    .read_text()
)["cases"]


def test_original_scalar_strict_classic_predicate_reaches_public_tech_checks():
    tol = 2.0**-52
    def operator(x):
        return jnp.sin(10.0 * (x - 0.1))
    for tech, kind in ((Chebtech1, 1), (Chebtech2, 2)):
        values_by_length = {}
        coeffs_by_length = {}
        for n in (39, 41):
            values = operator(chebpts(n, kind=kind))
            values_by_length[n] = values
            coeffs_by_length[n] = tech.vals2coeffs(values)
        strict_39 = tech.happiness_check(
            coeffs_by_length[39], values_by_length[39], tol=tol,
            check="strict", sample_test=False,
        )[0]
        strict_41 = tech.happiness_check(
            coeffs_by_length[41], values_by_length[41], tol=tol,
            check="strict", sample_test=False,
        )[0]
        classic_39 = tech.happiness_check(
            coeffs_by_length[39], values_by_length[39], tol=tol,
            check="classic", sample_test=False,
        )[0]
        assert (not strict_39) and strict_41 and classic_39


@pytest.mark.parametrize(
    "tech,modes",
    [
        ("chebtech1", ("standard", "classic", "strict", "plateau")),
        ("chebtech2", ("standard", "classic", "strict", "plateau")),
        ("trigtech", ("standard", "classic", "plateau")),
    ],
)
def test_public_pref_threads_each_native_named_mode(monkeypatch, tech, modes):
    class ReachedFromFunction(Exception):
        pass

    observed = []

    def capture(cls, *args, **kwargs):
        observed.append(kwargs)
        raise ReachedFromFunction

    monkeypatch.setattr(Chebfun3, "from_function", classmethod(capture))
    for mode in modes:
        pref = ChebfunPref(tech=tech, happinessCheck=mode)
        with pytest.raises(ReachedFromFunction):
            chebfun3(lambda x, y, z: x + y + z, pref=pref)
        assert observed[-1]["_happiness_check"] == mode
        assert observed[-1]["_happiness_pref"].happinessCheck == mode


@pytest.mark.parametrize("mode", ["classic", "plateau"])
def test_periodic_envelope_uses_native_supported_alternate_check(mode):
    n = 33
    values = jnp.sin(2.0 * jnp.pi * jnp.arange(n) / n)[:, None]
    assert isinstance(envelope_happiness(
        values, Trigtech, 1e-12, mode, ChebfunPref(tech="trigtech")
    ), bool)


@pytest.mark.parametrize("n", [16, 17])
@pytest.mark.parametrize("kind", ["periodic", "nonperiodic"])
@pytest.mark.parametrize("scales", [(1.0, 1e-3), (1e-3, 1.0)])
@pytest.mark.parametrize("mode", ["standard", "classic", "plateau"])
def test_trig_named_modes_match_native_multifiber_controls(n, kind, scales, mode):
    """24 native controls captured from trigtech.happinessCheck/classicCheck."""
    x = trigpts(n)
    if kind == "periodic":
        fibers = jnp.stack((
            scales[0] * (1 + 0.25*jnp.cos(2*jnp.pi*x) + 0.125*jnp.sin(3*jnp.pi*x)),
            scales[1] * (0.5 + 0.375*jnp.cos(4*jnp.pi*x)
                         - 0.0625*jnp.sin(2*jnp.pi*x)
                         + 0.05j*jnp.sin(3*jnp.pi*x)),
        ), axis=1)
    else:
        fibers = jnp.stack((
            scales[0] * (1 + 0.25*jnp.cos(2*x) + 0.125*jnp.sin(3*x)),
            scales[1] * (0.5 + 0.375*jnp.cos(4*x)
                         - 0.0625*jnp.sin(2*x) + 0.05j*jnp.sin(3*x)),
        ), axis=1)
    envelope = jnp.sum(jnp.abs(Trigtech.vals2coeffs(fibers)), axis=1)
    scale = jnp.max(jnp.abs(Trigtech.coeffs2vals(envelope)))
    checker = Trigtech.from_coeffs(envelope)
    pref = resolve_pref({"happinessCheck": mode, "chebfuneps": 2.0**-52})
    happy, cutoff = trig_happiness(
        checker, values=envelope, data={"vscale": scale}, pref=pref
    )
    expected_happy = kind == "periodic" and mode in {"classic", "plateau"}
    expected_cutoff = 17 if mode == "standard" else 9
    assert bool(happy) is expected_happy
    assert int(cutoff) == expected_cutoff


def test_periodic_classic_receives_source_multifiber_envelope(monkeypatch):
    import chebfunjax.tech._trig_constructor as trig_constructor

    n = 33
    grid = jnp.arange(n, dtype=jnp.float64) / n
    fibers = jnp.stack((2.0 + 0.3 * jnp.cos(2 * jnp.pi * grid),
                        1e-3 * jnp.sin(4 * jnp.pi * grid)), axis=1)
    coefficients = Trigtech.vals2coeffs(fibers)
    expected_envelope = jnp.sum(jnp.abs(coefficients), axis=1)
    expected_vscale = jnp.max(jnp.abs(Trigtech.coeffs2vals(expected_envelope)))
    seen = {}

    def capture(checker, op, values, data, pref):
        seen["coeffs"] = checker.coeffs
        seen["values"] = values
        seen["data"] = data
        seen["pref"] = pref
        return True, 7

    monkeypatch.setattr(trig_constructor, "happiness", capture)
    assert envelope_happiness(
        fibers, Trigtech, 1e-12, "classic", ChebfunPref(tech="trigtech")
    )
    assert jnp.array_equal(seen["coeffs"], expected_envelope)
    assert jnp.array_equal(seen["values"], expected_envelope)
    assert seen["data"]["vscale"] == expected_vscale
    assert seen["pref"]["happinessCheck"] == "classic"


@pytest.mark.parametrize(
    "tech,mode,message",
    [
        (Chebtech2, "loose", "CHEBFUN:CHEBTECH:looseCheck:notDone"),
        (Trigtech, "strict", "Strict check not implemented for TRIGTECH"),
        (Trigtech, "loose", "Loose check not implemented for TRIGTECH"),
    ],
)
def test_unimplemented_native_modes_keep_native_errors(tech, mode, message):
    values = jnp.sin(jnp.linspace(-1.0, 1.0, 17))[:, None]
    with pytest.raises(ValueError, match=message):
        envelope_happiness(values, tech, 1e-12, mode)


@pytest.mark.parametrize("tech", ["chebtech1", "chebtech2"])
@pytest.mark.parametrize("mode", ["standard", "classic", "strict", "plateau"])
def test_public_polynomial_modes_construct_simple_chebfun3(tech, mode, record_property):
    pref = ChebfunPref(tech=tech, happinessCheck=mode)
    native = next(
        case for case in _NATIVE_HAPPINESS
        if case["technology"] == tech and case["mode"] == mode
    )
    if tech == "chebtech1":
        with pytest.warns(
            RuntimeWarning, match="chebfun3f: max number of restarts reached"
        ) as observed:
            result = chebfun3(lambda x, y, z: x + 2 * y - z, pref=pref)
        assert str(observed[0].message) == native["warningMessage"]
        expected = native["pointValue"]
    else:
        result = chebfun3(lambda x, y, z: x + 2 * y - z, pref=pref)
        expected = native["pointValue"]
    assert isinstance(result, Chebfun3)
    x, y, z = 0.125, -0.25, 0.375
    actual_value = float(result(x, y, z))
    actual_lengths = tuple(result.length())
    actual_ranks = tuple(result.rank)
    diagnostic = {
        "technology": tech, "mode": mode,
        "native_point": native["pointValue"], "actual_point": actual_value,
        "native_lengths": native["lengths"], "actual_lengths": actual_lengths,
        "native_ranks": native["ranks"], "actual_ranks": actual_ranks,
        "native_warning": native["warningMessage"],
    }
    record_property("native_mode_diagnostic", json.dumps(diagnostic, sort_keys=True))
    assert actual_lengths == tuple(native["lengths"]), diagnostic
    assert actual_ranks == tuple(native["ranks"]), diagnostic
    tol = 100 * pref.cheb3Prefs.chebfun3eps
    assert abs(actual_value - expected) < tol, diagnostic


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_polynomial_custom_checker_preserves_native_chebfunpref_error(tech):
    called = []
    grid = chebpts(17, kind=1 if tech is Chebtech1 else 2)
    fibers = jnp.stack((2.0 + grid, 1.0 - 0.5 * grid**2), axis=1)

    def checker(f, values, data, pref):
        called.append(True)
        return True, 17

    pref = ChebfunPref(happinessCheck=checker)
    native = next(
        case for case in _NATIVE_HAPPINESS
        if case["technology"] == ("chebtech1" if tech is Chebtech1 else "chebtech2")
        and case["mode"] == "custom"
    )
    assert native["errorIdentifier"] == "MATLAB:maxlhs"
    assert native["errorMessage"] == "Too many output arguments."
    assert native["callbackCount"] == 0
    with pytest.raises(TypeError, match="too many outputs through ChebfunPref.subsref"):
        envelope_happiness(fibers, tech, 1e-8, checker, pref)
    assert called == []


@pytest.mark.parametrize("tech", ["chebtech1", "chebtech2", "trigtech"])
def test_public_callback_receives_full_owner_pref_and_does_not_mutate_it(tech):
    seen = []

    def checker(f, values, data, pref):
        assert isinstance(pref, ChebfunPref)
        assert pref.tech == tech
        assert pref.cheb3Prefs.maxRank == 128
        assert pref.happinessCheck is checker
        assert pref.chebfuneps != 2.0**-52
        assert data["hscale"] == 1.0
        if tech == "trigtech":
            assert f.coeffs.shape[1] == 1
            assert values.shape[0] == f.values.shape[0]
            assert values.shape[1] == 1
            assert data["vscale"] == f.vscale
            seen.append((pref.chebfuneps, f.values.shape[1], tuple(f.isReal)))
        else:
            seen.append(pref.chebfuneps)
        return True, 1

    pref = ChebfunPref(tech=tech, happinessCheck=checker)
    original = ChebfunPref(pref)
    if tech == "trigtech":
        def operator(x, y, z):
            return (jnp.cos(jnp.pi*x) + jnp.sin(jnp.pi*y)
                    + jnp.cos(2*jnp.pi*z))
    else:
        def operator(x, y, z):
            return x + 2*y - z
    if tech != "trigtech":
        with pytest.raises(
            TypeError, match="too many outputs through ChebfunPref.subsref"
        ):
            chebfun3(operator, pref=pref)
        assert seen == []
        assert pref == original
        return
    result = chebfun3(operator, pref=pref)
    assert isinstance(result, Chebfun3)
    assert seen
    x, y, z = 0.125, -0.25, 0.375
    expected = jnp.cos(jnp.pi*x) + jnp.sin(jnp.pi*y) + jnp.cos(2*jnp.pi*z)
    tol = 100 * pref.cheb3Prefs.chebfun3eps
    assert abs(float(result(x, y, z)) - float(expected)) < tol
    assert any(width > 1 for _, width, _ in seen)
    assert all(len(real_mask) == width for _, width, real_mask in seen)
    assert pref == original


def test_custom_trig_checker_receives_native_mutated_storage_and_scale():
    grid = jnp.arange(17, dtype=jnp.float64) / 17
    fibers = jnp.stack((jnp.sin(2 * jnp.pi * grid),
                        0.25 * jnp.cos(4 * jnp.pi * grid)
                        + 0.5j * jnp.sin(6 * jnp.pi * grid)), axis=1)
    original_values = fibers
    expected_envelope = jnp.sum(
        jnp.abs(Trigtech.vals2coeffs(fibers)), axis=1
    )
    expected_scale = jnp.max(
        jnp.abs(Trigtech.coeffs2vals(expected_envelope))
    )
    seen = []

    def checker(f, values, data, pref):
        assert isinstance(f, Trigtech)
        assert f.ishappy
        assert f.coeffs.shape == (17, 1)
        assert values.shape == (17, 1)
        assert jnp.array_equal(f.coeffs, expected_envelope[:, None])
        assert jnp.array_equal(values, expected_envelope[:, None])
        assert f.values.shape == original_values.shape
        assert jnp.array_equal(f.values, original_values)
        assert f.real_columns == (True, False)
        assert jnp.array_equal(f.isReal, jnp.asarray([True, False]))
        assert f.vscale == expected_scale
        assert data == {"vscale": expected_scale, "hscale": 1.0}
        assert isinstance(pref, ChebfunPref)
        assert pref.chebfuneps == 1e-8
        assert pref.cheb3Prefs.maxRank == 128
        seen.append(True)
        return True, 17

    pref = ChebfunPref(tech="trigtech", happinessCheck=checker)
    assert envelope_happiness(fibers, Trigtech, 1e-8, checker, pref)
    assert seen == [True]


def test_custom_trig_checker_uses_caller_initial_vscale_for_real_mask():
    n = 17
    fibers = jnp.stack((
        jnp.full((n,), 1e-16),
        jnp.full((n,), 1e-16j),
        jnp.ones((n,)),
    ), axis=1)
    masks = []

    def checker(f, values, data, pref):
        assert f.coeffs.shape == values.shape == (n, 1)
        assert f.values.shape == (n, 3)
        masks.append(tuple(f.isReal))
        return True, 1

    pref = ChebfunPref(tech="trigtech", happinessCheck=checker)
    assert envelope_happiness(fibers, Trigtech, 1e-8, checker, pref)
    assert envelope_happiness(
        fibers, Trigtech, 1e-8, checker, pref, initial_vscale=1.0
    )
    assert masks == [(True, False, True), (True, True, True)]


def test_classic_initial_btd_checker_order_and_scales(monkeypatch):
    import chebfunjax.chebfun3d._chebfun3classic as classic

    columns = jnp.asarray([[1.0, 0.5], [2.0, 0.0]])
    rows = jnp.asarray([[3.0, 0.0], [0.0, 0.0]])
    fibers = jnp.asarray([[4.0, 9.0], [1.0, 0.0]])
    skeleton = SimpleNamespace(blocks=[(columns, None, rows)], fibers=fibers)
    calls = []
    results = [True, False, True]

    def fake_happy(values, tech, relative, pref, *, initial_vscale=None):
        calls.append((values, float(initial_vscale)))
        return results[len(calls) - 1]

    monkeypatch.setattr(classic, "happy", fake_happy)
    resolved = happiness_btd(skeleton, Trigtech, 1e-8, ChebfunPref())
    assert all(actual is expected for (actual, _), expected in zip(
        calls, (fibers, columns, rows)
    ))
    assert [scale for _, scale in calls] == [4.0, 3.0, 3.0]
    assert resolved == [False, True, True]
