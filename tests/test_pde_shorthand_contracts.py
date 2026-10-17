"""Independent boundary ordering, offset, and public normalization contracts."""

import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._pde_ndf import pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from tests.test_pde_shorthand_native import assembly, close, initial


class AssemblyObserved(Exception):
    pass


def test_linear_right_precedes_nonlinear_left(monkeypatch):
    def observed(fun, mass, y, **kwargs):
        full, raw, scope, offset = assembly(fun)
        x = scope["coordinate"].vals
        probe = jnp.column_stack(
            (1 + x + 2 * x * x, 2 - 3 * x + 2 * x * x + x * x * x, 3 + x * x - 2 * x * x * x)
        )
        # On [2,6], source shorthand is width²/4=4 times physical derivative.
        expected = jnp.asarray(
            [
                4 * (1 + 4 * 6),
                4 * (-3 + 4 * 6 + 3 * 6**2),
                4 * (2 * 6 - 6 * 6**2),
                1 + 4 * 2,
                -3 + 4 * 2 + 3 * 2**2,
                2 * 2 - 6 * 2**2,
            ]
        )
        close(raw(0.003, probe.reshape(-1, order="F"))[:6], expected)
        assert scope["boundary_matrix"].shape == (3, 27)
        assert mass.shape == (27, 27)
        raise AssemblyObserved

    monkeypatch.setattr(pde_driver, "initialize_full_mass", observed)
    with pytest.raises(AssemblyObserved):
        pde15s(
            lambda u, v, w: (u.diff(2), v.diff(2), w.diff(2)),
            [0, 0.01],
            initial("neumann", (2.0, 6.0)),
            lbc=lambda u, v, w: (u.diff(), v.diff(), w.diff()),
            rbc="neumann",
            n=9,
        )


def test_scalar_numeric_target_preserves_both_q_subtractions(monkeypatch):
    def observed(fun, mass, y, **kwargs):
        full, raw, scope, offset = assembly(fun)
        targets = jnp.asarray([2.0, -3.0])
        endpoints = y[jnp.asarray([0, -1])]
        close(scope["linear_rhs"], targets)
        close(raw(0.0, y)[:2], endpoints - targets)
        close(offset, endpoints - 2 * targets)
        close(fun(y)[:2], targets)
        raise AssemblyObserved

    monkeypatch.setattr(pde_driver, "initialize_full_mass", observed)
    u0 = Chebfun.from_coeffs(jnp.asarray([1.0, 1.0]), domain=(-1.0, 1.0))
    with pytest.raises(AssemblyObserved):
        pde15s(lambda u: u.diff(2), [0, 0.01], u0, lbc=2.0, rbc=-3.0, n=9)


@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
def test_positional_keyword_and_case_normalization(monkeypatch, kind):
    module = importlib.import_module("chebfunjax.chebfun1d.pde15s")
    calls = []

    def solve(*args, **kw):
        calls.append(kw)
        return (["sentinel"], None)

    monkeypatch.setattr(module, "solve", solve)
    obj = object()
    assert module.pde15s(None, [0, 1], obj, kind.upper()) == ["sentinel"]
    assert module.pde15s(None, [0, 1], obj, bc=kind) == ["sentinel"]
    assert calls[0] == calls[1] and calls[0]["lbc"] == calls[0]["rbc"] == kind


@pytest.mark.parametrize(
    "bc,extra,error",
    [
        ("unknown", {}, NotImplementedError),
        ("periodic", {}, NotImplementedError),
        (7, {}, TypeError),
        ("neumann", {"lbc": 0.0}, ValueError),
        ("dirichlet", {"rbc": lambda u: u}, ValueError),
        ("neumann", {"method": "BDF"}, NotImplementedError),
    ],
)
def test_invalid_shorthand_rejected_before_solver(monkeypatch, bc, extra, error):
    module = importlib.import_module("chebfunjax.chebfun1d.pde15s")

    def forbidden(*args, **kwargs):
        raise AssertionError("solver reached")

    monkeypatch.setattr(module, "solve", forbidden)
    with pytest.raises(error):
        module.pde15s(None, [0, 1], object(), bc, **extra)


def test_existing_legacy_route_without_bc_is_unchanged(monkeypatch):
    legacy = importlib.import_module("chebfunjax.chebfun1d._pde_legacy")
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return ["legacy"]

    monkeypatch.setattr(legacy, "pde15s", spy)
    assert pde15s(None, [0, 1], object(), lbc=2.0, rbc=3.0, method="BDF") == ["legacy"]
    assert seen[0]["lbc"] == 2.0 and seen[0]["rbc"] == 3.0 and seen[0]["method"] == "BDF"


@pytest.mark.parametrize("side", ["lbc", "rbc"])
def test_endpoint_tokens_rejected_for_legacy_methods(side):
    with pytest.raises(NotImplementedError, match="method='NDF'"):
        pde15s(None, [0, 1], object(), method="BDF", **{side: "neumann"})
