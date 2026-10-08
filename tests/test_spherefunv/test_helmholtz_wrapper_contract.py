"""Independent call-protocol controls; not numerical decomposition acceptance.

Provenance
----------
MATLAB source : @spherefunv/helmholtzdecomp.m, @separableApprox/length.m
Chebfun commit: 7574c77
"""

import warnings

import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


@pytest.mark.parametrize(
    "residual, emits_warning",
    [(0.0, False), (300 * 2.0**-52, False), (301 * 2.0**-52, True), (1.0, True)],
)
def test_projected_rhs_literal_dimensions_and_warning(monkeypatch, residual, emits_warning):
    events = []

    class Scalar:
        def __init__(self, name, lengths):
            self.name, self.lengths = name, lengths

        def length(self):
            events.append(("length", self.name))
            return self.lengths

    div = Scalar("div", (7, 31))
    vort = Scalar("vort", (40, 9))

    class Tangent:
        def div(self):
            events.append("div")
            return div

        def vort(self):
            events.append("vort")
            return vort

    tangent = Tangent()

    class Component:
        def vscale(self):
            return 1.0

    class Residual:
        def norm(self):
            events.append("norm")
            return residual

    class Original:
        components = [Component(), Component(), Component()]

        def isempty(self):
            return False

        def tangent(self):
            events.append("tangent")
            return tangent

        def __sub__(self, other):
            assert other is tangent
            events.append("subtract")
            return Residual()

    def poisson(rhs, const, m, n):
        events.append(("poisson", rhs.name, const, m, n))
        return rhs.name

    monkeypatch.setattr(Spherefun, "poisson", staticmethod(poisson))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert Spherefunv.helmholtzdecomp(Original()) == ("div", "vort")
    assert len(seen) == int(emits_warning)
    if seen:
        assert str(seen[0].message).startswith("SPHEREFUNV:HELMHOLTZDECOMPOSITON:TANGENT:")
    assert events == [
        "tangent",
        "subtract",
        "norm",
        "div",
        ("length", "div"),
        ("poisson", "div", 0, 50, 62),
        "vort",
        ("length", "vort"),
        ("poisson", "vort", 0, 80, 50),
    ]


def test_empty_avoids_projection_and_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty source branch must return before numerical operations")

    monkeypatch.setattr(Spherefunv, "tangent", forbidden)
    monkeypatch.setattr(Spherefun, "poisson", staticmethod(forbidden))
    u, v = Spherefunv.empty().helmholtzdecomp()
    assert isinstance(u, Spherefun) and isinstance(v, Spherefun)
    assert u.isempty() and v.isempty()
