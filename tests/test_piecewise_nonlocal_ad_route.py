"""Source AD routing for piecewise scalar linear nonlocal equations.

Provenance
----------
MATLAB source: @chebop/linearize.m; @chebop/solvebvpLinear.m;
    @adchebfun/adchebfun.m compose; @chebcolloc/compose.m.
Example: ode-nonlin/DelayDifferentialEquations.m, discrete-delay sections.
Chebfun commit: 7574c77
The1e-10 analytic bound is an added control, not a page assertion.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


def test_one_delay_uses_source_ad_and_matches_analytic_solution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Linear composition must avoid legacy column residual assembly")

    monkeypatch.setattr(Chebop, "_solve_piecewise", forbidden)
    op = Chebop(lambda t, x: x.diff() - x((t - 0.5).maximum(0.0)), domain=(0.0, 0.5, 1.0))
    op.lbc = 1.0
    result = op.solve(0.0)
    t = jnp.linspace(0.0, 1.0, 101)
    expected = 1 + t + 0.5 * jnp.maximum(t - 0.5, 0.0) ** 2
    assert float(jnp.max(jnp.abs(result(t) - expected))) < 1e-10
    assert abs(float(result(0.0)) - 1.0) < 1e-10


@pytest.mark.parametrize("kind", ["local", "nonlinear", "periodic", "general_bc", "system"])
def test_other_piecewise_routes_remain_legacy(monkeypatch, kind):
    dom = (0.0, 0.5, 1.0)
    if kind == "nonlinear":
        op = Chebop(lambda t, x: x.diff() - x**2, domain=dom)
    elif kind == "system":
        op = Chebop(lambda t, x, y: [x.diff() - y, y.diff() + x], domain=dom)
    else:
        op = Chebop(lambda t, x: x.diff() - x, domain=dom)
    if kind == "periodic":
        op.bc = "periodic"
    elif kind == "general_bc":
        op.bc = lambda x: x(0.0) - 1.0
    else:
        # Native parseBC requires one endpoint value per system unknown.
        op.lbc = [1.0, 1.0] if kind == "system" else 1.0
    received = []
    sentinel = chebfun(1.0, domain=dom)

    def legacy(self, f, **kwargs):
        received.append(kwargs)
        return sentinel

    monkeypatch.setattr(Chebop, "_solve_piecewise", legacy)
    result = op.solve(0.0, n=19, max_iter=3)
    assert received == [{"n": 19, "max_iter": 3}]
    assert float(jnp.max(jnp.abs(result(jnp.asarray([0.0, 0.5, 1.0])) - 1.0))) < 1e-10
