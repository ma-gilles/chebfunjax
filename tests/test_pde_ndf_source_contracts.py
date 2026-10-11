"""Minimum-step termination from MATLAB R2025b ode15s source branches."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._pde_ndf.ndf import startup
from chebfunjax.chebfun1d._pde_ndf.ndf_segment import segment
from chebfunjax.chebfun1d._pde_ndf.pde_driver import solve
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref


@pytest.mark.parametrize("jacobian", [0.0, -100.0])
def test_hmin_warns_and_preserves_last_accepted_state(jacobian):
    y = jnp.asarray([1.0])
    state = startup(
        y,
        jnp.asarray([-100.0]),
        jnp.asarray([[jacobian]]),
        jnp.eye(1),
        rtol=1e-6,
        threshold=1.0,
        userhmin=1.0,
        userhmax=1.0,
        htspan=1.0,
    )
    state.update(tfinal=jnp.asarray(2.0), jac_threshold=jnp.asarray(1e-6))
    outputs = []
    with pytest.warns(RuntimeWarning, match="MATLAB:ode15s:IntegrationTolNotMet"):
        final, rows = segment(
            state,
            lambda t, u: -100 * u,
            100,
            jnp.asarray([1.0, 2.0]),
            lambda t, u: outputs.append(u) or False,
        )
    assert final["terminal_reason"] == "IntegrationTolNotMet"
    assert float(final["t"]) == 0.0
    assert bool(jnp.array_equal(final["y"], y))
    assert not outputs


def run(rhs, y0, yp, jac, tfinal, *, force_step=None):
    y = jnp.asarray([y0])
    mass = jnp.eye(1)
    s = startup(
        y,
        jnp.asarray([yp]),
        jnp.asarray([[jac]]),
        mass,
        rtol=1e-6,
        threshold=1.0,
        htspan=0.1,
        userhmax=0.1,
    )
    if force_step is not None:
        s = startup(
            y,
            jnp.asarray([yp]),
            jnp.asarray([[jac]]),
            mass,
            rtol=1.0,
            threshold=1.0,
            htspan=force_step,
            userhmax=force_step,
        )
        s["rtol"] = jnp.asarray(1e-6)
    s.update(tfinal=jnp.asarray(tfinal), jac_threshold=jnp.asarray(1e-6))
    values = []
    final, rows = segment(
        s, rhs, 1000, jnp.asarray([tfinal]), lambda t, y: values.append(y) or False
    )
    assert len(values) == 1
    return values[0][0], rows


def test_stiff_time_dependent_refresh():
    def rhs(t, y):
        rate = jnp.where(t > 0.2, 10000.0, 1.0)
        exact = jnp.exp(-t)
        return -rate * (y - exact) - exact

    value, rows = run(rhs, 1.0, -1.0, -1.0, 1.0)
    recoveries = [r for row in rows for r in row["newton_recoveries"]]
    assert "refresh_jacobian" in recoveries
    assert abs(float(value - jnp.exp(-1.0))) < 1e-5


def test_native_scalar_pde_with_restart():
    times = jnp.arange(11, dtype=jnp.float64) * 0.1
    def pde(t, x, u):
        return 0.1 * u.diff(2) + u.diff()
    runs = []
    solutions = []
    for kind in [1, 2]:
        f = chebfun(lambda x: jnp.sin(jnp.pi * x), domain=(-2.5, 3.0), chebkind=kind)
        values, trace = solve(pde, times, f, lbc=lambda u: u.diff(), rbc=0.0, rtol=1e-6, atol=1e-6)
        runs.append({"kind": kind, "trace": trace, "outputs": len(values)})
        solutions.append(values)
    err = max(float((u - v).norm(2)) for u, v in zip(*solutions))
    tol = 1e5 * ChebfunPref().chebfuneps
    report = {
        "runs": runs,
        "native_kind_comparison_error": err,
        "original_native_bound": tol,
        "solver_tolerances": {"rtol": 1e-6, "atol": 1e-6, "spatial": 1e-6},
    }
    assert all(x["outputs"] == 11 for x in runs)
    assert all([z["n"] for z in x["trace"]] == [33, 65] for x in runs), report
    assert all(
        x["trace"][0]["restart_time"] == 0 and not x["trace"][0]["accepted_times"] for x in runs
    ), report
    assert err < tol, report
