"""Independent controls for the native ode78/ode89 source candidate.

These controls test source-shaped solution structures and independent analytic
trajectories. The polynomial dense-output bound is a predeclared diagnostic
bound for the high-degree Verner continuous-extension coefficients; it is not
an original MATLAB assertion. The MATLAB reference runtime/output is not
available here, so this draft does not claim native trajectory parity.

Provenance
----------
MATLAB source: installed R2025b ode78.m, ode89.m, private/ntrp78.m,
private/ntrp89.m; source hashes and line anchors are in
native_high_order_rk_source_inventory_p80_20261006.md.
Chebfun source context: commit 7574c77, @chebfun/ode78.m / @chebfun/ode89.m.
"""

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp

# uses-numpy: host-side assertions and independent closed-form test oracles.
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils.native_rk import native_rk

_SOLVERS = ("ode78", "ode89")
_DENSE_DIAGNOSTIC_BOUND = 2e-8


@pytest.mark.parametrize("solver_name", _SOLVERS)
@pytest.mark.parametrize("backward", (False, True), ids=("forward", "backward"))
def test_native_rk_coupled_polynomial_and_solution_structure(solver_name, backward):
    """Independent [t,t²] solution checks for method and direction."""
    span = (1.0, 0.0) if backward else (0.0, 1.0)
    initial = [span[0], span[0] ** 2]
    result = native_rk(
        solver_name,
        lambda t, y: jnp.array([1.0, 2.0 * t]),
        span,
        initial,
        {"RelTol": 1e-8, "AbsTol": [1e-10, 1e-11]},
    )

    x = np.linspace(0.0, 1.0, 101)
    values, derivatives = result["sol"](x, return_derivative=True)
    np.testing.assert_allclose(
        values,
        np.stack((x, x**2)),
        rtol=0.0,
        atol=_DENSE_DIAGNOSTIC_BOUND,
    )
    np.testing.assert_allclose(
        derivatives,
        np.stack((np.ones_like(x), 2.0 * x)),
        rtol=0.0,
        atol=_DENSE_DIAGNOSTIC_BOUND,
    )

    # Native DEVAL returns stored mesh values directly. This is a structure
    # contract, separate from the approximate off-mesh dense polynomial.
    np.testing.assert_array_equal(result["sol"](result["x"]), result["y"])
    nstage = {"ode78": 12, "ode89": 14}[solver_name]
    f3d = result["idata"]["f3d"]
    assert f3d.shape == (2, nstage, result["x"].size)
    np.testing.assert_array_equal(f3d[:, :, 0], np.zeros((2, nstage)))
    assert result["stats"]["tfinal"] == span[-1]
    stats = result["stats"]
    total_stages, new_trial_stages = (17, 12) if solver_name == "ode78" else (21, 15)
    assert stats["nfevals"] == total_stages * stats["nsteps"] + new_trial_stages * stats["nfailed"]

    outside = -0.1 if not backward else 1.1
    with pytest.raises(ValueError, match="outside integration interval"):
        result["sol"](jnp.asarray([outside]))


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_exponential_with_norm_control(solver_name):
    """Independent smooth scalar trajectory exercising vector norm control."""
    result = native_rk(
        solver_name,
        lambda t, y: -y,
        (0.0, 2.0),
        [1.0],
        {"RelTol": 1e-8, "AbsTol": 1e-10, "NormControl": "on"},
    )
    x = np.linspace(0.0, 2.0, 41)
    np.testing.assert_allclose(
        result["sol"](x),
        np.exp(-x)[None, :],
        rtol=0.0,
        atol=_DENSE_DIAGNOSTIC_BOUND,
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_complex_rhs_with_real_initial_uses_source_endpoint_bound(solver_name):
    """Source pass(8) endpoint input and bound; this is not an external oracle."""
    result = native_rk(solver_name, lambda t, y: 1j * y, (0.0, 1.0), [1.0])
    assert jnp.iscomplexobj(result["y"])
    endpoint = complex(result["sol"](jnp.asarray(1.0))[0, 0])
    assert abs(endpoint - np.exp(1j)) < 2e-2


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_minimum_step_returns_only_accepted_prefix(solver_name):
    """Source hmin failure behavior: warn and return the accepted prefix."""
    with pytest.warns(UserWarning, match="returning accepted partial solution"):
        result = native_rk(
            solver_name,
            lambda t, y: 5.0 * y,
            (0.0, 1.0),
            [1.0],
            {"RelTol": 1e-8, "AbsTol": 1e-10, "MinStep": 0.2, "MaxStep": 0.2},
        )
    np.testing.assert_array_equal(result["x"], [0.0])
    np.testing.assert_array_equal(result["y"], [[1.0]])
    assert result["stats"]["nsteps"] == 0
    assert result["stats"]["nfailed"] == 1
    np.testing.assert_array_equal(result["sol"](jnp.asarray([0.0])), [[1.0]])


@pytest.mark.parametrize("solver_name", _SOLVERS)
@pytest.mark.parametrize(
    "span,initial,expected",
    [
        ([[0.0, 1.0]], [[1.0, 2.0]], [1.0, 2.0]),
        ([[0.0], [1.0]], [[1.0], [2.0]], [1.0, 2.0]),
        ([[0.0, 2.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [1.0, 3.0, 2.0, 4.0]),
    ],
    ids=("row-vectors", "column-vectors", "matrix-column-major"),
)
def test_native_rk_source_input_shapes_flatten_column_major(solver_name, span, initial, expected):
    """Independent constant solution for MATLAB tspan(:) and y0(:) layout."""
    result = native_rk(solver_name, lambda t, y: jnp.zeros_like(y), span, initial)
    expected_span = np.asarray(span).ravel(order="F")
    expected_y = np.broadcast_to(
        np.asarray(expected, dtype=np.float64)[:, None],
        (len(expected), expected_span.size),
    )
    np.testing.assert_array_equal(result["sol"](expected_span), expected_y)
    assert result["stats"]["tfinal"] == expected_span[-1]


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_vdp_odesol_original_representation_bound(solver_name, tmp_path):
    """Source VDP data/options and representation bound, without native-MAT oracle."""

    def rhs(t, y):
        return jnp.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])

    options = {"RelTol": 1e-6}
    start = time.monotonic()
    result = native_rk(solver_name, rhs, (0.0, 20.0), [2.0, 0.0], options)
    integrate_seconds = time.monotonic() - start
    start = time.monotonic()
    fitted_fun = cj.odesol(result, (0.0, 20.0))
    fit_seconds = time.monotonic() - start

    fitted = np.asarray(fitted_fun(result["x"]))
    raw = np.asarray(result["y"]).T
    max_error = float(np.max(np.abs(fitted - raw)))
    output = Path(os.environ.get("NATIVE_RK_CONTROL_OUTPUT", str(tmp_path)))
    np.savez(
        output / f"{solver_name}_vdp_arrays.npz",
        x=np.asarray(result["x"]),
        raw=raw,
        fitted=fitted,
        f3d=np.asarray(result["idata"]["f3d"]),
    )
    report = {
        "solver": solver_name,
        "source_input_options": options,
        "source_representation_bound": 1e-5,
        "max_fit_raw_at_native_nodes": max_error,
        "integrate_seconds": integrate_seconds,
        "fit_seconds": fit_seconds,
        "stats": result["stats"],
        "pieces": [
            {"interval": p.interval, "length": p.tech.coeffs.shape[0], "happy": p.ishappy}
            for p in fitted_fun.funs
        ],
        "no_native_MATLAB_fixture": True,
        "native_controller_parity_unqualified": True,
    }
    (output / f"{solver_name}_vdp_representation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))
    assert max_error < 1e-5


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_complex_initial_real_rhs_promotes_cached_derivative(solver_name):
    # Independent exact affine complex solution. Real derivative values must
    # retain a complex state and a consistent cached/reevaluated RHS dtype.
    result = native_rk(solver_name, lambda t, y: jnp.ones(y.shape), (0.0, 1.0), [1.0 + 2j])
    x = np.linspace(0.0, 1.0, 41)
    np.testing.assert_allclose(
        result["sol"](x), (1 + x + 2j)[None, :], rtol=0.0, atol=_DENSE_DIAGNOSTIC_BOUND
    )
    np.testing.assert_allclose(
        result["y"][-1, -1], 2 + 2j, rtol=0.0, atol=100 * np.finfo(float).eps
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_native_rk_source_column_absolute_tolerance(solver_name):
    # Native odearguments atol(:) admits a column-shaped tolerance vector.
    result = native_rk(
        solver_name,
        lambda t, y: jnp.array([1.0, 2 * t]),
        (0.0, 1.0),
        [0.0, 0.0],
        {"RelTol": 1e-8, "AbsTol": [[1e-10], [1e-11]]},
    )
    x = np.linspace(0.0, 1.0, 41)
    np.testing.assert_allclose(
        result["sol"](x), np.stack((x, x * x)), rtol=0.0, atol=_DENSE_DIAGNOSTIC_BOUND
    )
