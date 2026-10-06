"""Independent polynomial controls for the native ode113 dense interpolant.

These are diagnostic helper controls, not numbered tests from Chebfun's
MATLAB suite. They validate the source dense polynomial against exact
constant, linear, and quadratic functions; they do not validate Adams history
updates or a complete IVP solve.

Provenance
----------
MATLAB source: installed R2025b private/ntrp113.m (SHA-256
69da58404189f03f3ae608f9cc02ac5c3ecc1593defd951714265f8e789fd560).
Chebfun source context: commit 7574c77, @chebfun/odesol.m.
"""

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp

# uses-numpy: diagnostic host assertions of JAX dense output against exact polynomials.
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils.native_ode113 import _ntrp113, native_ode113

_EPS = np.finfo(np.float64).eps
_BOUND = 100.0 * _EPS


def _history(*, ynew, phi1, phi2=0.0):
    """Make source-shaped (neq,14) phi and 12-entry psi history arrays."""
    phi = jnp.zeros((1, 14), dtype=jnp.float64)
    phi = phi.at[0, 0].set(phi1)
    phi = phi.at[0, 1].set(phi2)
    psi = jnp.ones((12,), dtype=jnp.float64)
    return jnp.asarray([ynew], dtype=jnp.float64), phi, psi


def _assert_source_array(actual, expected):
    """Source ntrp113 returns state rows by queried-time columns."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected, dtype=np.float64).reshape(1, -1),
        rtol=0.0,
        atol=_BOUND * max(1.0, float(np.max(np.abs(expected)))),
    )


def test_ntrp113_constant_history_initial_interior_and_extrapolation():
    # y(t)=3, y'=0. At an accepted right endpoint tnew=3/2, the source
    # Newton history has klast=1, phi(:,1)=0 and nonzero prior step psi(1)=1/2.
    # Include the initial endpoint, tnew, and an out-of-step query: ntrp113
    # itself has no interval guard; public DEVAL owns range validation.
    ynew, phi, psi = _history(ynew=3.0, phi1=0.0)
    times = jnp.asarray([1.0, 1.25, 1.5, 1.75])
    values, derivatives = _ntrp113(
        times, 1.5, ynew, 1, phi, psi.at[0].set(0.5), return_derivative=True
    )
    _assert_source_array(values, [3.0, 3.0, 3.0, 3.0])
    _assert_source_array(derivatives, [0.0, 0.0, 0.0, 0.0])


def test_ntrp113_linear_history_at_start_endpoint_and_outside_step():
    # y(t)=1+2t. With tnew=3/2, ynew=4, constant derivative phi(:,1)=2,
    # and klast=1, the source polynomial is exactly ynew+(t-tnew)*2.
    # The t=1 value is the initial value for this synthetic step. t=7/4 is
    # outside [1,3/2] and checks the private interpolant's polynomial extension.
    ynew, phi, psi = _history(ynew=4.0, phi1=2.0)
    times = jnp.asarray([1.0, 1.25, 1.5, 1.75])
    values, derivatives = _ntrp113(
        times, 1.5, ynew, 1, phi, psi.at[0].set(0.5), return_derivative=True
    )
    _assert_source_array(values, [3.0, 3.5, 4.0, 4.5])
    _assert_source_array(derivatives, [2.0, 2.0, 2.0, 2.0])


def test_ntrp113_quadratic_history_and_derivative_use_right_endpoint_offset():
    # y(t)=t^2 at the right endpoint tnew=1/2. For equal prior steps h=1/4,
    # source divided-difference data are phi(:,1)=2*tnew=1,
    # phi(:,2)=2*h=1/2, phi(:,3)=0, psi=[h,2h,...], klast=2.
    # Substitution into ntrp113's Newton interpolant gives ynew+hi+hi^2;
    # its returned derivative is phi1+phi2*hi/psi1 = 1+2*hi.
    ynew, phi, psi = _history(ynew=0.25, phi1=1.0, phi2=0.5)
    psi = psi.at[0].set(0.25).at[1].set(0.5)
    times = jnp.asarray([0.0, 0.25, 0.5, 0.625])
    values, derivatives = _ntrp113(times, 0.5, ynew, 2, phi, psi, return_derivative=True)
    _assert_source_array(values, [0.0, 0.0625, 0.25, 0.390625])
    _assert_source_array(derivatives, [0.0, 0.5, 1.0, 1.25])


@pytest.mark.parametrize("backward", [False, True])
def test_native_adams_independent_coupled_polynomial(backward):
    # Independent exact trajectory [t,t^2], diagnostic 100eps scaled bound.
    span = (1.0, 0.0) if backward else (0.0, 1.0)
    initial = [span[0], span[0] ** 2]
    solved = native_ode113(
        lambda t, y: jnp.array([1.0, 2 * t]),
        span,
        initial,
        {"RelTol": 1e-8, "AbsTol": [1e-10, 1e-11]},
    )
    x = jnp.linspace(0, 1, 101)
    values, derivatives = solved["sol"](x, return_derivative=True)
    np.testing.assert_allclose(values, np.stack([x, x * x]), atol=100 * _EPS, rtol=0)
    np.testing.assert_allclose(
        derivatives, np.stack([np.ones(101), 2 * x]), atol=100 * _EPS, rtol=0
    )
    np.testing.assert_array_equal(solved["sol"](solved["x"]), solved["y"])
    orders = solved["idata"]["klastvec"]
    kmax = int(jnp.max(orders))
    nout = solved["x"].size
    assert solved["idata"]["phi3d"].shape == (2, kmax + 1, nout)
    assert solved["idata"]["psi2d"].shape == (kmax, nout)
    assert bool(jnp.all(solved["idata"]["phi3d"][:, :, 0] == 0))
    assert bool(jnp.all(solved["idata"]["psi2d"][:, 0] == 0))
    with pytest.raises(ValueError, match="outside integration interval"):
        solved["sol"](jnp.array([-0.1]))
    stats = solved["stats"]
    assert stats["nfevals"] == 1 + 2 * stats["nsteps"] + stats["nfailed"]
    assert stats["tfinal"] == span[-1]


def test_native_adams_complex_source_case_real_initial_dtype():
    # Same original real initial1, interval[0,1] and unchanged endpoint bound
    # as source tests/chebfun/test_ivp.m pass(8); analytic control, not MATLAB fixture.
    solved = native_ode113(lambda t, y: 1j * y, (0.0, 1.0), [1.0])
    assert jnp.iscomplexobj(solved["y"])
    assert abs(complex(solved["sol"](jnp.asarray(1.0))[0, 0]) - np.exp(1j)) < 2e-2


def test_native_adams_precise_exponential_and_normcontrol():
    # Supplemental analytic control: explicit RelTol1e-8/AbsTol1e-10, bound2e-7.
    solved = native_ode113(
        lambda t, y: -y, (0.0, 2.0), [1.0], {"RelTol": 1e-8, "AbsTol": 1e-10, "NormControl": "on"}
    )
    x = jnp.linspace(0, 2, 101)
    np.testing.assert_allclose(solved["sol"](x), np.exp(-np.asarray(x))[None, :], atol=2e-7, rtol=0)
    assert (
        solved["stats"]["nfevals"] == 1 + 2 * solved["stats"]["nsteps"] + solved["stats"]["nfailed"]
    )


def test_native_adams_minimum_step_warns_and_returns_accepted_prefix():
    with pytest.warns(UserWarning, match="returning accepted partial solution"):
        solved = native_ode113(
            lambda t, y: y,
            (0.0, 1.0),
            [1.0],
            {"RelTol": 1e-8, "AbsTol": 1e-10, "MinStep": 0.2, "MaxStep": 0.2},
        )
    np.testing.assert_array_equal(solved["x"], [0.0])
    np.testing.assert_array_equal(solved["y"], [[1.0]])
    assert solved["stats"]["nsteps"] == 0 and solved["stats"]["nfailed"] == 1
    np.testing.assert_array_equal(solved["sol"](jnp.array([0.0])), [[1.0]])


def test_native_adams_vdp_coupled_odesol_original_representation_bound(tmp_path):
    # Original source Van der Pol input/options and 1e-5 representation bound.
    # Both trajectories are this unqualified native-source port; no MATLAB
    # oracle or native node/controller trajectory equivalence is claimed.
    def rhs(t, y):
        return jnp.array([y[1], (1 - y[0] ** 2) * y[1] - y[0]])

    options = {"RelTol": 1e-6}
    start = time.monotonic()
    solved = native_ode113(rhs, (0.0, 20.0), [2.0, 0.0], options)
    integrate_seconds = time.monotonic() - start
    start = time.monotonic()
    y = cj.odesol(solved, (0.0, 20.0))
    fit_seconds = time.monotonic() - start
    fitted = np.asarray(y(solved["x"]))
    raw = np.asarray(solved["y"]).T
    error = float(np.max(np.abs(fitted - raw)))
    outdir = Path(os.environ.get("NATIVE_ODE113_CONTROL_OUTPUT", str(tmp_path)))
    np.savez(
        outdir / "vdp_arrays.npz",
        x=np.asarray(solved["x"]),
        raw=raw,
        fitted=fitted,
        order=np.asarray(solved["idata"]["klastvec"]),
        phi=np.asarray(solved["idata"]["phi3d"]),
        psi=np.asarray(solved["idata"]["psi2d"]),
    )
    report = {
        "source_input_options": options,
        "source_representation_bound": 1e-5,
        "max_fit_raw_at_native_nodes": error,
        "integrate_seconds": integrate_seconds,
        "fit_seconds": fit_seconds,
        "stats": solved["stats"],
        "pieces": [
            {"interval": p.interval, "length": p.tech.coeffs.shape[0], "happy": p.ishappy}
            for p in y.funs
        ],
        "no_native_MATLAB_fixture": True,
        "native_controller_parity_unqualified": True,
    }
    (outdir / "vdp_representation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert error < 1e-5


@pytest.mark.parametrize(
    "span,initial,expected",
    [
        ([[0.0, 1.0]], [[1.0, 2.0]], [1.0, 2.0]),
        ([[0.0], [1.0]], [[1.0], [2.0]], [1.0, 2.0]),
        ([[0.0, 2.0], [1.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [1.0, 3.0, 2.0, 4.0]),
    ],
    ids=["row", "column", "matrix-column-major"],
)
def test_native_adams_source_input_column_major_shapes(span, initial, expected):
    # Native odearguments uses tspan(:) and y0(:). Independent zero RHS
    # leaves the column-major initial state constant at the exact endpoints.
    solved = native_ode113(lambda t, y: jnp.zeros_like(y), span, initial)
    expected_span = np.asarray(span).ravel(order="F")
    np.testing.assert_array_equal(
        solved["sol"](expected_span),
        np.broadcast_to(np.asarray(expected)[:, None], (len(expected), len(expected_span))),
    )
    assert solved["stats"]["tfinal"] == expected_span[-1]
