"""Small actual 1D JAX gates for spin and the spincomp adapter."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.spincomp import spincomp
from chebfunjax.operators.spinop import Spinop, spin
from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.operators.spinpref import SpinPref, SpinPref2, SpinPref3


def _linear_decay(final_time):
    return Spinop(
        domain=(0.0, 2.0 * float(jnp.pi)),
        tspan=(0.0, final_time),
        lin_symbol=lambda omega: -jnp.ones_like(omega),
        nonlin=lambda values, omega, dx: 0.0,
        init=lambda x: jnp.sin(x),
    )


@pytest.mark.parametrize("scheme,dt,final_time", [
    ("etdrk4", 0.01, 0.04),
    ("pecec736", 0.01, 0.07),
])
def test_spin_jax_timing_is_transparent_and_linear_solution_is_exact(
        scheme, dt, final_time):
    op = _linear_decay(final_time)
    plain = spin(op, 8, dt, scheme=scheme, M=8, dealias=False)
    timed, elapsed = spin(
        op, 8, dt, scheme=scheme, M=8, dealias=False,
        return_computing_time=True)
    x = jnp.arange(8) * (2 * jnp.pi / 8)
    plain_values = np.asarray(plain(x))
    timed_values = np.asarray(timed(x))
    expected = np.exp(-final_time) * np.sin(np.asarray(x))
    assert plain_values.tobytes() == timed_values.tobytes()
    np.testing.assert_allclose(plain_values, expected, rtol=2e-12, atol=2e-12)
    assert np.isfinite(elapsed) and elapsed >= 0.0


def test_spin_pecec736_short_horizon_returns_unfiltered_initial():
    op = Spinop(
        domain=(0.0, 2.0 * float(jnp.pi)),
        tspan=(0.0, 0.04),
        lin_symbol=lambda omega: jnp.zeros_like(omega),
        nonlin=lambda values, omega, dx: 0.0,
        init=lambda x: jnp.sin(3.0 * x),
    )
    solution = spin(op, 8, 0.01, scheme="pecec736", M=8)
    x = jnp.arange(8) * (2 * jnp.pi / 8)
    actual = np.asarray(solution(x))
    initial = np.sin(3.0 * np.asarray(x))
    # q=7 startup ends at .06; tf=.04 skips the main loop. In particular,
    # the native initial output is not filtered by the dealias output mask.
    np.testing.assert_allclose(actual, initial, rtol=1e-12, atol=1e-12)


def test_spin_noninteger_horizon_saves_only_exact_checkpoints():
    op = _linear_decay(0.075)
    nonaligned = spin(op, 8, 0.02, scheme="etdrk4", M=8, dealias=False)
    x = jnp.arange(8) * (2 * jnp.pi / 8)
    initial = np.sin(np.asarray(x))
    # The literal native loop passes tf at t=.08 but never reaches a tspan
    # checkpoint at .075, so vOut still contains only the initial output.
    np.testing.assert_allclose(
        np.asarray(nonaligned(x)), initial, rtol=1e-12, atol=1e-12)

    op.tspan = (0.0, 0.08)
    aligned = spin(op, 8, 0.02, scheme="etdrk4", M=8, dealias=False)
    np.testing.assert_allclose(
        np.asarray(aligned(x)), np.exp(-0.08) * initial,
        rtol=2e-12, atol=2e-12)


def test_spin_endpoint_tolerance_stops_native_loop_immediately(monkeypatch):
    import importlib

    module = importlib.import_module("chebfunjax.operators.spinop")
    original_one_step = module.one_step
    calls = []

    def counted_one_step(*args, **kwargs):
        calls.append(None)
        return original_one_step(*args, **kwargs)

    monkeypatch.setattr(module, "one_step", counted_one_step)
    op = _linear_decay(0.08 + 1e-11)
    result, elapsed = spin(
        op, 8, 0.02, scheme="etdrk4", M=8, dealias=False,
        return_computing_time=True)
    assert len(calls) == 4
    x = jnp.arange(8) * (2 * jnp.pi / 8)
    np.testing.assert_allclose(
        np.asarray(result(x)), np.exp(-0.08) * np.sin(np.asarray(x)),
        rtol=2e-12, atol=2e-12)
    assert np.isfinite(elapsed) and elapsed >= 0.0


def test_spincomp_small_actual_1d_reference():
    final_time = 0.84
    op = _linear_decay(final_time)
    errors, elapsed = spincomp(
        op, 8, [0.12, 0.14],
        SpinPref(M=8, scheme=["etdrk4", "pecec736"]), plot=False)
    assert errors.shape == elapsed.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(errors)))
    assert bool(jnp.all(errors < 2e-12))
    assert bool(jnp.all(jnp.isfinite(elapsed)))
    assert bool(jnp.all(elapsed >= 0.0))


def test_spincomp_preserves_short_reference_startup_and_checkpoint_behavior():
    # The reference uses dt=min([.12,.14])/2=.06. PECEC736's q=7 startup
    # ends at .36, beyond tf=.28, so the native reference stays at u0.
    # ETDRK4 with dt=.12 misses the exact endpoint and also returns u0;
    # dt=.14 lands exactly at .28 and advances the heat mode by exp(-.28).
    op = _linear_decay(0.28)
    # Native spincomp asks solvepde for three outputs for each comparison.
    # When ETDRK4 dt=.12 misses .28, solvepde has no tOut(2) and errors.
    with pytest.raises(IndexError, match=r"tOut\(2\)"):
        spincomp(
            op, 8, [0.12, 0.14],
            SpinPref(M=8, scheme="etdrk4"), plot=False)


def _linear_decay_operator(dim, final_time=0.16):
    domain = tuple(value for _ in range(dim) for value in (0.0, 2.0 * float(jnp.pi)))
    if dim == 2:
        # Reference dt is half the requested step; its PECEC736 startup
        # needs six of those steps before the solver can emit a checkpoint.
        operator = Spinop2(domain, (0.0, final_time))
        operator.lin = "@(u) lap(u)"
        operator.nonlin = "@(u) 0*u"
        operator.init = lambda x, y: jnp.cos(x) * jnp.cos(y)
    else:
        operator = Spinop3(domain, (0.0, final_time))
        operator.lin = "@(u) lap(u)"
        operator.nonlin = "@(u) 0*u"
        operator.init = lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    return operator


@pytest.mark.parametrize("dim", [2, 3])
def test_spincomp_small_actual_2d_3d_timing_controls(dim):
    """Exercise source short-startup behavior and converged 2D/3D controls."""
    operator = _linear_decay_operator(dim)
    if dim == 2:
        plain = spin2(operator, 8, 0.02, "dealias", "off")
        timed, elapsed = spin2(
            operator, 8, 0.02, "dealias", "off", return_computing_time=True)
        coordinates = jnp.arange(8) * (2 * jnp.pi / 8)
        xx, yy = jnp.meshgrid(coordinates, coordinates)
        plain_values = np.asarray(plain(xx, yy))
        timed_values = np.asarray(timed(xx, yy))
        pref = SpinPref2(M=8, scheme="etdrk4")
    else:
        plain = spin3(operator, 8, 0.02, "dealias", "off")
        timed, elapsed = spin3(
            operator, 8, 0.02, "dealias", "off", return_computing_time=True)
        coordinates = jnp.arange(8) * (2 * jnp.pi / 8)
        xx, yy, zz = jnp.meshgrid(coordinates, coordinates, coordinates)
        plain_values = np.asarray(plain(xx, yy, zz))
        timed_values = np.asarray(timed(xx, yy, zz))
        pref = SpinPref3(M=8, scheme="etdrk4")
    assert plain_values.tobytes() == timed_values.tobytes()
    assert np.isfinite(elapsed) and elapsed >= 0.0

    # Native @spinoperator/solvepde.m initializes vOut{1}=vInit, tOut(1)=0;
    # for q=7 PECEC736 and tf=.04, its six-step startup leaves the loop
    # unentered and returns the initial output. The published 3D native q6
    # capture covers the same rule; this probe checks the current public 2D/3D
    # adapters on the corresponding short interval.
    short_operator = _linear_decay_operator(dim, final_time=0.04)
    if dim == 2:
        reference = spin2(
            short_operator, 8, 0.01, "dealias", "off", scheme="pecec736")
        x, y = jnp.meshgrid(coordinates, coordinates)
        initial = short_operator.init(x, y)
        reference_values = reference(x, y)
    else:
        reference = spin3(
            short_operator, 8, 0.01, "dealias", "off", scheme="pecec736")
        x, y, z = jnp.meshgrid(coordinates, coordinates, coordinates)
        initial = short_operator.init(x, y, z)
        reference_values = reference(x, y, z)
    assert np.max(np.abs(np.asarray(reference_values - initial))) < 1e-12
    short_errors, _ = spincomp(
        short_operator, 8, [0.02], pref, plot=False)
    expected_short_error = 1.0 - np.exp(-dim * 0.04)
    np.testing.assert_allclose(
        np.asarray(short_errors), [[expected_short_error]],
        rtol=1e-8, atol=1e-8)

    # At .16 the reference has passed its startup and sampled spincomp errors
    # measure the actual ETDRK4-vs-PECEC736 solution difference.
    errors, spincomp_time = spincomp(
        operator, 8, [0.02], pref, plot=False)
    assert errors.shape == spincomp_time.shape == (1, 1)
    assert bool(jnp.all(jnp.isfinite(errors)))
    assert bool(jnp.all(errors < 1e-10))
    assert bool(jnp.all(jnp.isfinite(spincomp_time)))
    assert bool(jnp.all(spincomp_time >= 0.0))
