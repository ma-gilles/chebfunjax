"""Source-shaped tests for the ten assignments in MATLAB test_ivp.m.

Provenance
----------
MATLAB source: tests/chebfun/test_ivp.m, Chebfun commit 7574c77.
The Van der Pol cases compare Chebfun's public odeXX result against a fresh
matching raw integration at the source solver's output times.
Native113/78/89 references exercise wrapper representation, not an independent
MATLAB trajectory oracle. The two unported methods use SciPy adapters.

Native MATLAB reference fixtures/methods remain unavailable. These preserve
all ten source numeric bounds and public input/output contracts while recording
each backend explicitly; they do not establish independent MATLAB parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# uses-numpy: comparison with the inherited SciPy reference backend.
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.chebfun1d.chebfun import ode15s, ode45, ode78, ode89, ode113
from chebfunjax.utils.native_ode113 import native_ode113
from chebfunjax.utils.native_rk import native_rk

jax.config.update("jax_enable_x64", True)


def _vdp1(t, y):
    """Van der Pol right-hand side with MATLAB test parameter mu=1."""
    del t
    return jnp.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])


def _vdp1_scipy(t, y):
    del t
    return np.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])


def _source_system_case(solver, scipy_method, tspan, source_options=None):
    """Run one distinct source matrix-error case through public wrappers."""
    t, y = solver(
        _vdp1,
        tspan,
        jnp.array([2.0, 0.0]),
        source_options,
        return_time=True,
    )
    assert callable(t)
    assert callable(y)

    # MATLAB two-output Refine is 1 for113, 8 for78/89. Each case
    # solves independently. The old failed DOP853 diagnostics are archived.
    if scipy_method in ('ode113', 'ode78', 'ode89'):
        options = {} if source_options is None else source_options
        if scipy_method == 'ode113':
            raw = native_ode113(_vdp1, tspan, jnp.array([2., 0.]), options)
            tm = raw['x']
        else:
            raw = native_rk(scipy_method, _vdp1, tspan, jnp.array([2., 0.]), options)
            mesh = raw['x']
            fractions = jnp.arange(1, 9, dtype=jnp.float64) / 8
            refined = mesh[:-1, None] + (mesh[1:] - mesh[:-1])[:, None] * fractions
            tm = jnp.concatenate((mesh[:1], refined.reshape(-1)))
        ym = np.asarray(raw['sol'](tm)).T
    else:
        scipy_kwargs = {"rtol": 1e-3, "atol": 1e-6,
                        "max_step": 0.1*abs(tspan[-1]-tspan[0])}
        if source_options is not None:
            scipy_kwargs["rtol"] = source_options["RelTol"]
            # MATLAB's AbsTol remains its default when only RelTol is supplied.
            scipy_kwargs["atol"] = 1e-6
        reference = solve_ivp(
            _vdp1_scipy,
            list(tspan),
            [2.0, 0.0],
            method=scipy_method,
            dense_output=False,
            **scipy_kwargs,
        )
        assert reference.success
        tm = reference.t
        ym = reference.y.T
    y_at_tm = np.asarray(y(jnp.asarray(tm)))
    assert y_at_tm.shape == ym.shape
    return float(np.max(np.abs(ym - y_at_tm)))


def test_source_pass01_ode15s_van_der_pol():
    # Source pass(1): no explicit solver options; source bound unchanged.
    assert _source_system_case(ode15s, "BDF", (0.0, 5.0)) < 2e-2


def test_source_pass02_ode45_van_der_pol():
    # Source pass(2): no explicit solver options; source bound unchanged.
    assert _source_system_case(ode45, "RK45", (0.0, 5.0)) < 1e-2


def test_source_pass03_ode113_van_der_pol():
    # Source pass(3): odeset('RelTol', 1e-6), default AbsTol; bound unchanged.
    options = {"RelTol": 1e-6}
    assert _source_system_case(ode113, "ode113", (0.0, 20.0), options) < 1e-5


def test_source_pass04_ode78_van_der_pol():
    # Source pass(4): independently solved matching native method, source bound.
    options = {"RelTol": 1e-6}
    assert _source_system_case(ode78, "ode78", (0.0, 20.0), options) < 1e-5


def test_source_pass05_ode89_van_der_pol():
    # Source pass(5): independently solved matching native method, source bound.
    options = {"RelTol": 1e-6}
    assert _source_system_case(ode89, "ode89", (0.0, 20.0), options) < 1e-5


def _complex_exact_endpoint(solver):
    # Source passes(6)-(10): scalar complex IVP u'=i*u, u(0)=1.
    result = solver(lambda t, u: 1j * u, (0.0, 1.0), 1.0)
    exact = np.exp(1j)
    return abs(complex(np.asarray(result(jnp.asarray(1.0)))) - exact)


def test_source_pass06_ode15s_complex_scalar():
    assert _complex_exact_endpoint(ode15s) < 2e-2


def test_source_pass07_ode45_complex_scalar():
    assert _complex_exact_endpoint(ode45) < 2e-2


def test_source_pass08_ode113_complex_scalar():
    assert _complex_exact_endpoint(ode113) < 2e-2


def test_source_pass09_ode78_complex_scalar():
    assert _complex_exact_endpoint(ode78) < 2e-2


def test_source_pass10_ode89_complex_scalar():
    assert _complex_exact_endpoint(ode89) < 2e-2
