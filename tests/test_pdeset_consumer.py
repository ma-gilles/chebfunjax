"""PDE option consumption through actual polynomial heat evolution.

Provenance
----------
Chebfun @chebfun/pde15s.m and @chebfun/pdeSolve.m; commit 7574c77.
These are authored consumer controls, not original MATLAB predicates.
"""

import warnings

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun, pdeset
from chebfunjax.chebfun1d.pde15s import pde15s


@pytest.mark.parametrize("absolute_tolerance", [1e-10, [1e-10] * 16])
def test_heat_options_grid_tolerances_and_solver_override(absolute_tolerance):
    """An options mapping selects N and accurately evolves a heat eigenmode."""
    initial = chebfun(lambda x: jnp.cos(jnp.pi * x / 2))
    times = jnp.asarray([0., .01, .02])
    options = pdeset(
        "N", 16, "Eps", 1e-8, "AbsTol", absolute_tolerance, "RelTol", 1e-9,
        "ODESolver", "ignored by native pde15s wrapper", "Plot", "off",
    )
    snapshots = pde15s(lambda u: .1 * u.diff(2), times, initial, "dirichlet", options)
    assert len(snapshots) == 3
    assert all(len(u) == 16 for u in snapshots)
    x = jnp.linspace(-1., 1., 31)
    for t, u in zip(times, snapshots):
        exact = jnp.cos(jnp.pi * x / 2) * jnp.exp(-.1 * (jnp.pi / 2) ** 2 * t)
        assert float(jnp.max(jnp.abs(u(x) - exact))) < 1e-7


def test_positional_grid_overrides_option_grid():
    """The sixth argument overrides options.N as in pdeSolve input parsing."""
    initial = chebfun(lambda x: jnp.cos(jnp.pi * x / 2))
    snapshots = pde15s(
        lambda u: .1 * u.diff(2), jnp.asarray([0., .01, .02]), initial,
        "dirichlet", pdeset("N", 8, "Eps", 1e-8), 16,
    )
    assert len(snapshots) == 3
    assert all(len(u) == 16 for u in snapshots)
    x = jnp.linspace(-1., 1., 31)
    exact = jnp.cos(jnp.pi * x / 2) * jnp.exp(-.1 * (jnp.pi / 2) ** 2 * .02)
    assert float(jnp.max(jnp.abs(snapshots[-1](x) - exact))) < 1e-7


def test_adjust_bcs_false_enforces_nominal_boundary_without_bad_ic_warning():
    """Disable the source offset for an inconsistent constant initial datum."""
    initial = chebfun(lambda x: .5 + 0. * x, domain=(0., 1.))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        snapshots = pde15s(
            lambda u: u.diff(2), jnp.asarray([0., .01, .02]), initial,
            "dirichlet", pdeset("N", 16, "AdjustBCs", False),
        )
    assert not any("BadIC" in str(w.message) for w in captured)
    assert len(snapshots) == 3
    # Initial output remains the supplied initial function; subsequent DAE
    # outputs obey the nominal zero conditions instead of the .5 offset.
    assert abs(float(snapshots[0](0.)) - .5) < 1e-14
    for u in snapshots[1:]:
        assert abs(float(u(0.))) < 1e-8
        assert abs(float(u(1.))) < 1e-8
