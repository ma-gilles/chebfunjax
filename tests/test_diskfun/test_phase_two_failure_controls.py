"""Source outer-loop failure state controls (@diskfun/constructor.m:94,147)."""
import importlib

import jax.numpy as jnp
import pytest

mod = importlib.import_module("chebfunjax.diskfun.diskfun")


def test_unresolved_slices_do_not_restart_phase_one(monkeypatch):
    calls = []
    phase_one = mod._phase_one_disk

    def observed_phase_one(*args, **kwargs):
        calls.append(args[0].shape)
        return phase_one(*args, **kwargs)

    monkeypatch.setattr(mod, "_phase_one_disk", observed_phase_one)
    with pytest.warns(RuntimeWarning, match="column slices not resolved"):
        f = mod.Diskfun.from_function(lambda theta, r: r, max_sample=128)
    # Source retains PhaseTwo failure even if its sample test fails. Prior
    # port restarts at successively larger phase-one grids for this cusp.
    assert calls == [(9, 16)]
    assert len(f.cols) == 1
    assert f.cols[0].n == 129


def test_resolved_slices_retain_success_state():
    theta = jnp.asarray([0.0, 0.5, 1.0])
    radius = jnp.asarray([0.0, 0.2, 1.0])
    f = mod.Diskfun.from_function(lambda theta, r: r*r, max_sample=128)
    assert float(jnp.max(jnp.abs(f(theta, radius) - radius**2))) < 1e-14


def test_zero_phase_two_has_no_failure():
    f = mod.Diskfun.from_function(lambda theta, r: jnp.zeros_like(r))
    values = f(jnp.asarray([0.0]), jnp.asarray([0.5]))
    assert float(jnp.max(jnp.abs(values))) == 0.0
