"""Source @trigtech/plotData line semantics (7574c77)."""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.plotting import trig_plot_data
from chebfunjax.tech.trigtech import trigpts


def test_shifted_real_grid_and_values():
    f = cj.chebfun(lambda x: jnp.sin(jnp.pi * (x - 2)) + .3 * jnp.cos(2 * jnp.pi * (x - 2)), domain=[2, 4], trig=True)
    data = trig_plot_data(f)
    x, y = data["xLine"][1:], data["yLine"][1:]
    assert len(x) == 501
    assert bool(jnp.isnan(data["xLine"][0]))
    assert float(x[0]) == 2 and float(x[-1]) < 4
    assert float(jnp.max(jnp.abs(y - (jnp.sin(jnp.pi * (x - 2)) + .3 * jnp.cos(2 * jnp.pi * (x - 2)))))) < 4e-14


def test_source_rounding_count():
    f = cj.chebfun(jnp.sin(24 * jnp.pi * trigpts(51)), trig=True)
    assert len(trig_plot_data(f)["xLine"]) == 642  # round(4*pi*51) + separator


def test_source_maxlength_keeps_step_grid(monkeypatch):
    values = .5 * (1 + jnp.sign(trigpts(65536)))
    f = cj.chebfun(values, domain=[-3, 3], trig=True)
    def forbid_dense_evaluation(*args, **kwargs):
        raise AssertionError("real plotData must use prolongation values")
    monkeypatch.setattr(type(f), "__call__", forbid_dense_evaluation)
    data = trig_plot_data(f)
    assert len(data["xLine"]) == 65537
    assert bool(jnp.array_equal(data["yLine"][1:], values))


def test_complex_curve_has_source_endpoints():
    f = cj.chebfun(lambda x: jnp.exp(1j * jnp.pi * x), trig=True)
    data = trig_plot_data(f)
    assert len(data["xLine"]) == 504
    z = data["xLine"][1:] + 1j * data["yLine"][1:]
    assert float(jnp.max(jnp.abs(jnp.abs(z) - 1))) < 4e-14
    assert abs(complex(z[0]) + 1) < 4e-14
    assert abs(complex(z[-1]) + 1) < 4e-14


def test_reject_chebyshev_and_invalid_cap():
    f = cj.chebfun(lambda x: x)
    with pytest.raises(TypeError, match="Trigtech"):
        trig_plot_data(f)
    with pytest.raises(ValueError, match="positive"):
        trig_plot_data(f, max_length=0)
