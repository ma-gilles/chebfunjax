"""Static mapped-interval JIT regression found in quadrature source review."""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import ultrapts


def test_mapped_static_jit_matches_eager():
    rule = partial(ultrapts, 8, .3, (0., 2.), bary=True)
    eager = rule()
    compiled = jax.jit(rule)()
    for a, b in zip(eager, compiled):
        assert jnp.max(jnp.abs(a-b)) < 1e-13


@pytest.mark.parametrize("lam", [.3, 0.])
def test_dynamic_interval_jit_matches_eager(lam):
    compiled = jax.jit(lambda a, b: ultrapts(8, lam, (a, b), angles=True))
    for interval in [(-1., 1.), (0., 2.)]:
        actual = compiled(*interval)
        expected = ultrapts(8, lam, interval, angles=True)
        for a, b in zip(actual, expected):
            assert jnp.max(jnp.abs(a-b)) < 1e-13


@pytest.mark.parametrize("interval", [(0., float("inf")), (0., float("nan")), (2., 1.)])
def test_eager_invalid_interval_still_rejected(interval):
    with pytest.raises(ValueError, match="Interval invalid"):
        ultrapts(8, .3, interval)


def test_small_asy_source_takes_two_corrections(monkeypatch):
    from chebfunjax.utils import ultraspherical_points as up

    # Source asy1 n<=21 empties its stopping residual after the first step,
    # then performs one final correction. A controlled Newton update makes
    # that branch observable independently of Bessel accuracy.
    monkeypatch.setattr(up, "_interior_eval",
                        lambda n, lam, theta: (.1*theta, jnp.ones_like(theta)))
    initial = up._initial(8, .3, jnp.arange(4, 0, -1, dtype=jnp.float64))
    _, _, _, theta = up._interior.__wrapped__(8, .3)
    assert jnp.max(jnp.abs(theta[4:]-.81*initial)) < 3e-16



def test_interior_source_stopping_has_no_iteration_cap(monkeypatch):
    from chebfunjax.utils import ultraspherical_points as up

    monkeypatch.setattr(up, "_interior_eval",
                        lambda n, lam, theta: (.1*theta, jnp.ones_like(theta)))
    _, _, _, theta = up._interior.__wrapped__(42, .3)
    # A linear contraction needs more than30 updates. Source stops on the
    # interior residual; the external supervisor supplies the runtime bound.
    assert jnp.max(jnp.abs(.1*theta[21:32])) <= jnp.sqrt(jnp.finfo(jnp.float64).eps)/1000



def test_boundary_underflow_warning_preserved(monkeypatch):
    from chebfunjax.utils import ultraspherical_points as up

    monkeypatch.setattr(up, "_boundary_eval",
                        lambda n, lam, theta, final=False:
                        (jnp.zeros_like(theta), jnp.full_like(theta, 1e200)))
    with pytest.warns(UserWarning, match="CHEBFUN:ultrapts:largeNLAMDBA"):
        up._asy(22, .3)
