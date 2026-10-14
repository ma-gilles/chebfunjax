"""Three-variable fallback controls; native trajectories remain diagnostic."""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils._fminsearch import fminsearch

DATA = json.loads(Path(__file__).with_name("fminsearch3_native.json").read_text())

@pytest.mark.parametrize("record", DATA["cases"], ids=lambda r: str(r["case"]))
def test_sine_mapped_native_fallback_objectives(record):
    case = record["case"]
    def objective(t):
        x = jnp.sin(t)
        if case == 1:
            return jnp.sum((x-jnp.asarray([.125, -.25, .5]))**2)
        if case == 2:
            return jnp.sum(jnp.asarray([1., 2., 4.])*(x-jnp.asarray([2., -.25, -.5]))**2)
        if case == 3:
            return jnp.sum(jnp.asarray([1., 2., 4.])*(x-jnp.asarray([2., -2., 2.]))**2)
        if case == 4:
            return (x[0]+2*x[1]-x[2]-.25)**2
        return jnp.asarray(1.)
    result = fminsearch(objective, jnp.arcsin(jnp.asarray(record["initial"])))
    assert result.exitflag == record["flag"] == 1
    assert result.iterations <= 600 and result.evaluations <= 600
    if case in (2, 3, 6):
        # Exactly representable analytical minimum, also captured natively.
        expected = {2: 1., 3: 7., 6: 1.}[case]
        assert result.fun == record["value"] == expected
    else:
        # Supplemental analytical objective control at source TolFun=eps.
        # No fitted angular-location or iteration-count envelope is asserted.
        assert 0 <= result.fun < jnp.finfo(jnp.float64).eps
        assert 0 <= record["value"] < jnp.finfo(jnp.float64).eps


def test_public_fallback_selects_sine_mapping(monkeypatch):
    from chebfunjax.chebfun3d import _extrema3_source as source
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.tech.chebtech import Chebtech2
    basis = [Chebtech2.from_coeffs(jnp.eye(2)[:, i]) for i in range(2)]
    core = jnp.zeros((2, 2, 2)).at[1, 0, 0].set(1.)
    f = Chebfun3(basis, basis, basis, core, (-1., 1., -1., 1., -1., 1.))
    calls = []
    real_solver = source.fminsearch
    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real_solver(*args, **kwargs)
    monkeypatch.setattr(source, "fminsearch", recording)
    values, locations = f.minandmax3(optimizer="fminsearch")
    assert len(calls) == 2
    assert all(c == {"max_iterations": 600, "max_evaluations": 600} for c in calls)
    assert jnp.array_equal(values, jnp.asarray([-1., 1.]))
    assert jnp.array_equal(locations[:, 0], jnp.asarray([-1., 1.]))
