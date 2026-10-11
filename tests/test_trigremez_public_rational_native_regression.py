"""Persistent public regressions from qualified source_output_gate_v1.

Native Chebfun7574c77680d7e82b79626300bf255498271a72df/R2025b.
Exact original constructions; unchanged original1e-12 comparison bound.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("case,degree", [(1, 2), (2, 6)])
def test_native_rational_public_output_iteration_warning(case, degree):
    x = cj.chebfun("x")
    f = abs(x) if case == 1 else (jnp.pi * x).sin().exp()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        p, q, evaluator, error, status = cj.trigremez(f, degree, degree)
    assert not any("trigremez:convergence" in str(w.message) for w in emitted)
    assert status["iter"] > 0
    assert set(status) == {"delta", "diffx", "iter", "xk"}
    fixtures = Path(__file__).parent / "fixtures/trigremez_rational_source_2025b"
    with np.load(fixtures / f"case{case}.npz") as native:
        samples = native["sample_x"]
        values = np.asarray(evaluator(jnp.asarray(samples)))
        np.testing.assert_allclose(values, native["r_values"], rtol=0, atol=1e-12)
        # Both iteration counts were observed in the qualified Python run
        # and captured native run; this catches the old zero-iteration exit.
        assert status["iter"] == int(native["status_iter"])
