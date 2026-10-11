"""Focused controls for the MATLAB normalized Legendre recurrence."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.random import _norm_legendre, _norm_legendre_triangle


def test_degree_210_finite_stress_and_exact_native_point_envelope():
    """Check finite stress and a pointwise envelope from captured native data.

    The exact point comparison is limited to one captured native input and is
    not a general accuracy bound over angles or degrees.
    """
    theta_grid = np.linspace(0.0, np.pi, 1001, dtype=np.float64)
    for order in (0, 1, 31, 105, 209, 210):
        got = np.asarray(_norm_legendre(210, order, theta_grid))
        assert np.isfinite(got).all()

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "references"
        / "random_legendre_degree210_native_point.json"
    )
    fixture = json.loads(fixture_path.read_text())
    theta_bits = np.asarray(int(fixture["theta_hex"], 16), dtype=np.uint64)
    theta = theta_bits.view(np.float64).item()
    x = float(np.asarray(jax.device_get(jnp.cos(jnp.asarray([theta], dtype=jnp.float64))))[0])
    assert np.asarray(x, dtype=np.float64).view(np.uint64).item() == int(
        fixture["x_ieee754_hex"], 16
    )

    candidate = np.asarray(
        _norm_legendre_triangle(210, jnp.asarray([theta], dtype=jnp.float64))[210, :, 0]
    )
    native = np.asarray(fixture["native_values"], dtype=np.float64)
    references = fixture["high_precision_values"]
    assert candidate.shape == native.shape == (211,)
    assert len(references) == 211
    assert np.isfinite(candidate).all()

    # Decimal.from_float preserves each computed binary64 value exactly.
    # The reference strings were emitted by an 80-digit recurrence evaluation.
    with localcontext() as ctx:
        ctx.prec = 80
        ref_decimal = [Decimal(value) for value in references]
        candidate_error = max(
            abs(Decimal.from_float(float(got)) - expected)
            for got, expected in zip(candidate, ref_decimal)
        )
        native_error = max(
            abs(Decimal.from_float(float(got)) - expected)
            for got, expected in zip(native, ref_decimal)
        )
    assert candidate_error <= native_error
