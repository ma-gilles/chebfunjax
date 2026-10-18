"""Native nobreaks exchange controls; Chebfun commit7574c77.

The public exchange control checks jump-root semantics. One-time preservation
of the previous Python outputs is retained in the qualification evidence;
the known symmetric reference mismatch is not a permanent output oracle.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

FIXTURES = Path(__file__).parent / "fixtures/trigremez_polynomial_exchange_source"


def _piecewise_constants(coefficients, point_values):
    return Chebfun(
        funs=[
            _Piece(tech=Chebtech2.from_coeffs(jnp.asarray([value])), interval=interval)
            for value, interval in zip(coefficients, [(-1.0, 0.0), (0.0, 1.0)])
        ],
        domain=Domain((-1.0, 0.0, 1.0)),
    ).set_point_values(jnp.asarray(point_values))


def test_native_nobreaks_preserves_true_jump_root():
    native = json.loads((FIXTURES / "root_controls.json").read_text())
    jump = _piecewise_constants([-1.0, 1.0], [-1.0, 1.0, 1.0])
    # nobreaks omits isolated stored zeros while retaining true jump roots.
    np.testing.assert_array_equal(jump.roots(nobreaks=True), native["jump_nobreaks"])
    np.testing.assert_array_equal(jump.roots(nojump=True), native["jump_nojump"])


def test_native_nobreaks_omits_isolated_stored_breakpoint_zero():
    native = json.loads((FIXTURES / "root_controls.json").read_text())
    pure_break = _piecewise_constants([1.0, 1.0], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(pure_break.roots(nobreaks=True), native["pure_break_nobreaks"])
    # Preserve the separate native default: it includes the stored zero.
    np.testing.assert_array_equal(pure_break.roots(), native["pure_break_default"])


def test_public_polynomial_exchange_includes_true_jump_root(monkeypatch):
    observed = []
    original = Chebfun.roots

    def capture_roots(self, *args, **kwargs):
        assert kwargs.get("nobreaks") is True
        result = original(self, *args, **kwargs)
        observed.append(np.asarray(result))
        return result

    f = abs(cj.chebfun("x"))
    monkeypatch.setattr(Chebfun, "roots", capture_roots)
    with pytest.warns(UserWarning, match="did not converge after 1 iterations"):
        cj.trigremez(f, 1, max_iter=1)
    assert len(observed) == 1
    assert len(observed[0]) == 5
    assert observed[0][2] == 0.0
