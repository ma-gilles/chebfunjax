"""Core-suite mirrors for one-sided Chebfun evaluation and ``jump``.

Exercises ``Chebfun(x, side)`` left/right limits at interior breakpoints and
the module-level ``jump`` helper (``f(x,'+') - f(x,'-') - c``), including the
detection recording used by the chebop jump solver.

Provenance
----------
MATLAB source : @chebfun/feval.m, @chebfun/jump.m; Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chebfunjax.chebfun1d.chebfun import (  # noqa: E402
    Chebfun,
    _Piece,
    jump,
    start_side_eval_record,
    stop_side_eval_record,
)
from chebfunjax.domain import Domain  # noqa: E402


def _piecewise_jump():
    """f = x on [-1,0] and x+1 on [0,1]: a jump of +1 in value at x=0."""
    dom = Domain((-1.0, 0.0, 1.0))
    m = 8
    kk = np.arange(m)
    tref = np.cos(np.pi * kk / (m - 1))[::-1]
    xl = -1.0 + (tref + 1.0) / 2.0
    xr = 0.0 + (tref + 1.0) / 2.0
    fl = _Piece.from_values(jnp.asarray(xl), -1.0, 0.0)
    fr = _Piece.from_values(jnp.asarray(xr + 1.0), 0.0, 1.0)
    return Chebfun(funs=[fl, fr], domain=dom)


class TestSideEval:
    def test_one_sided_limits(self):
        f = _piecewise_jump()
        assert abs(float(f(0.0, "left"))) < 1e-12
        assert abs(float(f(0.0, "right")) - 1.0) < 1e-12
        # Aliases.
        assert abs(float(f(0.0, "-"))) < 1e-12
        assert abs(float(f(0.0, "+")) - 1.0) < 1e-12

    def test_jump_value_and_offset(self):
        f = _piecewise_jump()
        assert abs(float(jump(f, 0.0)) - 1.0) < 1e-12
        assert abs(float(jump(f, 0.0, 1.0))) < 1e-12   # jump - c

    def test_jump_of_derivative_is_continuous(self):
        # Both pieces have slope 1, so the first derivative is continuous.
        f = _piecewise_jump()
        assert abs(float(jump(f.diff(), 0.0))) < 1e-10

    def test_side_eval_recording(self):
        f = _piecewise_jump()
        start_side_eval_record()
        _ = f(0.0, "left")
        _ = jump(f, 0.3)
        pts = stop_side_eval_record()
        assert any(abs(p - 0.0) < 1e-12 for p in pts)
        assert any(abs(p - 0.3) < 1e-12 for p in pts)
        # Recording is off again afterwards (no capture).
        _ = f(0.0, "right")
        assert stop_side_eval_record() == []
