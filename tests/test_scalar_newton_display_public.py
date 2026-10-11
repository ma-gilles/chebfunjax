"""Native scalarODE clause1 plus supplemental public display invariance."""

import contextlib
import io
import math
import struct
import warnings

import numpy as np

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


def _bits(value):
    if isinstance(value, dict):
        return {k: _bits(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_bits(v) for v in value]
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if hasattr(value, "dtype"):
        a = np.asarray(value)
        return dict(dtype=str(a.dtype), shape=a.shape, bytes=a.tobytes().hex())
    return value


def _compare_source_display():
    saved = ChebopPref._defaults
    observations = []
    try:
        ChebopPref.setDefaults("factory")
        for display in ["off", "iter"]:
            ChebopPref.setDefaults("display", display)
            # Literal test_scalarODE.m setup; default Chebcolloc2 clause1.
            op = Chebop(lambda u: u.diff(2) + (u - 0.2).sin(), domain=(0.0, math.pi))
            op.lbc = lambda u: u - 2
            op.rbc = lambda u: u - 3
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result, info = op.solvebvp(0)
            tol = 1e3 * ChebopPref().bvpTol
            residual = op(result).norm()
            assert residual < tol
            coefficients = np.asarray(result.funs[0].tech.coeffs).copy()
            observations.append(
                dict(
                    display=display,
                    stdout=stream.getvalue(),
                    coefficients=coefficients,
                    info=info,
                    info_bits=_bits(info),
                    residual=float(residual),
                    source_bound=tol,
                    warnings=[
                        {"category": type(w.message).__name__, "message": str(w.message)}
                        for w in caught
                    ],
                )
            )
        a, b = observations
        assert a["stdout"] == ""
        assert a["coefficients"].dtype == b["coefficients"].dtype
        assert a["coefficients"].shape == b["coefficients"].shape
        assert a["coefficients"].tobytes() == b["coefficients"].tobytes()
        assert a["info_bits"] == b["info_bits"]
        assert a["residual"] == b["residual"]
        assert a["warnings"] == b["warnings"]
        rows = [
            line
            for line in b["stdout"].splitlines()
            if line.startswith(" ") and len(line.split()) == 6
        ]
        assert len(rows) == len(b["info"]["iterationHistory"]) == len(b["info"]["normDelta"])
        assert b["stdout"].startswith(
            "Iter.   || du ||   Contraction    stepsize   len(du)   len(u)\n"
        )
        assert f"Newton's method converged in {len(rows)} iterations." in b["stdout"]
        assert "Discretization basis used: Values." in b["stdout"]
        for line, row in zip(rows, b["info"]["iterationHistory"]):
            fields = line.split()
            assert int(fields[0]) == row["iteration"]
            assert int(fields[4]) == row["lenDelta"] and int(fields[5]) == row["lenU"]
        return observations
    finally:
        ChebopPref._defaults = saved
        assert ChebopPref._defaults is saved


def test_literal_scalarODE_clause1_display_off_iter_exact_invariance():
    _compare_source_display()
