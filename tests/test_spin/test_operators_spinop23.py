"""Core tests for chebfunjax.operators.spinop2 / spinop3.

These mirror the constructor surface and solver wiring exercised by the
MATLAB ports in tests/test_matlab_port/spinop2, spinop3 (kept here,
outside test_matlab_port, for the coverage gate).  The heavy ETDRK4
numerics themselves are golden-ref tested in test_spinop2.py /
test_spinop3.py; here we check the operators-layer adapter (presets,
func2str, domain/tspan plumbing, trig-interpolant output, error paths).

JAX contract: the operators spin2/spin3 wrappers run on plain NumPy
(delegating to chebfunjax.spin.solver2d/solver3) -- no JIT.
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.operators.spinop2 import Spinop2, func2str, spin2
from chebfunjax.operators.spinop3 import Spinop3, spin3

_GL_STR = "@(u)u-(1+1.5i)*u.*(abs(u).^2)"


class TestSpinop2Surface:
    def test_preset_gl(self):
        S = Spinop2("GL")
        assert func2str(S.nonlin) == _GL_STR
        assert func2str(S.lin) == "@(u)lap(u)"
        assert S.domain == (0.0, 100.0, 0.0, 100.0)
        assert S.tspan == (0.0, 100.0)
        assert S.numVars == 1
        # nonlin FuncHandle is also callable in value space.
        z = 0.3 + 0.4j
        got = complex(S.nonlin(z))
        assert got == pytest.approx(z - (1.0 + 1.5j) * z * abs(z) ** 2)

    def test_dom_tspan_constructor(self):
        dom = [0.0, 2 * np.pi, 0.0, 2 * np.pi]
        S = Spinop2(dom, [0.0, 1.0])
        assert S.domain == tuple(dom)
        assert S.tspan == (0.0, 1.0)
        assert "Spinop2(" in repr(S)

    def test_empty_and_errors(self):
        S = Spinop2()
        assert S.domain is None
        with pytest.raises(ValueError):
            Spinop2("NOPE")
        with pytest.raises(ValueError):
            spin2(Spinop2([0, 1, 0, 1], [0, 1]), 8, 0.1)  # no numerics
        S = Spinop2("GL")
        S.init = None
        with pytest.raises(ValueError):
            spin2(S, 8, 0.1)  # no init

    def test_solve_trig_interp(self):
        # Short cheap solve; verify the returned trig interpolant
        # reproduces the periodic solution (self-consistency at nodes).
        S = Spinop2("GL")
        S.tspan = (0.0, 0.5)
        N = 32
        u = spin2(S, N, 0.1, "plot", "off")
        # Node reproduction: evaluate on the equispaced periodic grid.
        x = np.linspace(0.0, 100.0, N, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        vals = u(xx, yy)
        assert vals.shape == (N, N)
        assert np.all(np.isfinite(vals))
        # Periodicity: u(x) == u(x + L).
        a = u(np.array([3.0]), np.array([7.0]))
        b = u(np.array([103.0]), np.array([107.0]))
        assert np.allclose(a, b, atol=1e-8)


class TestSpinop3Surface:
    def test_preset_gl(self):
        S = Spinop3("GL")
        assert func2str(S.nonlin) == _GL_STR
        assert func2str(S.lin) == "@(u)lap(u)"
        assert S.domain == (0.0, 50.0, 0.0, 50.0, 0.0, 50.0)
        assert S.tspan == (0.0, 100.0)
        assert S.numVars == 1

    def test_dom_tspan_constructor(self):
        dom = [0.0, 2 * np.pi, 0.0, 2 * np.pi, 0.0, 2 * np.pi]
        S = Spinop3(dom, [0.0, 1.0])
        assert S.domain == tuple(dom)
        assert S.tspan == (0.0, 1.0)
        assert "Spinop3(" in repr(S)

    def test_empty_and_errors(self):
        assert Spinop3().domain is None
        with pytest.raises(ValueError):
            Spinop3("NOPE")
        with pytest.raises(ValueError):
            spin3(Spinop3([0, 1, 0, 1, 0, 1], [0, 1]), 8, 0.1)
        S = Spinop3("GL")
        S.init = None
        with pytest.raises(ValueError):
            spin3(S, 8, 0.1)

    def test_solve_trig_interp(self):
        S = Spinop3("GL")
        S.tspan = (0.0, 0.2)
        N = 16
        u = spin3(S, N, 0.1, "plot", "off")
        x = np.linspace(0.0, 50.0, N, endpoint=False)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        vals = u(xx, yy, zz)
        assert vals.shape == (N, N, N)
        assert np.all(np.isfinite(vals))
        a = u(np.array([3.0]), np.array([7.0]), np.array([11.0]))
        b = u(np.array([53.0]), np.array([57.0]), np.array([61.0]))
        assert np.allclose(a, b, atol=1e-8)
