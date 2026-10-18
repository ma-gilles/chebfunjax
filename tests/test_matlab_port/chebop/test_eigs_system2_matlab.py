"""Port of MATLAB Chebfun tests/chebop/test_eigs_system2.m (Fable 5).

The same Maxwell-inspired eigenproblem as test_eigs_system.m, which
MATLAB states in chebmatrix {} cell syntax (u{1}, u{2}); the cell
indexing is represented by a legacy multi-argument smooth-spectrum smoke
check here. All six literal original cell/autonomous clauses are in
tests/test_operators/test_eigs_system2_original_cells_source.py.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_system2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


class TestChebopEigsSystem2:
    def test_all_matlab_assertions(self):
        d = (0.0, np.pi)
        A = Chebop(lambda x, u, v: [-u + v.diff(), u.diff()], d)
        A.lbc = lambda u, v: u
        A.rbc = lambda u, v: u
        _, lam = A.eigs(k=5)
        lam = np.sort(np.abs(np.asarray(lam)))
        correct = np.sort(np.abs(np.array([
            0,
            -0.5 + np.sqrt(3) / 2 * 1j,
            -0.5 - np.sqrt(3) / 2 * 1j,
            -0.5 + np.sqrt(15) / 2 * 1j,
            -0.5 - np.sqrt(15) / 2 * 1j])))
        assert np.max(np.abs(lam - correct)) < 1e-10
