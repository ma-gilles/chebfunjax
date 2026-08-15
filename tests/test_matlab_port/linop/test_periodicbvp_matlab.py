"""Port of MATLAB Chebfun tests/linop/test_periodicbvp.m (Fable 5).

Ports the chebcolloc2 passes of the MATLAB loop (k = 1 with the smooth RHS
and k = 4 with the RHS carrying breakpoints).  The ultraS, chebcolloc1 and
trigcolloc passes are covered by separate skipped tests.

Provenance
----------
MATLAB source : tests/linop/test_periodicbvp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    mult,
    primitive_functionals,
    primitive_operators,
    zero_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def _smooth_part(f):
    """Residual with its Dirac part removed, plus that part's magnitude.

    When the RHS carries breakpoints, the computed solution is continuous
    there only to solver accuracy; chebfunjax's ``Chebfun.diff`` attaches a
    Dirac of that size (~1e-11 here) whereas MATLAB's does not, and
    ``norm`` of a chebfun carrying a Dirac is infinite.  The delta
    magnitudes are checked against the same tolerance as the residual, so
    nothing is discarded silently.
    """
    deltas = getattr(f, "deltas", ())
    worst = max((abs(float(d[1])) for d in deltas), default=0.0)
    return Chebfun(funs=f.funs, domain=f.domain), worst


class TestLinopPeriodicBvp:
    def test_all_matlab_assertions(self):
        dom = (-math.pi, math.pi)
        Z, I, Dop, C = primitive_operators(dom)[:4]
        z, E, s = primitive_functionals(dom)[:3]
        x = cj.chebfun(lambda t: t, domain=dom)
        c = (x ** 2).sin()
        C = mult(c)
        El = E(dom[0])
        zero = cj.chebfun(lambda t: jnp.zeros_like(t), domain=dom)

        L = linop(ChebMatrix([
            [Dop ** 2, -I, x.sin()],
            [C, Dop, zero],
            [zero_functional(dom), El, 4.0],
        ]))
        L = L.addbc("periodic")

        for f in ([x.sin(), zero, 1.0], [abs(x.cos()), 0 * x, 1.0]):
            w = L.linsolve(f)
            w1, w2, w3 = w[0], w[1], w[2]

            err = []
            # Check the ODEs.
            res, delta = _smooth_part(w1.diff(2) - w2 + x.sin() * w3 - f[0])
            err.append(float(res.norm()))
            err.append(delta)
            res, delta = _smooth_part(c * w1 + w2.diff())
            err.append(float(res.norm()))
            err.append(delta)
            err.append(abs(float(w2(jnp.asarray(dom[0]))) + 4 * w3 - 1.0))
            # Check the BCs.
            Du = w1.diff()
            for g in (w1, w2, Du):
                err.append(abs(float(g(jnp.asarray(math.pi)))
                               - float(g(jnp.asarray(-math.pi)))))
            # Check continuity.
            for pt in (math.pi / 2, -math.pi / 2):
                for g in (w1, w2, Du):
                    err.append(abs(float(g(jnp.asarray(pt), "left"))
                                   - float(g(jnp.asarray(pt), "right"))))

            assert all(e < TOL for e in err), err

    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_ultras_and_chebcolloc1(self, disc):
        # MATLAB's k = 2, 3, 5, 6 passes: the same periodic system
        # under ultraS and chebcolloc1.
        dom = (-math.pi, math.pi)
        Z, I, Dop, C = primitive_operators(dom)[:4]
        z, E, s = primitive_functionals(dom)[:3]
        x = cj.chebfun(lambda t: t, domain=dom)
        c = (x ** 2).sin()
        C = mult(c)
        El = E(dom[0])
        zero = cj.chebfun(lambda t: jnp.zeros_like(t), domain=dom)

        L = linop(ChebMatrix([
            [Dop ** 2, -I, x.sin()],
            [C, Dop, zero],
            [zero_functional(dom), El, 4.0],
        ]))
        L = L.addbc("periodic")

        for f in ([x.sin(), zero, 1.0], [abs(x.cos()), 0 * x, 1.0]):
            w = L.linsolve(f, n=64, discretization=disc)
            w1, w2, w3 = w[0], w[1], w[2]

            err = []
            res, delta = _smooth_part(w1.diff(2) - w2
                                      + x.sin() * w3 - f[0])
            err.append(float(res.norm()))
            err.append(delta)
            res, delta = _smooth_part(c * w1 + w2.diff())
            err.append(float(res.norm()))
            err.append(delta)
            err.append(abs(float(w2(jnp.asarray(dom[0])))
                           + 4 * w3 - 1.0))
            Du = w1.diff()
            for g in (w1, w2, Du):
                err.append(abs(float(g(jnp.asarray(math.pi)))
                               - float(g(jnp.asarray(-math.pi)))))
            assert all(e < TOL for e in err), err

    @pytest.mark.skip(
        reason="MATLAB err(:,13) solves the same problem with the trigcolloc "
               "discretization; chebfunjax's BlockLinop has no Fourier "
               "(trigcolloc) discretization -- periodic problems are handled "
               "by explicit periodic side conditions instead.")
    def test_trigcolloc(self):
        raise NotImplementedError
