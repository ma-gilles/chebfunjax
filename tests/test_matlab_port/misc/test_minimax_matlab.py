"""Independent ports of all twenty native minimax assertions.

MATLAB source: tests/misc/test_minimax.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Native seedRNG(6178) first 100 sites are captured in the binary64 fixture.
Clause 13 is the full rational degree (30, 30) case; bounds are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils._matlab_linspace import source_grid
from chebfunjax.utils.cfpade import cf
from chebfunjax.utils.minimax import minimax

jax.config.update("jax_enable_x64", True)
EPS = np.finfo(float).eps


def _sites():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "minimax_rng6178.json").read_text())
    return jnp.asarray(np.frombuffer(bytes.fromhex(fixture["binary64_hex"]), dtype="<f8").copy())


def _poly(result):
    return chebfun(jnp.asarray(result.coeffs), domain=result.domain, coeffs=True)


def _mm(f, n, **kwargs):
    bp = f.domain.breakpoints
    return minimax(f, n, domain=(float(bp[0]), float(bp[-1])), **kwargs)


def _rat(f, m, n, **kwargs):
    bp = f.domain.breakpoints
    return minimax(f, m, domain=(float(bp[0]), float(bp[-1])), rational=True, denom=n, **kwargs)


def _source_sort_roots(values):
    """MATLAB sort: real order, or complex magnitude then phase."""
    values = np.asarray(values)
    if np.any(np.imag(values) != 0):
        return values[np.lexsort((np.angle(values), np.abs(values)))]
    return values[np.argsort(np.real(values))]


class TestMiscMinimax:
    @pytest.mark.parametrize("clause", range(1, 21), ids=lambda n: f"native_{n:02d}")
    def test_matlab_clause(self, clause):
        # Each clause receives the native state it used, independently of
        # earlier failures. Source rational convenience p/q outputs and
        # supremum norms are retained rather than replaced by result.err.
        x = chebfun(lambda t: t, domain=(-1.0, 1.0))
        xx = _sites()
        if clause == 1:
            pbest = _poly(_mm(abs(x) + x, 1))
            assert float((pbest - (.5 + x)).norm(2)) < 1e-10
        elif clause == 2:
            f = (x.exp().sin()).exp()
            assert float((cf(f, 7)[0] - _poly(_mm(f, 7))).norm(2)) < .0003
        elif clause == 3:
            f = ((x + 3) * (x - .5)) / (x ** 2 - 4)
            result = _rat(f, 2, 2, tol=1e-12, max_iter=20)
            assert np.max(np.abs(np.asarray(f(xx)) - result.r(np.asarray(xx)))) < 1e-10
        elif clause == 4:
            x = chebfun(lambda t: t, domain=(-1.0, 0.0, 1.0))
            f = ((x - 3) * (x + .2) * (x - .7)) / ((x - 1.5) * (x + 2.1))
            result = _rat(f, 3, 2)
            assert np.max(np.abs(np.asarray(f(xx)) - result.r(np.asarray(xx)))) < 1e-10
        elif clause == 5:
            pbest = _poly(_mm(abs(x), 3))
            assert np.max(np.abs(np.asarray(pbest(xx) - (x ** 2 + 1 / 8)(xx)))) < 10 * EPS
        elif clause == 6:
            pbest = _poly(_mm(0 * x, 2))
            assert np.max(np.abs(np.asarray(pbest(xx) - (0 * x)(xx)))) < 10 * EPS
        elif clause in (7, 8):
            f = abs(x)
            degree = 0 if clause == 7 else 2
            p, q = _rat(f, degree, degree).as_chebfuns()
            err = float((f - p / q).norm(jnp.inf))
            assert abs(err - (.5 if clause == 7 else .043689)) < (1e-10 if clause == 7 else 1e-3)
        elif clause == 9:
            _rat(x ** 3, 0, 2)  # Native predicate is successful completion.
        elif clause == 10:
            f1, f2 = chebfun("exp(x)"), chebfun("1e100*exp(x)")
            err1 = float((f1 - _poly(_mm(f1, 7))).norm(jnp.inf))
            err2 = float((f2 - _poly(_mm(f2, 7))).norm(jnp.inf))
            assert abs(err1 - err2 / 1e100) < 1e-3
        elif clause == 11:
            f1, f2 = chebfun("exp(x)"), chebfun("1e-100*exp(x)")
            sample1 = float(_rat(f1, 1, 3).r(np.asarray(.3))) - float(f1(jnp.asarray(.3)))
            sample2 = float(_rat(f2, 1, 3).r(np.asarray(.3))) - float(f2(jnp.asarray(.3)))
            assert abs(sample1 - sample2 * 1e100) < 1e-3
        elif clause == 12:
            x = chebfun("x", domain=(-1e30, 2e30))
            f = abs(x)
            assert float((f - _poly(_mm(f, 30))).norm(jnp.inf)) / 1e30 - .0135210 < .01
        elif clause == 13:
            result = _rat(abs(x), 30, 30)
            assert abs(result.err - 2.1739878e-7) / 2.1739878e-7 < 1e-3
        elif clause == 14:
            f = chebfun("exp(x)")
            result = _mm(f, 7)
            assert abs(float((f - _poly(result)).norm(jnp.inf)) - result.err) < 1e-14
        elif clause == 15:
            err1 = minimax("exp(x)", 5).err
            err2 = minimax(lambda t: jnp.exp(t), 5).err
            err3 = _mm(chebfun("exp(x)"), 5).err
            assert err1 == err2 and abs(err1 - err3) < 1e-15
        elif clause in (16, 17):
            f = abs(x - .1).sqrt()
            xl = source_grid(-1, 1, 10000)
            if clause == 16:
                result = _mm(f, 5)
                values = _poly(result)(xl)
            else:
                result = _rat(f, 4, 4)
                values = result.r(np.asarray(xl))
            sampled_error = np.max(np.abs(np.asarray(f(xl)) - np.asarray(values)))
            assert abs(result.err - sampled_error) / result.err < 1e-4
        elif clause == 18:
            assert _rat(1e40 * abs(x), 5, 5).err < 1e38
        elif clause == 19:
            result = minimax(lambda t: jnp.sqrt(t), 4, domain=(0.0, 1.0), rational=True, denom=4)
            p, q = result.as_chebfuns()
            zer1 = _source_sort_roots(p.roots(all_roots=True))
            zer2 = _source_sort_roots(result.zeros)
            pol1 = _source_sort_roots(q.roots(all_roots=True))
            pol2 = _source_sort_roots(result.poles)
            assert zer1.shape == zer2.shape and pol1.shape == pol2.shape
            assert np.linalg.norm(zer1 - zer2, np.inf) / np.linalg.norm(zer1, np.inf) < 1e-5
            assert np.linalg.norm(pol1 - pol2, np.inf) / np.linalg.norm(pol1, np.inf) < 1e-5
        else:
            assert minimax(jnp.exp, 5).err == minimax(lambda t: jnp.exp(t), 5).err
