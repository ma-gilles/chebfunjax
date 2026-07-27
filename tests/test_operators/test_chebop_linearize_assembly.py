"""Core mirror tests for the structured linear-operator assembly.

``Chebop._linearize_op`` builds the collocation matrix of a linear (or
linearised) operator ``L[u] = c_0(x) u + ... + c_m(x) u^(m)`` without ever
tracing a single ``O(n)``-column ``jnp.stack`` graph (which blew up the XLA
compile -- an ~11 min ``jit_stack`` compile then an out-of-memory segfault --
for the stiff variable-coefficient BVPs of guide chapter 7).

The fast path probes ``L`` on the scaled monomials ``y^k/k!`` (``m + 1``
operator applications, independent of ``n``) and forward-substitutes for the
coefficient functions ``c_k``; ``_assembly_ok`` validates the result and, for
operators the differential form cannot represent (integral / nonlocal), the
code falls back to a general per-column probe materialised eagerly (never a
monolithic stack).

These tests pin, outside the MATLAB-port tree so the core coverage gate sees
them:
  * the structured matrix reproduces the reference collocation matrix (the
    eager column-by-column probe) to machine precision;
  * the forced-fallback path reproduces the structured path byte-for-byte;
  * ``_sniff_order`` detects the differential order and bails to ``None`` on
    nonlocal operators;
  * ``_assembly_ok`` accepts a faithful matrix and rejects a wrong one.

Provenance
----------
MATLAB source : @chebop/linearize.m, @linop/linop.m, diffmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.operators.chebop import Chebop, _chebfun_to_values


def _reference_probe_matrix(cop: Chebop, disc: ChebColloc2Disc) -> np.ndarray:
    """Reference collocation matrix: the eager column-by-column probe.

    This is the mathematical definition of the operator's collocation matrix
    (Frechet derivative at ``u = 0``) and the pre-optimisation behaviour of
    ``_linearize_op`` -- the correctness oracle the structured assembly must
    reproduce.
    """
    a, b = disc.domain
    n = disc.n
    dom = Domain((a, b))
    x_fun = Chebfun.identity(dom)
    zero = Chebfun.from_values(jnp.zeros(n, dtype=jnp.float64), dom)
    try:
        op0 = np.asarray(
            _chebfun_to_values(cop._apply_op(x_fun, zero), disc),
            dtype=np.float64,
        )
    except Exception:
        op0 = np.zeros(n)
    cols = []
    for j in range(n):
        e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(1.0)
        u_j = Chebfun.from_values(e_j, dom)
        v_j = np.asarray(
            _chebfun_to_values(cop._apply_op(x_fun, u_j), disc),
            dtype=np.float64,
        )
        cols.append(v_j - op0)
    return np.stack(cols, axis=1)


DOM = (-1.0, 1.0)

# (name, op, expected differential order)
_CASES = [
    ("const_uxx", lambda x, u: u.diff(2), 2),
    ("const_uxx_plus_u", lambda x, u: u.diff(2) + u, 2),
    ("helmholtz", lambda x, u: u.diff(2) + 100.0 * u, 2),
    ("conv_diff", lambda x, u: 0.01 * u.diff(2) + u.diff() + u, 2),
    ("varcoef_x_uxx", lambda x, u: x * u.diff(2) + u, 2),
    ("varcoef_sinx", lambda x, u: u.diff(2) + x.sin() * u.diff() + u, 2),
    ("affine_plus_f", lambda x, u: u.diff(2) + u + x.exp(), 2),
    ("first_order", lambda x, u: u.diff() + x * u, 1),
]


@pytest.mark.parametrize("name,op,order", _CASES, ids=[c[0] for c in _CASES])
def test_structured_matches_reference_probe(name, op, order):
    """Structured assembly reproduces the eager column probe (A-vs-A)."""
    cop = Chebop(op, DOM, 0.0, 0.0)
    disc = ChebColloc2Disc(24, DOM)

    # The structured fast path must actually fire for these well-conditioned
    # differential operators (otherwise the perf fix does nothing).
    m = cop._sniff_order(Chebfun.identity(Domain(DOM)), 8)
    assert m == order

    A_new = np.asarray(cop._linearize_op().matrix(disc), dtype=np.float64)
    A_ref = _reference_probe_matrix(cop, disc)
    scale = max(float(np.max(np.abs(A_ref))), 1e-30)
    assert np.max(np.abs(A_new - A_ref)) / scale < 1e-9


@pytest.mark.parametrize("name,op,order", _CASES, ids=[c[0] for c in _CASES])
def test_forced_fallback_matches_structured(name, op, order):
    """The eager per-column fallback reproduces the structured path exactly.

    Forcing ``_sniff_order`` to ``None`` routes assembly through the general
    column probe; its matrix must equal the reference probe bit-for-bit (same
    eager arithmetic) and therefore match the structured path.
    """
    cop = Chebop(op, DOM, 0.0, 0.0)
    disc = ChebColloc2Disc(20, DOM)
    A_struct = np.asarray(cop._linearize_op().matrix(disc), dtype=np.float64)

    cop._sniff_order = lambda *a, **k: None  # force the fallback branch
    A_fb = np.asarray(cop._linearize_op().matrix(disc), dtype=np.float64)
    A_ref = _reference_probe_matrix(cop, disc)

    # Fallback is the exact same eager computation as the reference probe.
    assert np.array_equal(A_fb, A_ref)
    # And it agrees with the structured fast path to machine precision.
    scale = max(float(np.max(np.abs(A_ref))), 1e-30)
    assert np.max(np.abs(A_struct - A_fb)) / scale < 1e-9


def test_sniff_order_bails_on_nonlocal():
    """`_sniff_order` returns None for an operator with a nonlocal term."""
    x_id = Chebfun.identity(Domain(DOM))
    # cumsum is not understood by the order sniffer -> bail to the general probe.
    cop = Chebop(lambda x, u: u.diff(2) + u.cumsum(), DOM, 0.0, 0.0)
    assert cop._sniff_order(x_id, 8) is None

    # A pure high-order-but-bounded operator is detected.
    cop2 = Chebop(lambda x, u: u.diff(4) + u, DOM, 0.0, 0.0)
    assert cop2._sniff_order(x_id, 8) == 4

    # Order above max_order also bails.
    assert cop2._sniff_order(x_id, 3) is None


def test_nonlocal_falls_back_and_is_correct():
    """A nonlocal operator routes through the fallback and stays correct."""
    cop = Chebop(lambda x, u: u.diff(2) + u.cumsum(), DOM, 0.0, 0.0)
    disc = ChebColloc2Disc(20, DOM)
    A_new = np.asarray(cop._linearize_op().matrix(disc), dtype=np.float64)
    A_ref = _reference_probe_matrix(cop, disc)
    assert np.array_equal(A_new, A_ref)


def test_assembly_ok_accepts_and_rejects():
    """`_assembly_ok` accepts a faithful matrix and rejects a wrong one."""
    n = 20
    disc = ChebColloc2Disc(n, DOM)
    a, b = DOM
    cop = Chebop(lambda x, u: u.diff(2) + x.sin() * u.diff() + u, DOM, 0.0, 0.0)

    from chebfunjax.utils.quadrature import chebpts

    t_ref = np.asarray(chebpts(n, kind=2))
    nodes = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
    x0 = 0.5 * (a + b)
    h = 0.5 * (b - a)
    y = (nodes - x0) / h

    x_fun = Chebfun.identity(Domain(DOM))

    def op_at(vals):
        uf = Chebfun.from_values(jnp.asarray(vals, dtype=jnp.float64),
                                 Domain(DOM))
        return np.asarray(_chebfun_to_values(cop._apply_op(x_fun, uf), disc),
                          dtype=np.float64)

    op0 = op_at(np.zeros(n))
    A_good = _reference_probe_matrix(cop, disc)
    assert Chebop._assembly_ok(A_good, op0, op_at, y) is True

    A_bad = A_good + 1.0  # clearly not the operator
    assert Chebop._assembly_ok(A_bad, op0, op_at, y) is False
