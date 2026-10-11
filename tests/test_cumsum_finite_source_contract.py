"""Source-derived API contracts, separate from the original cumsum30 clauses.

Pin7574c77 @chebfun/cumsum.m. Ordinary finite-array dispatch fails upstream;
Quasimatrix finite sums retain the literal shortened-loop endpoint.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain


def constants():
    breaks = [-0.75, -0.25, 0.25, 0.75]
    return Quasimatrix(
        [cj.chebfun(float(j), domain=[-1, b, 1]) for j, b in enumerate(breaks, 1)], Domain((-1, 1))
    )


def test_source_quasimatrix_shortened_loops_and_orientation():
    q = constants()
    sites = jnp.asarray([-0.9, -0.4, 0.0, 0.3, 0.9])
    before = [np.asarray(c.point_values).copy() for c in q.cols]
    for order, expected in [
        (0, [1, 2, 3, 4]),
        (1, [1, 3, 6, 4]),
        (2, [1, 4, 6, 4]),
        (3, [1, 4, 6, 4]),
    ]:
        g = q.cumsum(order, 2)
        np.testing.assert_array_equal(np.asarray(g(sites)), np.tile(expected, (5, 1)))
        assert not g.is_transposed
        row = q.T.cumsum(order, 1)
        np.testing.assert_array_equal(np.asarray(row(sites)), np.tile(expected, (5, 1)).T)
        assert row.is_transposed
    for column, values in zip(q.cols, before):
        np.testing.assert_array_equal(np.asarray(column.point_values), values)


def test_source_ordinary_finite_dispatch_error():
    a = cj.chebfun(lambda x: jnp.stack([x, 2 * x, 3 * x], axis=-1))
    cases = [(a, 1, 2), (a, 2, 2), (a.T, 1, 1), (cj.chebfun("x"), 1, 2), (a, 0, 2), (a, 1, 3)]
    for f, order, dim in cases:
        with pytest.raises(TypeError, match="MATLAB:TooManyInputs"):
            f.cumsum(order, dim)


def test_source_empty_and_fractional_dispatch_order(monkeypatch):
    empty = cj.chebfun()
    assert empty.cumsum(0.5, 2) is empty
    token = object()
    calls = []

    def frac(self, order):
        calls.append((self, order))
        return token

    monkeypatch.setattr(cj.Chebfun, "fracInt", frac)
    f = cj.chebfun("x")
    assert f.cumsum(0.5, 1) is token
    assert f.cumsum(0.5, 2) is token
    assert [order for _, order in calls] == [0.5, 0.5]
    q = constants()
    monkeypatch.setattr(Quasimatrix, "fracInt", frac)
    assert q.cumsum(0.5, 2) is token
    assert q.cumsum(0.5, 1) is token


def test_source_continuous_dimension_and_row_defaults():
    x = cj.chebfun("x")
    for f, dim in [(x, 1), (x.T, 2)]:
        g = f.cumsum(1, dim)
        assert g.is_transposed == f.is_transposed
        np.testing.assert_array_equal(
            np.asarray(g.funs[0].tech.coeffs).reshape(-1), [-0.25, 0, 0.25]
        )
    q = Quasimatrix([x, cj.chebfun(2.0)], Domain((-1, 1)))
    for f, dim in [(q, 1), (q.T, 2)]:
        g = f.cumsum(1, dim)
        assert g.is_transposed == f.is_transposed
        np.testing.assert_array_equal(
            np.asarray(g.cols[0].funs[0].tech.coeffs).reshape(-1), [-0.25, 0, 0.25]
        )
        np.testing.assert_array_equal(np.asarray(g.cols[1].funs[0].tech.coeffs).reshape(-1), [2, 2])


def test_default_continuous_delta_and_explicit_finite_limit():
    x = cj.chebfun("x")
    d = cj.dirac(x)
    sites = jnp.asarray([-0.75, -0.1, 0.1, 0.5, 0.75])
    for f in [d, d.T]:
        g = f.cumsum()
        np.testing.assert_array_equal(np.asarray(g(sites)).reshape(-1), [0, 0, 1, 1, 1])
        assert g.is_transposed == f.is_transposed
        assert tuple(g.domain.breakpoints) == (-1.0, 0.0, 1.0)
        finite_dim = 1 if f.is_transposed else 2
        for order in (0, 1, 2):
            with pytest.raises(NotImplementedError, match="per-FUN delta wrapper metadata"):
                f.cumsum(order, finite_dim)


def test_quasimatrix_two_columns_and_singleton_source_storage():
    q = constants()
    small = Quasimatrix(q.cols[:2], q.domain)
    np.testing.assert_array_equal(np.asarray(small.cumsum(1, 2)(jnp.array([0.0]))), [[1, 2]])
    singleton = Quasimatrix(q.cols[:1], q.domain)
    with pytest.raises(TypeError, match="MATLAB:TooManyInputs"):
        singleton.cumsum(1, 2)


def test_paired_native_affine_observations(tmp_path):
    import json
    import os
    from pathlib import Path

    with np.load(Path(__file__).parent / "fixtures/cumsum_finite_native_values.npz") as fixture:
        sites = jnp.asarray(fixture["sites"])
        breaks = [-0.75, -0.25, 0.25, 0.75]
        cols = [
            cj.chebfun(lambda x, j=j: j + x, domain=[-1, b, 1]) for j, b in enumerate(breaks, 1)
        ]
        q = cj.Chebfun.horzcat(*cols)
        assert isinstance(q, Quasimatrix)
        metrics = {}
        arrays = {}
        for name, order, row in [
            ("quasizero", 0, False),
            ("quasi1", 1, False),
            ("quasi2", 2, False),
            ("quasi3", 3, False),
            ("rowquasi1", 1, True),
        ]:
            f = q.T if row else q
            g = f.cumsum(order, 1 if row else 2)
            actual = np.asarray(g(sites))
            expected = fixture[name]
            assert actual.shape == expected.shape
            assert g.is_transposed == row
            # The last native column is never updated by the shortened loop.
            np.testing.assert_array_equal(
                np.asarray(g.cols[-1].point_values), np.asarray(f.cols[-1].point_values)
            )
            metrics[name] = {
                "max_abs_difference": float(np.max(np.abs(actual - expected))),
                "shape": list(actual.shape),
            }
            arrays[name] = actual
        # Observed provider differences have no invented native pass threshold.
        out = Path(os.environ.get("CHEBFUNJAX_CAPTURE_DIR", str(tmp_path)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "PAIRED_NATIVE_AFFINE.json").write_text(json.dumps(metrics, indent=2) + "\n")
        np.savez(out / "PAIRED_NATIVE_AFFINE.npz", **arrays)
