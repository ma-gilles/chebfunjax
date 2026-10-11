"""Diagnostic stability controls, separate from nine native source predicates.

80-digit direct products check weights up to a common factor;64*n*eps is
only a diagnostic roundoff budget. Large uniform grid checks the O(n)
conditional fast path. Clustered2500nodes check finite/max1normalization,
not interpolation conditioning. Scoped mpmath precision restores caller state.
"""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from chebfunjax.utils.trigutils import trigBaryWeights


def controls():
    rows = []
    rng = np.random.default_rng(3453)
    rng.random(1000)
    rng.random(8)
    rng.random(100)
    x = np.sort(rng.random(2500))
    w = np.asarray(trigBaryWeights(x))
    rows.append(
        {
            "control": "2500finite",
            "pass_": bool(np.all(np.isfinite(w))),
            "finite_count": int(np.isfinite(w).sum()),
        }
    )
    rows.append(
        {
            "control": "2500nonzero_normalization",
            "pass_": bool(np.all(np.isfinite(w)) and np.max(np.abs(w)) == 1 and np.any(w != 0)),
        }
    )
    context_before = mp.mp
    precision_before = mp.mp.dps
    with mp.workdps(80):
        for n in [16, 31]:
            nodes = np.random.default_rng(n).uniform(-2, 2, n)
            xx = [mp.mpf(float(v)) for v in nodes]
            wr = []
            for j in range(n):
                wr.append(1 / mp.fprod(mp.sin((xx[k] - xx[j]) / 2) for k in range(n) if k != j))
            scale = max(abs(v) for v in wr)
            ref = np.array([float(v / scale) for v in wr])
            got = np.asarray(trigBaryWeights(nodes))
            err = float(np.max(np.abs(got - ref)))
            bound = 64 * n * np.finfo(float).eps
            rows.append(
                {
                    "control": f"mp80directproduct_n{n}",
                    "absolute_normalized_error": err,
                    "diagnostic_roundoff_budget": bound,
                    "pass_": bool(err < bound),
                }
            )
    rows.append(
        {
            "control": "mp_context_identity_precision_restored",
            "pass_": bool(mp.mp is context_before and mp.mp.dps == precision_before),
        }
    )
    nodes = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    a = np.asarray(trigBaryWeights(nodes))
    try:
        b = np.asarray(jax.jit(trigBaryWeights)(jnp.asarray(nodes)))
    except Exception as e:
        rows.append(
            {"control": "jaxjit_fastpath_exact", "pass_": False, "exception": type(e).__name__}
        )
        return rows
    rows.append(
        {
            "control": "jaxjit_fastpath_exact",
            "pass_": bool(
                np.array_equal(a, b) and np.array_equal(a, np.where(np.arange(10) % 2, -1.0, 1.0))
            ),
        }
    )
    # n=16384: O(n) input/output each128KiB; old full float64 V would be2GiB.
    large_n = 16384
    uniform = 2 * np.pi * np.arange(large_n) / large_n
    large = np.asarray(jax.jit(trigBaryWeights)(jnp.asarray(uniform)))
    rows.append(
        {
            "control": "large_uniform_fastpath_16384",
            "pass_": bool(np.array_equal(large, np.where(np.arange(large_n) % 2, -1.0, 1.0))),
            "n": large_n,
        }
    )
    return rows


def test_trigbary_weight_stability_controls():
    for row in controls():
        assert row["pass_"], row
