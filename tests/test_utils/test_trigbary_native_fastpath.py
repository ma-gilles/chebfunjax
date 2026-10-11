"""Small deterministic fastpath classification diagnostics."""

import mpmath as mp
import numpy as np

from chebfunjax.utils.trigutils import trigBaryWeights


def classification_controls():
    x = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    x[4] += 2e-14
    native_tol = max(abs(x)) * np.finfo(float).eps
    native_uniform = bool(np.all(abs(np.diff(x) - 2 * np.pi / len(x)) < native_tol))
    got = np.asarray(trigBaryWeights(x))
    with mp.workdps(80):
        xx = [mp.mpf(float(v)) for v in x]
        ref = [
            1 / mp.fprod(mp.sin((xx[k] - xx[j]) / 2) for k in range(10) if k != j)
            for j in range(10)
        ]
        scale = max(abs(v) for v in ref)
        ref = np.asarray([float(v / scale) for v in ref])
    err = float(max(abs(got - ref)))
    budget = 10 * np.finfo(float).eps
    return [
        {
            "control": "nonuniform_native_rule_rejects",
            "pass_": not native_uniform,
            "native_tolerance": native_tol,
            "maximum_spacing_deviation": float(max(abs(np.diff(x) - 2 * np.pi / len(x)))),
        },
        {
            "control": "nonuniform_not_alternating_fastreturn",
            "pass_": not np.array_equal(got, np.where(np.arange(10) % 2, -1.0, 1.0)),
        },
        {
            "control": "nonuniform_mp80_products",
            "pass_": bool(err < budget),
            "error": err,
            "diagnostic_budget_10eps": budget,
        },
    ]


def test_native_fastpath_classification():
    import jax

    jax.config.update("jax_enable_x64", True)
    for row in classification_controls():
        assert row["pass_"], row
