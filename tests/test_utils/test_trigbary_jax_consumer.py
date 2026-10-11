"""Independent native-formula and JAX adapter controls, not source clauses."""

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.trigutils import trigBary


def literal(x, values, nodes, domain):
    # Independent direct products and literal native loop, original n parity.
    n = len(nodes)
    a, b = domain
    knots = np.pi / (b - a) * (2 * np.asarray(nodes) - a - b)
    targets = np.pi / (b - a) * (2 * np.asarray(x).ravel(order="F") - a - b)
    data = np.asarray(values, dtype=complex).copy()
    if data.ndim == 1:
        data = data[:, None]
    if np.linalg.norm([knots[0] + np.pi, knots[-1] - np.pi]) < 2 * np.pi * np.finfo(float).eps:
        data[0] = (data[0] + data[-1]) / 2
        data = data[:-1]
        knots = knots[:-1]
    weights = []
    for j in range(len(knots)):
        weights.append(
            1 / np.prod([np.sin((knots[k] - knots[j]) / 2) for k in range(len(knots)) if k != j])
        )
    weights = np.asarray(weights)
    weights /= max(abs(weights))
    s = sum(knots)
    c = 0 if abs(np.fmod(s, np.pi)) < 4 * np.pi * np.finfo(float).eps else 1 / np.tan(s / 2)
    if np.all(np.any(np.isnan(data), axis=0)):
        return np.full((len(targets), data.shape[1]), np.nan, dtype=complex)
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for target in targets:
            ratio = weights * (
                (1 / np.tan((target - knots) / 2) + c)
                if n % 2 == 0
                else 1 / np.sin((target - knots) / 2)
            )
            value = ratio @ data / sum(ratio)
            if np.isnan(value[0]):
                member = np.flatnonzero(target == knots)
                if len(member):
                    value = data[member[0]]
            out.append(value)
    return np.asarray(out)


def controls():
    rows = []

    def record(name, fn):
        try:
            detail = fn()
            rows.append({"control": name, "pass_": bool(detail[0]), "detail": detail[1]})
        except Exception as e:
            rows.append(
                {
                    "control": name,
                    "pass_": False,
                    "exception": type(e).__name__,
                    "message": str(e)[:180],
                }
            )

    def endpoint(n):
        nodes = np.linspace(-2, 2, n)
        data = np.exp(1j * np.pi * nodes) + 0.2 * np.sin(np.pi * nodes / 2)
        data[-1] += 0.01j
        x = np.array([-0.9, -0.2, 0.4, 1.3])
        ref = literal(x, data, nodes, (-2, 2))[:, 0]
        got = np.asarray(
            jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data), jnp.asarray(nodes), (-2.0, 2.0))
        )
        err = float(max(abs(got - ref)))
        budget = 64 * n * np.finfo(float).eps * max(1, max(abs(ref)))
        return err < budget, {"error": err, "diagnostic_roundoff_budget": budget, "original_n": n}

    record("native_literal_endpoint_even6", lambda: endpoint(6))
    record("native_literal_endpoint_odd5", lambda: endpoint(5))
    nodes = np.linspace(-np.pi, np.pi, 10, endpoint=False)

    def single_shape():
        data = np.exp(1j * nodes)
        x = np.array([[0.1, 0.3, 0.5], [-0.2, 0.7, 0.9]])
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data)))
        return got.shape == x.shape and max(abs((got - np.exp(1j * x)).ravel())) < 1e-12, {
            "shape": list(got.shape)
        }

    record("jit_complex_default_nodes_2d_shape", single_shape)

    def array_shape():
        data = np.column_stack([np.exp(1j * nodes), np.exp(-2j * nodes)])
        x = np.array([[0.1, 0.3, 0.5], [-0.2, 0.7, 0.9]])
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data), jnp.asarray(nodes)))
        ref = np.hstack([np.exp(1j * x), np.exp(-2j * x)])
        return got.shape == (2, 6) and max(abs((got - ref).ravel())) < 1e-12, {
            "shape": list(got.shape)
        }

    record("jit_multi_complex_native_horizontal_shape", array_shape)

    def exact():
        data = np.column_stack([np.exp(1j * nodes), 1 + np.exp(-2j * nodes)])
        got = np.asarray(
            jax.jit(trigBary)(jnp.asarray(nodes), jnp.asarray(data), jnp.asarray(nodes))
        )
        return np.array_equal(got, data), {"exact": bool(np.array_equal(got, data))}

    record("jit_exact_nodes_multiple_complex", exact)

    def scalar():
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(0.1), jnp.asarray(np.exp(1j * nodes))))
        return got.shape == () and abs(got - np.exp(0.1j)) < 1e-12, {"shape": list(got.shape)}

    record("jit_scalar_adapter", scalar)

    def constant():
        x = np.zeros((2, 3, 4))
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray([2 + 3j])))
        return got.shape == x.shape and np.all(got == 2 + 3j), {"shape": list(got.shape)}

    record("jit_single_column_nd_constant", constant)

    def empty():
        got = np.asarray(jax.jit(trigBary)(jnp.empty((0, 2)), jnp.asarray([1.0])))
        return got.shape == (0, 2), {"shape": list(got.shape)}

    record("jit_empty_shape", empty)

    def rejection(kind):
        try:
            if kind == "nd":
                trigBary(np.zeros((2, 2, 2)), np.ones((10, 2)), nodes)
            elif kind == "matrix":
                trigBary([0.0], np.ones(10), np.zeros((2, 5)))
            else:
                trigBary([0.0], np.ones(10), nodes + 10)
        except ValueError:
            return True, {"rejected": kind}
        return False, {"rejected": False}

    for kind in ["nd", "matrix", "outofdomain"]:
        record("eager_rejection_" + kind, lambda kind=kind: rejection(kind))

    def nd_column():
        x = np.zeros((2, 3, 4))
        data = np.exp(1j * nodes)[:, None]
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data), jnp.asarray(nodes)))
        return got.shape == x.shape and np.max(np.abs(got - 1)) < 1e-12, {"shape": list(got.shape)}

    record("jit_single_column_matrix_data_nd_shape", nd_column)

    def nan_shape():
        x = np.zeros((2, 3))
        data = np.full((10, 1), np.nan)
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data), jnp.asarray(nodes)))
        return got.shape == x.shape and np.all(np.isnan(got)), {"shape": list(got.shape)}

    record("jit_nan_shape_adapter", nan_shape)

    def nan_native_case(kind):
        knots = np.linspace(-np.pi, np.pi, 6, endpoint=False)
        data = np.column_stack([np.cos(knots), np.sin(knots)])
        if kind == "first_partial":
            data[1, 0] = np.nan
        elif kind == "second_partial":
            data[1, 1] = np.nan
        elif kind == "both_columns":
            data[1, 0] = np.nan
            data[2, 1] = np.nan
        else:
            data = data[:, 0]
            data[1] = np.nan
        x = np.concatenate([knots, [0.05]])
        ref = literal(x, data, knots, (-np.pi, np.pi))
        got = np.asarray(jax.jit(trigBary)(jnp.asarray(x), jnp.asarray(data), jnp.asarray(knots)))
        if got.ndim == 1:
            got = got[:, None]
        mask = np.isnan(ref)
        finite_error = float(np.max(np.abs(got[~mask] - ref[~mask]))) if np.any(~mask) else 0.0
        valid = np.array_equal(np.isnan(got), mask) and finite_error < 1e-12
        exact = np.array_equal(got[:6], ref[:6], equal_nan=True)
        return valid and exact, {
            "finite_error": finite_error,
            "exact_node_rows_equal": bool(exact),
            "kind": kind,
        }

    for kind in ["first_partial", "second_partial", "both_columns", "single_column"]:
        record("literal_native_nan_" + kind, lambda kind=kind: nan_native_case(kind))
    return rows


def test_trigbary_jax_consumer_controls():
    jax.config.update("jax_enable_x64", True)
    for row in controls():
        assert row["pass_"], row
