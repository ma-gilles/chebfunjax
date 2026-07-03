"""Continuous skeletonization study for bivariate function approximation.

This script compares fixed diagonal support skeletons with continuous CPLU,
continuous C2PLU, and randomized continuous RPLU on [-1, 1]^2.

The adaptive deterministic pivots are not selected by an argmax over an
equispaced or Chebyshev candidate grid. Chebyshev/Lobatto samples are used only
as initializer samples for continuous SciPy optimization of the residual.

Default run:

    python examples/approx2/continuous_skeletonization_study.py \
        --output-dir outputs/continuous_skeletonization

Smoke run:

    python examples/approx2/continuous_skeletonization_study.py \
        --smoke --output-dir outputs/continuous_skeletonization_smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    CHEBFUN2_AVAILABLE = True
except Exception:  # pragma: no cover - import diagnostics are written to report.
    jax = None
    jnp = None
    Chebfun2 = None
    CHEBFUN2_AVAILABLE = False


ArrayFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


ALL_ERROR_COLUMNS = [
    "function",
    "category",
    "method",
    "trial",
    "N",
    "rel_max_error",
    "abs_max_error",
    "cond_middle",
    "min_singular_value",
    "used_pinv",
    "pivot_alpha",
    "pivot_beta",
    "pivot_value",
]

SUMMARY_COLUMNS = [
    "function",
    "category",
    "predicted_slope",
    "method",
    "E20",
    "E40",
    "E80",
    "E120",
    "fitted_slope_10_80",
    "fitted_slope_20_120",
    "median_condition_number",
]


@dataclass(frozen=True)
class FunctionCase:
    """One test function and plotting metadata."""

    name: str
    category: str
    predicted_slope: float | None
    reference_label: str
    func: ArrayFunc


@dataclass(frozen=True)
class SolveInfo:
    """Conditioning diagnostics for the skeleton middle matrix."""

    cond: float
    min_singular_value: float
    used_pinv: bool
    numerical_rank: int


@dataclass
class Skeleton:
    """Continuous skeleton approximation from support sets A and B."""

    func: ArrayFunc
    alpha: np.ndarray
    beta: np.ndarray
    middle_inverse: np.ndarray
    info: SolveInfo

    @classmethod
    def from_supports(
        cls,
        func: ArrayFunc,
        alpha: Iterable[float],
        beta: Iterable[float],
        cond_threshold: float,
        svd_rcond: float,
    ) -> "Skeleton":
        alpha_arr = np.asarray(list(alpha), dtype=np.float64)
        beta_arr = np.asarray(list(beta), dtype=np.float64)
        if alpha_arr.size != beta_arr.size:
            raise ValueError("alpha and beta support sets must have the same length.")
        if alpha_arr.size == 0:
            return cls(
                func=func,
                alpha=alpha_arr,
                beta=beta_arr,
                middle_inverse=np.zeros((0, 0), dtype=np.float64),
                info=SolveInfo(cond=0.0, min_singular_value=np.inf, used_pinv=False, numerical_rank=0),
            )

        middle = eval_tensor(func, alpha_arr, beta_arr)
        singular_values = np.linalg.svd(middle, compute_uv=False)
        sigma_max = float(singular_values[0]) if singular_values.size else 0.0
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        if sigma_max == 0.0:
            cond = np.inf
        elif sigma_min == 0.0:
            cond = np.inf
        else:
            cond = sigma_max / sigma_min

        use_pinv = (not np.isfinite(cond)) or cond >= cond_threshold
        if use_pinv:
            cutoff = svd_rcond * sigma_max
            numerical_rank = int(np.sum(singular_values > cutoff))
            middle_inverse = np.linalg.pinv(middle, rcond=svd_rcond)
        else:
            numerical_rank = int(alpha_arr.size)
            middle_inverse = np.linalg.solve(middle, np.eye(alpha_arr.size, dtype=np.float64))

        return cls(
            func=func,
            alpha=alpha_arr,
            beta=beta_arr,
            middle_inverse=middle_inverse,
            info=SolveInfo(
                cond=float(cond),
                min_singular_value=float(sigma_min),
                used_pinv=bool(use_pinv),
                numerical_rank=numerical_rank,
            ),
        )

    @property
    def rank(self) -> int:
        return int(self.alpha.size)

    def evaluate_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        shape = np.broadcast_shapes(x_arr.shape, y_arr.shape)
        x_flat = np.broadcast_to(x_arr, shape).ravel()
        y_flat = np.broadcast_to(y_arr, shape).ravel()
        if self.rank == 0:
            return np.zeros_like(x_flat, dtype=np.float64).reshape(shape)
        left = self.func(x_flat[:, None], self.beta[None, :])
        right = self.func(self.alpha[:, None], y_flat[None, :])
        values = np.einsum("mi,ij,jm->m", left, self.middle_inverse, right, optimize=True)
        return values.reshape(shape)

    def evaluate_tensor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if self.rank == 0:
            return np.zeros((x_arr.size, y_arr.size), dtype=np.float64)
        left = self.func(x_arr[:, None], self.beta[None, :])
        right = self.func(self.alpha[:, None], y_arr[None, :])
        return left @ self.middle_inverse @ right


@dataclass
class ResidualEvaluator:
    """Continuous residual evaluator f - skeleton."""

    func: ArrayFunc
    skeleton: Skeleton | None = None

    def pairwise(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        exact = self.func(x_arr, y_arr)
        if self.skeleton is None or self.skeleton.rank == 0:
            return np.asarray(exact, dtype=np.float64)
        return np.asarray(exact, dtype=np.float64) - self.skeleton.evaluate_pairwise(x_arr, y_arr)

    def tensor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        exact = eval_tensor(self.func, x, y)
        if self.skeleton is None or self.skeleton.rank == 0:
            return exact
        return exact - self.skeleton.evaluate_tensor(x, y)


def cheb_lobatto_weights(n: int) -> np.ndarray:
    """Barycentric weights for ascending Chebyshev-Lobatto nodes."""
    if n <= 1:
        return np.array([1.0], dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    nodes_desc = np.cos(np.pi * j / float(n - 1))
    weights_desc = (-1.0) ** j
    weights_desc[0] *= 0.5
    weights_desc[-1] *= 0.5
    order = np.argsort(nodes_desc)
    return weights_desc[order].astype(np.float64)


@dataclass(frozen=True)
class ChebyshevLineModel:
    """High-order Chebyshev-Lobatto barycentric representation on [-1, 1]."""

    nodes: np.ndarray
    weights: np.ndarray
    values: np.ndarray

    @classmethod
    def from_evaluator(cls, evaluator: Callable[[float], float], n: int) -> "ChebyshevLineModel":
        nodes = cheb_lobatto_nodes(n)
        weights = cheb_lobatto_weights(n)
        values = np.array([float(evaluator(float(x))) for x in nodes], dtype=np.float64)
        return cls(nodes=nodes, weights=weights, values=values)

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        flat = x_arr.ravel()
        out = np.empty_like(flat, dtype=np.float64)
        for k, point in enumerate(flat):
            diffs = point - self.nodes
            hit = np.where(np.abs(diffs) <= 10.0 * np.finfo(np.float64).eps)[0]
            if hit.size:
                out[k] = self.values[int(hit[0])]
            else:
                terms = self.weights / diffs
                out[k] = float(np.sum(terms * self.values) / np.sum(terms))
        return out.reshape(x_arr.shape)


@dataclass(frozen=True)
class ChebyshevTensorModel:
    """High-order tensor Chebyshev-Lobatto barycentric residual model."""

    nodes: np.ndarray
    weights: np.ndarray
    values: np.ndarray

    @classmethod
    def from_residual(cls, residual: ResidualEvaluator, n: int) -> "ChebyshevTensorModel":
        nodes = cheb_lobatto_nodes(n)
        weights = cheb_lobatto_weights(n)
        values = residual.tensor(nodes, nodes)
        return cls(nodes=nodes, weights=weights, values=np.asarray(values, dtype=np.float64))

    def evaluate(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        shape = np.broadcast_shapes(x_arr.shape, y_arr.shape)
        x_flat = np.broadcast_to(x_arr, shape).ravel()
        y_flat = np.broadcast_to(y_arr, shape).ravel()
        out = np.empty_like(x_flat, dtype=np.float64)
        for k, (xx, yy) in enumerate(zip(x_flat, y_flat)):
            line_vals = np.empty(self.nodes.size, dtype=np.float64)
            for col in range(self.nodes.size):
                line = ChebyshevLineModel(self.nodes, self.weights, self.values[:, col])
                line_vals[col] = float(line.evaluate(xx))
            out[k] = float(ChebyshevLineModel(self.nodes, self.weights, line_vals).evaluate(yy))
        return out.reshape(shape)


def eval_tensor(func: ArrayFunc, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate func on a tensor product with row coordinate x and column coordinate y."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return np.asarray(func(x_arr[:, None], y_arr[None, :]), dtype=np.float64)


def cheb_lobatto_nodes(n: int) -> np.ndarray:
    """N Chebyshev-Lobatto nodes in ascending order; for n=1 use {0}."""
    if n <= 1:
        return np.array([0.0], dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    return np.sort(np.cos(np.pi * j / float(n - 1))).astype(np.float64)


def cheb_root_nodes(n: int) -> np.ndarray:
    """N Chebyshev root nodes in ascending order."""
    j = np.arange(n, dtype=np.float64)
    return np.sort(np.cos(np.pi * (j + 0.5) / float(n))).astype(np.float64)


def equispaced_nodes(n: int) -> np.ndarray:
    """N equispaced nodes in [-1, 1]; for n=1 use {0}."""
    if n <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(n), dtype=np.float64)


def top_k_flat_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values in a flattened array."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size <= k:
        return np.argsort(flat)[::-1]
    idx = np.argpartition(flat, -k)[-k:]
    return idx[np.argsort(flat[idx])[::-1]]


def maximize_abs_2d(
    residual: ResidualEvaluator,
    initializer_n: int,
    top_seeds: int,
    maxiter: int,
) -> tuple[float, float, float, dict[str, float]]:
    """Maximize |R(x,y)| continuously from Chebyshev initializer samples."""
    cheb_model = ChebyshevTensorModel.from_residual(residual, initializer_n)
    nodes = cheb_model.nodes
    vals = np.abs(cheb_model.values)
    seed_indices = top_k_flat_indices(vals, top_seeds)

    candidates: list[tuple[float, float, float, bool]] = []

    def objective(z: np.ndarray) -> float:
        xx = float(np.clip(z[0], -1.0, 1.0))
        yy = float(np.clip(z[1], -1.0, 1.0))
        val = float(np.abs(residual.pairwise(xx, yy)))
        if not np.isfinite(val):
            return 1e300
        return -val

    for idx in seed_indices:
        i, j = np.unravel_index(int(idx), vals.shape)
        seed = np.array([nodes[i], nodes[j]], dtype=np.float64)
        result = optimize.minimize(
            objective,
            seed,
            method="Powell",
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            options={"maxiter": int(maxiter), "xtol": 1e-11, "ftol": 1e-13, "disp": False},
        )
        point = np.asarray(result.x if np.all(np.isfinite(result.x)) else seed, dtype=np.float64)
        point = np.clip(point, -1.0, 1.0)
        value = float(np.abs(residual.pairwise(float(point[0]), float(point[1]))))
        candidates.append((value, float(point[0]), float(point[1]), bool(result.success)))

    if not candidates:
        raise RuntimeError("continuous 2D maximization had no initializer samples.")
    best = max(candidates, key=lambda item: item[0])
    diagnostics = {
        "initializer_n": float(initializer_n),
        "top_seeds": float(top_seeds),
        "optimizer_success_rate": float(sum(c[3] for c in candidates) / len(candidates)),
    }
    return best[1], best[2], best[0], diagnostics


def intervals_around_top_samples(nodes: np.ndarray, values: np.ndarray, top_seeds: int) -> list[tuple[float, float]]:
    """Small bounded intervals around top initializer samples for scalar refinement."""
    idxs = top_k_flat_indices(np.asarray(values, dtype=np.float64), top_seeds)
    intervals: list[tuple[float, float]] = []
    mids = 0.5 * (nodes[:-1] + nodes[1:]) if nodes.size > 1 else np.array([], dtype=np.float64)
    for idx_raw in idxs:
        idx = int(idx_raw)
        left = -1.0 if idx == 0 else float(mids[idx - 1])
        right = 1.0 if idx == nodes.size - 1 else float(mids[idx])
        if right < left:
            left, right = right, left
        intervals.append((left, right))
    return intervals


def maximize_scalar_from_samples(
    evaluator: Callable[[float], float],
    initializer_n: int,
    top_seeds: int,
    maxiter: int,
) -> tuple[float, float, dict[str, float]]:
    """Maximize a nonnegative scalar objective with Chebyshev initializer samples."""
    cheb_model = ChebyshevLineModel.from_evaluator(evaluator, initializer_n)
    nodes = cheb_model.nodes
    sample_values = cheb_model.values
    intervals = intervals_around_top_samples(nodes, sample_values, top_seeds)
    candidates: list[tuple[float, float, bool]] = []

    for left, right in intervals:
        if abs(right - left) < 1e-15:
            x_opt = 0.5 * (left + right)
            candidates.append((float(evaluator(x_opt)), x_opt, True))
            continue
        result = optimize.minimize_scalar(
            lambda t: -float(evaluator(float(t))),
            bounds=(left, right),
            method="bounded",
            options={"maxiter": int(maxiter), "xatol": 1e-12},
        )
        x_opt = float(np.clip(result.x, -1.0, 1.0))
        value = float(evaluator(x_opt))
        candidates.append((value, x_opt, bool(result.success)))

    if not candidates:
        raise RuntimeError("continuous scalar maximization had no initializer samples.")
    best = max(candidates, key=lambda item: item[0])
    diagnostics = {
        "initializer_n": float(initializer_n),
        "top_seeds": float(top_seeds),
        "optimizer_success_rate": float(sum(c[2] for c in candidates) / len(candidates)),
    }
    return best[1], best[0], diagnostics


def fixed_support_skeleton(
    case: FunctionCase,
    n: int,
    support_rule: str,
    cond_threshold: float,
    svd_rcond: float,
) -> Skeleton:
    """Build the prescribed fixed diagonal support skeleton for rank n."""
    if support_rule == "fixed_equispaced":
        pts = equispaced_nodes(n)
    elif support_rule == "fixed_cheb_lobatto":
        pts = cheb_lobatto_nodes(n)
    else:
        raise ValueError(f"unknown support rule {support_rule!r}")
    return Skeleton.from_supports(case.func, pts, pts, cond_threshold=cond_threshold, svd_rcond=svd_rcond)


def build_validation_set(
    grid_n: int,
    random_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent validation set used only for final error measurement."""
    grid_nodes = cheb_root_nodes(grid_n)
    if random_n <= 0:
        return grid_nodes, grid_nodes, np.empty(0), np.empty(0)
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    if random_n > 0 and (random_n & (random_n - 1)) == 0:
        sample = sampler.random_base2(int(math.log2(random_n)))
    else:
        sample = sampler.random(random_n)
    random_xy = 2.0 * sample - 1.0
    return grid_nodes, grid_nodes, random_xy[:, 0], random_xy[:, 1]


def validation_norm(case: FunctionCase, validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    x_grid, y_grid, x_random, y_random = validation
    grid_vals = eval_tensor(case.func, x_grid, y_grid)
    max_val = float(np.max(np.abs(grid_vals)))
    if x_random.size:
        random_vals = case.func(x_random, y_random)
        max_val = max(max_val, float(np.max(np.abs(random_vals))))
    return max(max_val, np.finfo(np.float64).tiny)


def evaluate_error(
    case: FunctionCase,
    skeleton: Skeleton,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    batch_size: int,
) -> tuple[float, float]:
    """Relative max validation error, using tensor roots plus random off-grid points."""
    x_grid, y_grid, x_random, y_random = validation
    exact_grid = eval_tensor(case.func, x_grid, y_grid)
    approx_grid = skeleton.evaluate_tensor(x_grid, y_grid)
    abs_max = float(np.max(np.abs(exact_grid - approx_grid)))

    for start in range(0, int(x_random.size), int(batch_size)):
        stop = min(start + int(batch_size), int(x_random.size))
        if start == stop:
            continue
        xr = x_random[start:stop]
        yr = y_random[start:stop]
        exact_random = case.func(xr, yr)
        approx_random = skeleton.evaluate_pairwise(xr, yr)
        abs_max = max(abs_max, float(np.max(np.abs(exact_random - approx_random))))
    return abs_max / f_norm, abs_max


def error_row(
    case: FunctionCase,
    method: str,
    trial: int,
    n: int,
    rel_error: float,
    abs_error: float,
    skeleton: Skeleton,
    pivot_alpha: float | None = None,
    pivot_beta: float | None = None,
    pivot_value: float | None = None,
) -> dict[str, object]:
    """CSV row for one error measurement."""
    return {
        "function": case.name,
        "category": case.category,
        "method": method,
        "trial": trial,
        "N": n,
        "rel_max_error": rel_error,
        "abs_max_error": abs_error,
        "cond_middle": skeleton.info.cond,
        "min_singular_value": skeleton.info.min_singular_value,
        "used_pinv": skeleton.info.used_pinv,
        "pivot_alpha": "" if pivot_alpha is None else pivot_alpha,
        "pivot_beta": "" if pivot_beta is None else pivot_beta,
        "pivot_value": "" if pivot_value is None else pivot_value,
    }


def run_fixed_method(
    case: FunctionCase,
    method: str,
    nmax: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    cond_threshold: float,
    svd_rcond: float,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in range(1, nmax + 1):
        skeleton = fixed_support_skeleton(case, n, method, cond_threshold, svd_rcond)
        rel_error, abs_error = evaluate_error(case, skeleton, validation, f_norm, batch_size)
        rows.append(error_row(case, method, 0, n, rel_error, abs_error, skeleton))
    return rows


def run_uniform_random_support(
    case: FunctionCase,
    trial: int,
    nmax: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    cond_threshold: float,
    svd_rcond: float,
    batch_size: int,
    rng: np.random.Generator,
    diagonal: bool,
) -> list[dict[str, object]]:
    """Prescribed uniform-random support skeletons, one independent attempt."""
    rows: list[dict[str, object]] = []
    for n in range(1, nmax + 1):
        alpha = np.sort(rng.uniform(-1.0, 1.0, size=n).astype(np.float64))
        beta = alpha.copy() if diagonal else np.sort(rng.uniform(-1.0, 1.0, size=n).astype(np.float64))
        skeleton = Skeleton.from_supports(case.func, alpha, beta, cond_threshold, svd_rcond)
        rel_error, abs_error = evaluate_error(case, skeleton, validation, f_norm, batch_size)
        rows.append(error_row(case, "fixed_uniform_random", trial, n, rel_error, abs_error, skeleton))
    return rows


def run_cplu(
    case: FunctionCase,
    nmax: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    cond_threshold: float,
    svd_rcond: float,
    batch_size: int,
    initializer_n: int,
    top_seeds: int,
    optimizer_maxiter: int,
    stop_factor: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alpha: list[float] = []
    beta: list[float] = []
    skeleton = Skeleton.from_supports(case.func, alpha, beta, cond_threshold, svd_rcond)
    stopped = False
    for n in range(1, nmax + 1):
        pivot_alpha = pivot_beta = pivot_value = None
        if not stopped:
            residual = ResidualEvaluator(case.func, skeleton if skeleton.rank else None)
            pivot_alpha, pivot_beta, abs_pivot, _ = maximize_abs_2d(
                residual, initializer_n=initializer_n, top_seeds=top_seeds, maxiter=optimizer_maxiter
            )
            pivot_value = abs_pivot
            if abs_pivot <= stop_factor * f_norm:
                stopped = True
            else:
                alpha.append(float(pivot_alpha))
                beta.append(float(pivot_beta))
                skeleton = Skeleton.from_supports(
                    case.func, alpha, beta, cond_threshold=cond_threshold, svd_rcond=svd_rcond
                )
        rel_error, abs_error = evaluate_error(case, skeleton, validation, f_norm, batch_size)
        rows.append(
            error_row(
                case,
                "continuous_cplu",
                0,
                n,
                rel_error,
                abs_error,
                skeleton,
                pivot_alpha,
                pivot_beta,
                pivot_value,
            )
        )
    return rows


def run_c2plu(
    case: FunctionCase,
    nmax: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    cond_threshold: float,
    svd_rcond: float,
    batch_size: int,
    initializer_n: int,
    top_seeds: int,
    optimizer_maxiter: int,
    row_quad_n: int,
    stop_factor: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alpha: list[float] = []
    beta: list[float] = []
    skeleton = Skeleton.from_supports(case.func, alpha, beta, cond_threshold, svd_rcond)
    quad_y, quad_w = np.polynomial.legendre.leggauss(row_quad_n)
    quad_y = quad_y.astype(np.float64)
    quad_w = quad_w.astype(np.float64)
    stopped = False

    for n in range(1, nmax + 1):
        pivot_alpha = pivot_beta = pivot_value = None
        if not stopped:
            residual = ResidualEvaluator(case.func, skeleton if skeleton.rank else None)

            def row_norm2(x: float) -> float:
                xx = np.full_like(quad_y, float(x), dtype=np.float64)
                vals = residual.pairwise(xx, quad_y)
                return float(np.sum(quad_w * np.abs(vals) ** 2))

            pivot_alpha, _, _ = maximize_scalar_from_samples(
                row_norm2,
                initializer_n=initializer_n,
                top_seeds=top_seeds,
                maxiter=optimizer_maxiter,
            )

            def row_abs(y: float) -> float:
                return float(np.abs(residual.pairwise(pivot_alpha, float(y))))

            pivot_beta, abs_pivot, _ = maximize_scalar_from_samples(
                row_abs,
                initializer_n=initializer_n,
                top_seeds=top_seeds,
                maxiter=optimizer_maxiter,
            )
            pivot_value = abs_pivot
            if abs_pivot <= stop_factor * f_norm:
                stopped = True
            else:
                alpha.append(float(pivot_alpha))
                beta.append(float(pivot_beta))
                skeleton = Skeleton.from_supports(
                    case.func, alpha, beta, cond_threshold=cond_threshold, svd_rcond=svd_rcond
                )
        rel_error, abs_error = evaluate_error(case, skeleton, validation, f_norm, batch_size)
        rows.append(
            error_row(
                case,
                "continuous_c2plu",
                0,
                n,
                rel_error,
                abs_error,
                skeleton,
                pivot_alpha,
                pivot_beta,
                pivot_value,
            )
        )
    return rows


def run_rplu_trial(
    case: FunctionCase,
    trial: int,
    nmax: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f_norm: float,
    cond_threshold: float,
    svd_rcond: float,
    batch_size: int,
    quadrature_n: int,
    rng: np.random.Generator,
    stop_factor: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    alpha: list[float] = []
    beta: list[float] = []
    skeleton = Skeleton.from_supports(case.func, alpha, beta, cond_threshold, svd_rcond)
    quad, weights = np.polynomial.legendre.leggauss(quadrature_n)
    quad = quad.astype(np.float64)
    weights = weights.astype(np.float64)
    tensor_weights = np.outer(weights, weights)
    stopped = False

    for n in range(1, nmax + 1):
        pivot_alpha = pivot_beta = pivot_value = None
        if not stopped:
            residual = ResidualEvaluator(case.func, skeleton if skeleton.rank else None)
            values = residual.tensor(quad, quad)
            density = np.abs(values) ** 2 * tensor_weights
            total = float(np.sum(density))
            if total <= (stop_factor * f_norm) ** 2 or not np.isfinite(total):
                stopped = True
            else:
                probs = (density / total).ravel()
                flat_idx = int(rng.choice(probs.size, p=probs))
                i, j = np.unravel_index(flat_idx, density.shape)
                pivot_alpha = float(quad[i])
                pivot_beta = float(quad[j])
                signed_pivot = float(residual.pairwise(pivot_alpha, pivot_beta))
                pivot_value = abs(signed_pivot)
                if pivot_value <= stop_factor * f_norm:
                    stopped = True
                else:
                    alpha.append(pivot_alpha)
                    beta.append(pivot_beta)
                    skeleton = Skeleton.from_supports(
                        case.func, alpha, beta, cond_threshold=cond_threshold, svd_rcond=svd_rcond
                    )
        rel_error, abs_error = evaluate_error(case, skeleton, validation, f_norm, batch_size)
        rows.append(
            error_row(
                case,
                "continuous_rplu",
                trial,
                n,
                rel_error,
                abs_error,
                skeleton,
                pivot_alpha,
                pivot_beta,
                pivot_value,
            )
        )
    return rows


def amplitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mild nonsymmetric amplitude used for skewed tests."""
    return 1.0 + 0.12 * x - 0.09 * y + 0.05 * x * y


def wendland_t(r: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - r, 0.0)


def make_cases() -> list[FunctionCase]:
    """Full function suite from the prompt."""
    rho = 1.0
    lam = 5.0

    def skew(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - 0.62 * y - 0.15

    def skew07(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - 0.7 * y + 0.1

    def skew065(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - 0.65 * y - 0.1

    def phi0(r: np.ndarray) -> np.ndarray:
        t = wendland_t(r)
        return t**2

    def phi2(r: np.ndarray) -> np.ndarray:
        t = wendland_t(r)
        return t**4 * (4.0 * r + 1.0)

    def phi6(r: np.ndarray) -> np.ndarray:
        t = wendland_t(r)
        return t**8 * (32.0 * r**3 + 25.0 * r**2 + 8.0 * r + 1.0)

    cases = [
        FunctionCase(
            "sym_triangular",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: np.maximum(1.0 - np.abs(x - y) / rho, 0.0),
        ),
        FunctionCase(
            "nonsym_triangular",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: amplitude(x, y) * np.maximum(1.0 - np.abs(skew(x, y)) / rho, 0.0),
        ),
        FunctionCase(
            "sym_exp_abs",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: np.exp(-lam * np.abs(x - y)),
        ),
        FunctionCase(
            "nonsym_exp_abs",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: amplitude(x, y) * np.exp(-lam * np.abs(skew07(x, y))),
        ),
        FunctionCase("sym_abs", "A_lipschitz_ridge", -1.0, "N^-1", lambda x, y: np.abs(x - y)),
        FunctionCase(
            "nonsym_abs",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: amplitude(x, y) * np.abs(skew065(x, y)),
        ),
        FunctionCase(
            "sym_relu",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: np.maximum(x - y, 0.0),
        ),
        FunctionCase(
            "nonsym_relu",
            "A_lipschitz_ridge",
            -1.0,
            "N^-1",
            lambda x, y: amplitude(x, y) * np.maximum(x - 0.7 * y - 0.05, 0.0),
        ),
        FunctionCase(
            "sym_holder_half",
            "B_holder_ridge",
            -0.5,
            "N^-1/2",
            lambda x, y: np.sqrt(np.abs(x - y)),
        ),
        FunctionCase(
            "nonsym_holder_half",
            "B_holder_ridge",
            -0.5,
            "N^-1/2",
            lambda x, y: amplitude(x, y) * np.sqrt(np.abs(x - 0.7 * y - 0.1)),
        ),
        FunctionCase(
            "sym_wendland_c0",
            "C_wendland",
            -1.0,
            "N^-1",
            lambda x, y: phi0(np.abs(x - y) / rho),
        ),
        FunctionCase(
            "nonsym_wendland_c0",
            "C_wendland",
            -1.0,
            "N^-1",
            lambda x, y: amplitude(x, y) * phi0(np.abs(skew(x, y)) / rho),
        ),
        FunctionCase(
            "sym_wendland_c2",
            "C_wendland",
            -3.0,
            "N^-3",
            lambda x, y: phi2(np.abs(x - y) / rho),
        ),
        FunctionCase(
            "nonsym_wendland_c2",
            "C_wendland",
            -3.0,
            "N^-3",
            lambda x, y: amplitude(x, y) * phi2(np.abs(skew(x, y)) / rho),
        ),
        FunctionCase(
            "sym_wendland_smooth",
            "C_wendland",
            -7.0,
            "N^-7",
            lambda x, y: phi6(np.abs(x - y) / rho),
        ),
        FunctionCase(
            "nonsym_wendland_smooth",
            "C_wendland",
            -7.0,
            "N^-7",
            lambda x, y: amplitude(x, y) * phi6(np.abs(skew(x, y)) / rho),
        ),
        FunctionCase(
            "cone_center",
            "D_2d_singular",
            -0.5,
            "N^-1/2",
            lambda x, y: np.sqrt((x - 0.2) ** 2 + (y + 0.15) ** 2),
        ),
        FunctionCase(
            "cone_offdiag_modulated",
            "D_2d_singular",
            -0.5,
            "N^-1/2",
            lambda x, y: amplitude(x, y) * np.sqrt((x - 0.35) ** 2 + 0.5 * (y + 0.2) ** 2),
        ),
        FunctionCase("abs_cross", "D_2d_singular", -0.5, "N^-1/2", lambda x, y: np.abs(x) + np.abs(y)),
        FunctionCase("product_abs", "D_2d_singular", -0.5, "N^-1/2", lambda x, y: np.abs(x) * np.abs(y)),
        FunctionCase(
            "radial_holder_half",
            "D_2d_singular",
            -0.5,
            "N^-1/2",
            lambda x, y: ((x - 0.1) ** 2 + (y + 0.2) ** 2) ** 0.25,
        ),
        FunctionCase(
            "runge",
            "E_analytic",
            None,
            "N^-8 visual reference only",
            lambda x, y: 1.0 / (1.0 + 25.0 * (x**2 + y**2)),
        ),
        FunctionCase(
            "nonsym_analytic",
            "E_analytic",
            None,
            "N^-8 visual reference only",
            lambda x, y: np.exp(1.8 * x * y + 0.45 * x - 0.25 * y)
            / (2.5 + 0.4 * x - 0.2 * y + 0.7 * (x - 0.3 * y) ** 2),
        ),
        FunctionCase(
            "boundary_layer",
            "E_analytic",
            None,
            "N^-8 visual reference only",
            lambda x, y: 1.0 / (1.0 + 100.0 * (x + 1.0) ** 2 + 10.0 * (y - 0.2) ** 2),
        ),
        FunctionCase(
            "oscillatory_analytic",
            "E_analytic",
            None,
            "N^-8 visual reference only",
            lambda x, y: np.cos(8.0 * x * y + 2.0 * x - y),
        ),
        FunctionCase(
            "gaussian_ridge",
            "E_analytic",
            None,
            "N^-8 visual reference only",
            lambda x, y: np.exp(-30.0 * (x - y) ** 2),
        ),
        FunctionCase(
            "rank_3",
            "F_low_rank",
            None,
            "exact after rank 3 if supports are nonsingular",
            lambda x, y: np.sin(x) * np.cos(y) + x**2 * np.exp(y) + np.cos(3.0 * x) * y,
        ),
        FunctionCase(
            "rank_5_nonsymmetric",
            "F_low_rank",
            None,
            "exact after rank 5 if supports are nonsingular",
            lambda x, y: (
                np.sin(x) * np.exp(y)
                + (x + 0.3) * np.cos(2.0 * y)
                + (x**2 + 0.1) * np.sin(3.0 * y)
                + np.cos(3.0 * x) * (y + 0.2 * y**2)
                + np.exp(0.3 * x) * (1.0 + y**3)
            ),
        ),
        FunctionCase(
            "diagonal_step",
            "G_discontinuous",
            None,
            "no smooth-convergence reference",
            lambda x, y: (x > y).astype(np.float64),
        ),
        FunctionCase(
            "nonsym_step",
            "G_discontinuous",
            None,
            "no smooth-convergence reference",
            lambda x, y: (x > 0.6 * y + 0.1).astype(np.float64),
        ),
    ]
    return cases


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_trial_curves(rows: list[dict[str, object]], function_name: str) -> dict[str, dict[int, list[float]]]:
    curves: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row["function"] != function_name:
            continue
        method = str(row["method"])
        n = int(row["N"])
        err = float(row["rel_max_error"])
        curves.setdefault(method, {}).setdefault(n, []).append(err)
    return curves


def median_curve(rows: list[dict[str, object]], function_name: str, method: str) -> dict[int, float]:
    data: dict[int, list[float]] = {}
    for row in rows:
        if row["function"] == function_name and row["method"] == method:
            data.setdefault(int(row["N"]), []).append(float(row["rel_max_error"]))
    return {n: float(np.median(vals)) for n, vals in data.items()}


def percentile_curve(
    rows: list[dict[str, object]],
    function_name: str,
    method: str,
    percentile: float,
) -> dict[int, float]:
    data: dict[int, list[float]] = {}
    for row in rows:
        if row["function"] == function_name and row["method"] == method:
            data.setdefault(int(row["N"]), []).append(float(row["rel_max_error"]))
    return {n: float(np.percentile(vals, percentile)) for n, vals in data.items()}


def representative_value(
    rows: list[dict[str, object]],
    function_name: str,
    method: str,
    n: int,
) -> float:
    values = [
        float(row["rel_max_error"])
        for row in rows
        if row["function"] == function_name and row["method"] == method and int(row["N"]) == int(n)
    ]
    if not values:
        return float("nan")
    return float(np.median(values))


def fitted_slope(curve: dict[int, float], n_min: int, n_max: int) -> float:
    xs = []
    ys = []
    for n, err in sorted(curve.items()):
        if n_min <= n <= n_max and err > 0 and np.isfinite(err):
            xs.append(math.log(float(n)))
            ys.append(math.log(float(err)))
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def make_summary_rows(rows: list[dict[str, object]], cases: list[FunctionCase]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    methods = [
        "fixed_equispaced",
        "fixed_cheb_lobatto",
        "fixed_uniform_random",
        "continuous_cplu",
        "continuous_c2plu",
        "continuous_rplu",
    ]
    for case in cases:
        for method in methods:
            curve = median_curve(rows, case.name, method)
            conds = [
                float(row["cond_middle"])
                for row in rows
                if row["function"] == case.name and row["method"] == method and np.isfinite(float(row["cond_middle"]))
            ]
            summary.append(
                {
                    "function": case.name,
                    "category": case.category,
                    "predicted_slope": "" if case.predicted_slope is None else case.predicted_slope,
                    "method": method,
                    "E20": curve.get(20, float("nan")),
                    "E40": curve.get(40, float("nan")),
                    "E80": curve.get(80, float("nan")),
                    "E120": curve.get(120, float("nan")),
                    "fitted_slope_10_80": fitted_slope(curve, 10, 80),
                    "fitted_slope_20_120": fitted_slope(curve, 20, 120),
                    "median_condition_number": float(np.median(conds)) if conds else float("nan"),
                }
            )
    return summary


def make_grid_comparison_rows(rows: list[dict[str, object]], cases: list[FunctionCase], rank: int = 80) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for case in cases:
        equi = representative_value(rows, case.name, "fixed_equispaced", rank)
        cheb = representative_value(rows, case.name, "fixed_cheb_lobatto", rank)
        ratio = cheb / equi if equi > 0 and np.isfinite(equi) else float("nan")
        if np.isfinite(equi) and np.isfinite(cheb):
            winner = "fixed_cheb_lobatto" if cheb < equi else "fixed_equispaced"
            comments = f"Chebyshev lower at N={rank}." if cheb < equi else f"Equispaced lower at N={rank}."
        else:
            winner = "unavailable"
            comments = f"Requested rank N={rank} not present in this run."
        out.append(
            {
                "function": case.name,
                "comparison_rank": rank,
                "fixed_equi_error": equi,
                "fixed_cheb_error": cheb,
                "cheb_over_equi": ratio,
                "winner": winner,
                "comments": comments,
            }
        )
    return out


def make_adaptive_vs_fixed_rows(rows: list[dict[str, object]], cases: list[FunctionCase], ranks: Iterable[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for case in cases:
        for rank in ranks:
            equi = representative_value(rows, case.name, "fixed_equispaced", rank)
            cheb = representative_value(rows, case.name, "fixed_cheb_lobatto", rank)
            uniform_random = representative_value(rows, case.name, "fixed_uniform_random", rank)
            cplu = representative_value(rows, case.name, "continuous_cplu", rank)
            c2plu = representative_value(rows, case.name, "continuous_c2plu", rank)
            rplu = representative_value(rows, case.name, "continuous_rplu", rank)
            out.append(
                {
                    "function": case.name,
                    "category": case.category,
                    "N": rank,
                    "fixed_equispaced": equi,
                    "fixed_cheb_lobatto": cheb,
                    "fixed_uniform_random_median": uniform_random,
                    "continuous_cplu": cplu,
                    "continuous_c2plu": c2plu,
                    "continuous_rplu_median": rplu,
                    "cplu_over_equi": cplu / equi if equi > 0 and np.isfinite(equi) else float("nan"),
                    "cplu_over_cheb": cplu / cheb if cheb > 0 and np.isfinite(cheb) else float("nan"),
                    "cplu_over_uniform_random": cplu / uniform_random
                    if uniform_random > 0 and np.isfinite(uniform_random)
                    else float("nan"),
                    "c2plu_over_cplu": c2plu / cplu if cplu > 0 and np.isfinite(cplu) else float("nan"),
                    "rplu_over_cplu": rplu / cplu if cplu > 0 and np.isfinite(cplu) else float("nan"),
                }
            )
    return out


def save_figure(fig: matplotlib.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_function_curves(rows: list[dict[str, object]], cases: list[FunctionCase], outdir: Path) -> None:
    colors = {
        "fixed_equispaced": "tab:blue",
        "fixed_cheb_lobatto": "tab:orange",
        "fixed_uniform_random": "tab:cyan",
        "continuous_cplu": "tab:green",
        "continuous_c2plu": "tab:red",
        "continuous_rplu": "tab:purple",
    }
    labels = {
        "fixed_equispaced": "fixed equispaced support",
        "fixed_cheb_lobatto": "fixed Chebyshev-Lobatto support",
        "fixed_uniform_random": "uniform random support median",
        "continuous_cplu": "continuous CPLU",
        "continuous_c2plu": "continuous C2PLU",
        "continuous_rplu": "continuous RPLU median",
    }
    for case in cases:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for method in ["fixed_equispaced", "fixed_cheb_lobatto", "continuous_cplu", "continuous_c2plu"]:
            curve = median_curve(rows, case.name, method)
            if not curve:
                continue
            ns = np.array(sorted(curve), dtype=np.float64)
            errs = np.array([curve[int(n)] for n in ns], dtype=np.float64)
            ax.loglog(ns, errs, marker="o", ms=3, lw=1.4, label=labels[method], color=colors[method])
        uniform_med = median_curve(rows, case.name, "fixed_uniform_random")
        if uniform_med:
            uniform_p10 = percentile_curve(rows, case.name, "fixed_uniform_random", 10.0)
            uniform_p90 = percentile_curve(rows, case.name, "fixed_uniform_random", 90.0)
            ns = np.array(sorted(uniform_med), dtype=np.float64)
            med = np.array([uniform_med[int(n)] for n in ns], dtype=np.float64)
            lo = np.array([uniform_p10[int(n)] for n in ns], dtype=np.float64)
            hi = np.array([uniform_p90[int(n)] for n in ns], dtype=np.float64)
            ax.loglog(
                ns,
                med,
                marker="o",
                ms=3,
                lw=1.2,
                color=colors["fixed_uniform_random"],
                label=labels["fixed_uniform_random"],
            )
            ax.fill_between(ns, lo, hi, color=colors["fixed_uniform_random"], alpha=0.16, linewidth=0)
        rplu_med = median_curve(rows, case.name, "continuous_rplu")
        if rplu_med:
            rplu_p10 = percentile_curve(rows, case.name, "continuous_rplu", 10.0)
            rplu_p90 = percentile_curve(rows, case.name, "continuous_rplu", 90.0)
            ns = np.array(sorted(rplu_med), dtype=np.float64)
            med = np.array([rplu_med[int(n)] for n in ns], dtype=np.float64)
            lo = np.array([rplu_p10[int(n)] for n in ns], dtype=np.float64)
            hi = np.array([rplu_p90[int(n)] for n in ns], dtype=np.float64)
            ax.loglog(ns, med, marker="o", ms=3, lw=1.4, color=colors["continuous_rplu"], label=labels["continuous_rplu"])
            ax.fill_between(ns, lo, hi, color=colors["continuous_rplu"], alpha=0.18, linewidth=0)

        slope = case.predicted_slope if case.predicted_slope is not None else (-8.0 if case.category == "E_analytic" else None)
        if slope is not None:
            all_curve = median_curve(rows, case.name, "continuous_cplu")
            if all_curve:
                ns = np.array(sorted(all_curve), dtype=np.float64)
                anchor_n = ns[min(len(ns) - 1, max(0, len(ns) // 3))]
                anchor_e = max(all_curve[int(anchor_n)], np.finfo(np.float64).tiny)
                ref = anchor_e * (ns / anchor_n) ** slope
                label = case.reference_label
                style = "--" if case.category == "E_analytic" else ":"
                ax.loglog(ns, ref, style, color="black", lw=1.0, label=label)
        ax.set_title(case.name)
        ax.set_xlabel("rank N")
        ax.set_ylabel("relative validation max error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, outdir / "plots" / "curves" / case.name)


def plot_summary_scatter(rows: list[dict[str, object]], cases: list[FunctionCase], outdir: Path, rank: int) -> None:
    def values(method: str) -> np.ndarray:
        return np.array([representative_value(rows, case.name, method, rank) for case in cases], dtype=np.float64)

    equi = values("fixed_equispaced")
    cheb = values("fixed_cheb_lobatto")
    uniform_random = values("fixed_uniform_random")
    cplu = values("continuous_cplu")
    c2plu = values("continuous_c2plu")
    rplu = values("continuous_rplu")

    plots = [
        (equi, cheb, f"fixed equispaced E{rank}", f"fixed Cheb E{rank}", "fixed_cheb_vs_equi"),
        (
            equi,
            uniform_random,
            f"fixed equispaced E{rank}",
            f"uniform random median E{rank}",
            "uniform_random_vs_fixed_equi",
        ),
        (
            cheb,
            uniform_random,
            f"fixed Cheb E{rank}",
            f"uniform random median E{rank}",
            "uniform_random_vs_fixed_cheb",
        ),
        (
            uniform_random,
            cplu,
            f"uniform random median E{rank}",
            f"continuous CPLU E{rank}",
            "cplu_vs_uniform_random",
        ),
        (equi, cplu, f"fixed equispaced E{rank}", f"continuous CPLU E{rank}", "cplu_vs_fixed_equi"),
        (cheb, cplu, f"fixed Cheb E{rank}", f"continuous CPLU E{rank}", "cplu_vs_fixed_cheb"),
        (cplu, c2plu, f"continuous CPLU E{rank}", f"continuous C2PLU E{rank}", "c2plu_vs_cplu"),
        (cplu, rplu, f"continuous CPLU E{rank}", f"RPLU median E{rank}", "rplu_vs_cplu"),
    ]
    for x, y, xlabel, ylabel, name in plots:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.any(mask):
            ax.loglog(x[mask], y[mask], "o", ms=5)
            lo = float(min(np.min(x[mask]), np.min(y[mask])))
            hi = float(max(np.max(x[mask]), np.max(y[mask])))
            ax.loglog([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        save_figure(fig, outdir / "plots" / "summary" / name)


def plot_ratio_bars(rows: list[dict[str, object]], cases: list[FunctionCase], outdir: Path, ranks: Iterable[int]) -> None:
    for rank in ranks:
        labels = [case.name for case in cases]
        cheb_ratios = []
        random_ratios = []
        for case in cases:
            equi = representative_value(rows, case.name, "fixed_equispaced", rank)
            cheb = representative_value(rows, case.name, "fixed_cheb_lobatto", rank)
            uniform_random = representative_value(rows, case.name, "fixed_uniform_random", rank)
            cheb_ratios.append(cheb / equi if equi > 0 and np.isfinite(equi) else np.nan)
            random_ratios.append(uniform_random / equi if equi > 0 and np.isfinite(equi) else np.nan)
        fig, ax = plt.subplots(figsize=(max(9.0, 0.32 * len(labels)), 5.0))
        ax.bar(np.arange(len(labels)), cheb_ratios, color="tab:orange")
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_ylabel(f"fixed Cheb / fixed equispaced error at N={rank}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        save_figure(fig, outdir / "plots" / "summary" / f"fixed_cheb_over_equi_N{rank}")

        if any(np.isfinite(random_ratios)):
            fig, ax = plt.subplots(figsize=(max(9.0, 0.32 * len(labels)), 5.0))
            ax.bar(np.arange(len(labels)), random_ratios, color="tab:cyan")
            ax.axhline(1.0, color="black", lw=1.0, ls="--")
            ax.set_yscale("log")
            ax.set_ylabel(f"uniform random median / fixed equispaced error at N={rank}")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
            ax.grid(True, axis="y", which="both", alpha=0.25)
            save_figure(fig, outdir / "plots" / "summary" / f"uniform_random_over_equi_N{rank}")


def generate_report(
    outdir: Path,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    grid_rows: list[dict[str, object]],
    adaptive_rows: list[dict[str, object]],
    cases: list[FunctionCase],
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Write a concise report with explicit interpretation questions."""
    def finite_values(key: str, source: list[dict[str, object]]) -> list[float]:
        vals = []
        for row in source:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                vals.append(value)
        return vals

    ratios = finite_values("cheb_over_equi", grid_rows)
    cheb_wins = sum(1 for row in grid_rows if row["winner"] == "fixed_cheb_lobatto")
    equi_wins = sum(1 for row in grid_rows if row["winner"] == "fixed_equispaced")
    ratio_med = float(np.median(ratios)) if ratios else float("nan")
    comparison_rank = int(grid_rows[0]["comparison_rank"]) if grid_rows else min(80, args.nmax)

    cplu_over_equi = finite_values("cplu_over_equi", adaptive_rows)
    cplu_over_cheb = finite_values("cplu_over_cheb", adaptive_rows)
    cplu_over_uniform_random = finite_values("cplu_over_uniform_random", adaptive_rows)
    c2_over_cplu = finite_values("c2plu_over_cplu", adaptive_rows)
    rplu_over_cplu = finite_values("rplu_over_cplu", adaptive_rows)

    pinv_count = sum(1 for row in rows if bool(row["used_pinv"]))
    total_rows = len(rows)
    slope_lines = []
    for case in cases:
        if case.predicted_slope is None:
            continue
        curve = median_curve(rows, case.name, "continuous_cplu")
        fit = fitted_slope(curve, 10, min(80, args.nmax))
        if np.isfinite(fit):
            slope_lines.append(f"- {case.name}: observed CPLU slope {fit:.2f}; reference {case.reference_label}.")

    report = f"""# Continuous Skeletonization Study

Generated by `examples/approx2/continuous_skeletonization_study.py`.

Runtime: {elapsed:.1f} seconds.

## Setup

- Rank range: 1 to {args.nmax}.
- RPLU trials per function: {args.rplu_trials}.
- Uniform-random support attempts per function: {args.uniform_random_trials}.
- Validation: {args.validation_grid} x {args.validation_grid} Chebyshev-root tensor grid plus {args.validation_random} Sobol off-grid points.
- Random seed: {args.seed}.
- Chebfun2 import available: {CHEBFUN2_AVAILABLE}.
- Middle matrix rule: direct solve if `cond(M_N) < {args.cond_threshold:.1e}`, otherwise truncated SVD pseudoinverse with `rcond={args.svd_rcond:.1e}`.
- Pseudoinverse rows: {pinv_count} of {total_rows}.
- Continuous optimizer setup: each CPLU step builds a high-order tensor
  Chebyshev-Lobatto barycentric residual model with {args.initializer_n} nodes
  per dimension to initialize local optimizers. Each C2PLU scalar maximization
  builds the analogous one-dimensional Chebyshev model. The objective refined
  by SciPy is the continuous residual evaluator, not the initializer model.

This follows the supplied continuous skeletonization note: the block skeleton
`f(x,B_N) f(A_N,B_N)^-1 f(A_N,y)` is the same approximation as the selected
Gaussian-elimination cross approximation, written in the note as
`L_l U_l = C_l M_l^-1 R_l`. The report uses that formula directly for fixed
support baselines and after each adaptive support update.

Throughout this report, "grid" means either a fixed prescribed support set for
the two fixed baselines or an independent validation/integration/optimizer
initialization device. Adaptive CPLU and C2PLU pivots are refined by continuous
optimization and are not chosen by searching an equispaced or Chebyshev
candidate pivot grid. RPLU samples from a quadrature approximation to the
continuous density proportional to `|R(x,y)|^2`; sampled points are not locally
refined in this implementation.

The uniform-random baseline is nonadaptive: each attempt draws support sets
uniformly from `[-1,1]` before validation error is evaluated, then forms the
same block skeleton. By default the random row and column supports are sampled
independently; use `--uniform-random-diagonal` to force `A_N=B_N`.

## Interpretation Questions

1. For fixed support skeletonization, Chebyshev-Lobatto diagonal support wins
   {cheb_wins} cases at N={comparison_rank} and equispaced support wins {equi_wins} cases
   where N={comparison_rank} is available. The median Cheb/equi error ratio is {ratio_med:.3g}.

2. The answer is function-dependent. Inspect `grid_support_comparison.csv` and
   the per-function plots for interior ridges, boundary layers, analytic
   functions, nonsymmetric skewed ridges, and discontinuities; the winner often
   changes with function geometry and conditioning.

3. Fixed supports do not automatically inherit the same constants or fitted
   slopes as continuous adaptive CPLU/C2PLU. The fitted slopes in `summary.csv`
   separate rate behavior from constant-factor shifts.

4. The gap can be a constant, a slope difference, or a conditioning failure.
   The `cond_middle`, `min_singular_value`, and `used_pinv` columns in
   `all_error_curves.csv` identify when fixed support errors are dominated by
   ill-conditioning rather than approximation rate alone.

5. For Lipschitz-only ridge and genuinely 2D singular functions, compare these
   observed slopes against the requested reference lines:
{chr(10).join(slope_lines) if slope_lines else "- Fitted slope ranges are unavailable in this shortened run."}

6. Nonsymmetry mostly changes constants and conditioning unless it changes the
   effective geometry seen by a fixed diagonal support rule. The paired
   symmetric/nonsymmetric rows in `summary.csv` make this visible.

7. Continuous RPLU should be compared by medians and 10/90 bands. In this run
   the median RPLU/CPLU ratio over available adaptive comparison rows is
   {float(np.median(rplu_over_cplu)) if rplu_over_cplu else float("nan"):.3g}.
   Small-pivot failures are visible as very large condition numbers,
   pseudoinverse use, or flat error curves in individual trials.

8. Robustness across functions is best judged by the summary scatter plots and
   `adaptive_vs_fixed.csv`. The median CPLU/fixed-equispaced ratio is
   {float(np.median(cplu_over_equi)) if cplu_over_equi else float("nan"):.3g},
   the median CPLU/fixed-Cheb ratio is
   {float(np.median(cplu_over_cheb)) if cplu_over_cheb else float("nan"):.3g},
   the median CPLU/uniform-random-support ratio is
   {float(np.median(cplu_over_uniform_random)) if cplu_over_uniform_random else float("nan"):.3g},
   and the median C2PLU/CPLU ratio is
   {float(np.median(c2_over_cplu)) if c2_over_cplu else float("nan"):.3g}
   over available comparison rows.

## Files

- `all_error_curves.csv`: every error curve and conditioning diagnostic.
- `summary.csv`: rank snapshots, fitted slopes, and median condition numbers.
- `grid_support_comparison.csv`: fixed Chebyshev versus fixed equispaced at N={comparison_rank}.
- `adaptive_vs_fixed.csv`: deterministic and randomized adaptive methods versus fixed supports.
- `plots/curves/`: per-function log-log plots, PNG and PDF.
- `plots/summary/`: scatter and bar summary plots, PNG and PDF.
"""
    (outdir / "report.md").write_text(report)


def generate_html_index(outdir: Path, cases: list[FunctionCase]) -> None:
    """Write a lightweight image gallery for browser inspection."""
    curve_blocks = []
    for case in cases:
        png = Path("plots") / "curves" / f"{case.name}.png"
        if (outdir / png).exists():
            curve_blocks.append(
                f"""
                <figure>
                  <a href="{png.as_posix()}"><img src="{png.as_posix()}" alt="{case.name}"></a>
                  <figcaption>{case.name}</figcaption>
                </figure>
                """
            )
    summary_blocks = []
    for png in sorted((outdir / "plots" / "summary").glob("*.png")):
        rel = png.relative_to(outdir)
        summary_blocks.append(
            f"""
            <figure>
              <a href="{rel.as_posix()}"><img src="{rel.as_posix()}" alt="{png.stem}"></a>
              <figcaption>{png.stem}</figcaption>
            </figure>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Continuous Skeletonization Gallery</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0.6rem 0; }}
    p {{ max-width: 920px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; border: 1px solid #ddd; padding: 10px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; font-size: 0.95rem; color: #333; }}
    code {{ background: #f4f4f4; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Continuous Skeletonization Gallery</h1>
  <p>
    Adaptive pivots are continuous optimizer outputs. Chebyshev/Lobatto samples
    are initializer samples for continuous maximization, not candidate pivot grids.
    See <a href="report.md">report.md</a>, <a href="summary.csv">summary.csv</a>,
    and <a href="all_error_curves.csv">all_error_curves.csv</a>.
  </p>
  <h2>Per-Function Error Curves</h2>
  <div class="grid">
    {''.join(curve_blocks)}
  </div>
  <h2>Summary Plots</h2>
  <div class="grid">
    {''.join(summary_blocks)}
  </div>
</body>
</html>
"""
    (outdir / "index.html").write_text(html)


def select_cases(cases: list[FunctionCase], requested: str | None, smoke: bool) -> list[FunctionCase]:
    if requested:
        wanted = {name.strip() for name in requested.split(",") if name.strip()}
        selected = [case for case in cases if case.name in wanted]
        missing = sorted(wanted - {case.name for case in selected})
        if missing:
            raise ValueError(f"unknown function names: {', '.join(missing)}")
        return selected
    if smoke:
        smoke_names = {"sym_abs", "runge", "rank_3"}
        return [case for case in cases if case.name in smoke_names]
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/continuous_skeletonization"))
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--rplu-trials", type=int, default=20)
    parser.add_argument("--uniform-random-trials", type=int, default=20)
    parser.add_argument("--uniform-random-diagonal", action="store_true")
    parser.add_argument("--validation-grid", type=int, default=400)
    parser.add_argument("--validation-random", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20240524)
    parser.add_argument("--functions", type=str, default=None, help="comma-separated subset of function names")
    parser.add_argument("--initializer-n", type=int, default=81)
    parser.add_argument("--top-seeds", type=int, default=12)
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--row-quad-n", type=int, default=161)
    parser.add_argument("--rplu-quad-n", type=int, default=121)
    parser.add_argument("--cond-threshold", type=float, default=1e12)
    parser.add_argument("--svd-rcond", type=float, default=1e-13)
    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument("--stop-factor", type=float, default=1e-14)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="small correctness run")
    args = parser.parse_args()
    if args.smoke:
        args.nmax = min(args.nmax, 5)
        args.rplu_trials = min(args.rplu_trials, 2)
        args.uniform_random_trials = min(args.uniform_random_trials, 2)
        args.validation_grid = min(args.validation_grid, 32)
        args.validation_random = min(args.validation_random, 1024)
        args.initializer_n = min(args.initializer_n, 21)
        args.top_seeds = min(args.top_seeds, 5)
        args.optimizer_maxiter = min(args.optimizer_maxiter, 35)
        args.row_quad_n = min(args.row_quad_n, 41)
        args.rplu_quad_n = min(args.rplu_quad_n, 41)
        args.batch_size = min(args.batch_size, 4096)
    if args.nmax < 1:
        raise ValueError("--nmax must be positive")
    return args


def run_study(args: argparse.Namespace) -> None:
    start_time = time.time()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(make_cases(), args.functions, args.smoke)
    validation = build_validation_set(args.validation_grid, args.validation_random, args.seed)
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, object]] = []

    config_text = "\n".join(f"{key}={value}" for key, value in sorted(vars(args).items()))
    (outdir / "run_config.txt").write_text(config_text + "\n")

    for case_index, case in enumerate(cases, start=1):
        case_start = time.time()
        print(f"[{case_index}/{len(cases)}] {case.name}", flush=True)
        f_norm = validation_norm(case, validation)
        all_rows.extend(
            run_fixed_method(
                case,
                "fixed_equispaced",
                args.nmax,
                validation,
                f_norm,
                args.cond_threshold,
                args.svd_rcond,
                args.batch_size,
            )
        )
        all_rows.extend(
            run_fixed_method(
                case,
                "fixed_cheb_lobatto",
                args.nmax,
                validation,
                f_norm,
                args.cond_threshold,
                args.svd_rcond,
                args.batch_size,
            )
        )
        for trial in range(args.uniform_random_trials):
            all_rows.extend(
                run_uniform_random_support(
                    case,
                    trial,
                    args.nmax,
                    validation,
                    f_norm,
                    args.cond_threshold,
                    args.svd_rcond,
                    args.batch_size,
                    rng,
                    args.uniform_random_diagonal,
                )
            )
        all_rows.extend(
            run_cplu(
                case,
                args.nmax,
                validation,
                f_norm,
                args.cond_threshold,
                args.svd_rcond,
                args.batch_size,
                args.initializer_n,
                args.top_seeds,
                args.optimizer_maxiter,
                args.stop_factor,
            )
        )
        all_rows.extend(
            run_c2plu(
                case,
                args.nmax,
                validation,
                f_norm,
                args.cond_threshold,
                args.svd_rcond,
                args.batch_size,
                args.initializer_n,
                args.top_seeds,
                args.optimizer_maxiter,
                args.row_quad_n,
                args.stop_factor,
            )
        )
        for trial in range(args.rplu_trials):
            all_rows.extend(
                run_rplu_trial(
                    case,
                    trial,
                    args.nmax,
                    validation,
                    f_norm,
                    args.cond_threshold,
                    args.svd_rcond,
                    args.batch_size,
                    args.rplu_quad_n,
                    rng,
                    args.stop_factor,
                )
            )
        write_csv(outdir / "all_error_curves.csv", all_rows, ALL_ERROR_COLUMNS)
        print(f"  done in {time.time() - case_start:.1f}s", flush=True)

    summary_rows = make_summary_rows(all_rows, cases)
    comparison_rank = min(80, args.nmax)
    snapshot_ranks = tuple(rank for rank in (20, 40, 80, 120) if rank <= args.nmax)
    bar_ranks = tuple(rank for rank in (40, 80, 120) if rank <= args.nmax)
    grid_rows = make_grid_comparison_rows(all_rows, cases, rank=comparison_rank)
    adaptive_rows = make_adaptive_vs_fixed_rows(all_rows, cases, ranks=snapshot_ranks)

    write_csv(outdir / "all_error_curves.csv", all_rows, ALL_ERROR_COLUMNS)
    write_csv(outdir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(
        outdir / "grid_support_comparison.csv",
        grid_rows,
        [
            "function",
            "comparison_rank",
            "fixed_equi_error",
            "fixed_cheb_error",
            "cheb_over_equi",
            "winner",
            "comments",
        ],
    )
    write_csv(
        outdir / "adaptive_vs_fixed.csv",
        adaptive_rows,
        [
            "function",
            "category",
            "N",
            "fixed_equispaced",
            "fixed_cheb_lobatto",
            "fixed_uniform_random_median",
            "continuous_cplu",
            "continuous_c2plu",
            "continuous_rplu_median",
            "cplu_over_equi",
            "cplu_over_cheb",
            "cplu_over_uniform_random",
            "c2plu_over_cplu",
            "rplu_over_cplu",
        ],
    )

    if not args.skip_plots:
        plot_function_curves(all_rows, cases, outdir)
        plot_summary_scatter(all_rows, cases, outdir, rank=comparison_rank)
        if bar_ranks:
            plot_ratio_bars(all_rows, cases, outdir, ranks=bar_ranks)

    elapsed = time.time() - start_time
    generate_report(outdir, all_rows, summary_rows, grid_rows, adaptive_rows, cases, args, elapsed)
    generate_html_index(outdir, cases)
    print(f"wrote outputs to {outdir}", flush=True)


if __name__ == "__main__":
    run_study(parse_args())
