"""Native tests/misc/test_fov.m predicates1–5 (Chebfun7574c77).

Use continuous Chebfun boundary, function norms and extrema. MATLAB
array-valued lineSegs maps to a Python list of Chebfun columns.
"""

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.fov import fov

jax.config.update("jax_enable_x64", True)
EPS = float(np.finfo(np.float64).eps)


class TestFov:
    def test_all_matlab_assertions(self):
        F = fov(np.array([[1, 2], [3, 2j]]))
        err = abs(float(F.real().max()[1]) - 3.049509756796393)
        tol = 1e2 * F.vscale * EPS
        assert err < tol  # pass(1)
        F = fov(np.array([[7.0]]))
        err = float((F - 7).abs().norm(2))
        tol = 10 * F.vscale * EPS
        assert err < tol  # pass(2)
        F = fov(np.diag(np.array([-1, 1, 1j])))
        err = abs(complex(F.mean()) - 0.25j)
        tol = 10 * F.vscale * EPS
        assert err < tol  # pass(3)
        F, lineSegs, theta = fov(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]]), line_segments=True)
        err = np.linalg.norm(theta - np.pi / 3 * np.array([1, 5]), ord=np.inf)
        assert err < tol  # pass(4)
        tmp = np.column_stack(
            [np.asarray(col(jnp.array([-1.0, 1.0]))).reshape(-1) for col in lineSegs]
        )
        err = np.linalg.norm(tmp.real - np.array([[1, 0.25], [0.25, 1]]), ord=np.inf)
        assert err < tol  # pass(5)

    def test_hermitian_is_real_segment(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        from chebfunjax.utils.fov import fov as sampled_fov

        _, F = sampled_fov(A)
        lam = np.linalg.eigvalsh(A)
        assert float(np.max(np.abs(np.asarray(F).imag))) < 1e-12
        assert abs(float(np.max(np.asarray(F).real)) - lam[-1]) < 1e-10
        assert abs(float(np.min(np.asarray(F).real)) - lam[0]) < 1e-10
