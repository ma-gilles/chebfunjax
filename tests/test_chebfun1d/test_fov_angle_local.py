"""fov as a Chebfun, angle/unwrap branch cuts, kink local extrema (Fable 5).

Provenance
----------
MATLAB source : fov.m, @chebfun/angle.m, @chebfun/unwrap.m,
    @chebfun/minandmax.m ('local'), @singfun/restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.fov import fov

jax.config.update("jax_enable_x64", True)


def test_fov_normal_matrix_is_polygon_hull():
    # A normal matrix: the field of values is the convex hull of its
    # eigenvalues; the boundary curve jumps between them.
    A = np.diag([1.0, 1j, -1.0, -1j])
    f, segs, theta = fov(A, line_segments=True)
    z = np.asarray(f(jnp.linspace(0.05, 2 * np.pi - 0.05, 9)))
    assert np.all(np.min(np.abs(z[:, None] - np.diag(A)[None, :]), axis=1) < 1e-10)
    assert len(segs) == len(theta) >= 3


def test_fov_hermitian_boundary_values():
    rs = np.random.RandomState(0)
    B = rs.randn(6, 6)
    A = B + 1j * rs.randn(6, 6)
    f = fov(A)
    z = complex(np.asarray(f(jnp.asarray(0.0))))
    H = (A + A.conj().T) / 2
    assert abs(z.real - np.max(np.linalg.eigvalsh(H))) < 1e-10


def test_angle_has_branch_cut_breakpoint_and_unwraps():
    t = cj.chebfun(lambda s: s, domain=(0.0, 2 * np.pi))
    c = (1j * t).exp()
    a = c.angle()
    assert len(a.funs) == 2
    u = a.unwrap()
    xs = jnp.linspace(0.1, 2 * np.pi - 0.1, 7)
    np.testing.assert_allclose(np.asarray(u(xs)), np.asarray(xs), atol=1e-12)


def test_local_min_at_kink():
    f = cj.chebfun(lambda x: jnp.abs(x) + 0.1 * x ** 2, domain=(-1.0, 0.0, 1.0))
    pos, val = f.min("local")
    pos = np.asarray(pos)
    assert np.any(np.abs(pos) < 1e-12)


def test_singular_restrict_keeps_exponent():
    f = cj.chebfun(lambda x: (7.0 - x) ** -0.5, domain=(-2.0, 7.0), exps=[0, -0.5])
    p = f.funs[-1]
    a, b = p.interval
    r = p.restrict(0.5 * (a + b), b)
    assert tuple(getattr(r.tech, "exponents", (0, 0))) == (0.0, -0.5)
