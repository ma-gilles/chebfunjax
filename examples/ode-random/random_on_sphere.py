"""Random trajectory on a sphere.

Solves the coupled ODE system
  dx/dt = f*y + g*z,   dy/dt = -f*x + h*z,   dz/dt = -g*x - h*y
where f, g, h are random functions, keeping ||u(t)|| = 1 exactly
because the coefficient matrices are skew-symmetric.

Following ode-random/RandomOnASphere.m by Kevin Burrage and Nick Trefethen (May 2017).

Original MATLAB: https://www.chebfun.org/examples/ode-random/RandomOnASphere.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    print("=" * 60)
    print("Random trajectory on a sphere")
    print("=" * 60)

    print("\nSystem: du/dt = f*A*u + g*B*u + h*C*u")
    print("A, B, C are skew-symmetric → ||u||=1 preserved exactly")

    # Skew-symmetric matrices (generators of SO(3) rotations)
    A = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float)
    B = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    C = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    domain = [0.0, 100.0]
    t_eval = np.linspace(0, 100, 2000)

    def solve_sphere(lam, seed_base=0):
        """Solve sphere ODE with random coefficients."""
        f_fn = cj.randnfun(lam, domain=domain, seed=seed_base, big=False)
        g_fn = cj.randnfun(lam, domain=domain, seed=seed_base + 1, big=False)
        h_fn = cj.randnfun(lam, domain=domain, seed=seed_base + 2, big=False)

        # Sample at evaluation points
        t_grid = t_eval
        f_vals = np.array([float(f_fn(np.array(ti))) for ti in t_grid])
        g_vals = np.array([float(g_fn(np.array(ti))) for ti in t_grid])
        h_vals = np.array([float(h_fn(np.array(ti))) for ti in t_grid])

        def rhs(t, u):
            fi = np.interp(t, t_grid, f_vals)
            gi = np.interp(t, t_grid, g_vals)
            hi = np.interp(t, t_grid, h_vals)
            M = fi * A + gi * B + hi * C
            return M @ u

        # Random initial condition on unit sphere
        rng = np.random.default_rng(seed_base + 10)
        u0 = rng.standard_normal(3)
        u0 /= np.linalg.norm(u0)

        sol = solve_ivp(rhs, [0, 100], u0, t_eval=t_eval,
                        method='RK45', rtol=1e-7, atol=1e-9,
                        max_step=0.1)
        return sol.y  # shape (3, N)

    # Lambda = 0.5
    print("\nSolving with lambda=0.5...")
    traj1 = solve_sphere(lam=0.5, seed_base=0)
    norms1 = np.linalg.norm(traj1, axis=0)
    print(f"  Max deviation from unit sphere: {np.max(np.abs(norms1 - 1)):.2e}")
    print(f"  Final position: ({traj1[0,-1]:.3f}, {traj1[1,-1]:.3f}, {traj1[2,-1]:.3f})")

    # Lambda = 0.125 (finer noise)
    print("\nSolving with lambda=0.125 (lambda/4)...")
    traj2 = solve_sphere(lam=0.125, seed_base=3)
    norms2 = np.linalg.norm(traj2, axis=0)
    print(f"  Max deviation from unit sphere: {np.max(np.abs(norms2 - 1)):.2e}")

    # Check energy conservation
    assert np.max(np.abs(norms1 - 1)) < 0.01, \
        f"Energy not conserved: max deviation = {np.max(np.abs(norms1 - 1)):.2e}"
    print("\n  PASS: unit norm preserved throughout trajectory")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory for lambda=0.5
    ax1 = fig.add_subplot(121, projection='3d')
    brown = (0.5, 0.25, 0.12)
    ax1.plot(traj1[0], traj1[1], traj1[2], color=brown, linewidth=1.2, alpha=0.8)
    # Draw wireframe sphere for reference
    u_sph = np.linspace(0, 2 * np.pi, 30)
    v_sph = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u_sph), np.sin(v_sph))
    ys = np.outer(np.sin(u_sph), np.sin(v_sph))
    zs = np.outer(np.ones(30), np.cos(v_sph))
    ax1.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.2, linewidth=0.5)
    ax1.set_title("Random walk on sphere (λ=0.5)", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # 3D trajectory for lambda=0.125
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(traj2[0], traj2[1], traj2[2], color='steelblue', linewidth=0.8, alpha=0.8)
    ax2.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.2, linewidth=0.5)
    ax2.set_title("Random walk on sphere (λ=0.125)", fontsize=11)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    fig.suptitle("Random trajectories on unit sphere", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random_on_sphere.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True


if __name__ == "__main__":
    run()
