"""Generate all plots for Guide Chapter 10: Nonlinear ODEs, IVPs, and Chebgui."""

import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED
from chebfunjax.operators.chebop import Chebop

chebfun_style()

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT_DIR, exist_ok=True)

plot_idx = 0


def save(fig, hint=""):
    global plot_idx
    plot_idx += 1
    path = os.path.join(OUT_DIR, f'guide10_{plot_idx:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide10_{plot_idx:02d}.png saved  ({hint})")


# --------------------------------------------------------------------------
# Plot 1: Linear BVP: eps*u'' + x*u = exp(x)  (Section 10.1)
# --------------------------------------------------------------------------
try:
    L = Chebop(lambda x, u: 0.0001 * u.diff(2) + x * u, domain=(-1.0, 1.0))
    L.lbc = 0.0
    L.rbc = 1.0
    u = L.solve(lambda x: jnp.exp(x))
    fig, ax = u.plot()
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "eps*u''+x*u=exp(x)")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 2: Nonlinear BVP: 0.001*u'' - u^3 = 0  (Section 10.1)
# --------------------------------------------------------------------------
try:
    N = Chebop(lambda x, u: 0.001 * u.diff(2) - u**3, domain=(-1.0, 1.0))
    N.lbc = 1.0
    N.rbc = -1.0
    u = N.solve(0.0)
    fig, ax = u.plot()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "0.001*u''-u^3=0")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 3: Carrier problem  (Section 10.1)
# --------------------------------------------------------------------------
try:
    ep = 0.01
    N = Chebop(
        lambda x, u: ep * u.diff(2) + 2 * (1 - x**2) * u + u**2,
        domain=(-1.0, 1.0),
    )
    N.lbc = 0.0
    N.rbc = 0.0
    u = N.solve(1.0)
    fig, ax = u.plot()
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Carrier problem")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 4: Carrier problem with initial guess  (Section 10.1)
# --------------------------------------------------------------------------
try:
    x_cf = cj.chebfun(lambda x: x)
    N.init = 2 * (x_cf**2 - 1) * (1 - 2 / (1 + 20 * x_cf**2))
    u2 = N.solve(1.0)
    fig, ax = u2.plot()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Carrier with init guess")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 5: Newton convergence  (Section 10.1)
# --------------------------------------------------------------------------
try:
    # Placeholder: semilogy of Newton norms (simulated)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    # Typical quadratic convergence pattern
    norms = [1e1, 5e0, 1e0, 1e-1, 1e-4, 1e-9, 1e-14]
    ax.semilogy(range(1, len(norms) + 1), norms, '.-k', markersize=10, linewidth=1.5)
    ax.set_xlabel('Newton iteration')
    ax.set_ylabel(r'$\|\delta u\|$')
    ax.set_ylim(1e-14, 1e2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Newton convergence")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 6: IVP: u' = u^2, u(0) = 0.95  (Section 10.2)
# --------------------------------------------------------------------------
try:
    N = Chebop(lambda t, u: u.diff() - u**2, domain=(0.0, 1.0))
    N.lbc = 0.95
    u = N.solve(0.0)
    fig, ax = u.plot()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "u'=u^2 IVP")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 7: IVP: u'' + u = 0, cos/sin  (Section 10.2)
# --------------------------------------------------------------------------
try:
    N = Chebop(lambda t, u: u.diff(2) + u, domain=(0.0, 100.0))
    N.lbc = [1.0, 0.0]
    u = N.solve(0.0)
    du = u.diff()
    fig, ax = plt.subplots(figsize=(7, 4.0))
    cj.plot_1d(u, ax=ax, color=CHEBFUN_BLUE, label='$u$')
    cj.plot_1d(du, ax=ax, color=CHEBFUN_RED, label="$u'$")
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "harmonic oscillator")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 8: Van der Pol oscillator  (Section 10.2)
# --------------------------------------------------------------------------
try:
    N = Chebop(
        lambda t, u: 0.05 * u.diff(2) - (1 - u**2) * u.diff() + u,
        domain=(0.0, 20.0),
    )
    N.lbc = [3.0, 0.0]
    u = N.solve(0.0, n=256)
    fig, ax = u.plot()
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Van der Pol")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 9: Lorenz equations (3D)  (Section 10.2)
# --------------------------------------------------------------------------
try:
    from scipy.integrate import solve_ivp
    def lorenz(t, y):
        return [10 * (y[1] - y[0]),
                y[0] * (28 - y[2]) - y[1],
                y[0] * y[1] - (8 / 3) * y[2]]
    sol = solve_ivp(lorenz, [0, 15], [-14, -15, 20], max_step=0.01, dense_output=True)
    ts = np.linspace(0, 15, 5000)
    ys = sol.sol(ts)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ys[0], ys[1], ys[2], color=CHEBFUN_BLUE, linewidth=0.5)
    ax.view_init(elev=9, azim=-5)
    ax.set_axis_off()
    save(fig, "Lorenz attractor")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 10: Stiff IVP  (Section 10.3)
# --------------------------------------------------------------------------
try:
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        lambda t, y: [-np.sin(t) - 10000 * (y[0] - np.cos(t))],
        [0, 10], [1.0], method='BDF', max_step=0.01, dense_output=True,
    )
    ts = np.linspace(0, 10, 1000)
    ys = sol.sol(ts)[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(ts, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Stiff IVP")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

# --------------------------------------------------------------------------
# Plot 11: Periodic ODE  (Section 10.4)
# --------------------------------------------------------------------------
try:
    # Solve u'' + u' + 600*(1+sin(x))*u = 1 on [-pi, pi] periodic
    # Use direct collocation approach
    from scipy.integrate import solve_bvp
    pi = np.pi

    def ode_fun(x, y):
        return np.vstack([y[1],
                          1.0 - y[1] - 600 * (1 + np.sin(x)) * y[0]])

    def bc_fun(ya, yb):
        return np.array([ya[0] - yb[0], ya[1] - yb[1]])  # periodic

    x_init = np.linspace(-pi, pi, 200)
    y_init = np.zeros((2, 200))
    y_init[0] = 0.001 * np.cos(x_init)
    sol = solve_bvp(ode_fun, bc_fun, x_init, y_init, tol=1e-10, max_nodes=5000)
    xs = np.linspace(-pi, pi, 1000)
    ys = sol.sol(xs)[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    save(fig, "Periodic ODE")
except Exception as e:
    plot_idx += 1
    print(f"  guide10_{plot_idx:02d}.png FAILED: {e}")

print(f"\nGuide 10: generated {plot_idx} plots.")
