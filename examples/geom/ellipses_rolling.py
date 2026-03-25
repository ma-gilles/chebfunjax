"""An ellipse rolling around another ellipse.

Computes the trajectory of the midpoint of a small ellipse rolling around
a larger ellipse without slipping. Translated from geom/Ellipses.m.

Original: https://www.chebfun.org/examples/geom/Ellipses.html
Author: Nick Trefethen, October 2011
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import odeint, solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Ellipse 1 (big): 3x1, semi-major L1=3
    # Ellipse 2 (small): 2x1, semi-major L2=2
    L1, L2 = 3.0, 2.0
    t_max = 7.5

    def theta1(z1):
        return np.arctan2(np.imag(z1), np.real(z1) / L1)

    def theta2(z2):
        return np.arctan2(np.imag(z2), np.real(z2) / L2)

    def ode1(t, y):
        z1 = y[0] + 1j * y[1]
        th = theta1(z1)
        dz1 = (-L1 * np.sin(th) + 1j * np.cos(th)) / np.sqrt(L1**2 * np.sin(th)**2 + np.cos(th)**2)
        return [np.real(dz1), np.imag(dz1)]

    def ode2(t, y):
        z2 = y[0] + 1j * y[1]
        th = theta2(z2)
        dz2 = (L2 * np.sin(th) - 1j * np.cos(th)) / np.sqrt(L2**2 * np.sin(th)**2 + np.cos(th)**2)
        return [np.real(dz2), np.imag(dz2)]

    # Solve ODEs
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)

    # z1 starts at right tip of big ellipse (L1/2, 0)
    sol1 = solve_ivp(ode1, t_span, [L1/2, 0], t_eval=t_eval,
                     method='RK45', rtol=1e-8, atol=1e-10)
    # z2 starts at left tip of small ellipse (-L2/2, 0)
    sol2 = solve_ivp(ode2, t_span, [-L2/2, 0], t_eval=t_eval,
                     method='RK45', rtol=1e-8, atol=1e-10)

    z1_traj = sol1.y[0] + 1j * sol1.y[1]
    z2_traj = sol2.y[0] + 1j * sol2.y[1]

    # Midpoint trajectory: w = z1 - z2 * (dz1/dz2)
    # Approximate derivative ratio
    dz1_dt = np.gradient(z1_traj, t_eval)
    dz2_dt = np.gradient(z2_traj, t_eval)
    dz1_dz2 = dz1_dt / (dz2_dt + 1e-15)
    w_traj = z1_traj - z2_traj * dz1_dz2

    # Trajectory arc length up to ~t=final (when imag(w) crosses 0 again)
    # Find approximate tfinal
    imag_w = np.imag(w_traj)
    sign_change = np.where(np.diff(np.sign(imag_w[500:])) != 0)[0]
    if len(sign_change) > 0:
        t_final_idx = 500 + sign_change[0]
        t_final = t_eval[t_final_idx]
    else:
        t_final = t_max
        t_final_idx = len(t_eval) - 1

    # Arc length of w up to t_final
    dw = np.diff(w_traj[:t_final_idx+1])
    arc_length = np.sum(np.abs(dw))
    print(f"Trajectory length ≈ {arc_length:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot big ellipse
    theta_e = np.linspace(0, 2 * np.pi, 500)
    big_ell_x = L1/2 * np.cos(theta_e)
    big_ell_y = 0.5 * np.sin(theta_e)

    axes[0].fill(big_ell_x, big_ell_y, color='lightblue', alpha=0.5)
    axes[0].plot(big_ell_x, big_ell_y, 'b-', linewidth=1.5)

    # Plot trajectory of midpoint w
    axes[0].plot(np.real(w_traj), np.imag(w_traj), 'k-', linewidth=2)

    # Plot some positions of the small ellipse
    for t_step in np.linspace(0, t_max * 0.8, 6):
        idx = np.argmin(np.abs(t_eval - t_step))
        z1_pos = z1_traj[idx]
        z2_pos = z2_traj[idx]
        w_pos = w_traj[idx]
        dz1 = dz1_dt[idx]
        dz2 = dz2_dt[idx]

        # Position of small ellipse (approximate)
        # Draw small ellipse centered at w_pos
        small_ell = (L2/2 * np.cos(theta_e) * np.exp(1j * np.angle(z1_pos))
                     + 1j * 0.5 * np.sin(theta_e) * np.exp(1j * np.angle(z1_pos)))
        axes[0].plot(np.real(small_ell + w_pos), np.imag(small_ell + w_pos),
                     'r-', linewidth=1, alpha=0.6)
        axes[0].plot(np.real(w_pos), np.imag(w_pos), '.k', markersize=6)

    axes[0].set_aspect('equal')
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_title('Ellipse rolling around ellipse', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Contact point trajectories
    axes[1].plot(np.real(z1_traj), np.imag(z1_traj), 'b-', linewidth=2, label='Contact on big')
    axes[1].plot(np.real(z2_traj), np.imag(z2_traj), 'r-', linewidth=2, label='Contact on small')
    axes[1].plot(np.real(w_traj), np.imag(w_traj), 'k-', linewidth=2, label='Midpoint w(t)')
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Contact points and midpoint trajectory\nArc length ≈ {arc_length:.3f}',
                      fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Ellipse Rolling around Ellipse', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ellipses_rolling.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("ellipses_rolling: done")
    return True


if __name__ == "__main__":
    run()
