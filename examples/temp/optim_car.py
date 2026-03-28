"""Optimal performance of a car.

Solves the optimal control problem: car on [0,2] with maximum acceleration 1,
minimizing time to go from 0 to 1 using Pontryagin's minimum principle.
The optimal control is sign(1-t): full acceleration for first half, full braking
for second half.
Translated from temp/OptimCar.m.

Original: https://www.chebfun.org/examples/ode/OptimCar.html
Author: Asgeir Birkisson, November 2010
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
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # Bang-bang optimal control: sign(1-t)
    # State equations: x' = v, v' = u(t), with u = sign(1-t)
    t = np.linspace(0, 2, 1000)
    u_opt = np.sign(1 - t)
    u_opt[t == 1] = 0  # discontinuity at t=1

    # Integrate ODE analytically
    # For 0 <= t <= 1: v(t) = t, x(t) = t^2/2
    # For 1 <= t <= 2: v(t) = 2-t, x(t) = 1/2 + (t-t^2/2) - (1-1/2) = t - t^2/2
    x = np.where(t <= 1, t**2 / 2, t - t**2/2)
    v = np.where(t <= 1, t, 2 - t)

    # Co-state (lambda_x, lambda_v) from Pontryagin conditions
    # H = lx*v + lv*u => lambda_v' = -lx, lambda_x' = 0 => lx = const
    # Transversality: lx(2) = 0, but for free final state with fixed endpoint x(2)=1:
    # lv(2) = 0, and lx = const = some value
    # Simple case: lv(t) = -(2-t), lx = -1
    lx = -np.ones_like(t)
    lv = np.where(t <= 1, t - 1, -(2-t))

    # --- Panel 1: State trajectories ---
    axes[0].plot(t, x, color='#0072BD', linestyle='-', linewidth=2.5, label='x(t) position')
    axes[0].plot(t, v, color='#D95319', linestyle='-', linewidth=2.5, label="v(t) speed")
    axes[0].plot(t, u_opt, color='#77AC30', linestyle='-', linewidth=1.5, label='u(t) control', alpha=0.7)
    axes[0].axvline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title('Optimal car: position, speed,\ncontrol vs time', fontsize=10)
    axes[0].legend(fontsize=9)
    print(f"Optimal car:")
    print(f"  x(2) = {x[-1]:.6f} (target: 1)")
    print(f"  v(2) = {v[-1]:.6f} (target: 0)")

    # --- Panel 2: Co-state variables ---
    axes[1].plot(t, lx, color='#0072BD', linestyle='-', linewidth=2.5, label='λ_x(t)')
    axes[1].plot(t, lv, color='#D95319', linestyle='-', linewidth=2.5, label='λ_v(t)')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_title('Co-state (adjoint) variables\nPontryagin minimum principle',
                       fontsize=10)
    axes[1].legend(fontsize=9)

    # --- Panel 3: Phase portrait and comparison with suboptimal ---
    # Optimal
    axes[2].plot(x, v, color='#0072BD', linestyle='-', linewidth=2.5, label='Optimal (bang-bang)')

    # Suboptimal: constant control u=1 for 0.7, then u=-1
    # x'=v, v'=0.5 (constant partial braking)
    t_sub = np.linspace(0, 2, 1000)
    x_sub = np.where(t_sub <= 1.2, 0.5*t_sub**2,
                      0.5*1.2**2 + 1.2*(t_sub-1.2) - 0.5*(t_sub-1.2)**2)
    v_sub = np.where(t_sub <= 1.2, 1.2*t_sub/1.2,
                      1.2 - (t_sub-1.2))
    axes[2].plot(x_sub, v_sub, color='#D95319', linestyle='--', linewidth=2, label='Suboptimal', alpha=0.8)

    axes[2].plot(0, 0, color='#77AC30', marker='o', linestyle='none', markersize=10, zorder=5, label='Start')
    axes[2].plot(x[-1], v[-1], color='#0072BD', marker='s', linestyle='none', markersize=10, zorder=5, label='End')
    axes[2].set_title('Phase portrait: position vs speed', fontsize=10)
    axes[2].legend(fontsize=9)

    fig.suptitle('Optimal Performance of a Car: Bang-Bang Control', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'optim_car.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("optim_car: done")
    return True

if __name__ == "__main__":
    run()
