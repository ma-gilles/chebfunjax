"""Location and stability of periodic orbits in the Van der Pol equation.

Solves the Van der Pol oscillator u'' - mu*(1-u^2)*u' + u = 0 for various
mu values, finding the periodic orbit (limit cycle) using scipy.
Translated from temp/VdPOrbit.m (original: ode-nonlin/VdPOrbit).

Original: https://www.chebfun.org/examples/ode-nonlin/VdPOrbit.html
Authors: Toby Driscoll and Hrothgar, April 2014
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def vdp_rhs(t, y, mu):
    """Van der Pol ODE: [u, u'] -> [u', mu*(1-u^2)*u' - u]."""
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]


def find_period(mu, t_end=100, n_cycles_skip=10):
    """Find the period of the Van der Pol limit cycle by Poincare section."""
    # Integrate for a long time to get onto the limit cycle
    sol = solve_ivp(vdp_rhs, [0, t_end], [1.0, 0.0], args=(mu,),
                    max_step=0.01, dense_output=True)

    # Find crossings of u'=0 with u>0 after transient
    t_check = sol.t[sol.t > t_end * 0.5]
    y_check = sol.sol(t_check)

    # Find sign changes of u'
    crossings = []
    for i in range(len(t_check)-1):
        if y_check[1, i] > 0 and y_check[1, i+1] <= 0 and y_check[0, i] > 0:
            # Refine with Brentq
            def f_cross(t):
                return sol.sol(t)[1]
            try:
                t_c = brentq(f_cross, t_check[i], t_check[i+1])
                crossings.append(t_c)
            except ValueError:
                pass

    if len(crossings) >= 2:
        periods = np.diff(crossings)
        return np.mean(periods[-3:])  # average last few
    return None


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Attraction to limit cycle (mu=1) ---
    mu = 1.0
    colors_ic = ['blue', 'green', 'orange', 'purple']
    initial_conds = [(0.1, 0), (3.0, 0), (0.5, 2.0), (2.5, -1.5)]

    for (u0, v0), col in zip(initial_conds, colors_ic):
        sol = solve_ivp(vdp_rhs, [0, 25], [u0, v0], args=(mu,),
                        max_step=0.05, dense_output=True)
        t_plot = np.linspace(0, 25, 1000)
        y_plot = sol.sol(t_plot)
        axes[0].plot(y_plot[0], y_plot[1], '-', color=col, linewidth=1.0,
                     alpha=0.7)
        axes[0].plot(u0, v0, 'o', color=col, markersize=6)

    axes[0].set_xlabel('u'); axes[0].set_ylabel("u'")
    axes[0].set_title(f'Van der Pol μ=1\nattraction to limit cycle', fontsize=10)
    axes[0].grid(True, alpha=0.3); axes[0].set_aspect('equal')
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-5, 5)

    # --- Panel 2: Limit cycles for different mu ---
    mu_vals = [0.5, 1, 2, 5]
    colors2 = ['blue', 'green', 'red', 'purple']
    periods = []

    for mu_v, col in zip(mu_vals, colors2):
        # Integrate long enough to converge to limit cycle
        t_end = 200 if mu_v <= 2 else 500
        sol = solve_ivp(vdp_rhs, [0, t_end], [1.0, 0.0], args=(mu_v,),
                        max_step=min(0.02, 0.1/mu_v), dense_output=True)
        t_cycle = np.linspace(t_end * 0.85, t_end, 3000)
        y_cycle = sol.sol(t_cycle)
        axes[1].plot(y_cycle[0], y_cycle[1], '-', color=col, linewidth=2,
                     label=f'μ={mu_v}')

        # Estimate period
        T = find_period(mu_v, t_end=t_end)
        if T is not None:
            periods.append((mu_v, T))
            print(f"  μ={mu_v:.1f}: period ≈ {T:.4f}")
        else:
            print(f"  μ={mu_v:.1f}: period not found")

    axes[1].set_xlabel('u'); axes[1].set_ylabel("u'")
    axes[1].set_title('Limit cycles for\nvarious μ', fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Period vs mu ---
    mu_range = [0.1, 0.5, 1, 2, 5, 10]
    found_periods = []
    for mu_v in mu_range:
        t_end = max(200, 50 * mu_v)
        T = find_period(mu_v, t_end=t_end)
        found_periods.append(T if T is not None else np.nan)
        if T:
            print(f"  μ={mu_v}: T={T:.4f}")

    # Theoretical: for mu->0, T->2*pi; for large mu, T~(3-2*log2)*mu
    mu_theory = np.array(mu_range, dtype=float)
    T_theory_small = 2 * np.pi * np.ones_like(mu_theory)
    T_large = (3 - 2*np.log(2)) * mu_theory

    axes[2].semilogx(mu_range, found_periods, 'b.-', markersize=10,
                      linewidth=2, label='Numerical period')
    axes[2].semilogx(mu_range, T_theory_small, 'r--', linewidth=1.5,
                      alpha=0.6, label='2π (μ→0 limit)')
    axes[2].semilogx([m for m in mu_range if m >= 1],
                      [(3-2*np.log(2))*m for m in mu_range if m >= 1],
                      'g--', linewidth=1.5, alpha=0.6, label='(3-2ln2)μ')
    axes[2].set_xlabel('μ'); axes[2].set_ylabel('Period T')
    axes[2].set_title('Period of Van der Pol\nlimit cycle vs μ', fontsize=10)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    # Assert mu=1 is close to known period (~6.6...)
    if found_periods[2] and not np.isnan(found_periods[2]):
        assert abs(found_periods[2] - 6.663) < 0.1, \
            f"mu=1 period {found_periods[2]:.4f} not close to 6.663"
        print(f"Assertion passed: μ=1 period = {found_periods[2]:.4f} ≈ 6.663")

    fig.suptitle('Van der Pol Oscillator: Periodic Orbits', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'vdp_orbit.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("vdp_orbit: done")
    return True


if __name__ == "__main__":
    run()
