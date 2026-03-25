"""Illustrating the mathematics of signal processing in Chebfun.

Demonstrates amplitude modulation (AM) and basic signal processing using
Chebfun arithmetic operations.

Credit: Mohsin Javed, August 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/CommunicationSystem.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Signal: low-frequency message
    dom = (0.0, 1.0)
    def message(t): return jnp.sin(2.0 * jnp.pi * 3.0 * t)
    def carrier(t): return jnp.cos(2.0 * jnp.pi * 50.0 * t)

    msg = cj.chebfun(message, domain=dom)
    car = cj.chebfun(carrier, domain=dom)

    # AM modulation: (1 + m(t)) * c(t)
    modulated = (1.0 + msg) * car

    # Envelope = 1 + msg
    envelope = 1.0 + msg

    xx = np.linspace(0.0, 1.0, 1200)
    msg_vals = np.array([float(msg(jnp.array(x))) for x in xx])
    mod_vals = np.array([float(modulated(jnp.array(x))) for x in xx])
    env_vals = np.array([float(envelope(jnp.array(x))) for x in xx])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(xx, msg_vals, 'b', lw=1.8)
    axes[0].set_title('Message signal: sin(6πt)', fontsize=11)
    axes[0].set_xlabel('t')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xx, mod_vals, 'b', lw=0.8)
    axes[1].plot(xx, env_vals, 'r', lw=1.5, label='+envelope')
    axes[1].plot(xx, -env_vals, 'r', lw=1.5)
    axes[1].set_title('AM modulated signal', fontsize=11)
    axes[1].set_xlabel('t')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Demodulation: multiply by carrier and low-pass filter (polyfit)
    demod_raw = modulated * car
    demod_raw_vals = np.array([float(demod_raw(jnp.array(x))) for x in xx])
    # Low-pass: polynomial fit at low degree
    demod_lp = demod_raw.polyfit(20)
    demod_lp_vals = np.array([float(demod_lp(jnp.array(x))) for x in xx])

    axes[2].plot(xx, msg_vals, 'k--', lw=1.5, label='original message')
    axes[2].plot(xx, 2.0 * demod_lp_vals, 'r', lw=1.5, label='recovered (×2)')
    axes[2].set_title('Demodulated signal', fontsize=11)
    axes[2].set_xlabel('t')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'CommunicationSystem.png'), dpi=150)
    plt.close(fig)

    print("CommunicationSystem: done.")
    return True


if __name__ == '__main__':
    run()
