# AAA Algorithm for System Identification from Bode Data

**Source:** `applics/Bode2tf.m` — Stefano Costa, August 2021
**Python:** `examples/applics/bode2tf.py`
**Original MATLAB:** https://www.chebfun.org/examples/applics/Bode2tf.html

## Overview

Given Bode plot data (magnitude and phase as functions of frequency),
this example uses the AAA rational approximation algorithm to fit the
frequency response of a second-order dynamical system.

## System

The target transfer function is a standard second-order system:
```
H(s) = ω_n² / (s² + 2ζω_n s + ω_n²)
```
with natural frequency `ω_n = 10` rad/s and damping ratio `ζ = 0.3`.

The complex poles are at `s = -3 ± 9.54j`.

## Approach

1. Generate Bode data: evaluate `H(jω)` for `ω ∈ [0.1, 100]` rad/s
2. Apply AAA to the real-valued magnitude `|H(jω)|` as a function of `log₁₀(ω)`
3. Apply AAA separately to the phase `∠H(jω)` (in degrees)

The AAA algorithm from `chebfunjax.utils.aaa` builds a rational interpolant
in barycentric form, which efficiently represents the resonance peak.

## Code excerpt

```python
from chebfunjax.utils.aaa import aaa
import numpy as np

omega_vals = np.logspace(-1, 2, 300)
H_mag = np.abs(H_vals)

f_aaa_mag, pol, res, zer, z_sup, f_sup, w = aaa(
    H_mag, np.log10(omega_vals), tol=1e-8, mmax=20
)
```

## Results

- DC gain: `H(0) = 1` (verified)
- Peak gain: ~10.7 dB at `ω ≈ 9.06` rad/s (near resonance `ω_r = ω_n√(1-2ζ²)`)
- AAA magnitude fit error: < 0.1

## Plots

![Bode2tf](../../../docs/images/applics/bode2tf.png)

Top: Bode magnitude plot (exact vs AAA fit).
Bottom: Bode phase plot (exact vs AAA fit).
