# AAA Algorithm for System Identification from Step Response

**Source:** `applics/Step2tf.m` — Stefano Costa, December 2021
**Python:** `examples/applics/step2tf.py`
**Original MATLAB:** https://www.chebfun.org/examples/applics/Step2tf.html

## Overview

Given step response data, identifies the transfer function using the AAA
rational approximation algorithm applied to Bode magnitude data.

## System

The target transfer function:
```
H(s) = 5(s + 3) / ((s + 1)(s + 2)(s + 4))
```

Properties:
- Poles at `s = -1, -2, -4`
- Zero at `s = -3`
- DC gain: `H(0) = 5·3/(1·2·4) = 1.875`

## Final value theorem

The step response `h(t)` satisfies `h(∞) = H(0) = 1.875`.
This is verified numerically using `scipy.signal.lti.step`.

## AAA identification

AAA is applied to the Bode magnitude `|H(jω)|` as a real-valued function
of `log₁₀(ω)`:

```python
from chebfunjax.utils.aaa import aaa
import numpy as np

omega_vals = np.logspace(-2, 2, 200)
H_mag = np.abs(H_exact(1j * omega_vals))

f_aaa, pol, res, zer, z_sup, f_sup, w = aaa(
    H_mag, np.log10(omega_vals), tol=1e-8, mmax=20
)
```

## Results

- Step response final value: `h(8) ≈ 1.875` (final value theorem verified)
- AAA magnitude fit error: < 0.1

## Plots

![Step2tf](../../../docs/images/applics/step2tf.png)

Left: step response `h(t)` on `[0, 8]`, with final value `H(0) = 1.875`.
Right: Bode magnitude of exact `H(jω)` vs AAA fit.
