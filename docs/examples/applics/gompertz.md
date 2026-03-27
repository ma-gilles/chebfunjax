# Exponential, Logistic, and Gompertz Growth

**Source:** `applics/Gompertz.m` — Toby Driscoll, June 2015
**Python:** `examples/applics/gompertz.py`
**Original MATLAB:** https://www.chebfun.org/examples/applics/Gompertz.html

## Overview

Three classical population growth models are compared on the time interval `[0, 25]`
with initial population `P₀ = 0.2`, growth rate `r = 0.5`, and carrying capacity `K = 6.0`.

## Models

**Exponential growth** — no resource limitation:
```
P' = r P
P(t) = P₀ e^{rt}
```

**Logistic growth** — bounded by carrying capacity:
```
P' = r P (K - P) / K
```
The solution approaches K from below, rapidly near P=K/2.

**Gompertz growth** — bounded, with slower approach to carrying capacity:
```
P' = r P log(K/P) / log(K/P₀)
```
Widely used for tumor growth modeling. Converges to K more slowly than logistic.

## Key results

At `t = 25`: logistic P ≈ 5.999, Gompertz P ≈ 5.504.
At `t = 10`: Gompertz < Logistic < Exponential, confirming that Gompertz
grows slowest near the carrying capacity.

## Code excerpt

```python
from scipy.integrate import solve_ivp
import numpy as np

P0, r, K, T = 0.2, 0.5, 6.0, 25.0
log_factor = np.log(K / P0)

# Gompertz: P' = r*P*log(K/P)/log(K/P0)
sol_gom = solve_ivp(
    lambda t, P: [r * P[0] * np.log(K / max(P[0], 1e-15)) / log_factor],
    [0, T], [P0], t_eval=np.linspace(0, T, 500)
)
```

## Plots

![Gompertz growth](../../images/applics/gompertz.png)

Left: population trajectories for all three models.
Right: per-capita growth rates `P'/P` as a function of `P`.
