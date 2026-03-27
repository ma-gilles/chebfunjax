# Chapter 8: Chebfun Preferences

*Based on [Chebfun Guide Chapter 8](https://www.chebfun.org/docs/guide/guide08.html)*

## 8.1 Introduction

Chebfunjax's behavior can be customized through a global preferences object. The preferences control aspects of the adaptive construction process such as the tolerance, the maximum polynomial length, and the default domain.

The preferences singleton is accessed via:

```python
from chebfunjax.pref import pref

print(pref)
```

This displays all current settings:

```
ChebPreferences(
    chop_tol=None,
    domain=(-1.0, 1.0),
    eps=2.220446049250313e-16,
    max_length=65537,
    tech='chebtech2',
)
```

To reset all preferences to their factory defaults:

```python
pref.reset()
```

## 8.2 `eps`: Construction Tolerance

The `eps` preference controls the tolerance used during adaptive chebfun construction. By default, it is set to machine epsilon ($\approx 2.22 \times 10^{-16}$):

```python
print(f"eps = {pref.eps}")
# eps = 2.220446049250313e-16
```

The adaptive constructor builds Chebyshev interpolants of increasing degree (17, 33, 65, 129, ...) until the tail of the Chebyshev coefficient series decays below `eps` relative to the function's vertical scale (`vscale`). The "standardChop" algorithm (Aurentz & Trefethen 2017) determines the cutoff.

### Changing the Tolerance

You can reduce accuracy requirements for faster construction or when high precision is unnecessary:

```python
pref.eps = 1e-8

import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(lambda x: jnp.exp(jnp.sin(10 * x)))
print(f"Length at eps=1e-8: {len(f)}")

pref.reset("eps")  # restore default
```

### Context Manager for Temporary Overrides

The preferred way to temporarily change preferences is with the `context` manager, which is thread-safe and automatically restores the previous values:

```python
with pref.context(eps=1e-6):
    f = cj.chebfun(lambda x: jnp.exp(jnp.sin(10 * x)))
    print(f"Length at eps=1e-6: {len(f)}")
    print(f"eps inside context: {pref.eps}")

print(f"eps after context: {pref.eps}")  # back to machine epsilon
```

## 8.3 `max_length`: Maximum Representation Length

The `max_length` preference sets the upper limit on the number of Chebyshev points used in the adaptive construction. The factory default is $2^{16} + 1 = 65537$:

```python
print(f"max_length = {pref.max_length}")
```

If a function cannot be resolved to the requested tolerance within `max_length` points, the constructor issues a warning and returns the best approximation it found.

### Example: Unresolvable Function

The `sign(x)` function is discontinuous and cannot be represented by a single polynomial to machine precision:

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f = cj.chebfun(lambda x: jnp.sign(x))
    if w:
        print(f"Warning: {w[0].message}")
```

### Increasing the Limit

For functions that are smooth but require very high degree, you can increase the limit:

```python
with pref.context(max_length=200000):
    f = cj.chebfun(lambda x: 1.0 / (1.0 + 1e8 * x**2))
    print(f"Length: {len(f)}")
```

### Fixed-Length Construction

You can bypass the adaptive process entirely by specifying a fixed number of points:

```python
f_fixed = cj.chebfun(lambda x: jnp.sign(x), n=65)
print(f"Length: {len(f_fixed)}")
```

## 8.4 `domain`: The Default Domain

By default, chebfuns are constructed on $[-1, 1]$:

```python
print(f"Default domain: {pref.domain}")
```

You can change this globally:

```python
pref.domain = (0.0, 1.0)
# Now all chebfuns without explicit domain use [0, 1]
f = cj.chebfun(lambda x: x**2)
print(f"f is on [{f.domain.a}, {f.domain.b}]")

pref.reset("domain")  # restore to [-1, 1]
```

Or use the context manager:

```python
import math
with pref.context(domain=(0.0, 2 * math.pi)):
    f = cj.chebfun(lambda t: jnp.sin(19 * t))
    print(f"f is on [{f.domain.a}, {f.domain.b}]")
```

In practice, it is usually clearer to specify the domain explicitly in each constructor call rather than relying on a global default:

```python
f = cj.chebfun(lambda t: jnp.sin(19 * t), domain=(0.0, 2 * math.pi))
```

## 8.5 `tech`: Representation Technology

The `tech` preference selects the underlying polynomial technology. The factory default is `"chebtech2"`, which uses Chebyshev points of the second kind (Gauss-Lobatto points):

$$x_k = \cos\!\left(\frac{k\pi}{n}\right), \quad k = 0, 1, \ldots, n.$$

```python
print(f"tech = {pref.tech}")
```

Chebfunjax also supports `"trigtech"` for periodic functions on equispaced grids (see Chapter 11). However, the `tech` preference primarily affects the internal Chebtech2 representation; for trigonometric representations, use the dedicated trigonometric API.

## 8.6 `chop_tol`: Coefficient Chopping Tolerance

The `chop_tol` preference provides a separate tolerance for the coefficient-chopping algorithm that truncates the Chebyshev series. When set to `None` (the factory default), the value of `eps` is used:

```python
print(f"chop_tol = {pref.chop_tol}")  # None
```

Setting `chop_tol` to a value different from `eps` allows independent control of the construction sampling and the final truncation:

```python
with pref.context(chop_tol=1e-12):
    f = cj.chebfun(lambda x: jnp.exp(x))
    print(f"Length with chop_tol=1e-12: {len(f)}")
```

## 8.7 Viewing and Resetting Preferences

### Viewing All Preferences

```python
print(pref)
# or equivalently:
print(pref.to_dict())
```

### Resetting Individual Preferences

```python
pref.max_length = 1000
print(f"max_length = {pref.max_length}")  # 1000

pref.reset("max_length")
print(f"max_length = {pref.max_length}")  # 65537 (factory)
```

### Resetting All Preferences

```python
pref.reset()
```

## 8.8 Thread Safety

The `pref` object uses Python's `contextvars` module, so overrides made within a `pref.context()` block are automatically scoped per-thread and per-async-task. This means that concurrent threads or coroutines can use different preference settings without interfering with each other:

```python
import threading

def worker(eps_val, label):
    with pref.context(eps=eps_val):
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(10 * x)))
        print(f"{label}: eps={pref.eps}, length={len(f)}")

t1 = threading.Thread(target=worker, args=(1e-6, "Thread-1"))
t2 = threading.Thread(target=worker, args=(1e-14, "Thread-2"))
t1.start()
t2.start()
t1.join()
t2.join()
```

## 8.9 Preferences for ODE Solving

The Chebop and Linop solvers have their own parameters that are passed directly as arguments rather than through the global preferences:

- **`tol`**: Convergence tolerance for the adaptive discretization loop (default `1e-10`).
- **`n_min`, `n_max`**: Minimum and maximum discretization sizes.
- **`max_iter`**: Maximum Newton iterations for nonlinear problems.
- **`newton_tol`**: Newton convergence tolerance.

```python
from chebfunjax.operators.chebop import Chebop

N = Chebop(lambda x, u: u.diff(2) + u, domain=(0.0, float(jnp.pi)))
N.lbc = 0.0
N.rbc = 0.0

# Use custom solver parameters
u = N.solve(0.0, tol=1e-14, n_min=16, n_max=4096)
```

## 8.10 Summary of Factory Defaults

| Preference   | Default Value                | Description                                  |
|-------------|------------------------------|----------------------------------------------|
| `eps`       | $2.22 \times 10^{-16}$      | Construction tolerance (machine epsilon)     |
| `max_length`| 65537                        | Maximum number of Chebyshev points           |
| `tech`      | `"chebtech2"`               | Chebyshev points of the second kind          |
| `domain`    | $(-1.0, 1.0)$               | Default interval                             |
| `chop_tol`  | `None` (uses `eps`)          | Coefficient chopping tolerance               |

## 8.11 References

- J. L. Aurentz and L. N. Trefethen, "Chopping a Chebyshev series," *ACM Trans. Math. Softw.* 43 (2017), p. 33.
