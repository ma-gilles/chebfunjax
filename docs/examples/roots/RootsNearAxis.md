# Complex roots near the real axis

**Nick Trefethen, October 2011**

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/roots/RootsNearAxis.html)

---

A chebfun may have no real roots while having complex roots very close to the
real axis. These complex roots influence the function's behaviour — the closer
they are, the more oscillatory the function.

## A wiggly chebfun with no real roots

Consider $f(x) = 3 + \sin(x) + \sin(\pi x)$ on $[0, 30]$. Since
$|\sin(x) + \sin(\pi x)| \le 2 < 3$, the function is strictly positive:

```python
import jax.numpy as jnp
import chebfunjax as cj

f = cj.chebfun(lambda x: 3 + jnp.sin(x) + jnp.sin(jnp.pi * x),
               domain=(0.0, 30.0))
r = f.roots()
print("Real roots:", len(r), "(should be 0)")
assert len(r) == 0
```

## Visualising complex roots via $\log|f|$

Although $f$ has no real roots, we can see where it *nearly* vanishes by
plotting $\log|f(z)|$ on a complex domain:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 30, 600)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

F = 3 + np.sin(Z) + np.sin(np.pi * Z)
plt.contourf(X, Y, np.log10(np.abs(F)), levels=30, cmap='RdYlBu_r')
plt.colorbar(label=r'$\log_{10}|f(z)|$')
plt.axhline(0, color='k', lw=0.5)
```

The dark regions (small $|f(z)|$) reveal the positions of the complex roots.

## Gallery

![Roots near axis](../../images/roots/roots_near_axis.png)

$\log_{10}|f(z)|$ over a complex strip, showing the near-axis complex roots as
dark spots. The real axis ($\text{Im}(z)=0$) has no zeros.
