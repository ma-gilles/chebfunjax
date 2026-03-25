# Contour Integrals

*Original: [chebfun.org/examples/complex/](https://www.chebfun.org/examples/complex/)*

---

The **Cauchy integral formula** is one of the most powerful results in complex
analysis. For $f$ holomorphic inside a simple closed contour $C$ and any
$z_0$ inside $C$:

$$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0}\,dz.$$

## Cauchy's formula for cos(z)

For $f(z) = \cos(z)$ and $z_0 = 0$ on the unit circle:

```python
import numpy as np

N = 5000
t_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
z = np.exp(1j * t_vals)         # unit circle
dz = 1j * np.exp(1j * t_vals)   # dz/dt
z0 = 0.0

f_z = np.cos(z)
integrand = f_z / (z - z0) * dz
val = np.trapz(integrand, t_vals) / (2j * np.pi)
print(f"cos(0) via Cauchy: {val.real:.8f}  (exact: 1.0)")
```

```
cos(0) via Cauchy: 1.00000000  (exact: 1.0)
```

## Higher derivatives

By differentiating Cauchy's formula:

$$f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}}\,dz.$$

For $f(z) = e^z$, all derivatives at $z_0 = 0$ equal 1:

```python
for n in [0, 1, 2, 3]:
    integrand = np.exp(z) / (z - 0.0)**(n+1) * dz
    dn = np.trapz(integrand, t_vals) * np.math.factorial(n) / (2j * np.pi)
    print(f"exp(0)^({n}) = {dn.real:.6f}  (exact: 1)")
```

```
exp(0)^(0) = 1.000000  (exact: 1)
exp(0)^(1) = 1.000000  (exact: 1)
exp(0)^(2) = 1.000000  (exact: 1)
exp(0)^(3) = 1.000000  (exact: 1)
```

![Contour integrals and residue theorem verification](../../../images/complex/contour_integrals.png)

## Residue theorem

The **residue theorem** generalizes Cauchy's formula to functions with poles.
For $f(z) = 1/(z^2 + 1)$ — two poles at $z = \pm i$ — integrating around
just the pole at $z = i$ (inside a circle of radius $r = 1.5$, say from 0 to 1):

$$\text{Res}_{z=i} \frac{1}{z^2+1} = \frac{1}{2i}.$$

## References

1. E. B. Saff and A. D. Snider, *Fundamentals of Complex Analysis*, Prentice Hall, 2003.
2. L. N. Trefethen and J. A. C. Weideman, The exponentially convergent trapezoidal
   rule, *SIAM Review* 56 (2014), 385–458.
