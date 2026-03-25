# Integrals over closed contours using periodic chebfuns

**Mohsin Javed, June 2014**

---

If a closed contour $\Gamma$ is parametrised by $z(t)$ for $t \in [a, b]$ with
$z(a) = z(b)$, then the integrand $f(z(t)) z'(t)$ is periodic and the
trapezoidal rule achieves spectral accuracy:

$$
\oint_\Gamma f(z)\, dz = \int_a^b f(z(t))\, z'(t)\, dt.
$$

## Example 1: Residue at a simple pole

For $f(z) = 1/z$ and a circle $z(t) = e^{it}$, the residue theorem gives
$\oint 1/z\, dz = 2\pi i$:

```python
import numpy as np

n = 100
t = np.linspace(0, 2*np.pi, n, endpoint=False)
z  = np.exp(1j * t)
dz = 1j * z
dt = 2 * np.pi / n

I1 = np.sum((1.0 / z) * dz) * dt
print(f"Residue 1/z: {I1:.6f}  (exact: {2j*np.pi:.6f})")
```

## Example 2: Cauchy's integral theorem

For an analytic function (no poles inside the contour), the integral is zero:

```python
# sin(z)/z is analytic inside |z|=1 (removable singularity at 0)
I2 = np.sum(np.sinc(z / np.pi) * dz) * dt
print(f"sin(z)/z integral: {I2:.2e}  (should be ≈ 0)")
```

## Example 3: Essential singularity

For $f(z) = e^{1/z}$ around $z = 0$, the residue is 1, so the integral equals
$2\pi i$:

```python
I3 = np.sum(np.exp(1.0 / z) * dz) * dt
print(f"exp(1/z) residue: {I3:.6f}  (exact: {2j*np.pi:.6f})")
```

## Example 4: Cauchy integral formula

$\oint f(z)/(z - z_0)\, dz = 2\pi i f(z_0)$ for $z_0$ inside $\Gamma$:

```python
z0 = 0.3 + 0.2j
f_vals = np.sin(z)
I4 = np.sum(f_vals / (z - z0) * dz) * dt
print(f"Cauchy formula: {I4:.6f}")
print(f"2πi·sin(z0)  : {2j*np.pi*np.sin(z0):.6f}")
```

## Gallery

![Closed contours](../../../examples/complex/closed_contours.png)

Four closed-contour integrals with their contours and exact values.
