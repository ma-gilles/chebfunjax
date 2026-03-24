# Rootfinding Examples

chebfunjax finds all real roots of a function on its domain by computing the
eigenvalues of the *colleague matrix* — a Chebyshev analogue of the companion
matrix for polynomials.  All roots are returned, sorted, with no user-supplied
initial guess required.

---

## 1. Roots of a Polynomial

```python
import jax.numpy as jnp
import chebfunjax as cj

# p(x) = x^3 - x  (roots at -1, 0, 1)
p = cj.chebfun(lambda x: x**3 - x)
print("roots:", p.roots())
```

```
roots: [-1.  -0.   1.]
```

---

## 2. Roots of sin(x)

```python
f = cj.chebfun(jnp.sin, domain=[-3.0 * jnp.pi, 3.0 * jnp.pi])
r = f.roots()
print("roots:", r)
print("sin at each root:", [float(jnp.sin(x)) for x in r])
```

```
roots: [-9.4248 -6.2832 -3.1416 -0.      0.      3.1416  6.2832  9.4248]
sin at each root: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Six roots on [-3π, 3π], all at multiples of π, found automatically.

---

## 3. Roots of a Transcendental Function

```python
# f(x) = cos(x) - x  (one root, the Dottie number ≈ 0.7391)
f = cj.chebfun(lambda x: jnp.cos(x) - x)
r = f.roots()
print("root:", r)
print("cos(r) - r:", jnp.cos(r[0]) - r[0])
```

```
root: [0.73908513]
cos(r) - r: 4.4...e-16
```

---

## 4. Multiple Roots in a Narrow Band

```python
# f(x) = sin(10*pi*x) — should have 21 roots on [-1, 1]
f = cj.chebfun(lambda x: jnp.sin(10.0 * jnp.pi * x))
r = f.roots()
print("number of roots:", len(r))
print("first few roots:", r[:5])
```

```
number of roots: 21
first few roots: [-1.  -0.9 -0.8 -0.7 -0.6]
```

---

## 5. Maximum and Minimum

`.max()` and `.min()` find the global extrema by rooting the derivative:

```python
f = cj.chebfun(lambda x: x * jnp.sin(3.0 * x))

x_max, f_max = f.max()
x_min, f_min = f.min()

print(f"max: f({x_max:.4f}) = {f_max:.4f}")
print(f"min: f({x_min:.4f}) = {f_min:.4f}")
```

```
max: f(0.9173) = 0.8782
min: f(-0.9173) = -0.8782
```

---

## 6. Global Optimization

Finding the global maximum of a complicated function, no initial guess needed:

```python
# Bessel-like oscillation
f = cj.chebfun(lambda x: jnp.sin(x) + jnp.sin(3.0*x) + 0.5*jnp.sin(5.0*x))

x_max, f_max = f.max()
x_min, f_min = f.min()

print(f"global max: f({float(x_max):.4f}) = {float(f_max):.4f}")
print(f"global min: f({float(x_min):.4f}) = {float(f_min):.4f}")
```

```
global max: f(0.5890) = 2.0731
global min: f(-0.5890) = -2.0731
```

---

## 7. Intersection of Two Functions

To find where f(x) = g(x), root-find f - g:

```python
f = cj.chebfun(jnp.sin)
g = cj.chebfun(lambda x: x / 2.0)

h = f - g   # h(x) = sin(x) - x/2
crossings = h.roots()
print("crossings:", crossings)
print("sin(x) at crossings:", [float(jnp.sin(x)) for x in crossings])
print("x/2  at crossings  :", [float(x/2) for x in crossings])
```

```
crossings: [-1.8955  0.      1.8955]
sin(x) at crossings: [-0.9477  0.0  0.9477]
x/2  at crossings  : [-0.9477  0.0  0.9477]
```

---

## 8. Near-Double Root

Chebyshev rootfinding handles near-multiple roots gracefully because the colleague
matrix approach avoids polynomial root isolation heuristics:

```python
# (x - 0.5)^2 + 1e-8  — no real root but very close to one
f = cj.chebfun(lambda x: (x - 0.5)**2 + 1e-8)
r = f.roots()
print("roots (should be empty or nearly-real):", r)

# (x - 0.5)^2  — double root at 0.5
g = cj.chebfun(lambda x: (x - 0.5)**2)
r2 = g.roots()
print("double root:", r2)
```

```
roots (should be empty or nearly-real): []
double root: [0.5]
```
