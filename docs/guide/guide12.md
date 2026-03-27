# 12. Chebfun2: Getting Started

*Based on [Chebfun Guide Chapter 12](https://www.chebfun.org/docs/guide/guide12.html) by Alex Townsend, March 2013, latest revision October 2019*

## 12.1 What is a chebfun2?

Chebfun2 is the part of chebfunjax that deals with functions of two variables defined on a rectangle $[a,b] \times [c,d]$. Just like chebfunjax in 1D, it is an extremely convenient tool for all kinds of computations including algebraic manipulation, optimization, integration, and rootfinding. It also extends to vector-valued functions of two variables, so that one can perform vector calculus.

For example, here is a test function that has been part of MATLAB for many years. MATLAB represents the "peaks" function by a $49\times 49$ matrix:

```python
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.chebfun2d import chebfun2

def peaks(x, y):
    return (3*(1-x)**2 * jnp.exp(-x**2 - (y+1)**2)
            - 10*(x/5 - x**3 - y**5) * jnp.exp(-x**2 - y**2)
            - 1/3 * jnp.exp(-(x+1)**2 - y**2))

f = chebfun2(peaks, domain=(-3.0, 3.0, -3.0, 3.0))
```

```python
cj.surf(f, title='Chebfun2 Peaks')
```

![Peaks surface](../images/guide/guide12_01.png)

In chebfunjax we can do all sorts of things with functions to a high accuracy, such as evaluate them

```python
print(f(0.5, 0.5))
```

```
0.375375578848315
```

or compute their maxima,

```python
import numpy as np
n = 500
xs = jnp.linspace(-3, 3, n)
ys = jnp.linspace(-3, 3, n)
xx, yy = jnp.meshgrid(xs, ys)
vals = f(xx.ravel(), yy.ravel()).reshape(n, n)
print(float(jnp.max(vals)))
```

```
8.106213589442337
```

A chebfun2, with a lower-case "c", is a Python object, the 2D analogue of a chebfun. The syntax for chebfun2 objects is similar to the syntax for matrices in MATLAB, and chebfun2 objects have many operations overloaded. For instance, `trace(A)` returns the sum of the diagonal entries of a matrix `A` and a 2D trace returns the integral of $f(x,x)$ when $f$ is a chebfun2.

Chebfun2 builds on chebfunjax's univariate representations and algorithms. Algorithmic details are given in [Townsend & Trefethen 2013b] and mathematical underpinnings in [Townsend & Trefethen 2014]. For more information, see Section 12.8.

## 12.2 What is a chebfun2v?

Chebfun2 can represent scalar-valued functions, such as $\exp(x+y)$, and vector-valued functions, such as $[\exp(x+y);\, \cos(x-y)]$. A vector-valued function is called a `Chebfun2v`, and `Chebfun2v` objects are useful for computations of vector calculus. For information about `Chebfun2v` objects and vector calculus, see Chapters 15 and 16 of this guide.

## 12.3 Constructing chebfun2 objects

A chebfun2 can be constructed by supplying the `chebfun2` constructor with a bivariate function. The default rectangular domain is $[-1,1] \times [-1,1]$. (An example showing how to specify a different domain is given in section 12.6.) For example, here we construct and plot a chebfun2 representing $\cos(2\pi xy)$ on $[-1,1] \times [-1,1]$.

```python
f = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y))
```

We could equally well have constructed chebfun2 objects for the variables `x` and `y` first and then computed `f` from these:

```python
x = chebfun2(lambda x, y: x)
y = chebfun2(lambda x, y: y)
```

Here is a surface plot of `f`:

```python
cj.surf(f)
```

![cos(2*pi*x*y) surface](../images/guide/guide12_02.png)

Along with `surf`, there is also the command `contour` for displaying a chebfun2. Here is a contour plot of `f`:

```python
cj.contour(f)
```

![cos(2*pi*x*y) contour](../images/guide/guide12_03.png)

One way to find the rank of the approximant used to represent `f`, discussed in Section 12.8, is like this:

```python
print(f.rank)
```

```
11
```

Alternatively, more information can be given by displaying the chebfun2 object:

```python
print(f)
```

```
Chebfun2(rank=11, domain=(-1.0, 1.0, -1.0, 1.0))
```

The vertical scale is used by operations to aim for close to machine precision relative to that number.

## 12.4 Basic operations

Once we have a chebfun2, we can compute quantities such as its definite double integral:

```python
print(f.sum2())
```

```
0.902823333580281
```

This matches well the exact answer obtained by calculus, which is $(2/\pi)\text{Si}(2\pi)$:

```python
exact = 0.9028233335802806267957003779
print(exact)
```

```
0.902823333580281
```

We can also evaluate a chebfun2 at a point $(x, y)$, or along a line. When evaluating along a line a function of one variable is returned because the answer depends on only one variable.

Evaluation at a point:

```python
import numpy as np
x_val = 2*np.random.rand() - 1
y_val = 2*np.random.rand() - 1
print(f(x_val, y_val))
```

There are plenty of other questions that may be of interest. For instance, what are the zero contours of $f(x,y) - 0.95$?

```python
g = chebfun2(lambda x, y: jnp.cos(2*jnp.pi*x*y) - 0.95)
curves = g.roots()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for c in curves:
    ax.plot(c[:, 0], c[:, 1], 'b-')
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title('Zero contours of f - 0.95')
```

![Zero contours of f-0.95](../images/guide/guide12_04.png)

What is the partial derivative $\partial f / \partial y$?

```python
fy = f.diff(dim=1)
cj.surf(fy)
```

![df/dy surface](../images/guide/guide12_05.png)

The syntax for the `diff` method can cause confusion because we are following the matrix syntax in MATLAB. In chebfunjax, `f.diff(dim=1)` differentiates with respect to $y$ and `f.diff(dim=2)` differentiates with respect to $x$. Chebfun2 also offers the more easily remembered `diffx(f,k)` and `diffy(f,k)` conventions in the MATLAB version, which differentiate $f(x,y)$ $k$ times with respect to the first and second variable, respectively.

What is the mean value of $f$ on $[-1,1] \times [-1,1]$?

```python
mean_val = f.sum2() / 4.0  # area of [-1,1]^2 is 4
print(float(mean_val))
```

```
0.225705833395070
```

## 12.5 Chebfun2 methods

There are many methods that can be applied to chebfun2 objects. The most important include:

- `f(x, y)` -- evaluate at points
- `f.sum2()` -- double integral
- `f.sum(dim=1)` / `f.sum(dim=2)` -- partial integrals
- `f.diff(dim=1)` / `f.diff(dim=2)` -- partial derivatives
- `f.norm()` -- L2 (Frobenius) norm
- `f.roots()` -- zero contours
- `f.rank` -- numerical rank
- `f.domain` -- domain tuple $(x_a, x_b, y_a, y_b)$

Most of these commands have been overloaded from their MATLAB counterparts.

## 12.6 Composition of chebfun2 objects

New chebfun2 objects can be constructed from existing ones by composing them with operations such as `+`, `-`, `*`, and elementary functions. For example,

```python
f = chebfun2(lambda x, y: 1.0 / (2 + jnp.cos(0.25 + x**2*y + y**2)),
             domain=(-4.0, 4.0, -2.0, 2.0))
cj.contour(f)
```

![Composition contour](../images/guide/guide12_06.png)

## 12.7 Analytic functions

An analytic function $f(z)$ can be thought of as a complex-valued function of two real variables, $f(x,y) = f(x+iy)$. If the chebfun2 constructor is given an anonymous function with one argument, it assumes that argument is a complex variable. For instance,

```python
f = chebfun2(lambda x, y: jnp.sin(x + 1j*y))
print(f(1.0, 1.0), jnp.sin(1.0 + 1j))
```

```
(1.298457581415977+0.634963914784736j) (1.298457581415977+0.634963914784736j)
```

These functions can be visualised by using a technique known as phase portrait plots. Given a complex number $z = re^{i\theta}$, the phase $e^{i\theta}$ can be represented by a colour. We follow Wegert's colour recommendations [Wegert 2012], using red for a phase $i$, then yellow, green, blue, and violet as the phase moves clockwise around the unit circle. For example,

```python
import numpy as np

def f_complex(z):
    return np.sin(z) - np.sinh(z)

cj.phaseplot(f_complex, region=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi],
             title=r'Phase portrait of $\sin(z) - \sinh(z)$')
```

![Phase portrait sin(z)-sinh(z)](../images/guide/guide12_07.png)

Many properties of analytic functions can be visualised by these types of plots, such as the location of zeros and their multiplicities. Can you work out the multiplicity of the root at $z=0$ from this plot?

Since chebfun2 only represents smooth functions, a trick is required to draw pictures like these for functions with poles [Trefethen 2013]. For functions with branch points or essential singularities, it is currently not possible in chebfun2 to draw phase plots.

## 12.8 Chebfun2 low rank approximations

Chebfun2 exploits the observation that many functions of two variables can be well approximated by low rank approximants. A rank 1 function, also known as separable, is of the form $u(y)v(x)$, and a rank $k$ function is one that can be written as the sum of $k$ rank 1 functions. Smooth functions tend to be well approximated by functions of low rank. Chebfun2 determines low rank function approximations automatically by means of an algorithm that can be viewed as an iterative application of Gaussian elimination with complete pivoting [Townsend & Trefethen 2013]. The underlying function representations are related to work by Carvajal, Chapman and Geddes [Carvajal, Chapman, & Geddes 2008] and others including Bebendorf [Bebendorf 2008], Hackbusch, Khoromskij, Oseledets, and Tyrtyshnikov. For further aspects of low-rank representations see [Trefethen 2017] and [Beckermann and Townsend 2019].

Here is an example adapted from [Townsend & Trefethen 2013]. The function $f(x,y) = \exp(-40(x^2 - xy + 2y^2 - 1/2)^2)$ has the shape of an elliptical ring in the unit square, and chebfun2 represents it by an approximation of reasonably high rank:

```python
ff = lambda x, y: jnp.exp(-40*(x**2 - x*y + 2*y**2 - 0.5)**2)
f = chebfun2(ff)
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

import matplotlib.pyplot as plt
n = 200
xs = jnp.linspace(-1, 1, n); ys = jnp.linspace(-1, 1, n)
XX, YY = jnp.meshgrid(xs, ys)
ZZ = f(XX.ravel(), YY.ravel()).reshape(n, n)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.contour(XX, YY, ZZ, levels=levels, colors='k')
ax.set_aspect('equal')
ax.set_title(f'rank {f.rank}')
```

![Smoke ring contour](../images/guide/guide12_08.png)

To illustrate the nature of low-rank approximations, rather than letting chebfun2 determine the rank adaptively, we can force it to take ranks 1, 2, ..., 9. Here are the results, plotted with black level curves at heights 0.2, 0.4, 0.6, 0.8:

```python
import numpy as np
from scipy.interpolate import RectBivariateSpline

ff_np = lambda x, y: np.exp(-40*(x**2 - x*y + 2*y**2 - 0.5)**2)
levels = [0.2, 0.4, 0.6, 0.8]
fig = plt.figure(figsize=(9, 9))
n_sample = 100
xs_s = np.linspace(-1, 1, n_sample)
ys_s = np.linspace(-1, 1, n_sample)
XXs, YYs = np.meshgrid(xs_s, ys_s)
ZZs = ff_np(XXs, YYs)
U, S, Vt = np.linalg.svd(ZZs, full_matrices=False)

for k in range(1, 10):
    ax = fig.add_axes([0.03 + 0.33*((k-1) % 3),
                       0.67 - 0.30*((k-1) // 3), 0.28, 0.28])
    ZZ_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    ax.contour(XXs, YYs, ZZ_k, levels=levels, colors='k')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'rank {k}', fontsize=9)
```

![Low-rank approximation grid](../images/guide/guide12_09.png)

For this function, "plotting accuracy" is achieved approximately at rank 16; the remaining terms are then required to get from 2-3 digits to 15.
