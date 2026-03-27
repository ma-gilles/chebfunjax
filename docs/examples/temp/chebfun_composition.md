# Composition with Multivariate Chebfuns

**Original:** [temp/ChebfunComposition](https://github.com/chebfun/examples/blob/master/temp/ChebfunComposition.m)
**Author(s):** Olivier Sete, February 2017

---

The composition of two functions, $h = g \circ f$, is a basic operation in
Chebfun. Since Chebfun v5.6.0, this works for all combinations of chebfun,
chebfun2, chebfun2v, chebfun3, chebfun3v, diskfun, diskfunv, spherefun, and
spherefunv objects, provided the range of $f$ lies in the domain of $g$.

## 1. Composition with chebfuns

The simplest case is composing two 1D chebfuns:

$$f(t) = \cos(t), \quad g(t) = e^t, \quad h = g(f) = e^{\cos t}.$$

When $g$ is a chebfun and $f$ is a chebfun2, the result is a chebfun2:

$$f(x,y) = x^2 + y, \quad g(t) = e^{\cos(10t)}, \quad h(x,y) = g\bigl(f(x,y)\bigr).$$

If the chebfun $g$ has two or three columns, the result is a chebfun2v or
chebfun3v object. Replacing $f$ by a chebfun3, diskfun, or spherefun works
analogously.

## 2. Composition of a chebfun2 object

A chebfun2 $g(x,y)$ can be composed with any object $f$ that maps to
$\mathbb{R}^2$. When $f$ is a curve in 2D, this restricts $g$ to the curve.
For example, with $g(x,y) = x^2 + y^2$ and $f(t) = (\cos t, \sin t)$ for
$t \in [-\pi, \pi]$, the composition $g(f)$ returns the constant chebfun $1$,
since $f$ traces the unit circle.

Similarly, composing $g$ with a chebfun2v parametrising a non-rectangular
domain gives a chebfun2 representing $g$ on that domain.

## 3. The complex plane

In two dimensions, the real and complex planes can be identified: for a
chebfun2 $g$, calling $g(z)$ is the same as calling $g(\operatorname{Re}(z),
\operatorname{Im}(z))$. A parametrisation of the complex unit circle
$f(t) = e^{it}$ gives the same result as the real parametrisation
$f(t) = (\cos t, \sin t)$.

## 4. Moving to 3D

A chebfun3 or chebfun3v object can be composed with anything mapping to
$\mathbb{R}^3$: a chebfun describing a curve, a chebfun2v describing a
surface, or a chebfun3v parametrising a domain. For example, given a helix

$$f(t) = \bigl(\cos(2\pi t),\; \sin(2\pi t),\; t\bigr), \quad t \in [0, 10],$$

and a distance function $g(x,y,z) = (x-1)^2 + (y-2)^2 + (z-3)^2$, the
composition $h = g(f)$ gives the distance from the helix to the point $(1,2,3)$
as a function of the parameter $t$.

The restriction of a chebfun3 to a surface is equally straightforward.

## Code

```python
from examples.temp.chebfun_composition import run
run()
```

## Output

![Composition with Multivariate Chebfuns](../../images/temp/chebfun_composition.png)
