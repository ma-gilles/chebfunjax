# Gravitational attraction to a sphere

*Nick Trefethen, May 2016*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/sphere/Gravity.html)

Python translation: [`examples/sphere/gravity.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/sphere/gravity.py)

## 1. A little geometry

Here's a vector $X$ with 2-norm 1.5, which we will think of as a spacecraft in orbit around the unit sphere:

```matlab
X = [-1  -1.1, -.2];
norm(X)
```

```text
ans =
   1.500000000000000
```

Let's use the vector-valued part of Spherefun to define the field of vector distances between $X$ and points on the sphere and $X$:

```matlab
d = spherefunv(@(x,y,z) X(1)-x, @(x,y,z) X(2)-y, @(x,y,z) X(3)-z);
```

Here is the scalar function representing $|d|$, that is, the the scalar distance between $X$ and points on the sphere:

```matlab
r = sqrt(dot(d,d));
```

We confirm that the closest point on the sphere to $X$ is at distance $0.5$:

```matlab
min_distance = min2(r)
```

```text
min_distance =
   0.500000000000001
```

Similarly, the farthest point is at distance $2.5$:

```matlab
max_distance = max2(r)
```

```text
max_distance =
   2.500000000000000
```

Here is a contour plot of $r$ on the sphere, together with a red dot showing our little spacecraft.

```matlab
contour(r,.6:.1:2,'k')
hold on, plot3(X(1),X(2),X(3),'.r','markersize',25), hold off
view(-10,35), axis equal, axis off
```

![Gravity figure 01](../../images/sphere/Gravity_01.png)

## 2. Inverse-square force

A great discovery of Newton (or was it Hooke?) is that the gravitational forces associated with a sphere of uniform mass distribution are the same as if all the mass were concentrated at the center. Accordingly, we know that if a unit mass is spread around the sphere and the spacecraft also has unit mass, then the inverse-square attraction between them should be $(1.5)^{-2}$:

```matlab
force_exact = 1/1.5^2
```

```text
force_exact =
   0.444444444444444
```

Let's confirm this prediction by computing the integral over the sphere. Since the area of the sphere is $4\pi$, the density of a uniformly distributed mass is

```matlab
rho = 1/(4*pi)
```

```text
rho =
   0.079577471545948
```

That gives us the following component of the force at each point, in the direction of $X$:

```matlab
Xnormalized = X/norm(X);
force_function = rho*(Xnormalized*d)./r.^3;
```

Summing, we get the expected answer:

```matlab
force = sum2(force_function)
```

```text
force =
   0.444444444444444
```

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
