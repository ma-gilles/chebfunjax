# Parametric surfaces

*Rodrigo Platte, March 2013*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/geom/ParametricSurfaces.html)

(Chebfun example geom/ParametricSurfaces.m)

Surfaces defined by coordinate functions of two parameters: a cone
and a hyperboloid (with a colored variant):

![ParametricSurfaces figure 1](../../images/geom/ParametricSurfaces_repl_01.png)

![ParametricSurfaces figure 3](../../images/geom/ParametricSurfaces_repl_03.png)

The sphere, a bumpy perturbation, and a colored one:

![ParametricSurfaces figure 5](../../images/geom/ParametricSurfaces_repl_05.png)

A seashell:

![ParametricSurfaces figure 7](../../images/geom/ParametricSurfaces_repl_07.png)

A Mobius strip; the tangent vectors $r_u$ and $r_v$ are orthogonal
everywhere:

![ParametricSurfaces figure 8](../../images/geom/ParametricSurfaces_repl_08.png)

```text
ans =
     1.318389841742373e-16
```

(MATLAB: 5.0e-14 from chebfun2-differentiated tangents; ours uses
the analytic tangents.)  And a Klein bottle:

![ParametricSurfaces figure 9](../../images/geom/ParametricSurfaces_repl_09.png)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
