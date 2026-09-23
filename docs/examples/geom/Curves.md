# Distance between two curves

*Nick Trefethen, November 2022*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/geom/Curves.html)

Python translation: [`examples/geom/curves.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/geom/curves.py)

Suppose we have two curves, like these,

```matlab
tic, rng(1), LW = 'linewidth'; MS = 'markersize';
t = chebfun('t');
f = 1i*t + .2*randnfun(.5) - 1;
g = 1i*t + .2*randnfun(.5) + 1;
plot([f g],'linewidth',2), axis equal, grid on
```

![Curves figure 01](../../images/geom/Curves_01.png)

and we want to know the closest distance between them. (This is a great simplification of a problem John Maddocks brought up at lunch today.) I am sure there is a lot known about how to compute this.

One approach is to simply make a chebfun2 $d(x,y)$ representing the distance between $f(x)$ and $g(y)\dots$

```matlab
d = chebfun2(@(x,y) abs(f(x)-g(y)));
contour(d,LW,1), axis equal, colorbar, xlabel x, ylabel y
```

![Curves figure 02](../../images/geom/Curves_02.png)

$\dots$ and find the global minimum:

```matlab
[mindist,pos] = min2(d); x = f(pos(1)); y = g(pos(2));
plot([f g],'linewidth',2), axis equal, grid on
hold on, plot([x y],'--k',LW,1.2), plot([x y],'.k',MS,20), hold off
title(['minimum distance: ' num2str(mindist)]), toc
```

```text
Elapsed time is 9.228925 seconds.
```

![Curves figure 03](../../images/geom/Curves_03.png)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
