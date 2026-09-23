# Splines

*Nick Trefethen, February 2013*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/Splines.html)

Python translation: [`examples/approx/splines.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/approx/splines.py)

Chebfun has an analogue of the MATLAB `spline` command. Here for example is a function on $[0,10]$:

```matlab
hold off
x = chebfun('x',[0 10]);
f = sin(x + 0.25*x.^2);
plot(f), axis([0 10 -1.2 1.2])
```

![Splines figure 01](../../images/approx/Splines_01.png)

Let us call Chebfun `spline` to make a cubic spline $s$ that interpolates $f$ at the integers, and superimpose it on the same plot in red. Since a spline is just piecewise smooth, the Chebfun is made of a collection of funs, and we include red dots to mark the knots that separate them.

```matlab
s = chebfun.spline(0:10,f(0:10));
hold on, plot(s,'r')
plot(s.ends,s(s.ends),'.r','markersize',12)
axis([0 10 -1.2 1.2])
```

![Splines figure 02](../../images/approx/Splines_02.png)

Here are the first, second, and third derivatives of $s$. The curves are respectively piecewise parabolic and continuously differentiable; piecewise linear and continuous; and piecewise constant.

```matlab
clf
for d = 1:3
   subplot(3,1,d)
   plot(diff(s,d))
   text(6.5,.8*2^(d+1),['derivative ' int2str(d)])
end
```

![Splines figure 03](../../images/approx/Splines_03.png)

Note that in the final image, there is no discontinuity at $x=1$ or $x=9$. This is because by default, MATLAB imposes the "not-a-knot condition" to use up the 2 available free parameters of an interpolating piecewise cubic with two continuous derivatives.

One of Chebfun's features is its ability to locate discontinuities automatically in `splitting on` mode. To illustrate this, suppose we construct a new chebfun `s2` by sampling `s` with splitting on. The accuracy is very good:

```matlab
s2 = chebfun(@(x) s(x),[0 10],'splitting','on');
norm(s-s2,inf)
```

```text
ans =
     7.878883183172377e-14
```

Here are the breakpoints of the automatically constructed function `s2`. Note that $x=1$ and $x=9$ are missing, again because of the not-a-knot condition.

```matlab
format long
s2.ends'
```

```text
ans =
  0.000000000000000
  2.000006888364444
  2.999994244090829
  4.000003615390170
  4.999998609190251
  5.999999487064672
  6.999917929432533
  8.000030396032123
  10.000000000000000
```

It is interesting that breakpoints near $2,3,\dots,8$ match these values not to $15$ or $16$ digits but to about the cube root of that precision. This does not reflect poor performance of the splitting algorithm, but just the fact that the function being sampled has two continuous derivatives. If it had only one continuous derivative, we'd get $8$ digits of accuracy, and if only the function itself were continuous, we'd expect $15$ or $16$ digits. When not even the function is continuous, Chebfun captures discontinuities down to the very last bit. For information about edge detection in Chebfun, see [1].

## References

1. R. Pachon, R. B. Platte, and L. N. Trefethen, Piecewise-smooth chebfuns, *IMA Journal of Numerical Analysis*, 30 (2010), 898-916.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
