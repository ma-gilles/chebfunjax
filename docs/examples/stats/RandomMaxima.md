# Maxima of a random function

*Nick Trefethen, February 2017*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/stats/RandomMaxima.html)

Python translation: [`examples/stats/random_maxima.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/stats/random_maxima.py)

How many maxima does a random function have? We can look at one answer to this question based on the notion of a random function implemented in the new Chebfun command `randnfun`. Here for example are the local maxima of a random function on the interval $[0,20]$.

```matlab
tic, rng(0), dx = 1;
f = randnfun(dx,[0,20]);
[val,pos] = max(f,'local');
MS = 'markersize'; LW = 'linewidth'; FS = 'fontsize';
ms = 32; lw = 5; fs = 36;
plot(f,'k',LW,lw), grid on, hold on
plot(pos,val,'.r',MS,ms), hold off
title([int2str(length(val)) ' maxima'],FS,fs)
```

![RandomMaxima figure 01](../../images/stats/RandomMaxima_01.png)

The random function in question is obtained as the restriction of a sum of a finite Fourier series with independent normally distributed random coefficients; the maximum wave number is about $2\pi /dx$, i.e., $2\pi$ in this case with $dx = 1$. Here is an analogous picture on an interval of length 40.

```matlab
f = randnfun(dx,[0,40]);
[val,pos] = max(f,'local');
plot(f,'k',LW,lw), grid on, hold on
plot(pos,val,'.r',MS,ms), hold off
title([int2str(length(val)) ' maxima'],FS,fs)
```

![RandomMaxima figure 02](../../images/stats/RandomMaxima_02.png)

Let's explore the number of maxima as a function of the length of the interval:

```matlab
Lvec = 2.^(0:10);
nmaxvec = [];
for L = Lvec
  f = randnfun(dx,[0 L]);
  nmaxvec = [nmaxvec length(max(f,'local'))];
end
loglog(Lvec,Lvec,'-r',LW,lw), hold on
loglog(Lvec,nmaxvec,'.',MS,44), grid on, hold off
axis([.8 1300 .8 1300]);
xlabel('length of interval',FS,fs)
ylabel('no. of maxima',FS,fs)
```

![RandomMaxima figure 03](../../images/stats/RandomMaxima_03.png)

It would seem that the expected number of maxima is asymptotic to $L$. Very likely there is a literature on this question.

Note that the first dot falls higher than the curve. This is because when Chebfun finds "local extrema", it includes extrema at the endpoints even though these (with probability 1) will not be points of zero derivative. We could improve the experiment to correct for this, though this is not too important since the effect diminishes proportionally when $L$ is large.

Execution time for this example:

```matlab
format short, time_in_seconds = toc
```

```text
time_in_seconds =
    63.3092
```

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
