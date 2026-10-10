"""Literal local reference routines from tests/misc/test_jac2jac.m7574c77.

Independent host arithmetic is test-only; no production conversion is reused.
Copyright 2017 The University of Oxford and The Chebfun Developers.
"""
import numpy as np
from scipy.special import beta, gamma


def _vandermonde(n, x, a, b):
    apb = a + b
    aa = a * a
    bb = b * b
    p = np.zeros((len(x), n))
    p[:, 0] = 1
    if n > 1:
        p[:, 1] = .5*(2*(a+1)+(apb+2)*(x-1))
    for k in range(2, n):
        k2 = 2*k
        kab = k2+apb
        q1 = k2*(k+apb)*(kab-2)
        q2 = (kab-1)*(aa-bb)
        q3 = (kab-2)*(kab-1)*kab
        q4 = 2*(k+a-1)*(k+b-1)*kab
        p[:, k] = ((q2+q3*x)*p[:, k-1]-q4*p[:, k-2])/q1
    return p


def _ccj_weights(n, a, b):
    if a == b and a == 0:
        c = 2/np.concatenate(([1.], 1-np.arange(2, n, 2, dtype=float)**2))
        c = np.concatenate((c, c[n//2-1:0:-1]))
        w = np.concatenate((np.fft.ifft(c).real, [0.]))
        w[[0, n-1]] = w[0]/2
    elif a == b:
        ell = a+.5
        g0 = gamma(ell+.5)*np.sqrt(np.pi)/gamma(ell+1)
        k = np.arange(1, (n-1)//2+1)
        c = g0*np.concatenate(([1.], np.cumprod((k-ell-1)/(k+ell))))
        c = np.concatenate((c, c[n//2-1:0:-1]))
        w = np.concatenate((np.fft.ifft(c).real, [0.]))
        w[[0, n-1]] = w[0]/2
    else:
        c = np.zeros(n+2)
        c[:2] = [1, (a-b)/(a+b+2)]
        for r in range(1, n+1):
            c[r+1] = -(2*(b-a)*c[r]+(a+b+2-r)*c[r-1])/(a+b+2+r)
        c *= 2**(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+2)
        v = np.fft.ifft(np.concatenate((c[:n], c[n-2:0:-1]))).real
        w = np.concatenate((v[:1], 2*v[1:n-1], v[n-1:n]))
    m = n-1
    x = np.sin(np.pi*np.arange(-m, m+1, 2)/(2*m))
    return w, x


def jac2cheb_direct(c, a, b):
    n = c.shape[0]
    if n <= 1:
        return c
    if n-1 > 2**11:
        raise ValueError("native direct reference degree exceeds 2048")
    x = np.sin(np.pi*(np.arange(-n+1, n, 2)/(2*n)))
    values = _vandermonde(n, x, a, b) @ c
    weights = 2*np.exp(1j*np.arange(n)*np.pi/(2*n))
    out = weights[:, None]*np.fft.ifft(np.concatenate((values[::-1], values)), axis=0)[:n]
    out[0] /= 2
    out = out.real
    even = np.max(np.abs(values-values[::-1]), axis=0) == 0
    odd = np.max(np.abs(values+values[::-1]), axis=0) == 0
    out[1::2, even] = 0
    out[0::2, odd] = 0
    return out


def cheb2jac_direct(c, a, b):
    n = c.shape[0]
    if n <= 1:
        return c
    if n-1 > 2**11:
        raise ValueError("native direct reference degree exceeds 2048")
    degree = n-1
    padded = np.concatenate((c, np.zeros((degree, c.shape[1]))))
    even = np.max(np.abs(padded[1::2]), axis=0) == 0
    odd = np.max(np.abs(padded[0::2]), axis=0) == 0
    padded[1:-1] /= 2
    mirrored = np.concatenate((padded, padded[-2:0:-1]))
    values = np.fft.fft(mirrored, axis=0).real[:2*degree+1][::-1]
    values[:, even] = (values[:, even]+values[::-1, even])/2
    values[:, odd] = (values[:, odd]-values[::-1, odd])/2
    weights, x = _ccj_weights(2*degree+1, a, b)
    p = _vandermonde(n, x, a, b)
    scale = np.zeros(n)
    scale[0] = beta(a+1, b+1)
    for k in range(degree):
        scale[k+1] = (2*k+a+b+1)*(k+a+1)*(k+b+1)/((k+1)*(2*k+a+b+3)*(k+a+b+1))*scale[k]
    scale *= 2**(a+b+1)
    return (p.T @ (values*weights[:, None]))/scale[:, None]
