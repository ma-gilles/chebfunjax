# Encrypting a message with chebfuns

*Nick Trefethen, December 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/fun/Encryption.html)

(Chebfun example fun/Encryption.m)

A message and a key, both piecewise-linear complex chebfuns from
`scribble`:

![Encryption figure 1](../../images/fun/Encryption_repl_01.png)

![Encryption figure 2](../../images/fun/Encryption_repl_02.png)

Adding the key encrypts; subtracting decrypts:

![Encryption figure 3](../../images/fun/Encryption_repl_03.png)

![Encryption figure 4](../../images/fun/Encryption_repl_04.png)

A nonlinear scrambling $e^{1.5iz}$, undone with `unwrap(log(.))`:

![Encryption figure 5](../../images/fun/Encryption_repl_05.png)

![Encryption figure 6](../../images/fun/Encryption_repl_06.png)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
