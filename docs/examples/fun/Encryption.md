# Encryption of a message with scribble

*Nick Trefethen, April 2012*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/fun/Encryption.html)

Python translation: [`examples/fun/encryption.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/fun/encryption.py)

The `scribble` command produces piecewise linear complex chebfuns whose plots look like words, like this:

```matlab
message = scribble('This is the message');
LW = 'linewidth'; lw = 2;
plot(message, LW, lw), axis equal
```

![Encryption figure 01](../../images/fun/Encryption_01.png)

Here's another string:

```matlab
key = scribble('Aardvarks eat ants');
plot(key, 'r', LW, lw), axis equal
```

![Encryption figure 02](../../images/fun/Encryption_02.png)

Now if we plot the sum of the two, we get nonsense:

```matlab
encrypted = message + key;
plot(encrypted, 'm', LW, lw), axis equal
```

![Encryption figure 03](../../images/fun/Encryption_03.png)

So we've invented a new encryption scheme! For of course the original message can be recovered by subtracting off that key:

```matlab
message2 = encrypted - key;
plot(message2, LW, lw), axis equal
```

![Encryption figure 04](../../images/fun/Encryption_04.png)

So long as we're investigating the world's most expensive and least secure method of encryption, we might as well tangle up the text in the complex plane a bit too. I'll bet you can't read this:

```matlab
scrambled = exp(1.5i*(encrypted));
plot(scrambled, 'g', LW, lw), axis equal
```

![Encryption figure 05](../../images/fun/Encryption_05.png)

But we can get the message back with a little unscramble:

```matlab
message3 = unwrap(log(scrambled))/1.5i - 1 - key;
plot(message3, LW, lw), axis equal
```

![Encryption figure 06](../../images/fun/Encryption_06.png)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
