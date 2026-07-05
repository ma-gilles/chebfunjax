"""Tests for the trigonometric gallery (Opus 4.8, task #20)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)


class TestGalleryTrig:
    def test_list(self):
        from chebfunjax.utils.gallerytrig import list_gallerytrig
        names = list_gallerytrig()
        for expected in ("amsignal", "fmsignal", "wavepacket", "sinefun1",
                         "gibbs", "weierstrass", "starburst"):
            assert expected in names

    def test_amsignal(self):
        from chebfunjax.utils.gallerytrig import gallerytrig
        f = gallerytrig("amsignal")
        xs = np.linspace(-3.0, 3.0, 60)
        npt.assert_allclose(
            np.asarray(f(jnp.asarray(xs))),
            np.cos(50 * xs) * (1 + 0.2 * np.cos(5 * xs)), atol=1e-11)

    def test_fmsignal(self):
        from chebfunjax.utils.gallerytrig import gallerytrig
        f = gallerytrig("fmsignal")
        xs = np.linspace(-3.0, 3.0, 60)
        npt.assert_allclose(
            np.asarray(f(jnp.asarray(xs))),
            np.cos(50 * xs + 4 * np.sin(5 * xs)), atol=1e-11)

    def test_gibbs(self):
        from chebfunjax.utils.gallerytrig import gallerytrig
        f = gallerytrig("gibbs")
        xs = np.linspace(-3.0, 3.0, 60)
        want = np.sum([4 / np.pi * np.sin(n * xs) / n
                       for n in range(1, 20, 2)], axis=0)
        npt.assert_allclose(np.asarray(f(jnp.asarray(xs))), want, atol=1e-11)

    def test_starburst_complex(self):
        from chebfunjax.utils.gallerytrig import gallerytrig
        f = gallerytrig("starburst")
        assert np.iscomplexobj(np.asarray(f(jnp.float64(0.3))))

    def test_unknown_raises(self):
        import pytest

        from chebfunjax.utils.gallerytrig import gallerytrig
        with pytest.raises(KeyError):
            gallerytrig("nonesuch")
