"""Keep the source sphere plotting tensor grid through actual evaluation."""

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# uses-numpy: plotting fixtures/assertions
from chebfunjax import plotting  # noqa: E402
from chebfunjax.spherefun.spherefun import Spherefun  # noqa: E402
from chebfunjax.tech import trigtech  # noqa: E402


def test_actual_sphere_plot_evaluates_source_tensor_without_repeated_nodes(monkeypatch):
    lam = jnp.linspace(-jnp.pi, jnp.pi, 11)[:-1]
    theta = jnp.linspace(0., jnp.pi, 9)
    ll, tt = jnp.meshgrid(lam, theta)
    f = Spherefun.from_values(2 + jnp.cos(tt) + .25 * jnp.sin(tt) * jnp.cos(ll))
    evaluated = []
    factor_point_counts = []
    original_call = Spherefun.__call__
    original_factor = trigtech._trig_eval_np

    def evaluate(self, longitude, colatitude):
        values = original_call(self, longitude, colatitude)
        if self is f:
            evaluated.append((np.asarray(longitude), np.asarray(colatitude), np.asarray(values)))
        return values

    def factor(coeffs, nodes, **kwargs):
        factor_point_counts.append(np.asarray(nodes).size)
        return original_factor(coeffs, nodes, **kwargs)

    monkeypatch.setattr(Spherefun, "__call__", evaluate)
    monkeypatch.setattr(trigtech, "_trig_eval_np", factor)
    n_pts = 11
    fig, _ = plotting.plot_sphere(f, n_pts=n_pts)
    try:
        longitude, colatitude, values = evaluated[0]
        assert longitude.shape == colatitude.shape == values.shape == (n_pts, n_pts)
        np.testing.assert_array_equal(longitude[0], np.linspace(-np.pi, np.pi, n_pts))
        np.testing.assert_array_equal(colatitude[:, 0], np.linspace(0., np.pi, n_pts))
        expected = 2 + np.cos(colatitude) + .25 * np.sin(colatitude) * np.cos(longitude)
        np.testing.assert_allclose(values, expected, rtol=0,
                                   atol=512 * np.finfo(float).eps)
        # Resource-shape contract: no one-dimensional Horner factor receives
        # n_pts**2 repeated points. Actual factors and real artists are used.
        assert factor_point_counts and max(factor_point_counts) <= n_pts
        fig.canvas.draw()
    finally:
        plt.close(fig)
