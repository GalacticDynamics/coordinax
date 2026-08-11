"""`EmbeddedChart`'s embedding parameters are live, not hidden from JAX.

`AbstractEmbeddingMap` used to be `jax.tree_util.register_static`, so a
`TwoSphereIn3D(radius=u.Q(...))` inside an `EmbeddedChart` was a single opaque
non-array leaf: the chart reported *zero* leaves, `jit` baked the radius in as a
constant, a tracer walked straight out through the boundary, and `hash` blew up
on the `ArrayImpl` it could see but JAX could not.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.base.charts import AbstractParameterizedChart
from coordinax._src.embedded.embedmap import AbstractEmbeddingMap

P = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(0.4, "rad")}


def test_embedded_chart_is_on_the_parameterized_branch() -> None:
    assert issubclass(cxm.EmbeddedChart, AbstractParameterizedChart)


def test_a_non_pytree_embedding_map_is_rejected() -> None:
    """Forgetting `equinox.Module` is what reintroduces the opaque leaf."""
    with pytest.raises(TypeError, match=r"must subclass `equinox\.Module`"):

        class Sneaky(AbstractEmbeddingMap):  # type: ignore[type-arg]
            intrinsic = cxc.sph2
            ambient = cxc.sph3d

            def embed(self, point, /, *, usys=None):  # type: ignore[no-untyped-def]
                return point

            def project(self, point, /, *, usys=None):  # type: ignore[no-untyped-def]
                return point


def test_dynamic_radius_gives_exactly_one_leaf() -> None:
    em = cxm.TwoSphereIn3D(radius=u.Q(2.0, "km"))
    assert len(jtu.tree_leaves(em)) == 1
    assert len(jax.tree.leaves(cxm.EmbeddedChart(em))) == 1


def test_dynamic_radius_is_hashable() -> None:
    """The dynamic parameter is excluded from the hash, not fed to it."""
    chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    assert isinstance(hash(chart), int)


def test_static_radius_still_gives_no_leaves() -> None:
    """A `StaticQuantity` radius keeps the chart leaf-free, as before."""
    chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.StaticQuantity(2.0, "km")))
    assert jax.tree.leaves(chart) == []
    assert isinstance(hash(chart), int)


def test_a_tracer_cannot_escape_through_the_embed_map() -> None:
    """Return the chart *across* the `jit` boundary, not via a side channel.

    A static chart travels in the output treedef, so the radius would come back
    as the trace's own `DynamicJaxprTracer`. As a pytree leaf it is a real
    output and comes back concrete.
    """

    @jax.jit
    def make(radius_value: jax.Array) -> cxm.EmbeddedChart:
        return cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(radius_value, "km")))

    chart = make(jnp.asarray(2.0))
    leaves = jtu.tree_leaves(chart.embed_map.radius)
    assert leaves, "radius contributed no leaf"
    assert not any(isinstance(x, jax.core.Tracer) for x in leaves)
    assert jnp.allclose(u.ustrip("km", chart.embed_map.radius), 2.0)


def _z_of_chart(chart: cxm.EmbeddedChart) -> jax.Array:
    return cxc.pt_map(P, chart, cxc.cart3d)["z"].ustrip("km")


def _z_of_radius(radius_value: float) -> jax.Array:
    em = cxm.TwoSphereIn3D(radius=u.Q(radius_value, "km"))
    return _z_of_chart(cxm.EmbeddedChart(em))


def test_grad_wrt_radius_matches_finite_differences() -> None:
    """Differentiate w.r.t. the *chart*, which only works if radius is a leaf."""
    chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    analytic = float(jax.grad(_z_of_chart)(chart).embed_map.radius.ustrip("km"))
    h = 1e-4
    numeric = (_z_of_radius(2.0 + h) - _z_of_radius(2.0 - h)) / (2 * h)
    assert jnp.allclose(analytic, numeric, rtol=1e-4), (analytic, numeric)


def test_jit_retraces_once_across_radii() -> None:
    """The chart crosses the boundary, so the radius must not be part of the key."""
    traces = []

    @jax.jit
    def f(chart: cxm.EmbeddedChart) -> jax.Array:
        traces.append(1)
        return _z_of_chart(chart)

    f(cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km"))))
    f(cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(3.0, "km"))))
    assert len(traces) == 1, (
        f"retraced {len(traces)} times; radius is being treated as static"
    )
