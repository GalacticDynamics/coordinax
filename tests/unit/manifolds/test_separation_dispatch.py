"""Tests for the separation() manifold API dispatches."""

import jax.numpy as jnp

import quaxed.numpy as qnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.manifolds as cxm


class TestSeparationDispatches:
    """The manifold-level `separation` accepts several input forms."""

    def test_chart_and_cdicts(self):
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        d = cxm.separation(cxc.cart3d, a, b)
        assert isinstance(d, cx.Distance)
        assert bool(qnp.isclose(d.ustrip("m"), 5.0))

    def test_metric_chart_and_cdicts(self):
        metric = cxm.FlatMetric(3)
        a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        assert bool(
            qnp.isclose(cxm.separation(metric, cxc.cart3d, a, b).ustrip("m"), 5.0)
        )

    def test_chart_and_packed_quantities(self):
        a = u.Q([3.0, 0.0, 0.0], "m")
        b = u.Q([0.0, 4.0, 0.0], "m")
        assert bool(qnp.isclose(cxm.separation(cxc.cart3d, a, b).ustrip("m"), 5.0))

    def test_chart_and_bare_arrays(self):
        a = jnp.array([3.0, 0.0, 0.0])
        b = jnp.array([0.0, 4.0, 0.0])
        assert bool(qnp.isclose(cxm.separation(cxc.cart3d, a, b), 5.0))

    def test_all_forms_agree_with_the_point_overload(self):
        p = cx.Point.from_([3.0, 0.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0, 0.0], "m")
        ref = cx.separation(p, q).ustrip("m")
        packed = cxm.separation(
            cxc.cart3d, u.Q([3.0, 0.0, 0.0], "m"), u.Q([0.0, 4.0, 0.0], "m")
        )
        assert bool(qnp.isclose(packed.ustrip("m"), ref))

    def test_packed_quantity_is_unit_invariant(self):
        a = u.Q([3.0, 0.0, 0.0], "m")
        b = u.Q([0.0, 0.004, 0.0], "km")
        assert bool(qnp.isclose(cxm.separation(cxc.cart3d, a, b).ustrip("m"), 5.0))

    def test_batched_packed_quantities(self):
        a = u.Q([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "m")
        b = u.Q([[0.0, 4.0, 0.0], [0.0, 1.0, 0.0]], "m")
        d = cxm.separation(cxc.cart3d, a, b).ustrip("m")
        assert bool(qnp.isclose(d[0], 5.0))
        assert bool(qnp.isclose(d[1], qnp.sqrt(2.0)))
