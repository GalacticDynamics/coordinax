"""Jit a parameterized-chart transition through each path a `chart ==` branch guards.

Task 2 made `AbstractChart.__eq__`/`__hash__` tracer-safe, which fixes those
branches indirectly. These tests are the direct evidence: without them, a site
that compares charts by identity, set membership, dict lookup -- or that
compares chart *parameters* and feeds the answer to something that only accepts
a plain array -- would still fail under `jit` and nothing would catch it.

Every test jits a transition through a `ProlateSpheroidal3D` whose `Delta` is a
traced `unxt.Quantity`, and asserts a finite answer. The point is chosen inside
the chart's validity domain (``mu >= Delta**2``, ``|nu| <= Delta**2``), so a
`nan` is a real failure and not a domain violation.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.representations as cxr

DELTA = 2.0  # metres; Delta**2 == 4 m2
AT = {"mu": u.Q(6.0, "m2"), "nu": u.Q(2.0, "m2"), "phi": u.Angle(0.3, "rad")}
VEL = {"mu": u.Q(1.0, "m2/s"), "nu": u.Q(0.5, "m2/s"), "phi": u.Q(0.2, "rad/s")}
PHYS_VEL = {"mu": u.Q(1.0, "m/s"), "nu": u.Q(0.5, "m/s"), "phi": u.Q(0.2, "m/s")}


def _chart(delta_value):
    """A prolate chart whose `Delta` is dynamic -- one pytree leaf."""
    return cxc.ProlateSpheroidal3D(Delta=u.Q(delta_value, "m"))


def _tangent(chart):
    return cx.Tangent(data=VEL, chart=chart, basis=cxr.coord_basis, semantic=cxr.vel)


def test_the_chart_under_test_is_actually_parameterized():
    """Guard the guard: these tests are worthless if `Delta` is static."""
    assert len(jax.tree.leaves(_chart(DELTA))) == 1


# ---------------------------------------------------------------------------
# charts/register_ptmap.py


def test_jit_pt_map_out_of_parameterized_chart():
    """Exercises the generic `if from_chart == to_chart` in `register_ptmap`."""

    @jax.jit
    def f(d):
        return cxc.pt_map(AT, _chart(d), cxc.cart3d)["x"].ustrip("m")

    assert jnp.isfinite(f(DELTA))


def test_jit_pt_map_identity_same_parameterized_chart():
    """Exercises the prolate->prolate rule's same-`Delta` short-circuit.

    The rule branches on ``to_chart.Delta == from_chart.Delta`` under
    `jax.lax.cond`. With a dynamic `Delta` that comparison is a dimensionless
    *Quantity*, which `lax.cond` rejects outright as a predicate.
    """

    @jax.jit
    def f(d):
        c = _chart(d)
        return cxc.pt_map(AT, c, c)["mu"].ustrip("m2")

    assert jnp.allclose(f(DELTA), 6.0)


def test_jit_pt_map_between_two_focal_lengths():
    """Same rule, the other `lax.cond` branch: via cylindrical, `Delta` changes."""

    @jax.jit
    def f(d_from, d_to):
        out = cxc.pt_map(AT, _chart(d_from), _chart(d_to))
        return jnp.asarray([v.value for v in out.values()])

    assert jnp.all(jnp.isfinite(f(DELTA, 3.0)))


def test_jit_jac_pt_map_out_of_parameterized_chart():
    """The Jacobian route every tangent path is built on."""

    @jax.jit
    def f(d):
        return cxc.jac_pt_map(AT, _chart(d), cxc.cart3d).value

    assert jnp.all(jnp.isfinite(f(DELTA)))


# ---------------------------------------------------------------------------
# _src/euclidean/scale_factors.py


def test_jit_scale_factors_on_parameterized_chart():
    """Exercises `scale_factors`' `chart == cart_chart` short-circuit."""

    @jax.jit
    def f(d):
        return cxm.scale_factors(cxm.FlatMetric(3), _chart(d), at=AT).value

    assert jnp.all(jnp.isfinite(f(DELTA)))


# ---------------------------------------------------------------------------
# representations/_src/tangent_map.py -- all three `from_chart == to_chart`


def test_jit_tangent_map_coordinate_basis():
    """`tangent_map(..., coord_basis, ...)`, the first branch."""

    @jax.jit
    def f(d):
        out = cxr.tangent_map(VEL, _chart(d), cxr.coord_basis, cxc.cart3d, at=AT)
        return jnp.asarray([v.value for v in out.values()])

    assert jnp.all(jnp.isfinite(f(DELTA)))


def test_jit_tangent_map_physical_basis():
    """`tangent_map(..., phys_basis, ...)`, the second branch."""

    @jax.jit
    def f(d):
        out = cxr.tangent_map(PHYS_VEL, _chart(d), cxr.phys_basis, cxc.cart3d, at=AT)
        return jnp.asarray([v.value for v in out.values()])

    assert jnp.all(jnp.isfinite(f(DELTA)))


def test_jit_tangent_map_between_representations():
    """The rep-to-rep overload, whose branch picks the base point's chart."""

    @jax.jit
    def f(d):
        out = cxr.tangent_map(
            VEL, _chart(d), cxr.coord_vel, cxc.cart3d, cxr.phys_vel, at=AT
        )
        return jnp.asarray([v.value for v in out.values()])

    assert jnp.all(jnp.isfinite(f(DELTA)))


# ---------------------------------------------------------------------------
# vectors/_src/register_cx.py -- `from_chart != vec.chart`


def test_jit_cconvert_point_out_of_parameterized_chart():
    """Exercises `cconvert(Point, chart)`'s source-chart check."""

    @jax.jit
    def f(d):
        return cxr.cconvert(cx.Point(AT, chart=_chart(d)), cxc.cart3d)["x"].value

    assert jnp.isfinite(f(DELTA))


def test_jit_cconvert_tangent_out_of_parameterized_chart():
    """Exercises the `at`-chart check on the `Tangent` overload."""

    @jax.jit
    def f(d):
        c = _chart(d)
        out = cxr.cconvert(_tangent(c), cxc.cart3d, at=cx.Point(AT, chart=c))
        return out["x"].value

    assert jnp.isfinite(f(DELTA))


# ---------------------------------------------------------------------------
# vectors/_src/{base,register_quax,register_compare,register_separation}.py
# and representations/_src/core.py


def test_jit_tangent_addition_in_parameterized_chart():
    """`+` routes through `register_quax`'s `ambient_chart == original_chart`."""

    @jax.jit
    def f(d):
        c = _chart(d)
        return (_tangent(c) + _tangent(c))["mu"].value

    assert jnp.isfinite(f(DELTA))


def test_jit_point_difference_in_parameterized_chart():
    """`-` routes through `representations/core.py`'s ambient-Cartesian branch."""

    @jax.jit
    def f(d):
        c = _chart(d)
        return (cx.Point(AT, chart=c) - cx.Point(AT, chart=c))["mu"].value

    assert jnp.isfinite(f(DELTA))


def test_jit_vector_equality_in_parameterized_chart():
    """`AbstractVector.__eq__` compares charts before comparing data."""

    @jax.jit
    def f(d):
        c = _chart(d)
        return jnp.asarray(cx.Point(AT, chart=c) == cx.Point(AT, chart=c))

    assert f(DELTA)


def test_jit_equivalent_across_a_parameterized_chart():
    """`equivalent` compares charts to decide whether to convert first."""

    @jax.jit
    def f(d):
        c = _chart(d)
        p = cx.Point(AT, chart=c)
        return jnp.asarray(cx.equivalent(p, cxr.cconvert(p, cxc.cart3d)))

    assert f(DELTA)


def test_jit_separation_in_parameterized_chart():
    """`separation` compares the two operands' charts."""

    @jax.jit
    def f(d):
        c = _chart(d)
        return cx.separation(cx.Point(AT, chart=c), cx.Point(AT, chart=c)).value

    assert jnp.isfinite(f(DELTA))


# ---------------------------------------------------------------------------
# vectors/_src/bundle.py -- a dict *keyed by chart*, so `__hash__` must hold up


def test_jit_coordinate_bundle_keyed_by_parameterized_chart():
    """`Coordinate.cconvert` caches base points in a dict *keyed by chart*.

    This is the one path that needs both halves of Task 2 at once. The base
    point and the fibre are given two *distinct* prolate instances, so
    ``vec.chart == self.point.chart`` compares two same-type charts that both
    carry a traced `Delta` -- the only shape in which `__eq__` reaches its
    dynamic-field guard -- and the miss then hashes the chart to fill the cache.
    """

    @jax.jit
    def f(d):
        point_chart, fibre_chart = _chart(d), _chart(d)
        bundle = cx.Coordinate(
            point=cx.Point(AT, chart=point_chart), velocity=_tangent(fibre_chart)
        )
        return bundle.cconvert(cxc.cart3d)["velocity"]["x"].value

    assert jnp.isfinite(f(DELTA))


# ---------------------------------------------------------------------------
# The payoff: none of the above may cost differentiability in `Delta`.


# `AT` is in-domain for Delta**2 in [|nu|, mu] == [2, 6] m2, i.e. Delta in
# [1.42, 2.44] m. Outside that the map is genuinely `nan` and says nothing.
@pytest.mark.parametrize("d", [1.5, 2.0, 2.4])
def test_grad_through_delta_is_finite_and_nonzero(d):
    """If any branch above went static in `Delta`, the gradient would vanish."""

    def f(delta):
        return cxc.pt_map(AT, _chart(delta), cxc.cart3d)["x"].ustrip("m")

    g = jax.grad(f)(d)
    assert jnp.isfinite(g)
    assert g != 0.0
