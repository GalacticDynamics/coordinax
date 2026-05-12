"""Tests for Coordinate.

Per the spec, a Coordinate stores:
  - a base ``Point`` (PointGeometry),
  - a named collection of fibre ``Tangent``s (TangentGeometry),
and on construction automatically converts every fibre vector into the
reference frame of the base point.
"""

__all__: tuple[str, ...] = ()

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.manifolds as cxm
import coordinax.representations as cxr
from coordinax.vectors import Coordinate, Point, Tangent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point(
    *xyz: float, unit: str = "m", frame: cxf.AbstractReferenceFrame = cxf.noframe
) -> Point:
    """Cartesian position point."""
    p = cx.Point.from_(list(xyz), unit)
    return replace(p, frame=frame)


def _make_vel(
    *xyz: float, unit: str = "m/s", frame: cxf.AbstractReferenceFrame = cxf.noframe
) -> Tangent:
    """Cartesian velocity vector."""
    data = {"x": u.Q(xyz[0], unit), "y": u.Q(xyz[1], unit), "z": u.Q(xyz[2], unit)}
    v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_vel)
    return replace(v, frame=frame)


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Test basic construction and type validation."""

    def test_base_only(self) -> None:
        """Coordinate with only a base point, no field vectors."""
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert pv.point is base
        assert len(list(pv.keys())) == 0

    def test_base_with_one_field(self) -> None:
        """Coordinate with base point + one field vector."""
        base = _make_point(1, 2, 3)
        vel = _make_vel(10, 20, 30)
        pv = Coordinate(point=base, velocity=vel)
        assert pv.point is base
        assert "velocity" in list(pv.keys())

    def test_base_with_multiple_fields(self) -> None:
        """Coordinate with base + two field vectors."""
        base = _make_point(1, 2, 3)
        vel = _make_vel(10, 20, 30)
        acc = cx.Tangent.from_(
            {"x": u.Q(0.1, "m/s^2"), "y": u.Q(0.2, "m/s^2"), "z": u.Q(0.3, "m/s^2")},
            cxc.cart3d,
            cxr.coord_acc,
        )
        pv = Coordinate(point=base, velocity=vel, acceleration=acc)
        assert "velocity" in list(pv.keys())
        assert "acceleration" in list(pv.keys())

    def test_point_must_be_Point(self) -> None:
        """Passing a bare Tangent (not Point) as point must raise TypeError."""
        vel = _make_vel(1, 2, 3)  # this is a Tangent, not a Point
        with pytest.raises(TypeError, match="point must be a Point"):
            Coordinate(point=vel)

    def test_field_must_not_be_Point(self) -> None:
        """Passing a Point as a field vector must raise TypeError."""
        base = _make_point(1, 2, 3)
        another_pos = _make_point(4, 5, 6)  # also a Point (PointGeometry)
        with pytest.raises(TypeError, match="must be a Tangent"):
            Coordinate(point=base, second_pos=another_pos)


# ---------------------------------------------------------------------------
# TestFrameAlignment
# ---------------------------------------------------------------------------


class TestFrameAlignment:
    """Test automatic frame alignment of field vectors on construction."""

    def test_noframe_noframe_no_conversion(self) -> None:
        """Field with noframe when point has noframe: no conversion (identity)."""
        base = _make_point(1, 0, 0)  # noframe
        vel = _make_vel(1, 0, 0)  # noframe
        pv = Coordinate(point=base, velocity=vel)
        # Same frame — no conversion, same object returned
        assert pv["velocity"] is vel

    def test_same_real_frame_no_conversion(self) -> None:
        """Field with same real frame as point: no conversion needed."""
        base = _make_point(1, 0, 0, frame=cxf.alice)
        vel = _make_vel(1, 0, 0, frame=cxf.alice)
        pv = Coordinate(point=base, velocity=vel)
        assert pv["velocity"].frame is cxf.alice

    def test_different_frame_converts(self) -> None:
        """Field in different frame is converted to point's frame."""
        base = _make_point(1, 0, 0, frame=cxf.alice)
        vel = _make_vel(1, 0, 0, frame=cxf.alex)  # alex != alice
        pv = Coordinate(point=base, velocity=vel)
        # After construction, velocity must be in alice's frame
        assert pv["velocity"].frame is cxf.alice

    def test_frame_property_is_point_frame(self) -> None:
        """pv.frame is always equal to point.frame."""
        base = _make_point(1, 0, 0, frame=cxf.alice)
        pv = Coordinate(point=base)
        assert pv.frame is cxf.alice

    def test_multiple_fields_all_converted(self) -> None:
        """All field vectors are converted to point's frame."""
        base = _make_point(1, 0, 0, frame=cxf.alice)
        vel = _make_vel(1, 0, 0, frame=cxf.alex)
        acc_data = {
            "x": u.Q(0.1, "m/s^2"),
            "y": u.Q(0.0, "m/s^2"),
            "z": u.Q(0.0, "m/s^2"),
        }
        acc = replace(
            cx.Tangent.from_(acc_data, cxc.cart3d, cxr.coord_acc),
            frame=cxf.alex,
        )
        pv = Coordinate(point=base, velocity=vel, acceleration=acc)
        assert pv["velocity"].frame is cxf.alice
        assert pv["acceleration"].frame is cxf.alice


# ---------------------------------------------------------------------------
# TestMappingAPI
# ---------------------------------------------------------------------------


class TestMappingAPI:
    """Test Mapping-like interface (fields only, not base point)."""

    @pytest.fixture
    def pv(self) -> Coordinate:
        base = _make_point(1, 2, 3)
        vel = _make_vel(10, 20, 30)
        return Coordinate(point=base, velocity=vel)

    def test_getitem_field_by_name(self, pv: Coordinate) -> None:
        assert isinstance(pv["velocity"], Tangent)

    def test_getitem_unknown_key_raises(self, pv: Coordinate) -> None:
        with pytest.raises(KeyError):
            _ = pv["nonexistent"]

    def test_keys_contains_fields(self, pv: Coordinate) -> None:
        assert "velocity" in list(pv.keys())

    def test_keys_excludes_point(self, pv: Coordinate) -> None:
        assert "point" not in list(pv.keys())

    def test_values(self, pv: Coordinate) -> None:
        vals = list(pv.values())
        assert len(vals) == 1
        assert isinstance(vals[0], Tangent)

    def test_items(self, pv: Coordinate) -> None:
        items = list(pv.items())
        assert len(items) == 1
        name, vec = items[0]
        assert name == "velocity"
        assert isinstance(vec, Tangent)

    def test_len_counts_fields_only(self, pv: Coordinate) -> None:
        assert len(pv) == 1  # one field vector, not counting point

    def test_iter_yields_field_names(self, pv: Coordinate) -> None:
        assert list(iter(pv)) == ["velocity"]


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test properties that delegate to the base point."""

    def test_chart_delegates_to_point(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert pv.chart == cxc.cart3d

    def test_rep_delegates_to_point(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert pv.rep == cxr.point

    def test_manifold_delegates_to_point(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert isinstance(pv.manifold, cxm.AbstractManifold)

    def test_frame_delegates_to_point(self) -> None:
        base = _make_point(1, 2, 3, frame=cxf.alice)
        pv = Coordinate(point=base)
        assert pv.frame is cxf.alice

    def test_shape_scalar(self) -> None:
        base = _make_point(1, 2, 3)
        vel = _make_vel(10, 20, 30)
        pv = Coordinate(point=base, velocity=vel)
        assert pv.shape == ()

    def test_shape_batched_base(self) -> None:
        base = cx.Point.from_(jnp.ones((2, 3)), "m")
        pv = Coordinate(point=base)
        assert pv.shape == (2,)

    def test_shape_broadcast(self) -> None:
        """Batched base (2,) and scalar velocity () broadcast to (2,)."""
        base = cx.Point.from_(jnp.ones((2, 3)), "m")
        vel = _make_vel(10, 20, 30)  # scalar shape ()
        pv = Coordinate(point=base, velocity=vel)
        assert pv.shape == (2,)


# ---------------------------------------------------------------------------
# TestCconvert
# ---------------------------------------------------------------------------


class TestCconvert:
    """Test chart conversion of Coordinate."""

    def test_base_only_cart_to_sph(self) -> None:
        """Base-only bundle converts base as a point map."""
        base = _make_point(1, 0, 0)
        pv = Coordinate(point=base)
        pv_sph = pv.cconvert(cxc.sph3d)
        assert pv_sph.point.chart == cxc.sph3d

    def test_base_cart_to_sph_values(self) -> None:
        """[1,0,0] in Cartesian → [r=1, theta=π/2, phi=0] in Spherical."""
        base = _make_point(1.0, 0.0, 0.0)
        pv = Coordinate(point=base)
        pv_sph = pv.cconvert(cxc.sph3d)
        assert jnp.allclose(pv_sph.point["r"].value, jnp.array(1.0), atol=1e-6)
        assert jnp.allclose(
            pv_sph.point["theta"].value, jnp.array(jnp.pi / 2), atol=1e-6
        )
        assert jnp.allclose(pv_sph.point["phi"].value, jnp.array(0.0), atol=1e-6)

    def test_velocity_uses_tangent_map(self) -> None:
        """Velocity at [1,0,0] in Cartesian converts via Jacobian to Spherical."""
        base = _make_point(1.0, 0.0, 0.0)
        vel = _make_vel(1.0, 0.0, 0.0)  # radial velocity in Cartesian
        pv = Coordinate(point=base, velocity=vel)
        pv_sph = pv.cconvert(cxc.sph3d)
        assert pv_sph.point.chart == cxc.sph3d
        assert pv_sph["velocity"].chart == cxc.sph3d
        # Radial velocity [1,0,0] at [1,0,0] → [vr=1, vθ=0, vφ=0] in spherical
        assert jnp.allclose(pv_sph["velocity"]["r"].value, jnp.array(1.0), atol=1e-6)

    def test_round_trip(self) -> None:
        """Cart → sph → cart should recover original base values."""
        base = _make_point(1.0, 2.0, 0.0)
        pv = Coordinate(point=base)
        pv_rt = pv.cconvert(cxc.sph3d).cconvert(cxc.cart3d)
        assert jnp.allclose(pv_rt.point["x"].value, base["x"].value, atol=1e-6)
        assert jnp.allclose(pv_rt.point["y"].value, base["y"].value, atol=1e-6)

    def test_identity_same_chart(self) -> None:
        """Converting to the same chart returns equivalent values."""
        base = _make_point(1.0, 2.0, 3.0)
        vel = _make_vel(4.0, 5.0, 6.0)
        pv = Coordinate(point=base, velocity=vel)
        pv_same = pv.cconvert(cxc.cart3d)
        assert jnp.allclose(pv_same.point["x"].value, base["x"].value, atol=1e-6)
        assert jnp.allclose(pv_same["velocity"]["x"].value, vel["x"].value, atol=1e-6)

    def test_pv_method_agrees_with_cx_function(self) -> None:
        """pv.cconvert(chart) and cx.cconvert(pv, chart) agree."""
        base = _make_point(1.0, 0.0, 0.0)
        pv = Coordinate(point=base)
        via_method = pv.cconvert(cxc.sph3d)
        via_func = cx.cconvert(pv, cxc.sph3d)
        assert via_method.point.chart == via_func.point.chart


# ---------------------------------------------------------------------------
# TestFromDispatches
# ---------------------------------------------------------------------------


class TestFromDispatches:
    """Test Coordinate.from_ constructor dispatches."""

    def test_from_pointed_vector_is_identity(self) -> None:
        """Coordinate.from_(pv) returns the same object."""
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        pv2 = Coordinate.from_(pv)
        assert pv2 is pv

    def test_from_point_wraps(self) -> None:
        """Coordinate.from_(Point) wraps it as a point-only bundle."""
        base = _make_point(1, 2, 3)
        pv = Coordinate.from_(base)
        assert isinstance(pv, Coordinate)
        assert pv.point is base
        assert len(list(pv.keys())) == 0

    def test_from_mapping_with_point_key(self) -> None:
        """Coordinate.from_(mapping) detects 'point' key for base."""
        base = _make_point(1, 2, 3)
        vel = _make_vel(4, 5, 6)
        pv = Coordinate.from_({"point": base, "velocity": vel})
        assert isinstance(pv, Coordinate)
        assert "velocity" in list(pv.keys())

    def test_from_mapping_explicit_point_kwarg(self) -> None:
        """Coordinate.from_(mapping, point=...) uses explicit base."""
        base = _make_point(1, 2, 3)
        vel = _make_vel(4, 5, 6)
        pv = Coordinate.from_({"velocity": vel}, point=base)
        assert pv.point is base


# ---------------------------------------------------------------------------
# TestJAXCompat
# ---------------------------------------------------------------------------


class TestJAXCompat:
    """Test JAX compatibility: pytree, jit, batch indexing."""

    def test_pytree_flatten_unflatten(self) -> None:
        """Pytree round-trip preserves structure."""
        base = _make_point(1.0, 2.0, 3.0)
        vel = _make_vel(4.0, 5.0, 6.0)
        pv = Coordinate(point=base, velocity=vel)
        leaves, treedef = jax.tree.flatten(pv)
        pv2 = jax.tree.unflatten(treedef, leaves)
        assert pv2.point.chart == pv.point.chart
        assert "velocity" in list(pv2.keys())

    def test_jit_cconvert(self) -> None:
        """Cconvert works under jax.jit."""
        base = _make_point(1.0, 0.0, 0.0)
        pv = Coordinate(point=base)

        @jax.jit
        def convert(p: Coordinate) -> Coordinate:
            return p.cconvert(cxc.sph3d)

        result = convert(pv)
        assert result.point.chart == cxc.sph3d

    def test_batch_indexing(self) -> None:
        """pv[0] returns a Coordinate sliced along the batch axis."""
        base = cx.Point.from_(jnp.ones((2, 3)), "m")
        vel_data = {
            "x": u.Q(jnp.ones((2,)), "m/s"),
            "y": u.Q(jnp.ones((2,)), "m/s"),
            "z": u.Q(jnp.ones((2,)), "m/s"),
        }
        vel = cx.Tangent.from_(vel_data, cxc.cart3d, cxr.coord_vel)
        pv = Coordinate(point=base, velocity=vel)
        pv0 = pv[0]
        assert isinstance(pv0, Coordinate)
        assert pv0.shape == ()

    def test_vmap_cconvert(self) -> None:
        """Vmap over a Coordinate converts correctly."""
        base = cx.Point.from_(jnp.ones((2, 3)), "m")
        vel_data = {
            "x": u.Q(jnp.ones((2,)), "m/s"),
            "y": u.Q(jnp.zeros((2,)), "m/s"),
            "z": u.Q(jnp.zeros((2,)), "m/s"),
        }
        vel = cx.Tangent.from_(vel_data, cxc.cart3d, cxr.coord_vel)
        pv = Coordinate(point=base, velocity=vel)

        def convert_one(p: Coordinate) -> Coordinate:
            return p.cconvert(cxc.sph3d)

        result = jax.vmap(convert_one)(pv)
        assert result.point.chart == cxc.sph3d
        assert result["velocity"].chart == cxc.sph3d

    def test_jit_with_fields(self) -> None:
        """Jit with velocity field runs correctly."""
        base = _make_point(1.0, 0.0, 0.0)
        vel = _make_vel(1.0, 0.0, 0.0)
        pv = Coordinate(point=base, velocity=vel)

        @jax.jit
        def convert(p: Coordinate) -> Coordinate:
            return p.cconvert(cxc.sph3d)

        result = convert(pv)
        assert result.point.chart == cxc.sph3d
        assert result["velocity"].chart == cxc.sph3d


# ---------------------------------------------------------------------------
# TestFieldCharts
# ---------------------------------------------------------------------------


class TestFieldCharts:
    """Test field_charts kwarg in cconvert."""

    def test_field_charts_override(self) -> None:
        """field_charts allows velocity to convert to a different chart than base."""
        base = _make_point(1.0, 0.0, 0.0)
        vel = _make_vel(1.0, 0.0, 0.0)
        pv = Coordinate(point=base, velocity=vel)
        pv_mixed = pv.cconvert(cxc.sph3d, field_charts={"velocity": cxc.cart3d})
        assert pv_mixed.point.chart == cxc.sph3d
        assert pv_mixed["velocity"].chart == cxc.cart3d

    def test_field_charts_empty_uses_default(self) -> None:
        """Empty field_charts dict behaves as default (all fields use to_chart)."""
        base = _make_point(1.0, 0.0, 0.0)
        vel = _make_vel(1.0, 0.0, 0.0)
        pv = Coordinate(point=base, velocity=vel)
        pv_a = pv.cconvert(cxc.sph3d, field_charts={})
        pv_b = pv.cconvert(cxc.sph3d)
        assert pv_a["velocity"].chart == pv_b["velocity"].chart


# ---------------------------------------------------------------------------
# TestFromErrors
# ---------------------------------------------------------------------------


class TestFromErrors:
    """Test Coordinate.from_ error cases."""

    def test_from_mapping_no_point_raises(self) -> None:
        """from_(mapping) without a 'point' key or kwarg raises ValueError."""
        vel = _make_vel(1, 2, 3)
        with pytest.raises(ValueError, match="point"):
            Coordinate.from_({"velocity": vel})

    def test_from_mapping_explicit_point_kwarg_takes_precedence(self) -> None:
        """Explicit point= kwarg wins over 'point' key in the mapping."""
        base1 = _make_point(1, 2, 3)
        base2 = _make_point(4, 5, 6)
        vel = _make_vel(1, 0, 0)
        pv = Coordinate.from_({"point": base1, "velocity": vel}, point=base2)
        assert pv.point is base2


# ---------------------------------------------------------------------------
# TestReprStr
# ---------------------------------------------------------------------------


class TestReprStr:
    """Test __repr__ and __str__ output."""

    def test_repr_contains_coordinate(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert "Coordinate" in repr(pv)

    def test_repr_contains_point(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert "Point" in repr(pv)

    def test_str_nonempty(self) -> None:
        base = _make_point(1, 2, 3)
        vel = _make_vel(4, 5, 6)
        pv = Coordinate(point=base, velocity=vel)
        assert str(pv)

    def test_repr_with_fields_contains_field_name(self) -> None:
        base = _make_point(1, 2, 3)
        vel = _make_vel(4, 5, 6)
        pv = Coordinate(point=base, velocity=vel)
        assert "velocity" in repr(pv)


# ---------------------------------------------------------------------------
# TestManifoldProperty
# ---------------------------------------------------------------------------


class TestManifoldProperty:
    """Test Coordinate.manifold delegates to point.M."""

    def test_manifold_is_abstract_manifold(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert isinstance(pv.manifold, cxm.AbstractManifold)

    def test_manifold_matches_point_M(self) -> None:
        base = _make_point(1, 2, 3)
        pv = Coordinate(point=base)
        assert pv.manifold == base.M
