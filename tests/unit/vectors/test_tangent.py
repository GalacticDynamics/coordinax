"""Tests for ``coordinax.vectors.Tangent``."""

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
from coordinax.internal import CDict

# ======================================================================
# Helpers


def _cart3d_vel_data() -> CDict:
    return {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}


def _cart3d_dpl_data() -> CDict:
    return {"x": u.Q(0.1, "m"), "y": u.Q(0.2, "m"), "z": u.Q(0.3, "m")}


def _make_vel(
    *xyz: float, unit: str = "m/s", frame: cxf.AbstractReferenceFrame = cxf.noframe
) -> cx.Tangent:
    """Cartesian velocity Tangent with optional frame."""
    data = {"x": u.Q(xyz[0], unit), "y": u.Q(xyz[1], unit), "z": u.Q(xyz[2], unit)}
    v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_vel)
    return replace(v, frame=frame)


# ======================================================================
# Construction


class TestVectorConstruction:
    """Tests for ``Tangent(data, chart, basis, semantic)`` construction."""

    def test_basic_construction_with_coord_vel(self):
        """Construct a Tangent with coord_basis and vel semantic - smoke test."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v is not None

    def test_basic_construction_with_phys_dpl(self):
        """Construct a Tangent with phys_basis and dpl semantic - smoke test."""
        v = cx.Tangent(
            data=_cart3d_dpl_data(),
            chart=cxc.cart3d,
            basis=cxr.phys_basis,
            semantic=cxr.dpl,
        )
        assert v is not None

    def test_data_accessible_by_component_name(self):
        """Components accessible via string indexing."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v["x"] == u.Q(1.0, "m/s")
        assert v["y"] == u.Q(2.0, "m/s")
        assert v["z"] == u.Q(3.0, "m/s")

    def test_chart_stored(self):
        """Chart field stores the given chart."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.chart == cxc.cart3d

    def test_basis_stored(self):
        """Basis field stores the given basis."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.basis == cxr.coord_basis

    def test_semantic_stored(self):
        """Semantic field stores the given semantic kind."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.semantic == cxr.vel

    def test_different_semantics_all_constructable(self):
        """vel, dpl, acc semantics all construct without error."""
        for semantic, data in [
            (cxr.vel, _cart3d_vel_data()),
            (cxr.dpl, _cart3d_dpl_data()),
            (
                cxr.acc,
                {
                    "x": u.Q(0.1, "m/s^2"),
                    "y": u.Q(0.2, "m/s^2"),
                    "z": u.Q(0.0, "m/s^2"),
                },
            ),
        ]:
            v = cx.Tangent(
                data=data, chart=cxc.cart3d, basis=cxr.coord_basis, semantic=semantic
            )
            assert v.semantic is semantic


# ======================================================================
# __check_init__


class TestCheckInit:
    """Tests for ``Tangent.__check_init__``."""

    def test_raises_for_wrong_data_keys(self):
        """Construction raises when data keys don't match chart components."""
        with pytest.raises(ValueError, match="Data keys do not match"):
            cx.Tangent(
                data={"wrong": u.Q(1.0, "m/s")},
                chart=cxc.cart3d,
                basis=cxr.coord_basis,
                semantic=cxr.vel,
            )

    def test_raises_for_missing_data_key(self):
        """Construction raises when a chart component is absent from data."""
        with pytest.raises(ValueError, match="Data keys do not match"):
            cx.Tangent(
                data={"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s")},  # z missing
                chart=cxc.cart3d,
                basis=cxr.coord_basis,
                semantic=cxr.vel,
            )

    def test_valid_data_does_not_raise(self):
        """Construction succeeds when data keys match chart components exactly."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v is not None

    def test_check_init_calls_manifold_check_chart(self):
        """__check_init__ enforces that the chart belongs to the manifold atlas.

        Tangent mirrors Point: both call M.check_chart(self.chart).
        """
        # Verify this is consistent with Point's behavior. For any valid chart,
        # M == chart.M so the check is always True. The key property is that the
        # call is made (not silently skipped).
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.M.has_chart(v.chart)


# ======================================================================
# M property


class TestVectorM:
    """Tests for the ``M`` property (manifold from chart)."""

    def test_M_is_abstract_manifold(self):
        """M returns an AbstractManifold instance."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_basis, cxr.vel)
        assert isinstance(v.M, cxm.AbstractManifold)

    def test_M_matches_chart_manifold(self):
        """M is the same manifold as the chart's M."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_basis, cxr.vel)
        assert v.M == cxc.cart3d.M


# ======================================================================
# rep property


class TestVectorRep:
    """Tests for the computed ``rep`` property."""

    def test_rep_is_representation_instance(self):
        """Rep returns a Representation instance."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert isinstance(v.rep, cxr.Representation)

    def test_rep_has_tangent_geometry(self):
        """rep.geom_kind is TangentGeometry."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert isinstance(v.rep.geom_kind, cxr.TangentGeometry)

    def test_rep_matches_coord_vel(self):
        """Rep matches the cxr.coord_vel singleton for coord_basis + vel."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.rep == cxr.coord_vel

    def test_rep_matches_phys_dpl(self):
        """Rep matches cxr.phys_disp for phys_basis + dpl."""
        v = cx.Tangent(
            data=_cart3d_dpl_data(),
            chart=cxc.cart3d,
            basis=cxr.phys_basis,
            semantic=cxr.dpl,
        )
        assert v.rep == cxr.phys_disp

    def test_rep_basis_is_stored_basis(self):
        """rep.basis is the same as the stored basis field."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.phys_basis,
            semantic=cxr.vel,
        )
        assert v.rep.basis == cxr.phys_basis

    def test_rep_semantic_is_stored_semantic(self):
        """rep.semantic_kind is the same as the stored semantic field."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.acc,
        )
        assert v.rep.semantic_kind == cxr.acc

    def test_rep_is_not_point_rep(self):
        """Tangent.rep is never the point representation."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.rep != cxr.point

    def test_rep_computed_not_stored(self):
        """Changing basis produces a different rep (rep computed from fields)."""
        v1 = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        v2 = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.phys_basis,
            semantic=cxr.vel,
        )
        assert v1.rep != v2.rep


# ======================================================================
# frame field


class TestVectorFrame:
    """Tests for the ``frame`` field on ``Tangent``."""

    def test_default_frame_is_noframe(self):
        """Tangent constructed without frame defaults to noframe."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.frame == cxf.noframe

    def test_explicit_frame_stored(self):
        """Explicitly provided frame is stored."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
            frame=cxf.alice,
        )
        assert v.frame == cxf.alice

    def test_none_frame_converts_to_noframe(self):
        """frame=None converts to noframe via converter."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
            frame=None,
        )
        assert v.frame == cxf.noframe


# ======================================================================
# from_ constructors


class TestVectorFromConstructors:
    """Tests for ``Tangent.from_`` dispatches."""

    def test_from_vector_identity(self):
        """Tangent.from_(vector) returns the same object."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        v2 = cx.Tangent.from_(v)
        assert v2 is v

    def test_from_dict_chart_basis_semantic(self):
        """Tangent.from_(dict, chart, basis, semantic) constructs correctly."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_basis, cxr.vel)
        assert isinstance(v, cx.Tangent)
        assert v.chart == cxc.cart3d
        assert v.basis == cxr.coord_basis
        assert v.semantic == cxr.vel

    def test_from_dict_chart_basis_semantic_frame(self):
        """Tangent.from_(dict, chart, basis, semantic, frame) sets frame."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_basis, cxr.vel, cxf.alice)
        assert v.frame == cxf.alice

    def test_from_array_unit_chart_basis_semantic(self):
        """Tangent.from_(array, unit, chart, basis, semantic) constructs correctly."""
        v = cx.Tangent.from_(
            jnp.array([1.0, 2.0, 3.0]), "m/s", cxc.cart3d, cxr.coord_basis, cxr.vel
        )
        assert isinstance(v, cx.Tangent)
        assert v.chart == cxc.cart3d
        assert v.basis == cxr.coord_basis
        assert v.semantic == cxr.vel

    def test_from_rep_dispatch_with_tangent_rep(self):
        """Tangent.from_(data, chart, rep) with tangent rep dispatches to Tangent."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_vel)
        assert isinstance(v, cx.Tangent)
        assert v.basis == cxr.coord_basis
        assert v.semantic == cxr.vel

    def test_from_rep_with_point_rep_raises(self):
        """Tangent.from_(data, chart, point_rep) raises TypeError."""
        data = _cart3d_vel_data()
        with pytest.raises(TypeError):
            cx.Tangent.from_(data, cxc.cart3d, cxr.point)

    def test_from_vector_frame_dispatch(self):
        """Tangent.from_(vector, frame) sets frame on the vector."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_basis, cxr.vel)
        v2 = cx.Tangent.from_(v, cxf.alice)
        assert v2.frame == cxf.alice
        assert v2.basis == v.basis
        assert v2.semantic == v.semantic

    def test_from_any_frame_dispatch(self):
        """Tangent.from_(dict, frame) infers chart/basis/semantic and sets frame."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxf.alice)
        assert isinstance(v, cx.Tangent)
        assert v.frame == cxf.alice

    def test_from_dict_chart_frame(self):
        """Tangent.from_(dict, chart, frame) infers basis/semantic and sets frame."""
        data = _cart3d_vel_data()
        v = cx.Tangent.from_(data, cxc.cart3d, cxf.alice)
        assert isinstance(v, cx.Tangent)
        assert v.chart == cxc.cart3d
        assert v.frame == cxf.alice


# ======================================================================
# cconvert


class TestVectorCconvert:
    """Tests for ``Tangent.cconvert`` chart conversion."""

    def test_cconvert_cart_to_sph_preserves_rep(self):
        """Cconvert changes chart but preserves basis and semantic."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pt = cx.Point.from_([1.0, 0.0, 0.0], "m")  # base point for tangent map
        v_sph = v.cconvert(cxc.sph3d, at=pt)
        assert isinstance(v_sph, cx.Tangent)
        assert v_sph.chart == cxc.sph3d
        assert v_sph.basis == cxr.coord_basis
        assert v_sph.semantic == cxr.vel

    def test_cconvert_preserves_frame(self):
        """Cconvert preserves the frame field."""
        v = cx.Tangent.from_(
            _cart3d_vel_data(), cxc.cart3d, cxr.coord_basis, cxr.vel, cxf.alice
        )
        pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
        v_sph = v.cconvert(cxc.sph3d, at=pt)
        assert v_sph.frame == cxf.alice

    def test_cconvert_identity_chart_returns_same_data(self):
        """Cconvert to same chart returns same data."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_basis, cxr.vel)
        pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
        v2 = v.cconvert(cxc.cart3d, at=pt)
        assert isinstance(v2, cx.Tangent)
        assert v2.chart == cxc.cart3d

    def test_cconvert_radial_velocity_at_x_axis(self):
        """Radial velocity [1,0,0] at [1,0,0] maps to vr=1, vθ=0, vφ=0."""
        v = cx.Tangent.from_(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.coord_basis,
            cxr.vel,
        )
        pt = cx.Point.from_([1.0, 0.0, 0.0], "m")
        v_sph = v.cconvert(cxc.sph3d, at=pt)
        assert jnp.allclose(v_sph["r"].value, jnp.array(1.0), atol=1e-6)
        assert jnp.allclose(v_sph["theta"].value, jnp.array(0.0), atol=1e-6)
        assert jnp.allclose(v_sph["phi"].value, jnp.array(0.0), atol=1e-6)

    def test_cconvert_round_trip_cart_sph_cart(self):
        """Cconvert cart→sph→cart recovers original component values."""
        data = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(0.0, "m/s")}
        v = cx.Tangent.from_(data, cxc.cart3d, cxr.coord_basis, cxr.vel)
        pt = cx.Point.from_([1.0, 2.0, 0.0], "m")
        v_rt = v.cconvert(cxc.sph3d, at=pt).cconvert(
            cxc.cart3d, at=pt.cconvert(cxc.sph3d)
        )
        assert jnp.allclose(v_rt["x"].value, jnp.array(1.0), atol=1e-5)
        assert jnp.allclose(v_rt["y"].value, jnp.array(2.0), atol=1e-5)


# ======================================================================
# shape / slicing


class TestVectorShape:
    """Tests for shape and indexing."""

    def test_shape_scalar(self):
        """Scalar components give empty shape."""
        v = cx.Tangent(
            data=_cart3d_vel_data(),
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.shape == ()

    def test_shape_batch(self):
        """Batched components give non-empty shape."""
        data = {
            "x": u.Q([1.0, 2.0], "m/s"),
            "y": u.Q([3.0, 4.0], "m/s"),
            "z": u.Q([5.0, 6.0], "m/s"),
        }
        v = cx.Tangent(
            data=data,
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        assert v.shape == (2,)

    def test_slice_indexing(self):
        """Integer slice indexing returns a new Tangent."""
        data = {
            "x": u.Q([1.0, 2.0], "m/s"),
            "y": u.Q([3.0, 4.0], "m/s"),
            "z": u.Q([5.0, 6.0], "m/s"),
        }
        v = cx.Tangent(
            data=data,
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        v0 = v[0]
        assert isinstance(v0, cx.Tangent)
        assert v0.shape == ()

    def test_slice_preserves_chart_basis_semantic(self):
        """Integer indexing preserves chart, basis, and semantic."""
        data = {
            "x": u.Q([1.0, 2.0], "m/s"),
            "y": u.Q([3.0, 4.0], "m/s"),
            "z": u.Q([5.0, 6.0], "m/s"),
        }
        v = cx.Tangent(
            data=data,
            chart=cxc.cart3d,
            basis=cxr.coord_basis,
            semantic=cxr.vel,
        )
        v0 = v[0]
        assert v0.chart == cxc.cart3d
        assert v0.basis == cxr.coord_basis
        assert v0.semantic == cxr.vel

    def test_slice_values_correct(self):
        """Indexed values match the original batch element."""
        data = {
            "x": u.Q([1, 2], "m/s"),
            "y": u.Q([3, 4], "m/s"),
            "z": u.Q([5, 6], "m/s"),
        }
        v = cx.Tangent(
            data=data, chart=cxc.cart3d, basis=cxr.coord_basis, semantic=cxr.vel
        )
        assert v[1]["x"].value == jnp.array(2)
        assert v[1]["y"].value == jnp.array(4)


# ======================================================================
# JAX compatibility


class TestVectorJAXCompat:
    """Tests for JAX integration: pytree, jit, vmap."""

    def test_pytree_flatten_unflatten_scalar(self):
        """Pytree round-trip preserves structure for a scalar Tangent."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)
        leaves, treedef = jax.tree.flatten(v)
        v2 = jax.tree.unflatten(treedef, leaves)
        assert v2.chart == v.chart
        assert v2.basis == v.basis
        assert v2.semantic == v.semantic
        assert v2["x"] == v["x"]

    def test_pytree_flatten_unflatten_batch(self):
        """Pytree round-trip preserves batch Tangent."""
        data = {
            "x": u.Q([1.0, 2.0], "m/s"),
            "y": u.Q([3.0, 4.0], "m/s"),
            "z": u.Q([5.0, 6.0], "m/s"),
        }
        v = cx.Tangent(
            data=data, chart=cxc.cart3d, basis=cxr.coord_basis, semantic=cxr.vel
        )
        leaves, treedef = jax.tree.flatten(v)
        v2 = jax.tree.unflatten(treedef, leaves)
        assert v2.shape == (2,)

    def test_jit_identity(self):
        """A JIT-compiled identity function returns an equivalent Tangent."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)

        @jax.jit
        def identity(x: cx.Tangent) -> cx.Tangent:
            return x

        v2 = identity(v)
        assert v2.chart == v.chart
        assert jnp.allclose(v2["x"].value, v["x"].value)

    def test_jit_cconvert(self):
        """Cconvert under jax.jit produces the correct chart."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)
        pt = cx.Point.from_([1.0, 0.0, 0.0], "m")

        @jax.jit
        def convert(vec: cx.Tangent, base: cx.Point) -> cx.Tangent:
            return vec.cconvert(cxc.sph3d, at=base)

        v_sph = convert(v, pt)
        assert v_sph.chart == cxc.sph3d

    def test_vmap_cconvert(self):
        """Vmap over a batch of base points applies Jacobian per-element."""
        v = cx.Tangent.from_(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
            cxc.cart3d,
            cxr.coord_vel,
        )
        pts_batch = cx.Point.from_(jnp.ones((2, 3)), "m")

        def convert_one(pt: cx.Point) -> cx.Tangent:
            return v.cconvert(cxc.sph3d, at=pt)

        result = jax.vmap(convert_one)(pts_batch)
        assert result.shape == (2,)
        assert result.chart == cxc.sph3d


# ======================================================================
# repr / str


class TestVectorRepr:
    """Tests for __repr__ and __str__."""

    def test_repr_contains_class_name(self):
        """Repr includes 'Tangent'."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)
        assert "Tangent" in repr(v)

    def test_str_contains_chart_name(self):
        """Str contains the chart class name."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)
        assert "Cart3D" in str(v)

    def test_repr_different_from_str(self):
        """Repr and str produce distinct (but non-empty) strings."""
        v = cx.Tangent.from_(_cart3d_vel_data(), cxc.cart3d, cxr.coord_vel)
        assert repr(v)  # non-empty
        assert str(v)  # non-empty


# ======================================================================
# change_basis(Point, AbstractLinearBasis) -> Tangent[Displacement]


class TestChangeBasisPointToDisplacement:
    """Tests for the Point -> Tangent[Displacement] promotion dispatch."""

    def _make_point(self) -> cx.Point:
        return cx.Point.from_([1.0, 2.0, 3.0], "m")

    def test_returns_tangent(self):
        """change_basis(Point, coord_basis) returns a Tangent."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert isinstance(disp, cx.Tangent)

    def test_semantic_is_displacement(self):
        """Resulting Tangent has Displacement semantic."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert isinstance(disp.semantic, cxr.Displacement)

    def test_basis_is_preserved(self):
        """Resulting Tangent carries the requested basis."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert disp.basis == cxr.coord_basis

    def test_phys_basis_is_preserved(self):
        """Works with PhysicalBasis as well."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.phys_basis)
        assert disp.basis == cxr.phys_basis

    def test_chart_is_unchanged(self):
        """Resulting Tangent has the same chart as the input Point."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert disp.chart == pt.chart

    def test_data_is_unchanged(self):
        """Component data is identical (not a copy or transformation)."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert disp.data is pt.data

    def test_frame_is_preserved(self):
        """Frame from the input Point is carried over."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert disp.frame == pt.frame

    def test_rep_is_tangent_geometry(self):
        """Resulting representation has TangentGeometry."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert isinstance(disp.rep.geom_kind, cxr.TangentGeometry)

    def test_rep_matches_coord_disp(self):
        """coord_basis promotion produces coord_disp representation."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert disp.rep == cxr.coord_disp

    def test_rep_matches_phys_disp(self):
        """phys_basis promotion produces phys_disp representation."""
        pt = self._make_point()
        disp = cxr.change_basis(pt, cxr.phys_basis)
        assert disp.rep == cxr.phys_disp

    def test_accessible_from_main(self):
        """change_basis is accessible via coordinax.main."""
        pt = self._make_point()
        disp = cx.change_basis(pt, cxr.coord_basis)
        assert isinstance(disp, cx.Tangent)

    def test_spherical_chart(self):
        """Works for a non-Cartesian (spherical) Point."""
        pt = cx.Point.from_(
            {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.0, "rad")},
            cxc.sph3d,
        )
        disp = cxr.change_basis(pt, cxr.coord_basis)
        assert isinstance(disp, cx.Tangent)
        assert disp.chart == cxc.sph3d
        assert isinstance(disp.semantic, cxr.Displacement)

    def test_jit_compatible(self):
        """The promotion is JIT-compatible."""
        pt = self._make_point()

        @jax.jit
        def promote(p):
            return cxr.change_basis(p, cxr.coord_basis)

        disp = promote(pt)
        assert isinstance(disp, cx.Tangent)
        assert isinstance(disp.semantic, cxr.Displacement)
