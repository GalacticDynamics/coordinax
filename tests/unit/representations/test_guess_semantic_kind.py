"""Tests for guess_geometry_kind, guess_rep, and guess_semantic_kind."""

__all__: tuple[str, ...] = ()

import pytest
from hypothesis import given

import unxt as u

import coordinax.charts as cxc
import coordinax.hypothesis.representations as cxrst
import coordinax.representations as cxr

# ===================================================================
# Identity dispatch


def test_identity_location() -> None:
    """guess_semantic_kind(Location()) returns the same object."""
    sem = cxr.Location()
    assert cxr.guess_semantic_kind(sem) is sem


def test_identity_canonical_instance() -> None:
    """guess_semantic_kind(loc) returns the canonical instance."""
    result = cxr.guess_semantic_kind(cxr.loc)
    assert result is cxr.loc


@given(sem=cxrst.semantics())
def test_identity_any_semantic(sem: cxr.AbstractSemanticKind) -> None:
    """guess_semantic_kind returns the input unchanged for any AbstractSemanticKind."""
    assert cxr.guess_semantic_kind(sem) is sem


# ===================================================================
# Dimension dispatch


def test_dimension_length_returns_loc() -> None:
    """guess_semantic_kind(u.dimension('length')) returns loc."""
    result = cxr.guess_semantic_kind(u.dimension("length"))
    assert result == cxr.loc


def test_dimension_angle_returns_loc() -> None:
    """guess_semantic_kind(u.dimension('angle')) returns loc."""
    result = cxr.guess_semantic_kind(u.dimension("angle"))
    assert result == cxr.loc


def test_dimension_unknown_raises() -> None:
    """guess_semantic_kind raises ValueError for an unregistered dimension."""
    with pytest.raises(ValueError, match="Cannot infer semantic kind"):
        cxr.guess_semantic_kind(u.dimension("time"))


# ===================================================================
# Quantity dispatch


def test_quantity_length_returns_loc() -> None:
    """guess_semantic_kind(Quantity in meters) returns loc."""
    result = cxr.guess_semantic_kind(u.Q(1.0, "m"))
    assert result == cxr.loc


def test_quantity_angle_returns_loc() -> None:
    """guess_semantic_kind(Quantity in radians) returns loc."""
    result = cxr.guess_semantic_kind(u.Q(0.5, "rad"))
    assert result == cxr.loc


def test_quantity_unknown_dim_raises() -> None:
    """guess_semantic_kind raises ValueError for a quantity with unknown dimension."""
    with pytest.raises(ValueError, match="Cannot infer semantic kind"):
        cxr.guess_semantic_kind(u.Q(1.0, "s"))


# ===================================================================
# CDict dispatch


def test_cdict_cartesian_returns_loc() -> None:
    """guess_semantic_kind on a Cartesian CDict returns loc."""
    d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.loc


def test_cdict_spherical_mixed_dims_returns_loc() -> None:
    """guess_semantic_kind on a spherical CDict (length + angle) returns loc."""
    d = {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1.0, "rad")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.loc


def test_cdict_pure_angle_returns_loc() -> None:
    """guess_semantic_kind on an angular CDict (lon/lat) returns loc."""
    d = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.loc


def test_cdict_empty_raises() -> None:
    """guess_semantic_kind raises ValueError for an empty CDict."""
    with pytest.raises(ValueError, match="Cannot infer semantic kind"):
        cxr.guess_semantic_kind({})


def test_cdict_unknown_dim_raises() -> None:
    """guess_semantic_kind raises ValueError for a CDict with unknown dimension."""
    with pytest.raises(ValueError, match="Cannot infer semantic kind"):
        cxr.guess_semantic_kind({"t": u.Q(1.0, "s")})


# ===================================================================
# Return type


@given(sem=cxrst.semantics())
def test_return_type_is_abstract_semantic_kind(
    sem: cxr.AbstractSemanticKind,
) -> None:
    """guess_semantic_kind always returns an AbstractSemanticKind instance."""
    result = cxr.guess_semantic_kind(sem)
    assert isinstance(result, cxr.AbstractSemanticKind)


# ===================================================================
# guess_geometry_kind
# ===================================================================


class TestGuessGeometryKind:
    """Tests for guess_geometry_kind."""

    # --- Identity dispatch ---

    def test_identity_point_geometry(self) -> None:
        """guess_geometry_kind(PointGeometry()) returns the same object."""
        geom = cxr.PointGeometry()
        assert cxr.guess_geometry_kind(geom) is geom

    def test_identity_canonical_instance(self) -> None:
        """guess_geometry_kind(point_geom) returns the canonical instance."""
        result = cxr.guess_geometry_kind(cxr.point_geom)
        assert result is cxr.point_geom

    @given(geom=cxrst.geometries())
    def test_identity_any_geometry(self, geom: cxr.AbstractGeometry) -> None:
        """guess_geometry_kind returns the input unchanged for any AbstractGeometry."""
        assert cxr.guess_geometry_kind(geom) is geom

    # --- Dimension dispatch ---

    def test_dimension_length_returns_point_geom(self) -> None:
        """guess_geometry_kind(dimension('length')) returns point_geom."""
        result = cxr.guess_geometry_kind(u.dimension("length"))
        assert result == cxr.point_geom

    def test_dimension_angle_returns_point_geom(self) -> None:
        """guess_geometry_kind(dimension('angle')) returns point_geom."""
        result = cxr.guess_geometry_kind(u.dimension("angle"))
        assert result == cxr.point_geom

    def test_dimension_unknown_raises(self) -> None:
        """guess_geometry_kind raises ValueError for an unregistered dimension."""
        with pytest.raises(ValueError, match="Cannot infer geometry kind"):
            cxr.guess_geometry_kind(u.dimension("time"))

    # --- Quantity dispatch ---

    def test_quantity_length_returns_point_geom(self) -> None:
        """guess_geometry_kind(Quantity in metres) returns point_geom."""
        result = cxr.guess_geometry_kind(u.Q(1.0, "m"))
        assert result == cxr.point_geom

    def test_quantity_angle_returns_point_geom(self) -> None:
        """guess_geometry_kind(Quantity in radians) returns point_geom."""
        result = cxr.guess_geometry_kind(u.Q(0.5, "rad"))
        assert result == cxr.point_geom

    def test_quantity_unknown_dim_raises(self) -> None:
        """Raises ValueError for a quantity with unknown dimension."""
        with pytest.raises(ValueError, match="Cannot infer geometry kind"):
            cxr.guess_geometry_kind(u.Q(1.0, "s"))

    # --- CDict dispatch (no chart) ---

    def test_cdict_cartesian_returns_point_geom(self) -> None:
        """guess_geometry_kind on a Cartesian CDict returns point_geom."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.point_geom

    def test_cdict_spherical_mixed_dims_returns_point_geom(self) -> None:
        """On a spherical CDict (length + angle) returns point_geom."""
        d = {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1.0, "rad")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.point_geom

    def test_cdict_pure_angle_returns_point_geom(self) -> None:
        """guess_geometry_kind on an angular CDict (lon/lat) returns point_geom."""
        d = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.point_geom

    def test_cdict_empty_raises(self) -> None:
        """guess_geometry_kind raises ValueError for an empty CDict."""
        with pytest.raises(ValueError, match="Cannot infer geometry kind"):
            cxr.guess_geometry_kind({})

    def test_cdict_unknown_dim_raises(self) -> None:
        """guess_geometry_kind raises ValueError for a CDict with unknown dimension."""
        with pytest.raises(ValueError, match="Cannot infer geometry kind"):
            cxr.guess_geometry_kind({"t": u.Q(1.0, "s")})

    # --- CDict + AbstractChart dispatch ---

    def test_cdict_with_chart_returns_point_geom(self) -> None:
        """guess_geometry_kind(CDict, chart) with matching keys returns point_geom."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        result = cxr.guess_geometry_kind(d, cxc.cart2d)
        assert result == cxr.point_geom

    def test_cdict_with_chart_wrong_keys_raises(self) -> None:
        """Raises when CDict keys don't match chart."""
        d = {"a": u.Q(1.0, "m"), "b": u.Q(2.0, "m")}
        with pytest.raises(ValueError, match="Data keys do not match chart components"):
            cxr.guess_geometry_kind(d, cxc.cart2d)

    # --- CDict + ProlateSpheroidal3D dispatch ---

    def test_cdict_prolate_spheroidal_returns_point_geom(self) -> None:
        """On prolate-spheroidal CDict {area, angle} returns point_geom."""
        d = {
            "mu": u.Q(1.0, "km2"),
            "nu": u.Q(0.5, "km2"),
            "phi": u.Q(1.0, "deg"),
        }
        chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1.0, "km"))
        result = cxr.guess_geometry_kind(d, chart)
        assert result == cxr.point_geom

    # --- Return type ---

    @given(geom=cxrst.geometries())
    def test_return_type_is_abstract_geometry(self, geom: cxr.AbstractGeometry) -> None:
        """guess_geometry_kind always returns an AbstractGeometry instance."""
        result = cxr.guess_geometry_kind(geom)
        assert isinstance(result, cxr.AbstractGeometry)


# ===================================================================
# guess_rep
# ===================================================================


class TestGuessRep:
    """Tests for guess_rep."""

    # --- Identity dispatch: Representation ---

    def test_identity_representation(self) -> None:
        """guess_rep(Representation(...)) returns the same object."""
        rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())
        assert cxr.guess_rep(rep) is rep

    def test_identity_canonical_instance(self) -> None:
        """guess_rep(point) returns the canonical point instance."""
        result = cxr.guess_rep(cxr.point)
        assert result is cxr.point

    @given(rep=cxrst.representations())
    def test_identity_any_representation(self, rep: cxr.Representation) -> None:
        """guess_rep returns the input unchanged for any Representation."""
        assert cxr.guess_rep(rep) is rep

    # --- PointGeometry dispatch ---

    def test_point_geometry_returns_point(self) -> None:
        """guess_rep(PointGeometry()) returns the point canonical instance."""
        result = cxr.guess_rep(cxr.PointGeometry())
        assert result == cxr.point

    def test_point_geometry_canonical_returns_point(self) -> None:
        """guess_rep(point_geom) returns the canonical point Representation."""
        result = cxr.guess_rep(cxr.point_geom)
        assert result == cxr.point

    # --- Dimension / Quantity / CDict dispatch ---

    def test_dimension_length_returns_point(self) -> None:
        """guess_rep(dimension('length')) returns point."""
        result = cxr.guess_rep(u.dimension("length"))
        assert result == cxr.point

    def test_quantity_length_returns_point(self) -> None:
        """guess_rep(Quantity in metres) returns point."""
        result = cxr.guess_rep(u.Q(1.0, "m"))
        assert result == cxr.point

    def test_cdict_cartesian_returns_point(self) -> None:
        """guess_rep on a Cartesian CDict returns point."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        result = cxr.guess_rep(d)
        assert result == cxr.point

    # --- CDict + chart dispatch ---

    def test_cdict_with_chart_returns_point(self) -> None:
        """guess_rep(CDict, chart) returns point."""
        d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        result = cxr.guess_rep(d, cxc.cart2d)
        assert result == cxr.point

    # --- Return type ---

    @given(rep=cxrst.representations())
    def test_return_type_is_representation(self, rep: cxr.Representation) -> None:
        """guess_rep always returns a Representation instance."""
        result = cxr.guess_rep(rep)
        assert isinstance(result, cxr.Representation)
