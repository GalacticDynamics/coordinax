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


def test_dimension_speed_returns_vel() -> None:
    """guess_semantic_kind(u.dimension('speed')) returns vel."""
    result = cxr.guess_semantic_kind(u.dimension("speed"))
    assert result == cxr.vel


def test_dimension_angular_speed_returns_vel() -> None:
    """guess_semantic_kind(u.dimension('angular speed')) returns vel."""
    result = cxr.guess_semantic_kind(u.dimension("angular speed"))
    assert result == cxr.vel


def test_dimension_tuple_speed_angular_speed_returns_vel() -> None:
    """guess_semantic_kind((speed, angular speed)) returns vel."""
    result = cxr.guess_semantic_kind(
        (u.dimension("speed"), u.dimension("angular speed"))
    )
    assert result == cxr.vel


def test_dimension_tuple_angular_speed_speed_returns_vel() -> None:
    """guess_semantic_kind((angular speed, speed)) returns vel."""
    result = cxr.guess_semantic_kind(
        (u.dimension("angular speed"), u.dimension("speed"))
    )
    assert result == cxr.vel


def test_dimension_acceleration_returns_acc() -> None:
    """guess_semantic_kind(u.dimension('acceleration')) returns acc."""
    result = cxr.guess_semantic_kind(u.dimension("acceleration"))
    assert result == cxr.acc


def test_dimension_angular_acceleration_returns_acc() -> None:
    """guess_semantic_kind(u.dimension('angular acceleration')) returns acc."""
    result = cxr.guess_semantic_kind(u.dimension("angular acceleration"))
    assert result == cxr.acc


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


def test_quantity_speed_returns_vel() -> None:
    """guess_semantic_kind(Quantity in m/s) returns vel."""
    result = cxr.guess_semantic_kind(u.Q(1.0, "m / s"))
    assert result == cxr.vel


def test_quantity_angular_speed_returns_vel() -> None:
    """guess_semantic_kind(Quantity in rad/s) returns vel."""
    result = cxr.guess_semantic_kind(u.Q(1.0, "rad / s"))
    assert result == cxr.vel


def test_quantity_acceleration_returns_acc() -> None:
    """guess_semantic_kind(Quantity in m/s^2) returns acc."""
    result = cxr.guess_semantic_kind(u.Q(1.0, "m / s**2"))
    assert result == cxr.acc


def test_quantity_angular_acceleration_returns_acc() -> None:
    """guess_semantic_kind(Quantity in rad/s^2) returns acc."""
    result = cxr.guess_semantic_kind(u.Q(1.0, "rad / s**2"))
    assert result == cxr.acc


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


def test_cdict_speed_returns_vel() -> None:
    """guess_semantic_kind on a speed CDict returns vel."""
    d = {"vx": u.Q(1.0, "m / s"), "vy": u.Q(2.0, "m / s")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.vel


def test_cdict_angular_speed_returns_vel() -> None:
    """guess_semantic_kind on an angular-speed CDict returns vel."""
    d = {"vphi": u.Q(1.0, "rad / s"), "vtheta": u.Q(0.5, "rad / s")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.vel


def test_cdict_mixed_speed_angular_speed_returns_vel() -> None:
    """guess_semantic_kind on a mixed speed+angular-speed CDict returns vel."""
    d = {"vr": u.Q(1.0, "m / s"), "vphi": u.Q(0.5, "rad / s")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.vel


def test_cdict_acceleration_returns_acc() -> None:
    """guess_semantic_kind on an acceleration CDict returns acc."""
    d = {"ax": u.Q(1.0, "m / s**2"), "ay": u.Q(2.0, "m / s**2")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.acc


def test_cdict_angular_acceleration_returns_acc() -> None:
    """guess_semantic_kind on an angular-acceleration CDict returns acc."""
    d = {"aphi": u.Q(1.0, "rad / s**2"), "atheta": u.Q(0.5, "rad / s**2")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.acc


def test_cdict_mixed_acceleration_angular_acceleration_returns_acc() -> None:
    """guess_semantic_kind on mixed acceleration/ang-acc CDict returns acc."""
    d = {"ar": u.Q(1.0, "m / s**2"), "aphi": u.Q(0.5, "rad / s**2")}
    result = cxr.guess_semantic_kind(d)
    assert result == cxr.acc


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

    def test_dimension_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(dimension('speed')) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.dimension("speed"))
        assert result == cxr.tangent_geom

    def test_dimension_angular_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(dimension('angular speed')) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.dimension("angular speed"))
        assert result == cxr.tangent_geom

    def test_dimension_tuple_speed_angular_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind((speed, angular speed)) returns tangent_geom."""
        result = cxr.guess_geometry_kind(
            (u.dimension("speed"), u.dimension("angular speed"))
        )
        assert result == cxr.tangent_geom

    def test_dimension_tuple_angular_speed_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind((angular speed, speed)) returns tangent_geom."""
        result = cxr.guess_geometry_kind(
            (u.dimension("angular speed"), u.dimension("speed"))
        )
        assert result == cxr.tangent_geom

    def test_dimension_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(dimension('acceleration')) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.dimension("acceleration"))
        assert result == cxr.tangent_geom

    def test_dimension_angular_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(angular acceleration) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.dimension("angular acceleration"))
        assert result == cxr.tangent_geom

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

    def test_quantity_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(Quantity in m/s) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.Q(1.0, "m / s"))
        assert result == cxr.tangent_geom

    def test_quantity_angular_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(Quantity in rad/s) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.Q(1.0, "rad / s"))
        assert result == cxr.tangent_geom

    def test_quantity_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(Quantity in m/s^2) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.Q(1.0, "m / s**2"))
        assert result == cxr.tangent_geom

    def test_quantity_angular_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind(Quantity in rad/s^2) returns tangent_geom."""
        result = cxr.guess_geometry_kind(u.Q(1.0, "rad / s**2"))
        assert result == cxr.tangent_geom

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

    def test_cdict_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind on a speed CDict returns tangent_geom."""
        d = {"vx": u.Q(1.0, "m / s"), "vy": u.Q(2.0, "m / s")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

    def test_cdict_angular_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind on an angular-speed CDict returns tangent_geom."""
        d = {"vphi": u.Q(1.0, "rad / s"), "vtheta": u.Q(0.5, "rad / s")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

    def test_cdict_mixed_speed_angular_speed_returns_tangent_geom(self) -> None:
        """guess_geometry_kind on mixed speed/angular-speed CDict."""
        d = {"vr": u.Q(1.0, "m / s"), "vphi": u.Q(0.5, "rad / s")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

    def test_cdict_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind on an acceleration CDict returns tangent_geom."""
        d = {"ax": u.Q(1.0, "m / s**2"), "ay": u.Q(2.0, "m / s**2")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

    def test_cdict_angular_acceleration_returns_tangent_geom(self) -> None:
        """guess_geometry_kind on an angular-acceleration CDict returns tangent_geom."""
        d = {"aphi": u.Q(1.0, "rad / s**2"), "atheta": u.Q(0.5, "rad / s**2")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

    def test_cdict_mixed_acceleration_angular_acceleration_returns_tangent_geom(
        self,
    ) -> None:
        """guess_geometry_kind on mixed acceleration/ang-acc CDict."""
        d = {"ar": u.Q(1.0, "m / s**2"), "aphi": u.Q(0.5, "rad / s**2")}
        result = cxr.guess_geometry_kind(d)
        assert result == cxr.tangent_geom

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

    # --- Tangent geometry: speed dimensions ---

    def test_dimension_speed_returns_tangent_vel(self) -> None:
        """guess_rep(dimension('speed')) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep(u.dimension("speed"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_dimension_tuple_speed_angular_speed_returns_tangent_vel(self) -> None:
        """guess_rep((speed, angular speed)) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep((u.dimension("speed"), u.dimension("angular speed")))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_dimension_tuple_angular_speed_speed_returns_tangent_vel(self) -> None:
        """guess_rep((angular speed, speed)) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep((u.dimension("angular speed"), u.dimension("speed")))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_dimension_angular_speed_returns_tangent_vel(self) -> None:
        """guess_rep(dimension('angular speed')) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep(u.dimension("angular speed"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_quantity_speed_returns_tangent_vel(self) -> None:
        """guess_rep(Quantity in m/s) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep(u.Q(1.0, "m / s"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_quantity_angular_speed_returns_tangent_vel(self) -> None:
        """guess_rep(Quantity in rad/s) returns TangentGeometry + Velocity."""
        result = cxr.guess_rep(u.Q(1.0, "rad / s"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_cdict_speed_returns_tangent_vel(self) -> None:
        """guess_rep on a speed CDict returns TangentGeometry + Velocity."""
        d = {"vx": u.Q(1.0, "m / s"), "vy": u.Q(2.0, "m / s")}
        result = cxr.guess_rep(d)
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    def test_cdict_mixed_speed_angular_speed_returns_tangent_vel(self) -> None:
        """guess_rep on mixed speed/angular-speed CDict returns tangent vel."""
        d = {"vr": u.Q(1.0, "m / s"), "vphi": u.Q(0.5, "rad / s")}
        result = cxr.guess_rep(d)
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.vel

    # --- Tangent geometry: acceleration dimensions ---

    def test_dimension_acceleration_returns_tangent_acc(self) -> None:
        """guess_rep(dimension('acceleration')) returns tangent acc."""
        result = cxr.guess_rep(u.dimension("acceleration"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_dimension_tuple_acceleration_angular_acceleration_returns_tangent_acc(
        self,
    ) -> None:
        """guess_rep((acceleration, angular acceleration)) returns tangent acc."""
        result = cxr.guess_rep(
            (u.dimension("acceleration"), u.dimension("angular acceleration"))
        )
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_dimension_tuple_angular_acceleration_acceleration_returns_tangent_acc(
        self,
    ) -> None:
        """guess_rep((angular acceleration, acceleration)) returns tangent acc."""
        result = cxr.guess_rep(
            (u.dimension("angular acceleration"), u.dimension("acceleration"))
        )
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_dimension_angular_acceleration_returns_tangent_acc(self) -> None:
        """guess_rep(dimension('angular acceleration')) returns tangent acc."""
        result = cxr.guess_rep(u.dimension("angular acceleration"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_quantity_acceleration_returns_tangent_acc(self) -> None:
        """guess_rep(Quantity in m/s^2) returns TangentGeometry + Acceleration."""
        result = cxr.guess_rep(u.Q(1.0, "m / s**2"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_quantity_angular_acceleration_returns_tangent_acc(self) -> None:
        """guess_rep(Quantity in rad/s^2) returns TangentGeometry + Acceleration."""
        result = cxr.guess_rep(u.Q(1.0, "rad / s**2"))
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_cdict_acceleration_returns_tangent_acc(self) -> None:
        """guess_rep on an acceleration CDict returns TangentGeometry + Acceleration."""
        d = {"ax": u.Q(1.0, "m / s**2"), "ay": u.Q(2.0, "m / s**2")}
        result = cxr.guess_rep(d)
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    def test_cdict_mixed_acceleration_angular_acceleration_returns_tangent_acc(
        self,
    ) -> None:
        """guess_rep on mixed acceleration/ang-acc CDict returns tangent acc."""
        d = {"ar": u.Q(1.0, "m / s**2"), "aphi": u.Q(0.5, "rad / s**2")}
        result = cxr.guess_rep(d)
        assert result.geom_kind == cxr.TangentGeometry()
        assert result.semantic_kind == cxr.acc

    # --- Return type ---

    @given(rep=cxrst.representations())
    def test_return_type_is_representation(self, rep: cxr.Representation) -> None:
        """guess_rep always returns a Representation instance."""
        result = cxr.guess_rep(rep)
        assert isinstance(result, cxr.Representation)
