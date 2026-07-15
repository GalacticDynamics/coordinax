"""Tests for ``coordinax.transforms.act`` dispatches.

The dispatch matrix — {Identity, Rotate, Reflect, Translate, Composed} ×
{Array, Quantity, QMatrix, CDict, Vector, Point+Frame, Point+XfmFrame} — is
covered by parametrized tests for:
  - correctness: known-value checks (also serves as cross-level consistency,
    since every level is compared to the same expected result)
  - return type: output matches input type
  - roundtrip:  act(op.inverse, None, act(op, None, x)) ≈ x
  - jit compat: wrapping in jit works

Level-specific structural checks (frame/chart preservation, mixed-unit
QMatrix) and the non-Cartesian tangent-geometry paths follow as their own
tests.
"""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .conftest import (
    EXPECTED_COMPOSED,
    EXPECTED_IDENTITY,
    EXPECTED_REFLECT,
    EXPECTED_ROTATE,
    EXPECTED_TRANSLATE,
)
from coordinax.internal import QMatrix

ATOL = 1e-5


# ===================================================================
# Helpers


def _extract_xyz(result):
    """Extract (x, y, z) floats from any result type for comparison."""
    if isinstance(result, dict):
        # CDict
        x = float(u.ustrip("km", result["x"]))
        y = float(u.ustrip("km", result["y"]))
        z = float(u.ustrip("km", result["z"]))
        return (x, y, z)

    if isinstance(result, cx.Point):
        d = result.data
        x = float(u.ustrip("km", d["x"]))
        y = float(u.ustrip("km", d["y"]))
        z = float(u.ustrip("km", d["z"]))
        return (x, y, z)

    if isinstance(result, QMatrix):
        x = float(u.ustrip("km", u.Q(result.value[0], result.unit[0])))
        y = float(u.ustrip("km", u.Q(result.value[1], result.unit[1])))
        z = float(u.ustrip("km", u.Q(result.value[2], result.unit[2])))
        return (x, y, z)

    if isinstance(result, u.AbstractQuantity):
        arr = u.ustrip("km", result)
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    # Bare array
    arr = jnp.asarray(result)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _assert_close(actual_xyz, expected_xyz, atol=ATOL):
    np.testing.assert_allclose(actual_xyz, expected_xyz, atol=atol)


# ===================================================================
# Dispatch matrix: {operator} × {input level}
#
# Every operator/level pair is compared against the same EXPECTED_* tuple, so
# this parametrized correctness test doubles as the cross-level consistency
# check (all input types represent the same fundamental action).
# ===================================================================

USYS = u.unitsystem("km", "s", "kg", "rad")

# (input fixture, expected isinstance type). A bare array carries no units, so
# translate/composed need `usys` supplied at that level only.
INPUT_LEVELS = [
    ("array_3d", jax.Array),
    ("quantity_3d", u.AbstractQuantity),
    ("qmatrix_3d", QMatrix),
    ("cdict_3d", dict),
    ("vector_3d", cx.Point),
    ("coord_3d", cx.Point),
    ("coord_xfm_3d", cx.Point),
]
LEVEL_FIXTURES = [name for name, _ in INPUT_LEVELS]
LEVEL_IDS = [name.removesuffix("_3d") for name, _ in INPUT_LEVELS]

# (op fixture, expected xyz, needs usys at the bare-array level)
OPS = [
    ("identity_op", EXPECTED_IDENTITY, False),
    ("rotate_op", EXPECTED_ROTATE, False),
    ("reflect_op", EXPECTED_REFLECT, False),
    ("translate_op", EXPECTED_TRANSLATE, True),
    ("composed_op", EXPECTED_COMPOSED, True),
]
OP_IDS = [name.removesuffix("_op") for name, _, _ in OPS]

# Operators with a non-trivial inverse worth round-tripping.
ROUNDTRIP_OPS = [("rotate_op", False), ("translate_op", True), ("composed_op", True)]

# Levels that accept an explicit chart / (chart, rep) as extra positional args.
CHART_LEVELS = ["quantity_3d", "qmatrix_3d", "cdict_3d"]


def _usys_kw(level_fixture, needs_usys):
    """`usys` is only required for the unit-less bare-array level."""
    return {"usys": USYS} if (needs_usys and level_fixture == "array_3d") else {}


@pytest.mark.parametrize("level_fixture", LEVEL_FIXTURES, ids=LEVEL_IDS)
@pytest.mark.parametrize(("op_fixture", "expected", "needs_usys"), OPS, ids=OP_IDS)
def test_act_matches_expected(request, op_fixture, expected, needs_usys, level_fixture):
    """Each operator gives its known result on every input level."""
    op = request.getfixturevalue(op_fixture)
    x = request.getfixturevalue(level_fixture)
    result = cxfm.act(op, None, x, **_usys_kw(level_fixture, needs_usys))
    _assert_close(_extract_xyz(result), expected)


@pytest.mark.parametrize(("level_fixture", "return_type"), INPUT_LEVELS, ids=LEVEL_IDS)
def test_act_returns_input_type(request, rotate_op, level_fixture, return_type):
    """The output type mirrors the input type."""
    x = request.getfixturevalue(level_fixture)
    assert isinstance(cxfm.act(rotate_op, None, x), return_type)


@pytest.mark.parametrize("level_fixture", LEVEL_FIXTURES, ids=LEVEL_IDS)
@pytest.mark.parametrize(
    ("op_fixture", "needs_usys"), ROUNDTRIP_OPS, ids=["rotate", "translate", "composed"]
)
def test_act_inverse_roundtrip(request, op_fixture, needs_usys, level_fixture):
    """act(op.inverse, act(op, x)) recovers x on every input level."""
    op = request.getfixturevalue(op_fixture)
    x = request.getfixturevalue(level_fixture)
    kw = _usys_kw(level_fixture, needs_usys)
    fwd = cxfm.act(op, None, x, **kw)
    back = cxfm.act(op.inverse, None, fwd, **kw)
    _assert_close(_extract_xyz(back), EXPECTED_IDENTITY)


@pytest.mark.parametrize(
    "level_fixture", CHART_LEVELS, ids=["quantity", "qmatrix", "cdict"]
)
def test_act_with_explicit_chart_and_rep(request, rotate_op, level_fixture):
    """A chart, and a (chart, rep) pair, may be passed as extra positionals."""
    x = request.getfixturevalue(level_fixture)
    _assert_close(
        _extract_xyz(cxfm.act(rotate_op, None, x, cxc.cart3d)), EXPECTED_ROTATE
    )
    _assert_close(
        _extract_xyz(cxfm.act(rotate_op, None, x, cxc.cart3d, cxr.point)),
        EXPECTED_ROTATE,
    )


@pytest.mark.parametrize("level_fixture", LEVEL_FIXTURES, ids=LEVEL_IDS)
def test_act_under_jit(request, rotate_op, level_fixture):
    """Wrapping act in jit works at every input level."""
    x = request.getfixturevalue(level_fixture)
    result = eqx.filter_jit(lambda y: cxfm.act(rotate_op, None, y))(x)
    _assert_close(_extract_xyz(result), EXPECTED_ROTATE)


# Level-specific structural checks that don't generalize across input types.


def test_qmatrix_heterogeneous_units_identity(identity_op):
    """A QMatrix with per-component (heterogeneous) units passes through Identity."""
    units = (u.unit("km"), u.unit("m"), u.unit("cm"))
    qm = QMatrix(jnp.array([1.0, 2.0, 3.0]), unit=units)
    result = cxfm.act(identity_op, None, qm)
    assert isinstance(result, QMatrix)
    np.testing.assert_allclose(np.asarray(result.value), [1.0, 2.0, 3.0])
    assert result.unit == units


def test_vector_preserves_chart(rotate_op, vector_3d):
    assert cxfm.act(rotate_op, None, vector_3d).chart == vector_3d.chart


def test_coordinate_preserves_frame(rotate_op, coord_3d):
    result = cxfm.act(rotate_op, None, coord_3d)
    assert isinstance(result.frame, type(coord_3d.frame))


def test_coordinate_xfm_preserves_transformed_frame(rotate_op, coord_xfm_3d):
    result = cxfm.act(rotate_op, None, coord_xfm_3d)
    assert isinstance(result.frame, cxf.TransformedReferenceFrame)


# ===================================================================
# Callable via __call__
# ===================================================================


class TestTransformCallable:
    """Verify transforms can be called directly as op(x) or op(tau, x)."""

    def test_rotate_call_vector(self, rotate_op, vector_3d):
        result = rotate_op(vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_rotate_call_with_tau_vector(self, rotate_op, vector_3d):
        result = rotate_op(None, vector_3d)
        _assert_close(_extract_xyz(result), EXPECTED_ROTATE)

    def test_translate_call_quantity(self, translate_op, quantity_3d):
        result = translate_op(quantity_3d)
        _assert_close(_extract_xyz(result), EXPECTED_TRANSLATE)

    def test_composed_call_cdict(self, composed_op, cdict_3d):
        result = composed_op(cdict_3d)
        _assert_close(_extract_xyz(result), EXPECTED_COMPOSED)

    def test_identity_call_coordinate(self, identity_op, coord_3d):
        result = identity_op(coord_3d)
        _assert_close(_extract_xyz(result), EXPECTED_IDENTITY)


# ===================================================================
# Tangent geometry on non-Cartesian charts (Jacobian pushforward)
# ===================================================================


class TestRotateTangentGeometryNonCartesian:
    """Verify that act(Rotate, TangentGeometry, sph3d) uses the Jacobian.

    The key invariant:
        cart(rotate(v, at_sph)) == R * cart(v, at_sph)

    where cart(*) denotes the tangent_map pushforward to Cartesian coords.
    """

    @pytest.fixture
    def rot90z(self):
        return cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    @pytest.fixture
    def at_sph(self):
        """Base point at the equator phi=0; Cartesian (1,0,0)."""
        return {"r": u.Q(1, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}

    @pytest.fixture
    def v_radial_sph(self):
        """Purely radial velocity in spherical coord-basis."""
        return {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}

    def test_cart_consistency(self, rot90z, at_sph, v_radial_sph):
        """cart(R*v at R*p) == R * cart(v at p)."""
        # Rotate tangent via Jacobian path
        v_rot_sph = cxfm.act(
            rot90z,
            None,
            v_radial_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph,
        )
        # Rotated base point in spherical (phi: 0 -> pi/2)
        at_sph_rot = {
            "r": u.Q(1, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(jnp.pi / 2, "rad"),
        }
        # Push rotated tangent to Cartesian
        v_rot_cart = cxr.tangent_map(
            v_rot_sph, cxc.sph3d, cxr.coord_vel, cxc.cart3d, at=at_sph_rot
        )
        # Directly compute R * cart(v) via public act on the Cartesian tangent
        v_cart = cxr.tangent_map(
            v_radial_sph, cxc.sph3d, cxr.coord_vel, cxc.cart3d, at=at_sph
        )
        v_expected = cxfm.act(
            rot90z, None, v_cart, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel
        )

        assert abs(float(v_rot_cart["x"].value) - float(v_expected["x"].value)) < ATOL
        assert abs(float(v_rot_cart["y"].value) - float(v_expected["y"].value)) < ATOL
        assert abs(float(v_rot_cart["z"].value) - float(v_expected["z"].value)) < ATOL

    def test_round_trip(self, rot90z, at_sph, v_radial_sph):
        """R⁻¹(R(v, at), R(at)) == v."""
        v_rot_sph = cxfm.act(
            rot90z,
            None,
            v_radial_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph,
        )
        at_sph_rot = {
            "r": u.Q(1, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(jnp.pi / 2, "rad"),
        }
        inv_op = cxfm.Rotate.from_euler("z", u.Q(-90, "deg"))
        v_recovered = cxfm.act(
            inv_op,
            None,
            v_rot_sph,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_vel,
            at=at_sph_rot,
        )
        assert abs(float(v_recovered["r"].to_value("m/s")) - 1) < ATOL
        assert abs(float(v_recovered["theta"].to_value("rad/s"))) < ATOL
        assert abs(float(v_recovered["phi"].to_value("rad/s"))) < ATOL

    def test_raises_without_at(self, rot90z, v_radial_sph):
        """act(Rotate, sph3d, TangentGeometry) raises TypeError without at=."""
        with pytest.raises(TypeError, match="requires 'at'"):
            cxfm.act(
                rot90z, None, v_radial_sph, cxc.sph3d, cxr.tangent_geom, cxr.coord_vel
            )

    def test_jit(self, rot90z, at_sph, v_radial_sph):
        """act(Rotate, sph3d, TangentGeometry) is JIT-compatible."""
        result = eqx.filter_jit(
            lambda v: cxfm.act(
                rot90z, None, v, cxc.sph3d, cxr.tangent_geom, cxr.coord_vel, at=at_sph
            )
        )(v_radial_sph)
        assert abs(float(result["r"].to_value("m/s")) - 1) < ATOL


# ===================================================================
# Coordinate.to_frame with non-Cartesian velocity
# ===================================================================


class TestCoordinateToFrameNonCartesianTangent:
    """Verify Coordinate.to_frame injects 'at' correctly for tangent fibres."""

    def test_cart3d_velocity_to_rotated_frame(self):
        """Coordinate with Cartesian velocity transforms correctly via to_frame."""
        rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot)

        point = cx.Point.from_([1, 0, 0], "m", cxf.alice)
        vel = cx.Tangent(
            {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
            cxc.cart3d,
            cxr.coord_basis,
            cxr.vel,
            frame=cxf.alice,
        )
        coord = cx.Coordinate(point=point, velocity=vel)
        result = coord.to_frame(rotated_frame)

        # Point (1,0,0) rotated 90° about z -> (0,1,0)
        _assert_close(
            (
                float(result.point.data["x"].ustrip("m")),
                float(result.point.data["y"].ustrip("m")),
                float(result.point.data["z"].ustrip("m")),
            ),
            (0, 1, 0),
        )
        # Velocity (1,0,0) m/s rotated -> (0,1,0) m/s
        _assert_close(
            (
                float(result["velocity"].data["x"].ustrip("m/s")),
                float(result["velocity"].data["y"].ustrip("m/s")),
                float(result["velocity"].data["z"].ustrip("m/s")),
            ),
            (0, 1, 0),
        )

    def test_coordinate_to_frame_then_cconvert_sph(self):
        """Coordinate.to_frame followed by .cconvert(sph3d) works correctly."""
        rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        rotated_frame = cxf.TransformedReferenceFrame(cxf.alice, rot)

        point = cx.Point.from_([1, 0, 0], "m", cxf.alice)
        vel = cx.Tangent(
            {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
            cxc.cart3d,
            cxr.coord_basis,
            cxr.vel,
            frame=cxf.alice,
        )
        coord = cx.Coordinate(point=point, velocity=vel)
        result = coord.to_frame(rotated_frame).cconvert(cxc.sph3d)

        # Point should land at (r=1, theta=pi/2, phi=pi/2)
        assert abs(float(result.point.data["r"].to_value("m")) - 1) < ATOL
        assert (
            abs(float(result.point.data["theta"].to_value("rad")) - jnp.pi / 2) < ATOL
        )
        assert abs(float(result.point.data["phi"].to_value("rad")) - jnp.pi / 2) < ATOL
        # Velocity should be purely radial (ṙ≈1, θ̇≈0, φ̇≈0)
        assert abs(float(result["velocity"].data["r"].to_value("m/s")) - 1) < ATOL
