"""Tests for ``coordinax.transforms.act`` dispatches.

The dispatch matrix — {Identity, Rotate, Reflect, Translate, Composed} ×
{Array, Quantity, QuantityMatrix, CDict, Vector, Point+Frame, Point+XfmFrame} — is
covered by parametrized tests for:
  - correctness: known-value checks (also serves as cross-level consistency,
    since every level is compared to the same expected result)
  - return type: output matches input type
  - roundtrip:  act(op.inverse, None, act(op, None, x)) ≈ x
  - jit compat: wrapping in jit works

Level-specific structural checks (frame/chart preservation, mixed-unit
QuantityMatrix) and the non-Cartesian tangent-geometry paths follow as their own
tests.
"""

__all__: tuple[str, ...] = ()

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plum import NotFoundLookupError

import unxt as u
import unxts.linalg as ul

import coordinax as cx
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .conftest import (
    EXPECTED_COMPOSED,
    EXPECTED_IDENTITY,
    EXPECTED_REFLECT,
    EXPECTED_ROTATE,
    EXPECTED_TRANSLATE,
)

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

    if isinstance(result, ul.QuantityMatrix):
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
    ("quantitymatrix_3d", ul.QuantityMatrix),
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
CHART_LEVELS = ["quantity_3d", "quantitymatrix_3d", "cdict_3d"]


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
    "level_fixture", CHART_LEVELS, ids=["quantity", "quantitymatrix", "cdict"]
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


def test_quantitymatrix_heterogeneous_units_identity(identity_op):
    """A QuantityMatrix with heterogeneous per-component units survives Identity."""
    units = (u.unit("km"), u.unit("m"), u.unit("cm"))
    qm = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=units)
    result = cxfm.act(identity_op, None, qm)
    assert isinstance(result, ul.QuantityMatrix)
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
# Non-JAX ArrayLike inputs
#
# `jaxtyping.ArrayLike` covers NumPy arrays as well as JAX arrays, and both
# dispatch equivalently through `guess_chart` and the act funnel. These guard
# that path, which the JAX-array fixtures above do not exercise. (A Python list
# is not an ArrayLike and is rejected — see test_act_rejects_python_list.)
# ===================================================================


@pytest.mark.parametrize(("op_fixture", "expected", "needs_usys"), OPS, ids=OP_IDS)
def test_act_accepts_numpy_array(request, op_fixture, expected, needs_usys):
    """A NumPy array dispatches equivalently to a JAX array."""
    op = request.getfixturevalue(op_fixture)
    x = np.asarray([1.0, 0.0, 0.0])
    kw = {"usys": USYS} if needs_usys else {}
    _assert_close(_extract_xyz(cxfm.act(op, None, x, **kw)), expected)


def test_act_numpy_matches_jax_array(rotate_op):
    """NumPy and JAX array inputs give identical results."""
    data = [1.0, 2.0, 3.0]
    out_np = np.asarray(cxfm.act(rotate_op, None, np.asarray(data)))
    out_jax = np.asarray(cxfm.act(rotate_op, None, jnp.asarray(data)))
    np.testing.assert_allclose(out_np, out_jax)


def test_act_rejects_python_list(rotate_op):
    """A Python list is not an ArrayLike, so it does not resolve.

    This is the documented boundary: callers pass ``jnp.asarray(...)`` or a
    Quantity, not a bare list.
    """
    with pytest.raises(NotFoundLookupError):
        cxfm.act(rotate_op, None, [1.0, 0.0, 0.0])


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


# ===================================================================
# gh#936: a bundle carrying an order >= 2 fibre must prolong jointly
# ===================================================================


class TestCoordinateBundleJointProlongation:
    r"""A `Coordinate` with an acceleration fibre must not go fibre-by-fibre.

    Walking the fibres one at a time transforms each by the frozen-$\tau$
    pushforward. That is exact at order 1, but the order-2 law carries
    $\partial_{xx}\phi(v, v)$ -- a term built from the *velocity* fibre, which
    a per-fibre pass never has in hand. So it was silently dropped, and the
    bundle returned a first-order acceleration with no indication (gh#936).

    The bundle is the one caller that always holds the lower fibres, so it
    hands the whole jet to `act_jet` instead. The reference here is `act_jet`,
    not a hardcoded number.
    """

    ROT = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg")) | cxfm.Rotate.from_euler(
        "x", u.Q(20.0, "deg")
    )
    CH = cxc.lonlat_sph3d
    Q0: ClassVar = {
        "lon": u.Q(25.0, "deg"),
        "lat": u.Q(40.0, "deg"),
        "distance": u.Q(2.0, "kpc"),
    }
    V0: ClassVar = {
        "lon": u.Q(0.7, "rad/Myr"),
        "lat": u.Q(0.3, "rad/Myr"),
        "distance": u.Q(1.0, "kpc/Myr"),
    }
    A0: ClassVar = {
        "lon": u.Q(-0.02, "rad/Myr2"),
        "lat": u.Q(0.05, "rad/Myr2"),
        "distance": u.Q(0.1, "kpc/Myr2"),
    }

    @staticmethod
    def _tangent(data, chart, kind):
        return cx.Tangent(data, chart, cxr.coord_basis, kind)

    def _bundle(self, *, velocity=True, chart=None, q=None, v=None, a=None):
        chart = self.CH if chart is None else chart
        fibres = {"acceleration": self._tangent(a or self.A0, chart, cxr.acc)}
        if velocity:
            fibres["velocity"] = self._tangent(v or self.V0, chart, cxr.vel)
        return cx.Coordinate(point=cx.Point(q or self.Q0, chart), **fibres)

    def test_acceleration_in_a_curved_chart_matches_act_jet(self):
        """The reported defect: was ~114% and ~155% off on lon/lat."""
        out = cxfm.act(self.ROT, None, self._bundle())["acceleration"].data
        ref = cxfm.act_jet(
            self.ROT, None, {0: self.Q0, 1: self.V0, 2: self.A0}, self.CH
        )[2]
        for k in ref:
            unit = u.unit_of(ref[k])
            assert jnp.allclose(u.ustrip(unit, out[k]), u.ustrip(unit, ref[k]))

    def test_the_velocity_fibre_genuinely_feeds_the_acceleration(self):
        """Discriminator: the old path was bit-identical under a 100x velocity.

        Without this, a fix that still ignored the velocity fibre would pass
        the test above whenever the reference happened to agree.
        """
        scaled = {k: 100 * v for k, v in self.V0.items()}
        base = cxfm.act(self.ROT, None, self._bundle())["acceleration"].data
        other = cxfm.act(self.ROT, None, self._bundle(v=scaled))["acceleration"].data
        assert not jnp.allclose(
            u.ustrip("rad/Myr2", base["lon"]), u.ustrip("rad/Myr2", other["lon"])
        )

    def test_a_velocity_only_bundle_still_goes_fibre_by_fibre(self):
        """Order 1 needs no joint jet: there the pushforward IS the law."""
        crd = cx.Coordinate(
            point=cx.Point(self.Q0, self.CH),
            velocity=self._tangent(self.V0, self.CH, cxr.vel),
        )
        out = cxfm.act(self.ROT, None, crd)["velocity"].data
        ref = cxfm.act_jet(
            self.ROT, None, {0: self.Q0, 1: self.V0, 2: self.A0}, self.CH
        )[1]
        for k in ref:
            unit = u.unit_of(ref[k])
            assert jnp.allclose(u.ustrip(unit, out[k]), u.ustrip(unit, ref[k]))

    def test_a_flat_chart_affine_bundle_keeps_the_cheap_path_and_stays_exact(self):
        r"""Where $\phi$ is affine in the chart's own coordinates the term is 0.

        This is the case that broke the first attempt at the numerically
        complete fix: routing it through the jet engine turns exact,
        anchor-free calls into missing-anchor errors. It must stay off that
        path *and* agree with it.
        """
        q = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
        v = {
            "x": u.Q(1.0, "kpc/Myr"),
            "y": u.Q(0.5, "kpc/Myr"),
            "z": u.Q(0.0, "kpc/Myr"),
        }
        a = {
            "x": u.Q(0.1, "kpc/Myr2"),
            "y": u.Q(0.2, "kpc/Myr2"),
            "z": u.Q(0.3, "kpc/Myr2"),
        }
        crd = self._bundle(chart=cxc.cart3d, q=q, v=v, a=a)
        out = cxfm.act(self.ROT, None, crd)["acceleration"].data
        ref = cxfm.act_jet(self.ROT, None, {0: q, 1: v, 2: a}, cxc.cart3d)[2]
        for k in ref:
            assert jnp.allclose(
                u.ustrip("kpc/Myr2", out[k]), u.ustrip("kpc/Myr2", ref[k])
            )

    def test_an_acceleration_without_a_velocity_is_refused_in_a_curved_chart(self):
        r"""The gap is a dead end, not a routing choice.

        $\partial_{xx}\phi(v, v)$ is built from the absent fibre, and absent
        means "not tracked", not "zero" -- so the honest answer is to refuse
        rather than return the first-order one.
        """
        with pytest.raises(TypeError, match=r"without the lower ladder fibre"):
            cxfm.act(self.ROT, None, self._bundle(velocity=False))

    def test_that_same_gap_is_fine_where_the_action_is_affine(self):
        """Flat chart, affine op: the missing fibre multiplies a zero term."""
        q = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
        a = {
            "x": u.Q(0.1, "kpc/Myr2"),
            "y": u.Q(0.2, "kpc/Myr2"),
            "z": u.Q(0.3, "kpc/Myr2"),
        }
        crd = cx.Coordinate(
            point=cx.Point(q, cxc.cart3d),
            acceleration=self._tangent(a, cxc.cart3d, cxr.acc),
        )
        out = cxfm.act(self.ROT, None, crd)["acceleration"].data
        assert set(out) == {"x", "y", "z"}

    def test_an_anchor_override_is_refused_rather_than_ignored(self):
        """The bundle supplies the anchors; a caller `at=` would be dropped."""
        with pytest.raises(TypeError, match=r"does not accept keyword overrides"):
            cxfm.act(self.ROT, None, self._bundle(), at=self.Q0)


class TestBundleFibreInAnotherChart:
    r"""A bundle may store a fibre in a chart other than the point's.

    The routing decision and both legs of the conversion have to ask about
    *that* fibre's chart, not the point's. A point in `cart3d` does not make
    an `sph3d` acceleration safe: the acceleration's own chart is where its
    curvature term lives. Getting this wrong returned the fibre completely
    untransformed.
    """

    ROT = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg")) | cxfm.Rotate.from_euler(
        "x", u.Q(20.0, "deg")
    )
    CQ: ClassVar = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
    CV: ClassVar = {
        "x": u.Q(0.3, "kpc/Myr"),
        "y": u.Q(-0.4, "kpc/Myr"),
        "z": u.Q(0.2, "kpc/Myr"),
    }
    CA: ClassVar = {
        "x": u.Q(0.1, "kpc/Myr2"),
        "y": u.Q(0.2, "kpc/Myr2"),
        "z": u.Q(0.3, "kpc/Myr2"),
    }

    def _all_cart(self):
        return cx.Coordinate(
            point=cx.Point(self.CQ, cxc.cart3d),
            velocity=cx.Tangent(self.CV, cxc.cart3d, cxr.coord_basis, cxr.vel),
            acceleration=cx.Tangent(self.CA, cxc.cart3d, cxr.coord_basis, cxr.acc),
        )

    def _mixed(self):
        """Same physics, but the acceleration fibre lives in `sph3d`."""
        allcart = self._all_cart()
        return cx.Coordinate._create_unchecked(
            allcart.point,
            {
                "velocity": allcart["velocity"],
                "acceleration": allcart.cconvert(cxc.sph3d)["acceleration"],
            },
        )

    def test_a_foreign_acceleration_matches_the_all_cartesian_route(self):
        """Where the fibre is *stored* must not change the physics.

        Both legs are second-order here -- the fibre is carried into the
        point's chart to build the jet, and back out afterwards -- and
        either one left as a bare Jacobian push loses the term.
        """
        got = cxfm.act(self.ROT, None, self._mixed())["acceleration"]
        ref = cxfm.act(self.ROT, None, self._all_cart()).cconvert(cxc.sph3d)
        ref_data = ref["acceleration"].data
        assert got.chart == cxc.sph3d
        for k in ref_data:
            unit = u.unit_of(ref_data[k])
            assert jnp.allclose(
                u.ustrip(unit, got.data[k]), u.ustrip(unit, ref_data[k])
            )

    def test_the_fibre_is_not_returned_untransformed(self):
        """Guard the guard: the defect returned the input verbatim."""
        before = self._mixed()["acceleration"].data
        after = cxfm.act(self.ROT, None, self._mixed())["acceleration"].data
        assert not jnp.allclose(
            u.ustrip("rad/Myr2", after["theta"]), u.ustrip("rad/Myr2", before["theta"])
        )

    def test_a_foreign_acceleration_without_a_velocity_is_refused(self):
        """Its jet cannot be assembled in its own chart without the slot below."""
        allcart = self._all_cart()
        lone = cx.Coordinate._create_unchecked(
            allcart.point, {"acceleration": allcart.cconvert(cxc.sph3d)["acceleration"]}
        )
        with pytest.raises(TypeError, match=r"without an order-1 fibre"):
            cxfm.act(self.ROT, None, lone)


class TestJointProlongationPreconditions:
    """What the joint path refuses, and whether it says why correctly.

    These are errors rather than first-order answers, so the message is the
    whole product: it is the only thing telling the caller which fibre to add.
    """

    SPH: ClassVar = cxc.sph3d
    Q0: ClassVar = {
        "r": u.Q(2.0, "kpc"),
        "theta": u.Q(0.9, "rad"),
        "phi": u.Q(0.4, "rad"),
    }
    A0: ClassVar = {
        "r": u.Q(0.1, "kpc/Myr2"),
        "theta": u.Q(0.05, "rad/Myr2"),
        "phi": u.Q(-0.02, "rad/Myr2"),
    }
    CQ: ClassVar = {"x": u.Q(1.0, "kpc"), "y": u.Q(2.0, "kpc"), "z": u.Q(3.0, "kpc")}
    CA: ClassVar = {
        "x": u.Q(0.1, "kpc/Myr2"),
        "y": u.Q(0.2, "kpc/Myr2"),
        "z": u.Q(0.3, "kpc/Myr2"),
    }

    @staticmethod
    def _gap(chart, q, a):
        return cx.Coordinate(
            point=cx.Point(q, chart),
            acceleration=cx.Tangent(a, chart, cxr.coord_basis, cxr.acc),
        )

    def test_a_static_curved_gap_blames_the_curvature(self):
        op = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg"))
        with pytest.raises(TypeError, match=r"is not affine in"):
            cxfm.act(op, None, self._gap(self.SPH, self.Q0, self.A0))

    def test_a_time_dependent_gap_blames_the_time_dependence(self):
        """It is joint because of tau, not curvature -- in a flat chart no less.

        `is_affine_in_chart` reports a `TimeDep` as non-affine in *any* chart,
        so asking about affinity first would blame curvature every time.
        """
        op = cxfm.TimeDep.from_(
            lambda t: cxfm.Translate(
                {
                    "x": u.Q(3.0, "kpc/Myr") * t,
                    "y": u.Q(0.0, "kpc"),
                    "z": u.Q(0.0, "kpc"),
                },
                chart=cxc.cart3d,
            )
        )
        with pytest.raises(TypeError, match=r"is time-dependent"):
            cxfm.act(op, u.Q(1.0, "Myr"), self._gap(cxc.cart3d, self.CQ, self.CA))
        # and specifically NOT the curvature story
        with pytest.raises(TypeError, match=r"^(?!.*not affine).*$"):
            cxfm.act(op, u.Q(1.0, "Myr"), self._gap(cxc.cart3d, self.CQ, self.CA))

    def test_a_physical_basis_fibre_is_refused_in_the_same_words_as_cconvert(self):
        """Its components are rescaled, so they are not jet slots.

        The units caught this downstream as a `UnitConversionError` from deep
        inside the engine, which says nothing about what the caller did.
        """
        crd = cx.Coordinate(
            point=cx.Point(self.Q0, self.SPH),
            velocity=cx.Tangent(
                {
                    "r": u.Q(1.0, "kpc/Myr"),
                    "theta": u.Q(0.3, "rad/Myr"),
                    "phi": u.Q(0.7, "rad/Myr"),
                },
                self.SPH,
                cxr.coord_basis,
                cxr.vel,
            ),
            acceleration=cx.Tangent(
                {
                    "r": u.Q(0.1, "kpc/Myr2"),
                    "theta": u.Q(0.05, "kpc/Myr2"),
                    "phi": u.Q(-0.02, "kpc/Myr2"),
                },
                self.SPH,
                cxr.phys_basis,
                cxr.acc,
            ),
        )
        op = cxfm.Rotate.from_euler("z", u.Q(37.0, "deg"))
        with pytest.raises(TypeError, match=r"non-coordinate basis holds rescaled"):
            cxfm.act(op, None, crd)
