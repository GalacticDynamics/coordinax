"""Tests that frame transforms follow active semantics."""

__all__: tuple[str, ...] = ()

import numpy as np
from hypothesis import given, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.hypothesis.vectors as cxvst
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinax.vectors as cxv


def _as_array(x: object, unit: str) -> np.ndarray:
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


# ===================================================================
# Known-value tests


def test_rotate_from_euler_uses_active_sign_convention() -> None:
    """A +90 degree z-rotation maps x-axis coordinates to positive y."""
    op = cxfm.Rotate.from_euler("z", u.Q(jnp.asarray(90), "deg"))

    q = u.Q(jnp.asarray([1.0, 0.0, 0.0]), "m")
    out = cxfm.act(op, None, q)

    np.testing.assert_allclose(
        _as_array(out, "m"), np.array([0.0, 1.0, 0.0]), rtol=0.0, atol=1e-12
    )


def test_transition_to_transformed_frame_applies_forward_operator() -> None:
    """`frame_transition(base, transformed)` applies `xop` to coordinates."""
    xop = cxfm.Rotate.from_euler("z", u.Q(jnp.asarray(90), "deg"))
    transformed = cxf.TransformedReferenceFrame(cxf.alice, xop)

    op = cxf.frame_transition(cxf.alice, transformed)
    q = u.Q(jnp.asarray([1.0, 0.0, 0.0]), "m")

    expected = cxfm.act(xop, None, q)
    got = cxfm.act(op, None, q)

    np.testing.assert_allclose(_as_array(got, "m"), _as_array(expected, "m"))


def test_transition_from_transformed_frame_applies_inverse_operator() -> None:
    """`frame_transition(transformed, base)` applies `xop.inverse` to coordinates."""
    xop = cxfm.Rotate.from_euler("z", u.Q(jnp.asarray(90), "deg"))
    transformed = cxf.TransformedReferenceFrame(cxf.alice, xop)

    op = cxf.frame_transition(transformed, cxf.alice)
    q = u.Q(jnp.asarray([1.0, 0.0, 0.0]), "m")

    expected = cxfm.act(xop.inverse, None, q)
    got = cxfm.act(op, None, q)

    np.testing.assert_allclose(_as_array(got, "m"), _as_array(expected, "m"))


def test_frame_transition_inverse_roundtrip_for_example_frames() -> None:
    """Alice->Alex and Alex->Alice operators are exact inverses on positions."""
    fwd = cxf.frame_transition(cxf.alice, cxf.alex)
    bwd = cxf.frame_transition(cxf.alex, cxf.alice)

    q = u.Q(jnp.asarray([3.0, -2.0, 5.0]), "m")
    back = cxfm.act(bwd, None, cxfm.act(fwd, None, q))

    np.testing.assert_allclose(_as_array(back, "m"), _as_array(q, "m"))


# ===================================================================
# Property-based tests


class TestActiveSemanticsProperty:
    """Hypothesis-driven property tests for active-transform semantics."""

    @given(angle_deg=st.floats(-360, 360, allow_nan=False, allow_infinity=False))
    @settings(deadline=None)
    def test_euler_z_rotation_sign_convention(self, angle_deg: float) -> None:
        """Active Euler-z rotation by θ maps (1,0,0) to (cos(θ), sin(θ), 0).

        Verifies that `Rotate.from_euler` uses the active convention:
        rotating points by +θ moves the x-axis toward +y.
        """
        op = cxfm.Rotate.from_euler("z", u.Q(jnp.asarray(angle_deg), "deg"))
        q = u.Q(jnp.asarray([1.0, 0.0, 0.0]), "m")
        out = _as_array(cxfm.act(op, None, q), "m")

        theta = jnp.deg2rad(jnp.asarray(angle_deg))
        expected = np.array([float(jnp.cos(theta)), float(jnp.sin(theta)), 0.0])
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)

    @given(
        q=ust.quantities(
            "m",
            shape=(3,),
            elements={"min_value": -1e6, "max_value": 1e6},
        )
    )
    @settings(deadline=None)
    def test_alice_alex_roundtrip(self, q: u.AbstractQuantity) -> None:
        """Alice→Alex→Alice is the identity for any Cartesian 3-vector."""
        fwd = cxfm.act(cxf.frame_transition(cxf.alice, cxf.alex), None, q)
        back = cxfm.act(cxf.frame_transition(cxf.alex, cxf.alice), None, fwd)

        np.testing.assert_allclose(_as_array(back, "m"), _as_array(q, "m"), atol=1e-6)

    @given(
        angle_deg=st.floats(-360, 360, allow_nan=False, allow_infinity=False),
        q=cxvst.vectors(
            cxc.cart3d,
            cxr.point,
            elements={"min_value": -1e6, "max_value": 1e6, "allow_nan": False},
        ),
    )
    @settings(deadline=None)
    def test_transformed_frame_roundtrip(self, angle_deg: float, q: object) -> None:
        """TransformedReferenceFrame→base→transformed is the identity on positions."""
        assert isinstance(q, cxv.Point)

        xop = cxfm.Rotate.from_euler("z", u.Q(jnp.asarray(angle_deg), "deg"))
        transformed = cxf.TransformedReferenceFrame(cxf.alice, xop)

        fwd = cxfm.act(cxf.frame_transition(cxf.alice, transformed), None, q)
        back = cxfm.act(cxf.frame_transition(transformed, cxf.alice), None, fwd)

        assert isinstance(back, cxv.Point)
        for key in ("x", "y", "z"):
            np.testing.assert_allclose(
                float(u.ustrip("m", q.data[key])),
                float(u.ustrip("m", back.data[key])),
                atol=1e-10,
            )
