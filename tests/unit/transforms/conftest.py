"""Shared fixtures for frames/act tests."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.transforms as cxfm
from coordinax.internal import QMatrix

# ===================================================================
# Transform fixtures


@pytest.fixture
def identity_op():
    """Identity transform."""
    return cxfm.Identity()


@pytest.fixture
def rotate_op():
    """90-degree rotation about z-axis.

    Maps: (1,0,0) -> (0,1,0), (0,1,0) -> (-1,0,0)
    """
    return cxfm.Rotate.from_euler("z", u.Q(90, "deg"))


@pytest.fixture
def reflect_op():
    """Reflection across the yz-plane.

    Maps: (1,0,0) -> (-1,0,0), (0,1,0) -> (0,1,0)
    """
    return cxfm.Reflect.from_normal([1, 0, 0])


@pytest.fixture
def translate_op():
    """Translate by (1, 0, 0) km."""
    return cxfm.Translate.from_([1, 0, 0], "km")


@pytest.fixture
def composed_op(translate_op, rotate_op):
    """Composed: translate then rotate.

    Step 1: translate (1,0,0) -> (2,0,0)
    Step 2: rotate  (2,0,0) -> (0,2,0)

    So (1,0,0) km  ->  (0,2,0) km
    """
    return translate_op | rotate_op


# ===================================================================
# Object fixtures — all represent the point (1, 0, 0) in km


@pytest.fixture
def array_3d():
    """Bare JAX array [1, 0, 0]."""
    return jnp.array([1, 0, 0])


@pytest.fixture
def quantity_3d():
    """Quantity [1, 0, 0] km."""
    return u.Q([1, 0, 0], "km")


@pytest.fixture
def qmatrix_3d():
    """QMatrix [1, 0, 0] with uniform km units."""
    return QMatrix(
        jnp.array([1, 0, 0]), unit=(u.unit("km"), u.unit("km"), u.unit("km"))
    )


@pytest.fixture
def cdict_3d():
    """CDict {"x": 1 km, "y": 0 km, "z": 0 km}."""
    return {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}


@pytest.fixture
def vector_3d():
    """Vector from [1, 0, 0] km."""
    return cx.Point.from_([1, 0, 0], "km")


@pytest.fixture
def coord_3d():
    """Coordinate with Alice frame."""
    return cx.Point.from_([1, 0, 0], "km", cxf.alice)


@pytest.fixture
def coord_xfm_3d():
    """Coordinate with a TransformedReferenceFrame.

    The frame is Alice's frame, rotated 90 degrees about z-axis.
    The coordinates (1, 0, 0) km are *in* the rotated frame.
    """
    rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    frame = cxf.TransformedReferenceFrame(cxf.alice, rot)
    return cx.Point.from_([1, 0, 0], "km", frame)


# ===================================================================
# Expected results after each transform applied to (1, 0, 0) km


EXPECTED_IDENTITY = (1, 0, 0)  # no change
EXPECTED_ROTATE = (0, 1, 0)  # 90° z-rotation
EXPECTED_REFLECT = (-1, 0, 0)  # reflection across yz-plane
EXPECTED_TRANSLATE = (2, 0, 0)  # +1 km in x
EXPECTED_COMPOSED = (0, 2, 0)  # translate then rotate
