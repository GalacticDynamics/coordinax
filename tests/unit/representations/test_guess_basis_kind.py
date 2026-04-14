"""Tests for guess_basis_kind."""

__all__: tuple[str, ...] = ()

import pytest
from hypothesis import given

import unxt as u

import coordinax.hypothesis.representations as cxrst
import coordinax.representations as cxr

# ===================================================================
# Identity dispatch


def test_identity_no_basis() -> None:
    """guess_basis_kind(NoBasis()) returns the same object."""
    basis = cxr.NoBasis()
    assert cxr.guess_basis_kind(basis) is basis


def test_identity_canonical_instance() -> None:
    """guess_basis_kind(no_basis) returns the canonical instance."""
    result = cxr.guess_basis_kind(cxr.no_basis)
    assert result is cxr.no_basis


@given(basis=cxrst.bases())
def test_identity_any_basis(basis: cxr.AbstractBasis) -> None:
    """guess_basis_kind returns the input unchanged for any AbstractBasis."""
    assert cxr.guess_basis_kind(basis) is basis


# ===================================================================
# Dimension dispatch


def test_dimension_length_returns_no_basis() -> None:
    """guess_basis_kind(u.dimension('length')) returns no_basis."""
    result = cxr.guess_basis_kind(u.dimension("length"))
    assert result == cxr.no_basis


def test_dimension_angle_returns_no_basis() -> None:
    """guess_basis_kind(u.dimension('angle')) returns no_basis."""
    result = cxr.guess_basis_kind(u.dimension("angle"))
    assert result == cxr.no_basis


def test_dimension_unknown_raises() -> None:
    """guess_basis_kind raises ValueError for an unregistered dimension."""
    with pytest.raises(ValueError, match="Cannot infer basis kind"):
        cxr.guess_basis_kind(u.dimension("time"))


# ===================================================================
# Quantity dispatch


def test_quantity_length_returns_no_basis() -> None:
    """guess_basis_kind(Quantity in meters) returns no_basis."""
    result = cxr.guess_basis_kind(u.Q(1.0, "m"))
    assert result == cxr.no_basis


def test_quantity_angle_returns_no_basis() -> None:
    """guess_basis_kind(Quantity in radians) returns no_basis."""
    result = cxr.guess_basis_kind(u.Q(0.5, "rad"))
    assert result == cxr.no_basis


def test_quantity_unknown_dim_raises() -> None:
    """guess_basis_kind raises ValueError for a quantity with unknown dimension."""
    with pytest.raises(ValueError, match="Cannot infer basis kind"):
        cxr.guess_basis_kind(u.Q(1.0, "s"))


# ===================================================================
# CDict dispatch


def test_cdict_cartesian_returns_no_basis() -> None:
    """guess_basis_kind({'x': Q(m), 'y': Q(m)}) returns no_basis."""
    d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    result = cxr.guess_basis_kind(d)
    assert result == cxr.no_basis


def test_cdict_cartesian3d_returns_no_basis() -> None:
    """guess_basis_kind({'x': Q(m), 'y': Q(m), 'z': Q(m)}) returns no_basis."""
    d = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    result = cxr.guess_basis_kind(d)
    assert result == cxr.no_basis


def test_cdict_angular_returns_no_basis() -> None:
    """guess_basis_kind({'lon': Q(deg), 'lat': Q(deg)}) returns no_basis."""
    d = {"lon": u.Q(1.0, "deg"), "lat": u.Q(2.0, "deg")}
    result = cxr.guess_basis_kind(d)
    assert result == cxr.no_basis


def test_cdict_empty_raises() -> None:
    """guess_basis_kind({}) raises ValueError."""
    with pytest.raises(ValueError, match="Cannot infer basis kind without dimensions"):
        cxr.guess_basis_kind({})
