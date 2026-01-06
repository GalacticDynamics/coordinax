"""Tests for the representations_like strategy."""

from coordinax_hypothesis import representations_like
from hypothesis import given, settings

import coordinax as cx


@given(rep=representations_like(cx.r.cart3d))
@settings(max_examples=50)
def test_representations_like_cart3d(rep: cx.r.AbstractRep) -> None:
    """Test representations_like with Cart3D template."""
    assert isinstance(rep, cx.r.Abstract3D)
    assert rep.dimensionality == 3


@given(rep=representations_like(cx.r.sph3d))
@settings(max_examples=50)
def test_representations_like_spherical3d(rep: cx.r.AbstractRep) -> None:
    """Test representations_like with Spherical3D template."""
    assert isinstance(rep, cx.r.Abstract3D)
    assert isinstance(rep, cx.r.AbstractSpherical3D)
    assert rep.dimensionality == 3


@given(rep=representations_like(cx.r.polar2d))
@settings(max_examples=50)
def test_representations_like_polar2d(rep: cx.r.AbstractRep) -> None:
    """Test representations_like with Polar2D template."""
    assert isinstance(rep, cx.r.Abstract2D)
    assert rep.dimensionality == 2


@given(rep=representations_like(cx.r.radial1d))
@settings(max_examples=50)
def test_representations_like_radial1d(rep: cx.r.AbstractRep) -> None:
    """Test representations_like with Radial1D template."""
    assert isinstance(rep, cx.r.Abstract1D)
    assert rep.dimensionality == 1


@given(rep=representations_like(cx.r.twosphere))
@settings(max_examples=50)
def test_representations_like_twosphere(rep: cx.r.AbstractRep) -> None:
    """Test representations_like with TwoSphere template."""
    assert isinstance(rep, cx.r.Abstract2D)
    assert rep.dimensionality == 2
