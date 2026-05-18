"""Tests for static transform group classifications."""

__all__: tuple[str, ...] = ()

import unxt as u

import coordinax.transforms as cxfm


def test_concrete_transform_groups_match_spec() -> None:
    """Concrete transforms expose their primary transformation groups."""
    assert cxfm.Identity.groups() == frozenset(
        (cxfm.IdentityGroup, cxfm.DiffeomorphismGroup)
    )
    assert cxfm.Translate.groups() == frozenset(
        (cxfm.EuclideanGroup, cxfm.DiffeomorphismGroup)
    )
    assert cxfm.Rotate.groups() == frozenset(
        (cxfm.SpecialOrthogonalGroup, cxfm.DiffeomorphismGroup)
    )
    assert cxfm.Reflect.groups() == frozenset(
        (cxfm.OrthogonalGroup, cxfm.DiffeomorphismGroup)
    )
    assert cxfm.Scale.groups() == frozenset(
        (cxfm.AffineGroup, cxfm.DiffeomorphismGroup)
    )
    assert cxfm.Shear.groups() == frozenset(
        (cxfm.AffineGroup, cxfm.DiffeomorphismGroup)
    )


def test_composed_rotate_and_translate_promotes_to_euclidean_group() -> None:
    """Rotation composed with translation is Euclidean, not merely rotational."""
    op = cxfm.Rotate.from_euler("z", u.Q(0, "deg")) | cxfm.Translate.from_(
        [1, 0, 0], "m"
    )
    assert op.groups() == frozenset((cxfm.EuclideanGroup, cxfm.DiffeomorphismGroup))


def test_composed_rotate_and_scale_promotes_to_affine_group() -> None:
    """Adding a scale lifts a rigid motion into the affine group."""
    op = cxfm.Rotate.from_euler("z", u.Q(0, "deg")) | cxfm.Scale.from_factors([2, 1, 1])
    assert op.groups() == frozenset((cxfm.AffineGroup, cxfm.DiffeomorphismGroup))


def test_composed_reflect_and_rotate_promotes_to_orthogonal_group() -> None:
    """Reflection composed with rotation stays inside the orthogonal group."""
    op = cxfm.Reflect.from_normal([1, 0, 0]) | cxfm.Rotate.from_euler(
        "z", u.Q(0, "deg")
    )
    assert op.groups() == frozenset((cxfm.OrthogonalGroup, cxfm.DiffeomorphismGroup))


def test_composed_identity_is_neutral_for_group_inference() -> None:
    """Identity should not widen the inferred group of a composition."""
    op = cxfm.Identity() | cxfm.Rotate.from_euler("z", u.Q(0, "deg"))
    assert op.groups() == frozenset(
        (cxfm.SpecialOrthogonalGroup, cxfm.DiffeomorphismGroup)
    )
