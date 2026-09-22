"""Tests for static transform group classifications."""

__all__: tuple[str, ...] = ()

import equinox as eqx
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.transforms as cxfm


def test_concrete_transform_groups_match_spec() -> None:
    """Concrete transforms expose their primary transformation groups."""
    assert cxfm.Identity.groups() == frozenset(
        (cxfm.groups.IdentityGroup, cxfm.groups.DiffeomorphismGroup)
    )
    assert cxfm.Translate.groups() == frozenset(
        (cxfm.groups.EuclideanGroup, cxfm.groups.DiffeomorphismGroup)
    )
    # `Rotate.groups` is an instance method: the group depends on `sign(det R)`,
    # not on the type. See `test_rotate_groups_follow_the_determinant_sign`.
    assert cxfm.Rotate(jnp.eye(3)).groups() == frozenset(
        (cxfm.groups.SpecialOrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
    )
    assert cxfm.Reflect.groups() == frozenset(
        (cxfm.groups.OrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
    )
    assert cxfm.Scale.groups() == frozenset(
        (cxfm.groups.AffineGroup, cxfm.groups.DiffeomorphismGroup)
    )
    assert cxfm.Shear.groups() == frozenset(
        (cxfm.groups.AffineGroup, cxfm.groups.DiffeomorphismGroup)
    )


def test_composed_rotate_and_translate_promotes_to_euclidean_group() -> None:
    """Rotation composed with translation is Euclidean, not merely rotational."""
    op = cxfm.Rotate.from_euler("z", u.Q(0, "deg")) | cxfm.Translate.from_(
        [1, 0, 0], "m"
    )
    assert op.groups() == frozenset(
        (cxfm.groups.EuclideanGroup, cxfm.groups.DiffeomorphismGroup)
    )


def test_composed_rotate_and_scale_promotes_to_affine_group() -> None:
    """Adding a scale lifts a rigid motion into the affine group."""
    op = cxfm.Rotate.from_euler("z", u.Q(0, "deg")) | cxfm.Scale.from_factors([2, 1, 1])
    assert op.groups() == frozenset(
        (cxfm.groups.AffineGroup, cxfm.groups.DiffeomorphismGroup)
    )


def test_composed_reflect_and_rotate_promotes_to_orthogonal_group() -> None:
    """Reflection composed with rotation stays inside the orthogonal group."""
    op = cxfm.Reflect.from_normal([1, 0, 0]) | cxfm.Rotate.from_euler(
        "z", u.Q(0, "deg")
    )
    assert op.groups() == frozenset(
        (cxfm.groups.OrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
    )


def test_composed_identity_is_neutral_for_group_inference() -> None:
    """Identity should not widen the inferred group of a composition."""
    op = cxfm.Identity() | cxfm.Rotate.from_euler("z", u.Q(0, "deg"))
    assert op.groups() == frozenset(
        (cxfm.groups.SpecialOrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
    )


def test_rotate_is_always_special_orthogonal() -> None:
    """`Rotate` is SO(n) by construction, so `groups` needs no determinant.

    Regression for #938: `groups` once answered `SpecialOrthogonalGroup`
    unconditionally while the constructor accepted improper matrices, so a
    `det = -1` matrix claimed an orientation it flips -- and the claim
    propagated through `Composed.groups` and `least_common_supergroup`. The
    constructor now rejects those, which makes the constant answer true.
    """
    proper = cxfm.Rotate(
        jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert proper.groups() == frozenset(
        (cxfm.groups.SpecialOrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="det R"):
        cxfm.Rotate(jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])))


def test_negating_a_rotation_leaves_so_n_in_odd_dimensions() -> None:
    """``-R`` flips the determinant in 3D, so it is a `Linear` in O(n).

    `Rotate` is not closed under negation, which is why `__neg__` returns the
    wider type carrying the group that does survive.
    """
    R = cxfm.Rotate(jnp.eye(3))
    assert cxfm.groups.SpecialOrthogonalGroup in R.groups()

    neg = -R
    assert isinstance(neg, cxfm.Linear)
    assert cxfm.groups.OrthogonalGroup in neg.groups()
    assert cxfm.groups.SpecialOrthogonalGroup not in neg.groups()

    # `Linear` *is* closed under negation: the matrix round-trips.
    assert bool(jnp.allclose((-neg).matrix, R.matrix))
