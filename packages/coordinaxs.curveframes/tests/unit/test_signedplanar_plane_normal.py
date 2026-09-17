"""`SignedPlanarBuilder` must not take a string for its gauge.

`plane_normal` is a *dynamic* field, so a string sat there as a pytree leaf:
construction, `jax.tree.leaves` and `location` all succeeded, and the failure
arrived only in `_plane_normal` as ``not a valid JAX array type``. That is the
defect #921 fixed for `BishopBuilder`, which was still live on its sibling --
and reached most easily by copying Bishop's ``normal_0="auto"`` across.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def cubic(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """A cubic with an inflection at the origin."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "m")


@pytest.mark.parametrize("bad", ["auto", "z", "", "[0, 0, 1]"])
def test_a_string_plane_normal_is_refused(bad: str) -> None:
    """`"auto"` is the likely slip: it is `BishopBuilder`'s sentinel, not one here."""
    with pytest.raises(TypeError, match="is not a plane normal"):
        cxfc.SignedPlanarBuilder(cubic, "s", plane_normal=bad)


def test_the_message_says_omission_is_the_z_axis() -> None:
    """The way out is to drop the argument, not to spell the sentinel better."""
    with pytest.raises(TypeError) as excinfo:
        cxfc.SignedPlanarBuilder(cubic, "s", plane_normal="auto")

    assert "z-axis" in str(excinfo.value)


def test_omitting_it_still_means_the_z_axis() -> None:
    """The documented default, unchanged by the guard."""
    omitted = cxfc.SignedPlanarBuilder(cubic, "s")
    z_axis = cxfc.SignedPlanarBuilder(
        cubic, "s", plane_normal=jnp.asarray([0.0, 0.0, 1.0])
    )

    assert np.allclose(
        np.asarray(omitted.rotation_matrix(u.Q(0.5, "s"))),
        np.asarray(z_axis.rotation_matrix(u.Q(0.5, "s"))),
    )


def test_a_vector_leaves_no_string_in_the_tree() -> None:
    """What the guard is protecting: the pytree holds arrays, not text."""
    b = cxfc.SignedPlanarBuilder(cubic, "s", plane_normal=jnp.asarray([0.0, 1.0, 0.0]))

    assert not any(isinstance(leaf, str) for leaf in jax.tree.leaves(b))
