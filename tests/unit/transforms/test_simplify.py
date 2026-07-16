"""Tests for ``simplify`` pairwise collapse and trace-safety (#539)."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Real

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


def _xyz(d):
    return np.array([float(d[k].ustrip("m")) for k in "xyz"])


def _point(x=1.0, y=2.0, z=3.0):
    return {"x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")}


def _acts_equal(a, b):
    """Two operators act identically on a representative point."""
    p = _point()
    ra = cxfm.act(a, None, p, cxc.cart3d, cxr.point)
    rb = cxfm.act(b, None, p, cxc.cart3d, cxr.point)
    np.testing.assert_allclose(_xyz(ra), _xyz(rb), atol=1e-6)


# ===================================================================
# Pairwise merges


def test_adjacent_rotations_merge_to_one() -> None:
    R1 = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    R2 = cxfm.Rotate.from_euler("z", u.Q(60, "deg"))
    pipe = cxfm.Composed((R1, R2))
    out = cxfm.simplify(pipe)
    assert isinstance(out, cxfm.Rotate)
    # The single rotation acts exactly like the two-step pipe.
    _acts_equal(out, pipe)
    np.testing.assert_allclose(np.asarray(out.R), np.asarray((R1 @ R2).R))


def test_adjacent_translations_merge_to_one() -> None:
    T1 = cxfm.Translate.from_([1, 2, 3], "km")
    T2 = cxfm.Translate.from_([4, 5, 6], "km")
    pipe = cxfm.Composed((T1, T2))
    out = cxfm.simplify(pipe)
    assert isinstance(out, cxfm.Translate)
    _acts_equal(out, pipe)


def test_inverse_pair_cancels_to_identity() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
    assert cxfm.simplify(cxfm.Composed((R, R.inverse))) is cxfm.identity


def test_identity_strip_re_exposes_adjacency() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
    pipe = cxfm.Composed((R, cxfm.Identity(), R.inverse))
    assert cxfm.simplify(pipe) is cxfm.identity


def test_non_mergeable_pair_is_preserved() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    T = cxfm.Translate.from_([1, 0, 0], "km")
    out = cxfm.simplify(cxfm.Composed((R, T)))
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2
    _acts_equal(out, cxfm.Composed((R, T)))


def test_different_semantic_kind_translates_do_not_merge() -> None:
    disp = cxfm.Translate.from_([1, 0, 0], "km")
    vel = replace(disp, semantic_kind=cxr.vel)
    out = cxfm.simplify(cxfm.Composed((disp, vel)))
    # A displacement and a velocity-kick are different actions: not merged.
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2


def test_time_dependent_rotations_do_not_merge() -> None:
    def R_of_t(t) -> Real[Array, "3 3"]:
        th = t.ustrip("s")
        c, s = jnp.cos(th), jnp.sin(th)
        return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    td = cxfm.Rotate.from_(R_of_t)
    out = cxfm.simplify(cxfm.Composed((td, td)))
    assert isinstance(out, cxfm.Composed)
    assert len(out.transforms) == 2


# ===================================================================
# Trace-safety: the approx flag


def test_approx_false_merges_but_skips_identity_collapse() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    pipe = cxfm.Composed((R, R.inverse))
    # Structural merge happens (Rotate @ Rotate) but the value-inspecting
    # collapse to Identity is skipped.
    out = cxfm.simplify(pipe, approx=False)
    assert isinstance(out, cxfm.Rotate)
    # It is still numerically the identity rotation.
    np.testing.assert_allclose(np.asarray(out.R), np.eye(3), atol=1e-6)


def test_approx_false_works_under_jit() -> None:
    R = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    pipe = cxfm.Composed((R, R.inverse))
    out = jax.jit(lambda op: cxfm.simplify(op, approx=False))(pipe)
    assert isinstance(out, cxfm.Rotate)


def test_default_simplify_is_not_jit_safe() -> None:
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda op: cxfm.simplify(op))(cxfm.Rotate(jnp.eye(3)))
