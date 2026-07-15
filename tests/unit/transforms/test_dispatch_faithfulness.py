"""Guard plum dispatch-cache faithfulness of the hot-path generic functions.

plum only uses its (fast, type-keyed) method cache when every registered
signature of a generic function is "faithful". Two things can silently break
that and force a full (~200x slower) resolution on every call:

1. A **parametric** annotation such as ``dict[str, Any]`` — hence ``CDict`` is
   the bare ``dict`` (see ``custom_types.py``).
2. An **unfaithful class**: ``jax.Array``'s metaclass overrides
   ``__instancecheck__``, so plum treats it — and every ``ArrayLike`` union
   containing it — as unfaithful *unless* ``jax.Array.__faithful__`` is set.
   ``quax >= 0.3.6`` sets it (in ``quax._compat``), which is why plain
   ``ArrayLike`` / ``jax.Array`` annotations are safe in dispatch signatures.

The first test pins the upstream ``quax`` guarantee directly; the rest exercise
each hot-path function once (resolvers populate lazily) and assert the whole
resolver is faithful, so a future registration or dependency regression cannot
silently degrade dispatch performance. If one fails, find the offending method
via::

    [m for m in fn._resolver.methods if not m.signature.is_faithful]
"""

from jaxtyping import ArrayLike

import jax
import jax.numpy as jnp
from plum import Signature

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


def _point():
    return {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}


def test_jax_array_and_arraylike_are_faithful():
    # `ArrayLike` here is `jaxtyping.ArrayLike` — the exact type the hot-path
    # dispatch signatures annotate with (e.g. register_apply / translate) — so
    # this pins the type that actually matters, not merely `jax.typing`'s.
    # Provided by quax >= 0.3.6 (`jax.Array.__faithful__ = True`). If this
    # fails, the quax floor regressed and every ArrayLike/jax.Array dispatch
    # signature below silently loses the method cache.
    assert Signature(jax.Array).is_faithful
    assert Signature(ArrayLike).is_faithful


def test_act_dispatch_is_faithful():
    op = cxfm.Translate.from_([1, 2, 3], "km")
    cxfm.act(op, None, _point(), cxc.cart3d, cxr.point)
    cxfm.act(op, None, _point())
    cxfm.act(op, None, u.Q([1.0, 0.0, 0.0], "km"))
    cxfm.act(op, None, jnp.array([1.0, 0.0, 0.0]), usys=u.unitsystems.si)
    assert cxfm.act._resolver.is_faithful


def test_pushforward_dispatch_is_faithful():
    op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    v = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel)
    assert cxfm.pushforward._resolver.is_faithful


def test_prolong_dispatch_is_faithful():
    op = cxfm.Translate.from_([1, 2, 3], "km")
    jet = {0: _point()}
    cxfm.prolong(op, None, jet, cxc.cart3d)
    assert cxfm.prolong._resolver.is_faithful


def test_pt_map_dispatch_is_faithful():
    cxc.pt_map(_point(), cxc.cart3d, cxc.sph3d)
    cxc.pt_map(jnp.array([1.0, 0.0, 0.0]), cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
    assert cxc.pt_map._resolver.is_faithful
