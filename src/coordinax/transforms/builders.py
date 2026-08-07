"""Builders for `~coordinax.transforms.TimeDep` transforms.

A builder is a one-parameter *family* of transforms: a callable
``builder(tau) -> AbstractTransform``. It is **not** itself a transform — it has
no ``act``, no ``inverse``, no ``@``. Only the `~coordinax.transforms.TimeDep`
that wraps it is a transform. That distinction is why builders live in their own
namespace instead of alongside the transform classes.

`RotationAboutAxis` and `UniformTranslation` are the built-in families you
construct yourself. `FnBuilder`, `ConstBuilder`, `ComposedBuilder`, and
`InverseBuilder` are returned by the `TimeDep` algebra; you rarely build them by
hand, but they are public because they appear in ``repr``, in `jax.tree` paths,
and in tracing errors.

Examples
--------
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import coordinax.transforms as cxfm

>>> b = cxfm.builders.RotationAboutAxis(u.Q(90, "deg/s"), axis=jnp.array([0., 0., 1.]))
>>> cxfm.TimeDep(b)
TimeDep(RotationAboutAxis(...))

"""

__all__ = (
    "RotationAboutAxis",
    "UniformTranslation",
    "FnBuilder",
    "ConstBuilder",
    "ComposedBuilder",
    "InverseBuilder",
)

from ._src.actions.builders import RotationAboutAxis, UniformTranslation
from ._src.actions.timedep import (
    ComposedBuilder,
    ConstBuilder,
    FnBuilder,
    InverseBuilder,
)
