# `coordinax.transforms`

The `coordinax.transforms` module (commonly imported as `cxfm`) provides transform operators, transform composition APIs, and transformation-group marker classes.

## Overview

`coordinax.transforms` is the canonical transform namespace. `coordinax.frames` depends on it to build frame-transition operators.

## Quick Start

```python
import coordinax.frames as cxf
import coordinax.transforms as cxfm
import coordinax as cx
import unxt as u

op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
v = cx.Point.from_([1, 0, 0], "m")

rotated = cxfm.act(op, None, v)

# frame transitions still come from coordinax.frames
frame_op = cxf.frame_transition(cxf.alice, cxf.alex)
out = cxfm.act(frame_op, None, v)
```

## Functional API

- `act(transform, tau, x)`: apply a transform to data (the kinematic prolongation for tangent data — see the [transforms guide](../guides/transforms.md#time-dependence-couples-the-ladder-kinematic-prolongation))
- `pushforward(transform, tau, v, chart, rep, *, at)`: the frozen-tau spatial differential — the transformation law for displacement data
- `act_jet(transform, tau, jet, chart)`: joint action on a jet `{0: point, 1: velocity, 2: acceleration, ...}`
- `simplify(transform)`: simplify transform structure
- `compose(*transforms)`: compose transforms into `Composed`
- `evaluate_at(transform, tau)`: evaluate every `TimeDep` part of a transform at `tau`, returning a constant transform
- `is_time_dependent(transform)`: declared trait — whether the transform's point action depends on `tau` (`True` for `TimeDep` and `Boost`; the disjunction of children for `Composed`)
- `tau_derivative(fn, tau, n=1)`: unit-aware n-th time derivative of a parameter function

## Transform Types

- `AbstractTransform`: base class for transforms
- `Identity`: null transform
- `Translate`: additive offset on the semantic-kind ladder (displacement by default; a velocity kick with `semantic_kind=vel`)
- `Boost`: Galilean boost (moves points by `dv * tau`, shifts velocities by `dv`)
- `Rotate`: pure rotation
- `Reflect`: Householder hyperplane reflection
- `Scale`: Cartesian linear scaling
- `Shear`: Cartesian linear shear
- `Composed`: ordered transform composition
- `TimeDep`: a one-parameter family of transforms, `builder(tau) -> AbstractTransform` — the mechanism for all time-dependent transforms (see the [transforms guide](../guides/transforms.md#time-dependent-parameters)). `TimeDep.from_(fn, *args, **kw)` builds one from a user-defined function, binding `args`/`kw` as differentiable leaves (`fn` takes `tau` **last**)
- `identity`: convenience instance of `Identity`

(transforms-builders)=

## `coordinax.transforms.builders`

A builder is a one-parameter family `builder(tau) -> AbstractTransform`. It is **not** a transform: it has no `act`, no `inverse`, no `@` — only the `TimeDep` wrapping it does. That is why builders have their own namespace.

Built-in families you construct yourself:

- `cxfm.builders.RotationAboutAxis`: uniform rotation about a fixed axis
- `cxfm.builders.UniformTranslation`: straight-line motion at constant velocity

Builders the algebra _returns_ — you receive these rather than construct them, but they show up in `repr`, `jax.tree` paths, and tracing errors:

- `cxfm.builders.FnBuilder`: wraps a bare function, from `TimeDep.from_(fn)` (static field; see its docstring for the differentiable alternative)
- `cxfm.builders.ConstBuilder`: constant family `b(tau) = op`, from `TimeDep @ static_transform`
- `cxfm.builders.ComposedBuilder`: pointwise composition `(a @ b)(tau) = a(tau) @ b(tau)`, from `TimeDep @ TimeDep`
- `cxfm.builders.InverseBuilder`: pointwise inverse `b(tau).inverse`, from `TimeDep.inverse`

(transforms-groups)=

## `coordinax.transforms.groups`

Transformation-group marker classes, used for classification and dispatch; not instantiated directly:

- `cxfm.groups.AbstractTransformGroup`
- `cxfm.groups.IdentityGroup`
- `cxfm.groups.DiffeomorphismGroup`
- `cxfm.groups.AffineGroup`
- `cxfm.groups.EuclideanGroup`
- `cxfm.groups.OrthogonalGroup`
- `cxfm.groups.SpecialOrthogonalGroup`
- `cxfm.groups.LorentzGroup`
- `cxfm.groups.ProperOrthochronousLorentzGroup`
- `cxfm.groups.PoincareGroup`

```{eval-rst}

.. currentmodule:: coordinax.transforms

.. automodule:: coordinax.transforms
    :exclude-members: aval, default, materialise, enable_materialise, builders, groups

.. automodule:: coordinax.transforms.builders
    :exclude-members: aval, default, materialise, enable_materialise

.. automodule:: coordinax.transforms.groups
    :exclude-members: aval, default, materialise, enable_materialise

```
