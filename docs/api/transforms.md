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
- `RotationAboutAxis`: built-in `TimeDep` builder for uniform rotation about a fixed axis
- `UniformTranslation`: built-in `TimeDep` builder for straight-line motion at constant velocity
- `identity`: convenience instance of `Identity`

## Transformation Group Classes (Markers)

These live in the `coordinax.transforms.groups` sub-namespace, reached as `cxfm.groups.<Name>`. They are used for classification and dispatch, and are never instantiated:

- `groups.AbstractTransformGroup`
- `groups.IdentityGroup`
- `groups.DiffeomorphismGroup`
- `groups.AffineGroup`
- `groups.EuclideanGroup`
- `groups.OrthogonalGroup`
- `groups.SpecialOrthogonalGroup`
- `groups.LorentzGroup`
- `groups.ProperOrthochronousLorentzGroup`
- `groups.PoincareGroup`

```{eval-rst}

.. currentmodule:: coordinax.transforms

.. automodule:: coordinax.transforms
    :exclude-members: aval, default, materialise, enable_materialise

.. currentmodule:: coordinax.transforms.groups

.. automodule:: coordinax.transforms.groups

```
