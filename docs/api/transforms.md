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
- `prolong(transform, tau, jet, chart)`: joint action on a jet `{0: point, 1: velocity, 2: acceleration, ...}`
- `simplify(transform)`: simplify transform structure
- `compose(*transforms)`: compose transforms into `Composed`
- `materialize_transform(transform, tau)`: materialize time-dependent transform parameters
- `is_time_dependent(transform)`: whether any parameter is a callable of `tau`
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
- `identity`: convenience instance of `Identity`

## Transformation Group Classes (Markers)

Used for classification and dispatch; not instantiated directly:

- `AbstractTransformGroup`
- `IdentityGroup`
- `DiffeomorphismGroup`
- `AffineGroup`
- `EuclideanGroup`
- `OrthogonalGroup`
- `SpecialOrthogonalGroup`
- `LorentzGroup`
- `ProperOrthochronousLorentzGroup`
- `PoincareGroup`

```{eval-rst}

.. currentmodule:: coordinax.transforms

.. automodule:: coordinax.transforms
    :exclude-members: aval, default, materialise, enable_materialise

```
