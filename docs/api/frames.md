# `coordinax.frames`

The `coordinax.frames` module provides reference frames and frame-transition construction for relating different spatial observers and coordinate perspectives.

## Overview

Frames represent spatial observers. The transform operators themselves live in `coordinax.transforms` (typically imported as `cxfm`), and `coordinax.frames` constructs frame transitions using those operators.

For design philosophy, reference frame models, and practical workflows, see [Working With Frames](../guides/frames.md). For background math, see {ref}`spec § Frame Transforms <frame-transforms>`.

## Transformation Groups

Transformation groups classify transformations by the geometric properties they preserve:

| Group | Parent | Preserves | Intuition |
| --- | --- | --- | --- |
| **Identity** | — | Everything (null) | Do nothing |
| **Diffeomorphism** | — | Smooth structure | Any smooth invertible map |
| **Affine** | Diffeomorphism | Parallelism, hyperplanes | Linear + translation |
| **Euclidean** | Affine | Distances, angles | Rotations + slides |
| **Orthogonal** | Affine | Angles (fixes origin) | Rotations/reflections through origin |
| **Special Orthogonal** | Orthogonal | Orientation (det=+1) | Proper rotations only |
| **Poincaré** | Diffeomorphism | Spacetime intervals | Lorentz boosts + translations |
| **Lorentz** | — | Minkowski metric | Proper & improper boosts |
| **Proper Orthochronous Lorentz** | Lorentz | Space+time orient. | Physical relativistic transforms |

For detailed semantics and use cases of each group, see [Working With Frames](../guides/frames.md#transformation-groups-mathematical-classification).

## Quick Start

```python
import coordinax.frames as cxf
import coordinax.transforms as cxfm
import coordinax.main as cx

# Create transformation between frames
transform = cxf.frame_transition(cxf.alice, cxf.alex)

# Apply to vector
v = cx.Point.from_([1, 2, 3], "m")
v_transformed = cxfm.act(transform, None, v)
```

See [Working With Frames](../guides/frames.md#building-transformations) for composition, inversion, and simplification patterns.

## Functional API

### Frame Operations

- `frame_transition(frame1, frame2)`: construct a transformation operator from one frame to another
- `AbstractReferenceFrame`: base class for defining reference frames

### Transform Operations

Transform operations and transform classes are in [`coordinax.transforms`](transforms.md).

### Built-in Frames

- `Alice`, `Alex`: example stationary frames
- `NoFrame`: identity frame
- `TransformedReferenceFrame`: frame defined relative to a base frame

### Transformation Group Classes (Markers)

Transformation-group marker classes are in [`coordinax.transforms`](transforms.md).

## Design & Integration

For reference frame models, custom frame design, and active transformation semantics, see [Working With Frames](../guides/frames.md). For JAX integration patterns (vmap, jit), see [Working With Frames](../guides/frames.md#jax-integration-patterns).

```{eval-rst}

.. currentmodule:: coordinax.frames

.. automodule:: coordinax.frames
    :exclude-members: aval, default, materialise, enable_materialise

```
