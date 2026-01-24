# `coordinax.metrics`

The `coordinax.metrics` module provides metric tensors for measuring distances
and angles in coordinate systems.

## Overview

A **metric** defines the inner product on tangent spaces, enabling:

- Distance measurements
- Angle calculations
- Index raising/lowering operations
- Norm computations

## Quick Start

```python
import coordinax.charts as cxc
import coordinax.metrics as cxm
import unxt as u

# Get the metric for a chart
metric = cxm.metric_of(cxc.cart3d)

# Compute the metric matrix at a point
p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
g = metric.metric_matrix(cxc.cart3d, p)

# Compute the norm of a vector
v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
magnitude = cxm.norm(metric, cxc.cart3d, v)  # Returns 5 m
```

## Built-in Metrics

- **Euclidean**: Standard flat metric (default for Cartesian charts)
- **Sphere intrinsic**: Metric on the 2-sphere surface
- **Minkowski**: Spacetime metric for special relativity

## Metric Operations

### Getting the Metric for a Chart

```python
import coordinax.metrics as cxm
import coordinax.charts as cxc

# Euclidean metric for Cartesian coordinates
metric = cxm.metric_of(cxc.cart3d)

# Sphere intrinsic metric
sphere_metric = cxm.metric_of(cxc.twosphere)
```

### Computing Norms

The `norm` function computes the magnitude of a vector using the metric tensor:

```python
import coordinax.charts as cxc
import coordinax.metrics as cxm
import unxt as u

# Euclidean norm (position-independent)
metric = cxm.metric_of(cxc.cart3d)
v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
magnitude = cxm.norm(metric, cxc.cart3d, v)  # 5 m
```

For curved metrics like the sphere, the norm depends on position:

```python
import jax.numpy as jnp

# On a sphere, the metric varies with latitude
sphere_metric = cxm.metric_of(cxc.twosphere)
p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0, "rad")}
v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(1, "rad/s")}
magnitude = cxm.norm(sphere_metric, cxc.twosphere, v, at=p)
```

```{eval-rst}

.. currentmodule:: coordinax.metrics

.. automodule:: coordinax.metrics
    :exclude-members: aval, default, materialise, enable_materialise

```
