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

### Index Raising and Lowering

```python
# Raise an index (covariant to contravariant)
v_up = cxm.raise_index(metric, chart, v_down, at=p)

# Lower an index (contravariant to covariant)
v_down = cxm.lower_index(metric, chart, v_up, at=p)
```

### Computing Norms

```python
# Compute the norm of a vector using the metric
norm = cxm.norm(metric, chart, v, at=p)
```

```{eval-rst}

.. currentmodule:: coordinax.metrics

.. automodule:: coordinax.metrics
    :exclude-members: aval, default, materialise, enable_materialise

```
