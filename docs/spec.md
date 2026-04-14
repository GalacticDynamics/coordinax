# Coordinax Core Specification

This document is the **normative** specification for the mathematical and software design of the `coordinax` coordinate and vector system. It defines a low-level, mathematically correct framework—**Charts**, **Metrics**, **Manifolds**, **Frames**, **Embeddings**, and **Maps**—and a high-level user API built on top of them -- **Vector**, **Reference Frames**, **Coordinate**. The goals are:

1. **Correctness-first foundation**: rigorous definitions of points, tangents, their transformations, etc.
2. **Ergonomic high-level API**: common tasks should not expose low-level details.
3. **Extensibility via multiple dispatch**: functional API for existing functions are added by registration, not by editing core logic.
4. **JAX compatibility**: dispatch resolves on static Python objects; numerical kernels are pure and traceable.

## Table of Contents

```{contents}
:depth: 2
```

</br>

---

(math-spec)=

# The Math

In this section we fix the _mathematical objects_ that `coordinax` represents and the _transformation laws_ that determine how component data moves. This is a high-level overview; later sections specify each object and API in detail.

(math-spec-angles-and-distances)=

## Angles and Distances

<details>
<summary>Software Specification</summary>

- [Angles](#software-spec-angles)
- [Distances](#software-spec-distances)

</details>

A **quantity** is a pair $(x, u)$ where $x \in \mathbb{R}$ (or $\mathbb{R}^n$) is a numerical value and $u$ is a unit of measurement. Two quantities are physically equivalent when their values scale inversely with their units; unit conversion is a bijection on $\mathbb{R}$. _(Unit arithmetic is handled by `unxt`.)_

An **angular quantity** is special because it represents an element of the circle $S^1 = \mathbb{R}/2\pi\mathbb{Z}$, not of $\mathbb{R}$. $S^1$ is compact and has no boundary: there is no intrinsically "first" or "last" angle, and the space wraps. To store an angle as a real number one must choose a **branch cut** — a point excluded from the circle used as the interval endpoint. The two standard choices are $[0, 2\pi)$ and $(-\pi, \pi]$; both represent the same $S^1$. The cut is an artifact of representation: arithmetic that crosses it requires explicit wrapping, and the same physical angle has different numerical values under the two conventions.

A **distance quantity** has dimensions of length and, in the strict metric sense, is non-negative: it measures the magnitude of a displacement. A **radial coordinate**, however, can be signed — either because the coordinate system parameterizes a full line ($r \in \mathbb{R}$), or because a negative value is given physical meaning via the antipodal map $(r, \theta, \phi) \equiv (-r,\, \pi - \theta,\, \phi + \pi)$. The distinction between "distance" ($r \geq 0$) and "signed radial coordinate" ($r \in \mathbb{R}$) is a domain constraint on the same dimensional quantity. The point $r = 0$ is a coordinate singularity in spherical and cylindrical systems: the angular coordinates are degenerate there.

</br>

(math-spec-points)=

## Points

A point is an abstract geometric object. In this section we will explore how points can be concretized into coordinates -- e.g. "x", "y", "z" -- in coordinate representations --- e.g. Cartesian --- on manifolds -- e.g. $\mathbb{R}^3$. We will see how these coordinate representations of these points can be changed in the same manifold and across manifolds. We will also explore (time-dependent) actions on these coordinates, e.g. translations, and how to build these actions into changes of reference frames.

</br>

---

# Packages

```{contents}
:local:
:depth: 1
```

## Main

This package collects all the most-commonly used parts of `coordinax` into a single convenient location.

It should be imported as `import coordinax.main as cx`

A non-exhaustive table of exported objects are:

| Package               | Object                              |
| --------------------- | ----------------------------------- |
| `coordinax.angles`    | `AbstractAngle`, `Angle`, `wrap_to` |
| `coordinax.distances` | `AbstractDistance`, `Distance`      |

</br>

(software-spec-angles)=

## Angles

The `coordinax.angles` module provides the angle-facing scalar API used by charts, vectors, and frame transformations.

!!! info `AbstractAngle`

    Abstract base type for angular quantities. It defines the shared angle interface used by dispatch and typing.

!!! info `Angle`

     Concrete angular scalar type (value + angular unit), built on `unxt`. Angles represent directions on $S^1$ and do not encode branch-cut convention in the type itself.

    - `wrap_to` method that calls `coordinax.angles.wrap_to` on the `Angle`.

!!! info `wrap_to`

    Functional API for explicit interval wrapping. It remaps an angle into a caller-specified interval (for example $[0, 2\pi)$ or $(-\pi, \pi]$).

!!! info `Parallax`

    Re-exported from `coordinax.distances` for convenience. `Parallax` is an angular quantity that is commonly used as a distance proxy in astronomy.

</br>

(software-spec-distances)=

## Distances

The `coordinax.distances` module provides the distance-facing scalar API used by coordinate charts and astronomy-oriented conversions.

!!! info `AbstractDistance`

    Abstract base type for distance-like quantities. It defines the shared interface used by dispatch and typing.

    Properties:

    - `distance`: convert to a [`coordinax.distances.Distance`](#software-spec-distance) with length units. This is the canonical distance representation for all distance-like quantities.

(software-spec-distance)=

!!! info `Distance`

    Concrete length quantity type (value + length unit), built on `unxt`, for physical distances.

!!! info `DistanceModulus`

    Magnitude-space distance representation used in astronomical workflows and conversions.

!!! info `Parallax`

    Angular distance proxy represented in angular units and linked to distance conversion flows.
