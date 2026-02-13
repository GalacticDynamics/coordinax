<h1 align='center'> coordinax </h1>
<h3 align="center">Coordinates in JAX</h3>

<p align="center">
    <a href="https://pypi.org/project/coordinax/"> <img alt="PyPI: coordinax" src="https://img.shields.io/pypi/v/coordinax?style=flat" /> </a>
    <a href="https://pypi.org/project/coordinax/"> <img alt="PyPI versions: coordinax" src="https://img.shields.io/pypi/pyversions/coordinax" /> </a>
    <a href="https://coordinax.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://img.shields.io/badge/read_docs-here-orange" /> </a>
    <a href="https://pypi.org/project/coordinax/"> <img alt="coordinax license" src="https://img.shields.io/github/license/GalacticDynamics/coordinax" /> </a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/coordinax/actions"> <img alt="CI status" src="https://github.com/GalacticDynamics/coordinax/workflows/CI/badge.svg" /> </a>
    <a href="https://coordinax.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://readthedocs.org/projects/coordinax/badge/?version=latest" /> </a>
    <a href="https://codecov.io/gh/GalacticDynamics/coordinax"> <img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/coordinax/graph/badge.svg" /> </a>
    <a href="https://scientific-python.org/specs/spec-0000/"> <img alt="ruff" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /> </a>
    <a href="https://docs.astral.sh/ruff/"> <img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /> </a>
    <a href="https://pre-commit.com"> <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /> </a>
</p>

---

Coordinax enables calculations with coordinates in
[`JAX`](https://jax.readthedocs.io/en/latest/). Built on
[`equinox`](https://docs.kidger.site/equinox/) and
[`quax`](https://github.com/patrick-kidger/quax), with unit-support using
[`unxt`](https://github.com/GalacticDynamics/unxt)

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install coordinax
```

## Documentation

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://coordinax.readthedocs.io/en/)

## Concepts

- Charts: a coordinate chart / component schema (names + physical dimensions). A
  chart does not store numerical values.
- Roles: semantic interpretation of vector data (`Point`, `PhysDisp`, `PhysVel`,
  `PhysAcc`, `CoordDisp`, etc.). A role is not a chart.
- `Vector`: data + chart + role. Data are the coordinate values or physical
  components.
- Metrics: a bilinear form _g_ on the tangent space defining inner products and
  norms (Euclidean, sphere intrinsic, Minkowski).
- Transformations: convert between coordinate charts and roles, including
  velocity and acceleration transformations
- Operators: frame-aware vector operations (rotation, boost, translation, etc.)
- `PointedVector`: combines a vector with a reference point or frame for context
- Frames: reference systems (ICRS, Galactocentric, etc.) and their
  transformations
- `Coordinate`: encapsulates a vector with its reference frame, enabling
  frame-aware transformations and comparisons

## Modules

```
import coordinax.charts as cxc  # Charts
import coordinax.roles as cxr  # Roles
import coordinax.metrics as cxm  # Metric
import coordinax.transforms as cxt  # Transformations
```

## Quantities

Distances and angles are first-class quantities:

```
import coordinax.distances as cxd
import unxt as u

d = cxd.Distance(10.0, "kpc")
a = u.Angle(30.0, "deg")
```

## Representations and coordinate maps

Transform coordinate dictionaries between reps:

```
import coordinax.charts as cxc
import coordinax.transforms as cxt

q = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
q_sph = cxt.point_transform(cxc.sph3d, cxc.cart3d, q)
```

We can also transform physical vector components between reps:

```
v = {"x": u.Q(4.0, "km/s"), "y": u.Q(5.0, "km/s"), "z": u.Q(6.0, "km/s")}
v_sph = cxt.physical_tangent_transform(cxc.sph3d, cxc.cart3d, v, at=q)
```

## Metrics

- Euclidean metric is default for Euclidean reps exposed in `coordinax.metrics`.
- `TwoSphere` uses intrinsic sphere metric.
- `SpaceTimeCT` uses Minkowski metric with signature `(-,+,+,+)`.
- `SpaceTimeEuclidean` uses Euclidean metric in 4D.

## Embedded manifolds

`EmbeddedManifold` wraps an intrinsic chart and an ambient rep. Use
`embed_point` and `project_point` for positions, and `embed_tangent` /
`project_tangent` for physical components.

```
import jax.numpy as jnp
import coordinax.embeddings as cxe

rep = cxe.EmbeddedManifold(
    intrinsic_chart=cxc.twosphere,
    ambient_chart=cxc.cart3d,
    params={"R": u.Q(2.0, "km")},
)
p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
q = cxe.embed_point(rep, p)
r2 = (q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2)
bool(jnp.allclose(r2.ustrip("km"), 4.0))
```

## Physical vector conversion (vconvert)

Physical components (not coordinate derivatives) transform via orthonormal
frames.

```
import jax.numpy as jnp
import coordinax.roles as cxr
from coordinax.objs import Vector

q = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
v = {"x": u.Q(4.0, "km/s"), "y": u.Q(5.0, "km/s"), "z": u.Q(6.0, "km/s")}
qvec = Vector(data=q, chart=cxc.cart3d, role=cxr.phys_disp)
vvec = Vector(data=v, chart=cxc.cart3d, role=cxr.phys_vel)
v_sph = vvec.vconvert(cxc.sph3d, qvec)
v_back = v_sph.vconvert(cxc.cart3d, qvec)
bool(jnp.allclose(u.ustrip("km/s", v_back.data["x"]),
                  u.ustrip("km/s", vvec.data["x"])))
```

Different vector roles transform via different mechanisms:

| Role       | Transformation                      | Requires Base Point? |
| ---------- | ----------------------------------- | -------------------- |
| `Point`    | Position transform (coordinate map) | No                   |
| `PhysDisp` | Tangent transform (physical vector) | Sometimes[^1]        |
| `PhysVel`  | Tangent transform (physical vector) | Sometimes[^1]        |
| `PhysAcc`  | Tangent transform (physical vector) | Sometimes[^1]        |

[^1]:
    Required when converting between representations (e.g., Cartesian ↔
    Spherical), not required for unit conversions within the same
    representation.

## PointedVector: Ergonomic Tangent Vector Conversions

`PointedVector` provides a container for vectors anchored at a common base
point, automatically managing the base point dependency required for tangent
vector transformations:

```

base = Vector.from_([1, 2, 3], "km")
vel = Vector.from_([10, 20, 30], "km/s")
acc = Vector.from_([0.1, 0.2, 0.3], "km/s^2")

# Create bundle - base is explicit, fields are tangent vectors
bundle = cx.PointedVector(base=base, velocity=vel, acceleration=acc)

# Convert entire bundle - automatically handles `at=` for tangent vectors
sph_bundle = bundle.vconvert(cxc.sph3d)

# Equivalent to manual:
# base_sph = base.vconvert(cxc.sph3d)
# at_for_vel = base.vconvert(vel.chart)
# vel_sph = vel.vconvert(cxc.sph3d, at_for_vel)
# (and similarly for acc)
```

**Key features:**

- Base must have role `PhysDisp`; fields must not (enforced at construction)
- Automatic `at=` parameter handling for tangent transformations
- JAX-compatible (works with `jax.jit`, `vmap`)
- Physical components with uniform units (not coordinate derivatives)

See the
[PointedVector guide](https://coordinax.readthedocs.io/en/latest/guides/anchored_vector_bundle.html)
for details.

## Vector Algebra: Points vs Displacements

`coordinax` distinguishes between **positions** (points) and **displacements**
(tangent vectors). Vector addition follows affine geometry rules:

```
import coordinax as cx
import unxt as u

# A position (point)
pos = Vector.from_([0, 0, 0], "m")

# A displacement (offset / tangent vector)
disp = Vector(
    {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    cxc.cart3d,
    cxr.phys_disp,
)

# Position + Displacement → Position
new_pos = pos.add(disp)  # role is Pos

# Displacement + Displacement → Displacement
d1 = Vector(
    {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    cxc.cart3d,
    cxr.phys_disp,
)
d2 = Vector(
    {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")},
    cxc.cart3d,
    cxr.phys_disp,
)
d_sum = d1.add(d2)  # role is Displacement

# Convert position to displacement from origin
disp_from_origin = cx.as_disp(new_pos)
```

| Operation             | Result     | Allowed? |
| --------------------- | ---------- | -------- |
| `PhysDisp + PhysDisp` | `PhysDisp` | ✅       |
| `Point + PhysDisp`    | `Point`    | ✅       |
| `PhysDisp + Point`    | —          | ❌       |
| `Point + Point`       | —          | ❌       |

> **⚠️ Critical: Physical Components with Uniform Units**
>
> `PhysDisp` stores **physical vector components in an orthonormal frame**, not
> coordinate increments. All components must have uniform dimension `[length]`.
>
> For example, in cylindrical coordinates:
>
> - ✅ **Correct**: `PhysDisp(rho=1m, phi=2m, z=3m)` — physical components,
>   where `phi=2m` means "2 meters in the tangential direction"
> - ❌ **Wrong**: `PhysDisp(rho=1m, phi=0.5rad, z=3m)` — coordinate increments
>   with mixed units
>
> This applies to all tangent vectors (`PhysDisp`, `PhysVel`, `PhysAcc`), which
> transform via orthonormal frame transformations, not coordinate chart
> transformations.

## Metrics and Representations

Representations are coordinate charts; roles (PhysDisp, PhysVel, PhysAcc, ...)
give vectors their physical meaning. A chart’s default metric defines how
physical components are interpreted.

- Euclidean charts are exposed in `cx.r` (Cart, Cylindrical, Spherical, etc.).
- Manifold charts are exposed in `cx.r` (e.g. `TwoSphere`).
- Spacetime charts are exposed in `cx.r` (Minkowski `SpaceTimeCT`, Euclidean
  `SpaceTimeEuclidean`).

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful in academic work, then please cite.

## Development

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![codecov][codecov-badge]][codecov-link]
[![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link]
[![pre-commit][pre-commit-badge]][pre-commit-link]
[![ruff][ruff-badge]][ruff-link]

We welcome contributions!

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/coordinax/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/coordinax/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/unxt/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/unxt
[pre-commit-badge]:         https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit-link]:          https://pre-commit.com
[pypi-link]:                https://pypi.org/project/coordinax/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/coordinax
[pypi-version]:             https://img.shields.io/pypi/v/coordinax
[rtd-badge]:                https://readthedocs.org/projects/coordinax/badge/?version=latest
[rtd-link]:                 https://coordinax.readthedocs.io/en/latest/?badge=latest
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/DOI/10.5281/zenodo.15320465.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850557

<!-- prettier-ignore-end -->
