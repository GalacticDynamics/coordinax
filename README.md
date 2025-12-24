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
[JAX](https://jax.readthedocs.io/en/latest/). Built on
[Equinox](https://docs.kidger.site/equinox/) and
[Quax](https://github.com/patrick-kidger/quax).

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

- Representation (Rep): a coordinate chart / component schema (names + physical
  dimensions). A rep does not store numerical values.
- Vector: data + rep + role. Data are the coordinate values or physical
  components.
- Role: semantic interpretation of vector data (Pos, Vel, Acc, etc.). A role is
  not a rep.
- Metric: a bilinear form g on the tangent space defining inner products and
  norms (Euclidean, sphere intrinsic, Minkowski).
- Physical components: components of a geometric vector expressed in an
  orthonormal frame with respect to the active metric. Components have uniform
  units (e.g. all speed or all acceleration).
- Coordinate derivatives: time derivatives of coordinate components (e.g.
  \dot\theta, \ddot\phi); these may have heterogeneous units and are not what
  diff_map/vconvert uses for physical vectors.

## Quantities

Distances and angles are first-class quantities:

```python
import coordinax as cx
import unxt as u

d = cx.distance.Distance(10.0, "kpc")
a = u.Angle(30.0, "deg")
```

## Representations and coordinate maps

Transform coordinate dictionaries between reps:

```python
import coordinax as cx
import unxt as u

rep_cart = cx.r.cart3d
rep_sph = cx.r.sph3d
q = {"x": u.Quantity(1.0, "km"), "y": u.Quantity(2.0, "km"), "z": u.Quantity(3.0, "km")}
q_sph = cx.r.coord_map(rep_sph, rep_cart, q)
```

## Metrics

- Euclidean metric is default for Euclidean reps exposed in `cx.r`.
- `TwoSphere` uses intrinsic sphere metric.
- `SpaceTimeCT` uses Minkowski metric with signature `(-,+,+,+)`.
- `SpaceTimeEuclidean` uses Euclidean metric in 4D.

## Embedded manifolds

`EmbeddedManifold` wraps an intrinsic chart and an ambient rep. Use `embed_pos`
and `project_pos` for positions, and `embed_dif` / `project_dif` for physical
components.

```python
import jax.numpy as jnp
import coordinax as cx
import unxt as u

rep = cx.r.EmbeddedManifold(
    chart_kind=cx.r.twosphere,
    ambient_kind=cx.r.cart3d,
    params={"R": u.Quantity(2.0, "km")},
)
p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
q = cx.r.embed_pos(rep, p)
r2 = (
    u.uconvert("km", q["x"]) ** 2
    + u.uconvert("km", q["y"]) ** 2
    + u.uconvert("km", q["z"]) ** 2
)
bool(jnp.allclose(r2.value, 4.0))
```

## Physical vector conversion (vconvert)

Physical components (not coordinate derivatives) transform via orthonormal
frames.

```python
import jax.numpy as jnp
import coordinax as cx
import unxt as u

q = {"x": u.Quantity(1.0, "km"), "y": u.Quantity(2.0, "km"), "z": u.Quantity(3.0, "km")}
v = {
    "x": u.Quantity(4.0, "km/s"),
    "y": u.Quantity(5.0, "km/s"),
    "z": u.Quantity(6.0, "km/s"),
}
qvec = cx.Vector(data=q, rep=cx.r.cart3d, role=cx.r.Pos())
vvec = cx.Vector(data=v, rep=cx.r.cart3d, role=cx.r.Vel())
v_sph = vvec.vconvert(cx.r.sph3d, qvec)
v_back = v_sph.vconvert(cx.r.cart3d, qvec)
bool(
    jnp.allclose(
        u.uconvert("km/s", v_back.data["x"]).value,
        u.uconvert("km/s", vvec.data["x"]).value,
    )
)
```

# Coordinate(

# KinematicSpace({

# 'length': <Cart3D: (x, y, z) [kpc]

# [[-1.732e+01 5.246e+00 3.614e+00]

# ...

# [-3.004e+01 1.241e+01 -1.841e+00]]>,

# 'speed': <CartVel3D: (x, y, z) [km / s]

# [[ 3.704 250.846 11.373]

# ...

# [ -9.02 258.012 5.918]]>

# }),

# frame=Galactocentric( ... )

# )

```

## Metrics and Representations

Representations are coordinate charts; roles (Pos, Vel, Acc, ...) give vectors
their physical meaning. A chart’s default metric defines how physical components
are interpreted.

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
```
