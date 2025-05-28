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

## Quick example

```python
import jax.numpy as jnp
import unxt as u
import coordinax as cx

q = cx.CartesianPos3D(
    x=u.Quantity(jnp.arange(0, 10.0), "kpc"),
    y=u.Quantity(jnp.arange(5, 15.0), "kpc"),
    z=u.Quantity(jnp.arange(10, 20.0), "kpc"),
)
print(q)
# <CartesianPos3D: (x, y, z) [kpc]
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

q2 = cx.vconvert(cx.SphericalPos, q)
print(q2)
# <SphericalPos: (r[kpc], theta[rad], phi[rad])
#     [[11.18   0.464  1.571]
#      [12.57   0.505  1.406]
#      ...
#      [23.601  0.703  1.019]
#      [25.259  0.719  0.999]]>

p = cx.CartesianVel3D(
    x=u.Quantity(jnp.arange(0, 10.0), "km/s"),
    y=u.Quantity(jnp.arange(5, 15.0), "km/s"),
    z=u.Quantity(jnp.arange(10, 20.0), "km/s"),
)
print(p)
# <CartesianVel3D: (x, y, z) [km / s]
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

p2 = cx.vconvert(cx.SphericalVel, p, q)
print(p2)
# <SphericalVel: (r[km / s], theta[km rad / (km s)], phi[km rad / (km s)])
#     [[ 1.118e+01 -3.886e-16  0.000e+00]
#      [ 1.257e+01 -1.110e-16  0.000e+00]
#      ...
#      [ 2.360e+01  0.000e+00  0.000e+00]
#      [ 2.526e+01 -2.776e-16  0.000e+00]]>


# Transforming between frames
icrs_frame = cx.frames.ICRS()
gc_frame = cx.frames.Galactocentric()
op = cxf.frame_transform_op(icrs_frame, gc_frame)
q_gc, p_gc = op(q, p)
print(q_gc, p_gc, sep="\n")
# <CartesianPos3D: (x, y, z) [kpc]
#     [[-1.732e+01  5.246e+00  3.614e+00]
#      ...
#      [-3.004e+01  1.241e+01 -1.841e+00]]>
# <CartesianVel3D: (x, y, z) [km / s]
#      [[  3.704 250.846  11.373]
#       ...
#       [ -9.02  258.012   5.918]]>

coord = cx.Coordinate(cx.Space(length=q, speed=p), frame=icrs_frame)
print(coord)
# Coordinate(
#     data=Space({
#        'length': <CartesianPos3D: (x, y, z) [kpc]
#             [[ 0.  5. 10.]
#              ...
#              [ 9. 14. 19.]]>,
#        'speed': <CartesianVel3D: (x, y, z) [km / s]
#             [[ 0.  5. 10.]
#              ...
#              [ 9. 14. 19.]]>
#     }),
#     frame=ICRS()
# )

print(coord.to_frame(gc_frame))
# Coordinate(
#     data=Space({
#        'length': <CartesianPos3D: (x, y, z) [kpc]
#             [[-1.732e+01  5.246e+00  3.614e+00]
#              ...
#              [-3.004e+01  1.241e+01 -1.841e+00]]>,
#        'speed': <CartesianVel3D: (x, y, z) [km / s]
#             [[  3.704 250.846  11.373]
#              ...
#              [ -9.02  258.012   5.918]]>
#     }),
#     frame=Galactocentric( ... )
# )
```

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
