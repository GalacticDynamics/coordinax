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

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://unxt.readthedocs.io/en/)

Coming soon. In the meantime, if you've used `astropy.coordinates`, then
`coordinax` should be fairly intuitive.

## Quick example

```python
import jax.numpy as jnp
import unxt as u
import coordinax as cx

q = cx.CartesianPos3D(
    x=u.Quantity(jnp.arange(0, 10.0), "km"),
    y=u.Quantity(jnp.arange(5, 15.0), "km"),
    z=u.Quantity(jnp.arange(10, 20.0), "km"),
)
print(q)
# <CartesianPos3D (x[km], y[km], z[km])
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

q2 = cx.vconvert(cx.SphericalPos, q)
print(q2)
# <SphericalPos (r[km], theta[rad], phi[rad])
#     [[11.18   0.464  1.571]
#      [12.57   0.505  1.406]
#      ...
#      [23.601  0.703  1.019]
#      [25.259  0.719  0.999]]>

p = cx.CartesianVel3D(
    d_x=u.Quantity(jnp.arange(0, 10.0), "m/s"),
    d_y=u.Quantity(jnp.arange(5, 15.0), "m/s"),
    d_z=u.Quantity(jnp.arange(10, 20.0), "m/s"),
)
print(p)
# <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

p2 = cx.vconvert(cx.SphericalVel, p, q)
print(p2)
# <SphericalVel (d_r[m / s], d_theta[m rad / (km s)], d_phi[m rad / (km s)])
#     [[ 1.118e+01 -3.886e-16  0.000e+00]
#      [ 1.257e+01 -1.110e-16  0.000e+00]
#      ...
#      [ 2.360e+01  0.000e+00  0.000e+00]
#      [ 2.526e+01 -2.776e-16  0.000e+00]]>
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
[zenodo-badge]:             https://zenodo.org/badge/755708966.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850557

<!-- prettier-ignore-end -->
