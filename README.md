<h1 align='center'> coordinax </h1>
<h2 align="center">Coordinates in JAX</h2>

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

[![Documentation Status][rtd-badge]][rtd-link]

Coming soon. In the meantime, if you've used `astropy.coordinates`, then
`coordinax` should be fairly intuitive.

## Quick example

```python
import coordinax as cx
import jax.numpy as jnp
from unxt import Quantity

q = cx.CartesianPos3D(
    x=Quantity(jnp.arange(0, 10.0), "km"),
    y=Quantity(jnp.arange(5, 15.0), "km"),
    z=Quantity(jnp.arange(10, 20.0), "km"),
)
print(q)
# <CartesianPos3D (x[km], y[km], z[km])
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

q2 = cx.represent_as(q, cx.SphericalPos)
print(q2)
# <SphericalPos (r[km], theta[rad], phi[rad])
#     [[11.18   0.464  1.571]
#      [12.57   0.505  1.406]
#      ...
#      [23.601  0.703  1.019]
#      [25.259  0.719  0.999]]>

p = cx.CartesianVel3D(
    d_x=Quantity(jnp.arange(0, 10.0), "m/s"),
    d_y=Quantity(jnp.arange(5, 15.0), "m/s"),
    d_z=Quantity(jnp.arange(10, 20.0), "m/s"),
)
print(p)
# <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
#     [[ 0.  5. 10.]
#      [ 1.  6. 11.]
#      ...
#      [ 8. 13. 18.]
#      [ 9. 14. 19.]]>

p2 = cx.represent_as(p, cx.SphericalVel, q)
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

We welcome contributions!

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/coordinax/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/coordinax/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/coordinax
[conda-link]:               https://github.com/conda-forge/coordinax-feedstock
[pypi-link]:                https://pypi.org/project/coordinax/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/coordinax
[pypi-version]:             https://img.shields.io/pypi/v/coordinax
[rtd-badge]:                https://readthedocs.org/projects/coordinax/badge/?version=latest
[rtd-link]:                 https://coordinax.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/755708966.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850557

<!-- prettier-ignore-end -->
