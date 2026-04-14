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

Coordinax enables calculations with coordinates in [`JAX`](https://jax.readthedocs.io/en/latest/). Built on [`equinox`](https://docs.kidger.site/equinox/) and [`quax`](https://github.com/patrick-kidger/quax), with unit-support using [`unxt`](https://github.com/GalacticDynamics/unxt)

## Installation &nbsp; [![PyPI platforms][pypi-platforms]][pypi-link] [![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install coordinax
```

## Quick Start &nbsp; [![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://coordinax.readthedocs.io/en/)

### Concepts

- Specialized quantities: scalar coordinate quantities with units, including `Angle` (directional values on $S^1$ with explicit wrapping) and `Distance` (length-valued quantity), plus astronomy-facing forms like `Parallax` and `DistanceModulus`.
- Charts: a coordinate chart / component schema (names + physical dimensions). A chart does not store numerical values.
- Representation: geometric meaning of components, encoded as (geometry, basis, semantics), e.g. `point`.
- Point: data + chart + representation, with conversion and arithmetic behavior defined by chart transition maps and tangent pushforwards.

## Modules

The most common import is the high-level user API:

```python
import coordinax.main as cx
```

### Specialized Quantities

```pycon
>>> import coordinax.main as cx
>>> import unxt as u

>>> a = cx.Angle(30.0, "deg")
>>> d = cx.Distance(10.0, "kpc")
>>> u.uconvert("rad", a)
Angle(0.52359878, 'rad')
```

```pycon
>>> import unxt as u
>>> u.uconvert("rad", a)
Angle(0.52359878, 'rad')

```

### Charts and Point Maps

Transform point coordinates between charts with `pt_map`:

```pycon
>>> import coordinax.main as cx
>>> import unxt as u

>>> q = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
>>> q_sph = cx.pt_map(q, cx.cart3d, cx.sph3d)
>>> q_sph
{'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'), 'phi': Q(1.10714872, 'rad')}
```

### Point Conversion

`Point` carries chart + representation metadata, so conversions preserve semantics:

```pycon
>>> import coordinax.main as cx

>>> vec = cx.Point.from_([1, 2, 3], "m")
>>> print(vec)
<Point: chart=Cart3D (x, y, z) [m]
    [1 2 3]>

>>> sph_vec = vec.cconvert(cx.sph3d)
>>> print(sph_vec)
<Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
    [3.742 0.641 1.107]>
```

### Representations

Common representation constants are available from the high-level module:

```python
import coordinax.main as cx

cx.point  # point location data
```

### Manifolds

Define an explicit custom atlas and manifold:

```pycon
>>> import coordinax.main as cx
>>> import unxt as u

>>> atlas = cx.CustomAtlas(
...     charts=(type(cx.cart2d), type(cx.polar2d)),
...     chart_default=cx.cart2d,
... )
>>> cx.polar2d in atlas
True
>>> M = cx.CustomManifold(atlas)
>>> q = {"x": u.Q(1.0, "km"), "y": u.Q(1.0, "km")}
>>> M.pt_map(q, cx.cart2d, cx.polar2d)
{'r': Q(1.41421356, 'km'), 'theta': Q(0.78539816, 'rad')}

```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful in academic work, then please cite.

## Development

[![Actions Status][actions-badge]][actions-link] [![Documentation Status][rtd-badge]][rtd-link] [![codecov][codecov-badge]][codecov-link] [![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link] [![pre-commit][pre-commit-badge]][pre-commit-link] [![ruff][ruff-badge]][ruff-link]

We welcome contributions!

For the local development workflow, see `docs/dev.md`. For pull request expectations, see `docs/contributing.md`.

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
