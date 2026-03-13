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

- Specialized Quantities: scalar coordinate quantities with units, including `Angle` (directional values on $S^1$ with explicit wrapping) and `Distance` (length-valued quantities), plus astronomy-facing forms like `Parallax` and `DistanceModulus`.

## Modules

The most common import is this module which aggregates all the most-commonly used functionality. Chances are this has what you need.

```python
import coordinax.main as cx
```

### Specialized Quantities

The specific sub-packages, with the full functionality are:

```python
import coordinax.angles as cxa
import coordinax.distances as cxd
```

Distances and angles are first-class quantities:

```pycon
>>> a = cx.Angle(30.0, "deg")
>>> d = cx.Distance(10.0, "kpc")

```

```pycon
>>> import unxt as u
>>> u.uconvert("rad", a)
Angle(Array(0.52359878, dtype=float64, ...), unit='rad')

```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful in academic work, then please cite.

## Development

[![Actions Status][actions-badge]][actions-link] [![Documentation Status][rtd-badge]][rtd-link] [![codecov][codecov-badge]][codecov-link] [![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link] [![pre-commit][pre-commit-badge]][pre-commit-link] [![ruff][ruff-badge]][ruff-link]

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
