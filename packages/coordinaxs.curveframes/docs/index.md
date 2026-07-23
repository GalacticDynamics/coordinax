# coordinaxs.curveframes

```{toctree}
:maxdepth: 1
:hidden:

guide.md
visualizing
tutorial.md
tutorial_circle_equivalence.md
api.md
spec.md
```

Curve-attached reference frames for [coordinax](https://github.com/GalacticDynamics/coordinax).

This package implements Frenet–Serret and Bishop frames for τ-parameterized smooth curves in 3D Euclidean space, letting you transform coordinate data into and out of the moving frame that travels along a space curve.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinaxs.curveframes
```

:::

:::{tab-item} uv

```bash
uv add coordinaxs.curveframes
```

:::

::::

## Learn More

- {doc}`Working With Curve Frames <guide>` — concepts, and a walkthrough of building a curve frame and transforming data through it.
- {doc}`Visualizing Curve Frames <visualizing>` — plots of the moving Frenet–Serret and Bishop frames along a helix.
- {doc}`Tutorial: Frenet–Serret Curve Frames <tutorial>` — a complete worked example, including chaining frames and using JAX for JIT and vectorisation.
- {doc}`Tutorial: Parallel Transport vs Corotating Frame on a Circle <tutorial_circle_equivalence>` — why the Frenet–Serret, Bishop, and rigid-rotation frames coincide on a circle.
- {doc}`API Reference <api>` — complete API documentation.
- {doc}`Specification <spec>` — the normative specification.

## License

MIT License. See [LICENSE](../../../LICENSE) for details.
