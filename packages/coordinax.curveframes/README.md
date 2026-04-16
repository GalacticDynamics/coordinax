# coordinax.curveframes

Curve-attached reference frames for [coordinax](https://github.com/GalacticDynamics/coordinax).

This package implements Frenet-Serret and Bishop frames for tau-parameterized smooth curves in 3D Euclidean space.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax.curveframes
```

:::

:::{tab-item} uv

```bash
uv add coordinax.curveframes
```

:::

::::

## Quick Start

```python
import jax.numpy as jnp
import unxt as u
import coordinax.curveframes as cxfc
import coordinax.frames as cxf
import coordinax.transforms as cxfm


def helix(tau: u.Q) -> u.Q:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.5 * t]), "m")


frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), helix)
op = cxf.frame_transition(cxf.Alice(), frame)
tau = u.Q(0.5, "s")
out = cxfm.act(op, tau, u.Q(jnp.array([1.0, 0.0, 0.0]), "m"))
```

## License

MIT License. See [LICENSE](LICENSE) for details.
