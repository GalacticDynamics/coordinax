---
jupytext:
  formats: md:myst
kernelspec:
  display_name: "Python 3"
  language: "python"
  name: "python3"
---

# Visualizing Curve Frames

A curve frame is easiest to understand by _seeing_ it: a little set of axes that rides along a curve, reorienting itself as the curve bends and twists. This page plots the {doc}`Frenet–Serret and Bishop frames <guide>` directly so you can watch the frame move.

```{code-cell} python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import unxt as u
import coordinaxs.curveframes as cxfc
```

## A moving Frenet–Serret triad

We use a helix, whose steady bending and climbing exercises all three frame axes. The `location`, `tangent`, `normal`, and `binormal` fields are evaluated at a handful of parameter values and drawn as arrows: the tangent $\mathbf{T}$ in red, the normal $\mathbf{N}$ in green, and the binormal $\mathbf{B}$ in blue.

```{code-cell} python
def helix(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


fs = cxfc.FrenetSerretTransform.from_curve(helix)

# Dense sampling for the curve line, sparse sampling for the frame triads.
ts = np.linspace(0.0, 4.0 * np.pi, 300)
curve = np.stack([np.asarray(helix(u.Q(t, "s")).ustrip("km")) for t in ts])
sample_ts = np.linspace(0.4, 4.0 * np.pi - 0.4, 9)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(*curve.T, color="0.6", lw=1.0)

for t in sample_ts:
    tau = u.Q(float(t), "s")
    loc = np.asarray(fs.location(tau).ustrip("km"))
    axes = {
        "tab:red": np.asarray(fs.tangent(tau).value),
        "tab:green": np.asarray(fs.normal(tau).value),
        "tab:blue": np.asarray(fs.binormal(tau).value),
    }
    for color, vec in axes.items():
        ax.quiver(*loc, *vec, length=0.6, color=color, lw=1.5)

ax.set_title("Frenet–Serret frame along a helix\nT (red), N (green), B (blue)")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
plt.tight_layout()
plt.show()
```

The tangent always points along the direction of travel, the normal points toward the inside of the bend, and the binormal completes the right-handed triad. Together they twist steadily around the helix.

## Frenet–Serret vs. Bishop

The {doc}`Bishop frame <guide>` is _rotation-minimising_: instead of tracking the principal normal (which spins around the tangent as the curve twists), it transports its cross-section axes as smoothly as possible. On the helix this shows up as a visible lag between the Frenet normal and the Bishop `normal1` axis.

```{code-cell} python
bishop = cxfc.BishopTransform.from_curve(helix)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(*curve.T, color="0.6", lw=1.0)

for t in sample_ts:
    tau = u.Q(float(t), "s")
    loc = np.asarray(fs.location(tau).ustrip("km"))
    fs_n = np.asarray(fs.normal(tau).value)
    bi_n = np.asarray(bishop.normal1(tau).value)
    ax.quiver(*loc, *fs_n, length=0.6, color="tab:green", lw=1.5)
    ax.quiver(*loc, *bi_n, length=0.6, color="tab:purple", lw=1.5)

ax.set_title(
    "Cross-section axis: Frenet normal (green)\nvs Bishop normal1 (purple)"
)
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
plt.tight_layout()
plt.show()
```

The two normals start aligned but drift apart: the Frenet normal keeps rotating with the curve's torsion, while the Bishop axis stays as untwisted as the geometry allows. That rotation-minimising property is exactly why the Bishop frame is preferred for tubes, ribbons, and camera paths — and why it stays well-defined where the curvature (and hence the Frenet normal) vanishes.
