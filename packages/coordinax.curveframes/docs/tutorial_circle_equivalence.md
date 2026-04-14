# Tutorial: Parallel Transport vs Corotating Frame on a Circle

A circle in the $xy$-plane is a special curve: the Frenet–Serret frame, the Bishop (parallel-transport) frame, and a plain rotation by $\tau$ all produce the **same** moving triad. This tutorial builds all three from scratch and shows they agree numerically.

**Why they coincide.** On a planar circle with constant curvature $\kappa$ and zero torsion, the Frenet–Serret angular velocity equals the parallel-transport angular velocity, which in turn equals the constant rate of a rigid rotation about $+z$. In general these three differ, but for a circle they collapse to a single thing.

```pycon
>>> import jax.numpy as jnp
>>> import numpy as np

>>> import unxt as u

>>> import coordinax.charts as cxc
>>> import coordinax.frames as cxf
>>> import coordinax.transforms as cxfm
>>> import coordinax.curveframes as cxfc
```

## Step 1: Define the Circle

A unit-radius circle in the $xy$-plane, parameterised by $\tau$ in seconds:

```pycon
>>> def circle(tau: u.Q) -> u.Q:
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")
...

>>> circle(u.Q(0.0, "s"))
Q([1., 0., 0.], 'km')
```

## Step 2: Build the Three Frames

### 2a — Frenet–Serret frame

The Frenet–Serret frame has axes $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ computed from the curve's first and second derivatives:

```pycon
>>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.alice, circle)
```

### 2b — Bishop (parallel-transport) frame

The Bishop frame $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$ is obtained by parallel-transporting an initial normal along the curve. We choose the initial normal to match the Frenet–Serret normal at $\tau = 0$, which for this circle is $\mathbf{N}_0 = (-1, 0, 0)$:

```pycon
>>> bishop_frame = cxfc.BishopFrame.from_curve(
...     cxf.alice,
...     circle,
...     initial_normal=jnp.array([-1.0, 0.0, 0.0]),
... )
```

### 2c — Corotating frame via Translate | Rotate

A rigid rotation by angle $\tau$ about the $z$-axis, centred on $\gamma(\tau)$, is defined without any curve-frame machinery — just a `Translate` and a `Rotate`:

```pycon
>>> def neg_gamma(tau: u.Q) -> u.Q:
...     """Translate by -gamma(tau)."""
...     return cxc.cdict(-circle(tau), cxc.cart3d)
...

>>> def R_z(tau: u.Q) -> jnp.ndarray:
...     """Rotation matrix for angle tau about +z.
...     This is the same matrix R = [T; N; B] that the
...     Frenet-Serret frame produces for this particular circle.
...     """
...     t = tau.ustrip("s")
...     ct, st = jnp.cos(t), jnp.sin(t)
...     return jnp.array([[-st, ct, 0.0], [-ct, -st, 0.0], [0.0, 0.0, 1.0]])
...

>>> xop = cxfm.Translate(neg_gamma, chart=cxc.cart3d) | cxfm.Rotate(R_z)
>>> corot_frame = cxf.TransformedReferenceFrame(cxf.alice, xop)
```

## Step 3: Compare the Rotation Matrices

At any $\tau$, all three frames should produce the same $3 \times 3$ rotation matrix $R(\tau)$. Let's check at several values:

```pycon
>>> taus = [u.Q(0.0, "s"), u.Q(0.5, "s"), u.Q(1.0, "s"), u.Q(jnp.pi, "s")]
```

Extract the rotation matrix from each transform:

```pycon
>>> for tau in taus:
...     R_fs = jnp.stack(
...         [
...             fs_frame.xop.tangent(tau).value,
...             fs_frame.xop.normal(tau).value,
...             fs_frame.xop.binormal(tau).value,
...         ]
...     )
...     R_bp = jnp.stack(
...         [
...             bishop_frame.xop.tangent(tau).value,
...             bishop_frame.xop.normal1(tau).value,
...             bishop_frame.xop.normal2(tau).value,
...         ]
...     )
...     R_co = R_z(tau)
...     np.testing.assert_allclose(R_fs, R_co, atol=1e-6)
...     np.testing.assert_allclose(R_bp, R_co, atol=1e-5)
...
```

All three rotation matrices agree to numerical precision.

## Step 4: Transform a Point Through Each Frame

Pick a test point and a specific $\tau$:

```pycon
>>> tau = u.Q(1.0, "s")
>>> p = u.Q([2.0, 0.5, 0.3], "km")
```

Apply the forward transform through each frame:

```pycon
>>> op_fs = cxf.frame_transition(cxf.alice, fs_frame)
>>> op_bp = cxf.frame_transition(cxf.alice, bishop_frame)
>>> op_co = cxf.frame_transition(cxf.alice, corot_frame)

>>> p_fs = cxfm.act(op_fs, tau, p)
>>> p_bp = cxfm.act(op_bp, tau, p)
>>> p_co = cxfm.act(op_co, tau, p)
```

All three give the same result:

```pycon
>>> np.testing.assert_allclose(p_fs.value, p_co.value, atol=1e-6)
>>> np.testing.assert_allclose(p_bp.value, p_co.value, atol=1e-5)
```

## Step 5: Round-Trip Through Each Frame

Going forward then backward should recover the original point:

```pycon
>>> op_fs_inv = cxf.frame_transition(fs_frame, cxf.alice)
>>> op_bp_inv = cxf.frame_transition(bishop_frame, cxf.alice)
>>> op_co_inv = cxf.frame_transition(corot_frame, cxf.alice)

>>> p_rt_fs = cxfm.act(op_fs_inv, tau, p_fs)
>>> p_rt_bp = cxfm.act(op_bp_inv, tau, p_bp)
>>> p_rt_co = cxfm.act(op_co_inv, tau, p_co)

>>> np.testing.assert_allclose(p_rt_fs.value, p.value, atol=1e-6)
>>> np.testing.assert_allclose(p_rt_bp.value, p.value, atol=1e-5)
>>> np.testing.assert_allclose(p_rt_co.value, p.value, atol=1e-6)
```

## Step 6: Why They Agree — and When They Don't

For any **planar curve with constant curvature** $\kappa$ and zero torsion $\tau_{\text{geom}} = 0$:

- The Frenet–Serret rotation rate equals $\kappa$ (constant),
- parallel transport has no torsion-induced drift, and
- a rigid rotation at rate $\kappa$ about the curve normal to the plane matches exactly.

For a **helix** (constant $\kappa > 0$, constant $\tau_{\text{geom}} \ne 0$), the Frenet–Serret and Bishop frames differ — the FS normal $\mathbf{N}$ rotates relative to the parallel-transported $\mathbf{U}_1$. A rigid-rotation frame would need a more complex axis to match either.

Let's verify on a helix that the three frames **disagree**:

```pycon
>>> def helix(tau: u.Q) -> u.Q:
...     t = tau.ustrip("s")
...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "km")
...

>>> fs_helix = cxfc.FrenetSerretTransform.from_curve(helix)
>>> bp_helix = cxfc.BishopTransform.from_curve(
...     helix,
...     initial_normal=jnp.array([-1.0, 0.0, 0.0]),
... )

>>> tau_h = u.Q(1.0, "s")
>>> T_fs = fs_helix.tangent(tau_h).value
>>> N_fs = fs_helix.normal(tau_h).value
>>> U1_bp = bp_helix.normal1(tau_h).value

>>> assert not np.allclose(
...     N_fs, U1_bp, atol=1e-3
... ), "On a helix the FS normal and Bishop normal should differ!"
```

## Summary

| Quantity | Circle ($\kappa$ const, $\tau_\text{geom} = 0$) | Helix ($\kappa$ const, $\tau_\text{geom} \ne 0$) |
| --- | --- | --- |
| Frenet–Serret $R(\tau)$ | $=$ Rotation by $\kappa\tau$ about $\hat{z}$ | FS-specific |
| Bishop $R(\tau)$ | same as FS | differs from FS by torsion drift |
| Rigid rotation | same as FS | does not match either |

The circle is the unique case where all three coincide — a useful sanity check and a gateway to understanding the differences that emerge on more complex curves.
