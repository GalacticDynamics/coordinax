# coordinaxs.curveframes Specification

This document is the normative specification for `coordinaxs.curveframes`.

`coordinaxs.curveframes` is subordinate to [docs/spec.md](../../../docs/spec.md). If behavior differs, the root spec is authoritative.

---

# The Math

(curveframes-math-curves)=

## Curves in Euclidean 3-Space

A **smooth parameterized curve** is a smooth map

$$
\boldsymbol{\gamma} : \Lambda \to \mathbb{R}^3,
\qquad \tau \mapsto \boldsymbol{\gamma}(\tau),
$$

where $\Lambda \subseteq \mathbb{R}$ is an open interval and $\tau$ is a smooth **evolution parameter** (arc length, time, proper time, etc.). The curve is **regular** when $\boldsymbol{\gamma}'(\tau) \neq 0$ for all $\tau \in \Lambda$.

(curveframes-math-frenet-serret)=

## Frenet–Serret Frame

For a regular curve $\boldsymbol{\gamma}$ with non-vanishing curvature, the **Frenet–Serret frame** attaches an oriented orthonormal triad $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ to each point:

$$
\mathbf{T}(\tau) = \frac{\boldsymbol{\gamma}'(\tau)}
                        {\lVert\boldsymbol{\gamma}'(\tau)\rVert},
$$

$$
\mathbf{N}(\tau) = \frac{\boldsymbol{\gamma}''(\tau)
  - \bigl(\boldsymbol{\gamma}''(\tau) \cdot \mathbf{T}(\tau)\bigr)\,
    \mathbf{T}(\tau)}
  {\bigl\lVert \boldsymbol{\gamma}''(\tau)
  - \bigl(\boldsymbol{\gamma}''(\tau) \cdot \mathbf{T}(\tau)\bigr)\,
    \mathbf{T}(\tau) \bigr\rVert},
$$

$$
\mathbf{B}(\tau) = \mathbf{T}(\tau) \times \mathbf{N}(\tau).
$$

| Symbol | Name | Definition |
| --- | --- | --- |
| $\mathbf{T}$ | Tangent | Unit tangent: $\boldsymbol{\gamma}'/\lVert\boldsymbol{\gamma}'\rVert$ |
| $\mathbf{N}$ | Normal | Unit principal normal: Gram–Schmidt rejection of $\boldsymbol{\gamma}''$ onto $\mathbf{T}$, then normalised |
| $\mathbf{B}$ | Binormal | $\mathbf{T} \times \mathbf{N}$ (right-handed completion) |

**Properties.** For every $\tau$:

1. **Orthonormality**: $\mathbf{T} \cdot \mathbf{N} = \mathbf{T} \cdot \mathbf{B} = \mathbf{N} \cdot \mathbf{B} = 0$ and $\lVert\mathbf{T}\rVert = \lVert\mathbf{N}\rVert = \lVert\mathbf{B}\rVert = 1$.
2. **Right-handedness**: $\mathbf{B} = \mathbf{T} \times \mathbf{N}$.
3. **Singularity**: The frame is undefined when the curvature $\kappa(\tau) = 0$ (i.e.\ the curve is locally straight).

(curveframes-math-frenet-transform)=

## Frenet–Serret Transform

The Frenet–Serret frame defines a $\tau$-dependent **rigid-body transform** (translation + rotation) between the ambient Cartesian frame and the curve-attached frame.

### Forward Transform

At each $\tau$, define the rotation matrix

$$
R(\tau) = \begin{pmatrix}
  \mathbf{T}(\tau)^T \\
  \mathbf{N}(\tau)^T \\
  \mathbf{B}(\tau)^T
\end{pmatrix}
\in SO(3).
$$

The **forward transform** maps an ambient point $\mathbf{p}$ to curve-frame coordinates:

$$
\mathbf{p}' = R(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}(\tau)\bigr).
$$

### Inverse Transform

Since $R \in SO(3)$, we have $R^{-1} = R^T$. The **inverse transform** maps curve-frame coordinates back to the ambient frame:

$$
\mathbf{p} = R^T(\tau)\,\mathbf{p}' + \boldsymbol{\gamma}(\tau).
$$

**Double-inverse identity.** Because $(R^T)^T = R$:

$$
\bigl(F^{-1}\bigr)^{-1} = F.
$$

### Applying the Transform

A `FrenetSerretBuilder` does not store $\boldsymbol{\gamma}$, $\mathbf{T}$, $\mathbf{N}$, $\mathbf{B}$ as separate fields. It stores the curve $\boldsymbol{\gamma}$ itself (and `tau_unit`), and builds the rigid-body transform on demand:

$$
F(\tau) = \mathrm{Translate}\bigl(-\boldsymbol{\gamma}(\tau)\bigr)\;\big|\;\mathrm{Rotate}\bigl(R(\tau)\bigr),
$$

evaluated left-to-right (translate, then rotate), so applying $F(\tau)$ to a point $\mathbf{p}$ gives exactly the forward-transform formula:

$$
\text{act}(F(\tau), \mathbf{p}) = R(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}(\tau)\bigr).
$$

The `FrenetSerretBuilder` itself is wrapped in a `coordinax.transforms.Parametric`, `Parametric(F)`, which is what `act(op, tau, p)` actually dispatches on: `act` calls `F(tau)` to materialise the rigid-body transform, then applies it.

**Inversion is generic, not builder-specific.** `Parametric(F).inverse` does not construct a second `FrenetSerretBuilder` with swapped fields; it wraps `F` in a pointwise-inverse combinator whose `__call__(tau)` returns `F(tau).inverse`. Inverting the composed `Translate(-\gamma) | Rotate(R)` reverses order and inverts each factor:

$$
F(\tau)^{-1} = \mathrm{Rotate}\bigl(R(\tau)\bigr)^{-1} \;\big|\; \mathrm{Translate}\bigl(-\boldsymbol{\gamma}(\tau)\bigr)^{-1}
  = \mathrm{Rotate}\bigl(R^T(\tau)\bigr) \;\big|\; \mathrm{Translate}\bigl(\boldsymbol{\gamma}(\tau)\bigr),
$$

which applied to $\mathbf{p}'$ (rotate, then translate) gives exactly $R^T(\tau)\,\mathbf{p}' + \boldsymbol{\gamma}(\tau)$ — the inverse transform above. Because this combinator is an involution, `Parametric(F).inverse.inverse.builder is F` — no closure chain accumulates no matter how many times `.inverse` is taken.

(curveframes-math-frenet-ref-frame)=

## Frenet–Serret Reference Frame

A **Frenet–Serret reference frame** $\mathcal{F}_\gamma$ is a curve-attached reference frame defined relative to an ambient **base frame** $\mathcal{B}$. At each parameter value $\tau$, the frame is centred at $\boldsymbol{\gamma}(\tau)$ with oriented axes $(\mathbf{T}, \mathbf{N}, \mathbf{B})$.

The **frame transition** from the base frame to the curve frame is the forward Frenet–Serret transform:

$$
\mathcal{B} \xrightarrow{F(\tau)} \mathcal{F}_\gamma.
$$

The reverse transition is its inverse:

$$
\mathcal{F}_\gamma \xrightarrow{F^{-1}(\tau)} \mathcal{B}.
$$

**Composition with other frames.** Given an arbitrary frame $\mathcal{A}$ with a known transition to $\mathcal{B}$, the transition from $\mathcal{A}$ to $\mathcal{F}_\gamma$ composes:

$$
\mathcal{A} \to \mathcal{F}_\gamma
  = (\mathcal{A} \to \mathcal{B}) \circ
    (\mathcal{B} \to \mathcal{F}_\gamma).
$$

The evolution parameter $\tau$ is **not** stored on the frame object. It is supplied at evaluation time when the frame transition operator is applied to coordinates via `act(op, tau, x)`.

(curveframes-math-bishop)=

## Bishop Frame

The **Bishop frame** (also called the **rotation-minimising frame** or **parallel-transport frame**) attaches an orthonormal triad $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$ to each point of a regular curve $\boldsymbol{\gamma}(\tau)$. Unlike the Frenet–Serret frame, it is well-defined even when the curvature vanishes ($\kappa = 0$), because it does not depend on $\boldsymbol{\gamma}''$.

### Definition via Parallel Transport

Given a regular curve with unit tangent $\mathbf{T}(\tau) = \boldsymbol{\gamma}'/\lVert\boldsymbol{\gamma}'\rVert$, choose an **initial** orthonormal pair $\mathbf{U}_1(\tau_0), \mathbf{U}_2(\tau_0)$ in the normal plane at $\tau_0$ (i.e.\ perpendicular to $\mathbf{T}(\tau_0)$).

The Bishop frame vectors $\mathbf{U}_1(\tau), \mathbf{U}_2(\tau)$ are the unique solutions of the **parallel-transport ODE**:

$$
\frac{d\mathbf{U}_i}{d\tau}
  = -\bigl(\mathbf{U}_i \cdot \mathbf{T}'\bigr)\,\mathbf{T},
  \qquad i \in \{1, 2\},
$$

subject to the initial conditions $\mathbf{U}_i(\tau_0) = \mathbf{U}_{i,0}$.

Equivalently, at each point the derivative of $\mathbf{U}_i$ is its projection onto the tangent direction, negated — this keeps $\mathbf{U}_i$ perpendicular to $\mathbf{T}$ while minimising rotation.

### Choosing the Initial Normal

When the user does not supply an explicit initial normal, one is selected automatically via **Gram–Schmidt**: pick the standard basis vector $\mathbf{e}_k$ ($k \in \{0,1,2\}$) that is _least aligned_ with $\mathbf{T}(\tau_0)$, i.e.\
$k = \arg\min_j\, |\mathbf{T}(\tau_0) \cdot \mathbf{e}_j|$. Then project out the tangent component and normalise:

$$
\mathbf{U}_{1,0}
  = \frac{\mathbf{e}_k - (\mathbf{e}_k \cdot \mathbf{T}_0)\,\mathbf{T}_0}
         {\lVert \mathbf{e}_k - (\mathbf{e}_k \cdot \mathbf{T}_0)\,\mathbf{T}_0 \rVert},
\qquad
\mathbf{U}_{2,0} = \mathbf{T}_0 \times \mathbf{U}_{1,0}.
$$

### Properties

For every $\tau$:

1. **Orthonormality**: $\mathbf{T} \cdot \mathbf{U}_1 = \mathbf{T} \cdot \mathbf{U}_2
   = \mathbf{U}_1 \cdot \mathbf{U}_2 = 0$ and $\lVert\mathbf{T}\rVert = \lVert\mathbf{U}_1\rVert
   = \lVert\mathbf{U}_2\rVert = 1$.
2. **Right-handedness**: $\mathbf{U}_2 = \mathbf{T} \times \mathbf{U}_1$.
3. **Non-singularity**: The frame is defined for _every_ regular curve, including at points where $\kappa = 0$. (In contrast, the Frenet–Serret normal is singular at such points.)
4. **Rotation-minimising**: The angular velocity of the frame about the tangent is zero — the normal-plane vectors do not twist around $\mathbf{T}$.

### Relation to the Frenet–Serret Frame

When $\kappa(\tau) \neq 0$, a Bishop frame and the Frenet–Serret frame are related by a $\tau$-dependent rotation $\theta(\tau)$ in the normal plane:

$$
\begin{pmatrix} \mathbf{N} \\ \mathbf{B} \end{pmatrix}
  = \begin{pmatrix}
      \cos\theta & \sin\theta \\
      -\sin\theta & \cos\theta
    \end{pmatrix}
  \begin{pmatrix} \mathbf{U}_1 \\ \mathbf{U}_2 \end{pmatrix}.
$$

The Bishop frame is the unique frame in this family for which $d\theta/d\tau = 0$ (no torsion-induced twist).

(curveframes-math-bishop-transform)=

## Bishop Transform

The Bishop frame defines a $\tau$-dependent **rigid-body transform** (translation + rotation) between the ambient Cartesian frame and the curve-attached frame, with the same algebraic structure as the Frenet–Serret transform.

### Forward Transform

At each $\tau$, define the rotation matrix

$$
R(\tau) = \begin{pmatrix}
  \mathbf{T}(\tau)^T \\
  \mathbf{U}_1(\tau)^T \\
  \mathbf{U}_2(\tau)^T
\end{pmatrix}
\in SO(3).
$$

The **forward transform** maps an ambient point $\mathbf{p}$ to curve-frame coordinates:

$$
\mathbf{p}' = R(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}(\tau)\bigr).
$$

### Inverse Transform

Since $R \in SO(3)$, $R^{-1} = R^T$ and the **inverse transform** is:

$$
\mathbf{p} = R^T(\tau)\,\mathbf{p}' + \boldsymbol{\gamma}(\tau).
$$

**Double-inverse identity.** $(R^T)^T = R$, so $\bigl(B^{-1}\bigr)^{-1} = B$.

### Applying the Transform

Exactly as for `FrenetSerretBuilder` ({ref}`curveframes-math-frenet-transform`): a `BishopBuilder` stores the curve $\boldsymbol{\gamma}$ (and `tau_unit`, `tau_0`, `initial_normal`), not $\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2$ as separate fields, and builds

$$
B(\tau) = \mathrm{Translate}\bigl(-\boldsymbol{\gamma}(\tau)\bigr)\;\big|\;\mathrm{Rotate}\bigl(R(\tau)\bigr),
\qquad R = [\mathbf{T};\,\mathbf{U}_1;\,\mathbf{U}_2] \text{ (rows)}
$$

on demand, wrapped in `Parametric(B)`. The **uniform act formula** is identical in form to the Frenet–Serret case:

$$
\text{act}(B(\tau), \mathbf{p}) = R(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}(\tau)\bigr).
$$

Inversion is the same generic `Parametric` pointwise-inverse combinator described for `FrenetSerretBuilder` — there is no separate `BishopBuilder` for the inverse direction, and `Parametric(B).inverse.inverse.builder is B`.

(curveframes-math-bishop-ref-frame)=

## Bishop Reference Frame

A **Bishop reference frame** $\mathcal{B}_\gamma$ is a curve-attached reference frame defined relative to an ambient **base frame** $\mathcal{B}$. At each parameter value $\tau$, the frame is centred at $\boldsymbol{\gamma}(\tau)$ with oriented axes $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$.

Frame transitions and composition rules are identical in structure to the Frenet–Serret case:

$$
\mathcal{B} \xrightarrow{B(\tau)} \mathcal{B}_\gamma,
\qquad
\mathcal{B}_\gamma \xrightarrow{B^{-1}(\tau)} \mathcal{B}.
$$

The evolution parameter $\tau$ is **not** stored on the frame object. It is supplied at evaluation time via `act(op, tau, x)`.

---

# The Software

(curveframes-sw-overview)=

## Overview

The public API lives under `coordinaxs.curveframes` (typically imported as `import coordinaxs.curveframes as cxfc`).

| Symbol | Kind | Description |
| --- | --- | --- |
| `AbstractParallelTransportFrame` | `abstract` | Base class for curve-attached orthonormal frames |
| `AbstractCurveFrameBuilder` | `abstract` | `equinox.Module` builder ABC: `tau -> Translate(-gamma) \| Rotate(R)` |
| `FrenetSerretBuilder` | `@final` | Builder for the $(\mathbf{T},\mathbf{N},\mathbf{B})$ triad |
| `FrenetSerretFrame` | `@final` | Frenet–Serret curve-attached reference frame |
| `BishopBuilder` | `@final` | Builder for the $(\mathbf{T},\mathbf{U}_1,\mathbf{U}_2)$ triad |
| `BishopFrame` | `@final` | Bishop (rotation-minimising) curve frame |

Every curve frame is built from a `coordinax.transforms.Parametric` wrapping one of these builders — the same single mechanism for time dependence used everywhere else in `coordinax.transforms` (see [Parametric](../../../docs/spec.md#software-spec-transforms-parametric) in the root spec). `AbstractCurveFrameBuilder` is an `equinox.Module`, so every field is a genuine pytree leaf: differentiable and `vmap`-able, including the curve's own parameters when the curve is itself an `equinox.Module`.

(curveframes-sw-abstract-curve-frame)=

!!! info `AbstractParallelTransportFrame`

    Abstract base class for curve-attached orthonormal frames in 3D.

    Inherits from `coordinax.frames.AbstractTransformedReferenceFrame[FrameT]` and therefore carries three fields:

    - `base_frame : FrameT` — the ambient reference frame relative to which the curve frame is defined.
    - `xop : Parametric` — the forward transform (base frame → curve frame), wrapping an `AbstractCurveFrameBuilder`.
    - `xop_inv : Parametric` — the pre-computed inverse of `xop` (curve frame → base frame).

    `AbstractParallelTransportFrame` is **not instantiable directly**; concrete subclasses (e.g. `FrenetSerretFrame`, `BishopFrame`) must be `@final`.

    Because `AbstractParallelTransportFrame` IS-A `AbstractTransformedReferenceFrame`, the generic `frame_transition` dispatches registered for `AbstractTransformedReferenceFrame` apply automatically. No additional frame-transition dispatches are needed for concrete curve-frame subclasses.

    The evolution parameter $\tau$ is **not** stored on the frame object. It is supplied at evaluation time when a frame-transition operator is applied to coordinates via `act(op, tau, x)`.

(curveframes-sw-abstract-builder)=

!!! info `AbstractCurveFrameBuilder`

    Abstract `equinox.Module` base class for curve-frame builders: `tau -> Translate(-gamma) | Rotate(R)`. This is what a `Parametric` wraps; it is not itself a `coordinax.transforms.AbstractTransform`.

    Fields (declared `eqx.AbstractVar`, defined by concrete subclasses):

    - `curve : Callable[[Any], Any]` — the curve $\gamma \mapsto \boldsymbol{\gamma}(\gamma)$, mapping a parameter `Quantity` to a Cartesian 3-vector `Quantity`. A pytree leaf: make `curve` itself an `equinox.Module` to get differentiable/vmappable curve parameters.
    - `tau_unit : unxt.AbstractUnit` — physical unit of the curve parameter. **Static**: it selects the differentiation units, not a numeric value.
    - `gamma : Any` — an optional *fixed* curve parameter. When `None` (the default), $\tau$ itself is the curve parameter — the classic moving-frame usage. When set, the frame sits at the fixed point $\boldsymbol{\gamma}(\gamma)$ and is $\tau$-independent: a frame *field* along the curve, differentiable and `vmap`-able in `gamma`.

    Methods:

    - `rotation_matrix(tau) -> Array` — **abstract**, implemented by concrete subclasses; the $3\times3$ rotation matrix $R$ whose rows are the frame vectors.
    - `__call__(tau) -> Composed` — builds `Translate(-gamma(param)) | Rotate(rotation_matrix(tau))`, where `param` is `tau` or the fixed `gamma`.
    - `location(tau)`, `tangent(tau)` — convenience accessors; `location` evaluates $\boldsymbol{\gamma}$ at the resolved parameter, `tangent` returns row 0 of `rotation_matrix(tau)`.

(curveframes-sw-frenet-transform)=

!!! info `FrenetSerretBuilder`

    A `@final` subclass of `AbstractCurveFrameBuilder` computing the $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ triad.

    Fields (in addition to the ABC's `curve`, `tau_unit`, `gamma`): none — `FrenetSerretBuilder` adds no fields beyond the base class.

    - `curve : Callable[[Any], Any]` — the constructing curve.
    - `tau_unit : unxt.AbstractUnit` — unit of the curve parameter, used by `unxt.experimental.jacfwd` to compute unit-correct derivatives. Defaults to `"s"`. Static.
    - `gamma : Any` — optional fixed curve parameter (a leaf); see `AbstractCurveFrameBuilder`.

    `rotation_matrix(tau)` computes $R = [\mathbf{T}; \mathbf{N}; \mathbf{B}]$: unit-aware first and second derivatives of `curve` via `unxt.experimental.jacfwd`, then $\mathbf{T} = \gamma'/\lVert\gamma'\rVert$, Gram–Schmidt rejection of $\gamma''$ onto $\mathbf{T}$ normalised to give $\mathbf{N}$, and $\mathbf{B} = \mathbf{T}\times\mathbf{N}$.

    Convenience accessors: `normal(tau)` (row 1), `binormal(tau)` (row 2); `location(tau)`, `tangent(tau)` are inherited from `AbstractCurveFrameBuilder`.

    Constructed directly — `FrenetSerretBuilder(curve, tau_unit="s", gamma=None)` — there is no `from_curve`/`from_` classmethod on the builder; that convenience lives on `FrenetSerretFrame`.

    JAX compatibility: `FrenetSerretBuilder` is an `equinox.Module`, so it is a valid pytree. `curve`, `gamma` are dynamic leaves (differentiable, `vmap`-able); `tau_unit` is static. `rotation_matrix` and `__call__` operate on scalar $\tau$; batching is via `jax.vmap`. A plain `jax.jit` cannot hash a builder holding array leaves (e.g. an `equinox.Module` curve with array fields, or a `gamma`); use `eqx.filter_jit` in that case.

    `act` dispatches on `Parametric(FrenetSerretBuilder(...))`, not on the builder directly — see [`Parametric`](../../../docs/spec.md#software-spec-transforms-parametric) in the root spec. `act(Parametric(F), tau, x)` materialises `F(tau)` and applies the resulting `Composed` transform.

(curveframes-sw-frenet-frame)=

!!! info `FrenetSerretFrame`

    A `@final` subclass of `AbstractParallelTransportFrame[FrameT]` representing a Frenet–Serret curve-attached reference frame.

    Fields (all inherited):

    - `base_frame : FrameT` — the ambient reference frame (e.g. `Alice()`).
    - `xop : Parametric` — the $\tau$-dependent rigid-body transform from the base frame to the curve frame, wrapping a `FrenetSerretBuilder`.
    - `xop_inv : Parametric` — its pre-computed inverse, `xop.inverse`.

    At evaluation time, the evolution parameter $\tau$ is passed via `act(op, tau, x)`, not stored on the frame.

    Constructors:

    - `FrenetSerretFrame(base_frame, xop, xop_inv)` — direct construction from a base frame and a `Parametric`-wrapped `FrenetSerretBuilder` (forward and inverse).
    - `from_curve(base_frame, curve, /, tau_unit="s", *, gamma=None)` — convenience constructor that builds `FrenetSerretBuilder(curve, tau_unit, gamma)`, wraps it in `Parametric`, and sets `xop_inv = xop.inverse`.

    Frame transitions:

    - Handled entirely by the generic `AbstractTransformedReferenceFrame` dispatches in `coordinax.frames`.
    - `frame_transition(base, fs_frame)` returns `(base → base_frame) | xop`.
    - `frame_transition(fs_frame, base)` returns `xop_inv | (base_frame → base)`.
    - `frame_transition(fs_frame_1, fs_frame_2)` composes through both base frames.

    Usage pattern:

    ```python
    import jax.numpy as jnp
    import unxt as u
    import coordinax.frames as cxf
    import coordinax.transforms as cxfm
    import coordinaxs.curveframes as cxfc


    def circle(tau: u.Q) -> u.Q:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "m")


    fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
    op = cxf.frame_transition(cxf.Alice(), fs_frame)
    tau = u.Q(0.0, "s")
    p_ambient = u.Q(jnp.array([1.0, 0.0, 0.0]), "m")
    p_curve = cxfm.act(op, tau, p_ambient)
    ```

(curveframes-sw-bishop-transform)=

!!! info `BishopBuilder`

    A `@final` subclass of `AbstractCurveFrameBuilder` computing the $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$ triad via parallel transport.

    Fields (the ABC's `curve`, `tau_unit`, `gamma`, plus two more):

    - `curve : Callable[[Any], Any]` — the constructing curve.
    - `tau_unit : unxt.AbstractUnit` — unit of the curve parameter. Defaults to `"s"`. Static.
    - `gamma : Any` — optional fixed curve parameter (a leaf); see `AbstractCurveFrameBuilder`.
    - `tau_0 : unxt.AbstractQuantity | None` — reference parameter where the initial frame is defined (a leaf). `None` is resolved to `Q(0.0, tau_unit)` by `__post_init__`.
    - `initial_normal : Any` — initial $\mathbf{U}_{1,0}$ (dimensionless 3-vector, a leaf), or `None` for auto-selection via Gram–Schmidt.

    `rotation_matrix(tau)` computes $R = [\mathbf{T}; \mathbf{U}_1; \mathbf{U}_2]$: $\mathbf{T}$ from the unit-aware first derivative of `curve`; $\mathbf{U}_1$ by solving the parallel-transport ODE $d\mathbf{U}_1/d\tau = -(\mathbf{U}_1\cdot\mathbf{T}')\,\mathbf{T}$ from `tau_0` to the resolved parameter via `jax.experimental.ode.odeint` (skipped via `jax.lax.cond` when the parameter equals `tau_0`); $\mathbf{U}_2 = \mathbf{T}\times\mathbf{U}_1$.

    Convenience accessors: `normal1(tau)` (row 1), `normal2(tau)` (row 2); `location(tau)`, `tangent(tau)` inherited.

    Constructed directly — `BishopBuilder(curve, tau_unit="s", gamma=None, tau_0=None, initial_normal=None)` — there is no `from_curve`/`from_` classmethod on the builder; that convenience lives on `BishopFrame`.

    JAX compatibility: same as `FrenetSerretBuilder` — `curve`, `gamma`, `tau_0`, `initial_normal` are dynamic leaves; `tau_unit` is static. A plain `jax.jit` cannot hash a builder holding array leaves; use `eqx.filter_jit`.

    `act` dispatches on `Parametric(BishopBuilder(...))`, identically to `FrenetSerretBuilder`.

(curveframes-sw-bishop-frame)=

!!! info `BishopFrame`

    A `@final` subclass of `AbstractParallelTransportFrame[FrameT]` representing a Bishop (rotation-minimising) curve-attached reference frame.

    Fields (all inherited):

    - `base_frame : FrameT` — the ambient reference frame (e.g. `Alice()`).
    - `xop : Parametric` — the $\tau$-dependent rotation-minimising transform from the base frame to the curve frame, wrapping a `BishopBuilder`.
    - `xop_inv : Parametric` — its pre-computed inverse, `xop.inverse`.

    Constructors:

    - `BishopFrame(base_frame, xop, xop_inv)` — direct construction.
    - `from_curve(base_frame, curve, /, tau_unit="s", *, gamma=None, tau_0=None, initial_normal=None)` — convenience constructor that builds `BishopBuilder(curve, tau_unit, gamma, tau_0, initial_normal)`, wraps it in `Parametric`, and sets `xop_inv = xop.inverse`.

    Frame transitions:

    - Handled by the generic `AbstractParallelTransportFrame` dispatches (same as FrenetSerretFrame).

    Usage pattern:

    ```python
    import jax.numpy as jnp
    import unxt as u
    import coordinax.frames as cxf
    import coordinax.transforms as cxfm
    import coordinaxs.curveframes as cxfc


    def curve(tau: u.Q) -> u.Q:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "m")


    b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), curve)
    op = cxf.frame_transition(cxf.Alice(), b_frame)
    tau = u.Q(0.0, "s")
    p_ambient = u.Q(jnp.array([1.0, 0.0, 0.0]), "m")
    p_curve = cxfm.act(op, tau, p_ambient)
    ```
