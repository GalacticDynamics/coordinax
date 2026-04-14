# coordinax.curveframes Specification

This document is the normative specification for `coordinax.curveframes`.

`coordinax.curveframes` is subordinate to [docs/spec.md](../../../docs/spec.md). If behavior differs, the root spec is authoritative.

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

Expressing the inverse as a `FrenetSerretTransform` with its own location / tangent / normal / binormal fields:

$$
\boldsymbol{\gamma}_{\text{inv}}(\tau)
  = -R(\tau)\,\boldsymbol{\gamma}(\tau),
$$

$$
\mathbf{T}_{\text{inv}}(\tau)
  = \text{column 0 of } R(\tau)
  = \bigl(T_0,\; N_0,\; B_0\bigr),
$$

$$
\mathbf{N}_{\text{inv}}(\tau)
  = \text{column 1 of } R(\tau)
  = \bigl(T_1,\; N_1,\; B_1\bigr),
$$

$$
\mathbf{B}_{\text{inv}}(\tau)
  = \text{column 2 of } R(\tau)
  = \bigl(T_2,\; N_2,\; B_2\bigr),
$$

where the subscripts denote Cartesian component indices.

**Double-inverse identity.** Because $(R^T)^T = R$:

$$
\bigl(F^{-1}\bigr)^{-1} = F.
$$

### Applying the Transform

Every `FrenetSerretTransform` instance — whether forward or inverse — stores its own location, tangent, normal, and binormal fields. The **uniform act formula** for any such instance is:

$$
\text{act}(F, \tau, \mathbf{p})
  = R_F(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}_F(\tau)\bigr),
$$

where $R_F = [\mathbf{T}_F;\,\mathbf{N}_F;\,\mathbf{B}_F]$ (rows) and $\boldsymbol{\gamma}_F$ are the fields of the instance $F$.

**Verification for the inverse instance.** Substituting the inverse fields into the uniform formula recovers the mathematical inverse:

$$
R_{\text{inv}}(\mathbf{p}' - \boldsymbol{\gamma}_{\text{inv}})
  = R^T \mathbf{p}' + \boldsymbol{\gamma},
$$

which is exactly the inverse transform defined above.

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

The inverse fields expressed as a `BishopTransform`:

$$
\boldsymbol{\gamma}_{\text{inv}}(\tau)
  = -R(\tau)\,\boldsymbol{\gamma}(\tau),
$$

$$
\mathbf{T}_{\text{inv}}(\tau)
  = \text{column 0 of } R(\tau)
  = \bigl(T_0,\; U_{1,0},\; U_{2,0}\bigr),
$$

$$
\mathbf{U}_{1,\text{inv}}(\tau)
  = \text{column 1 of } R(\tau)
  = \bigl(T_1,\; U_{1,1},\; U_{2,1}\bigr),
$$

$$
\mathbf{U}_{2,\text{inv}}(\tau)
  = \text{column 2 of } R(\tau)
  = \bigl(T_2,\; U_{1,2},\; U_{2,2}\bigr).
$$

**Double-inverse identity.** $(R^T)^T = R$, so $\bigl(B^{-1}\bigr)^{-1} = B$.

### Applying the Transform

The **uniform act formula** is identical in form to the Frenet–Serret case:

$$
\text{act}(B, \tau, \mathbf{p})
  = R_B(\tau)\bigl(\mathbf{p} - \boldsymbol{\gamma}_B(\tau)\bigr),
$$

where $R_B = [\mathbf{T}_B;\,\mathbf{U}_{1,B};\,\mathbf{U}_{2,B}]$ (rows).

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

The public API lives under `coordinax.curveframes` (typically imported as `import coordinax.curveframes as cxfc`).

| Symbol | Kind | Description |
| --- | --- | --- |
| `AbstractParallelTransportFrame` | `abstract` | Base class for curve-attached orthonormal frames |
| `FrenetSerretTransform` | `@final` | $\tau$-dependent rigid-body curve-frame transform |
| `FrenetSerretFrame` | `@final` | Frenet–Serret curve-attached reference frame |
| `BishopTransform` | `@final` | $\tau$-dependent rotation-minimising transform |
| `BishopFrame` | `@final` | Bishop (rotation-minimising) curve frame |

(curveframes-sw-abstract-curve-frame)=

!!! info `AbstractParallelTransportFrame`

    Abstract base class for curve-attached orthonormal frames in 3D.

    Inherits from `coordinax.frames.AbstractTransformedReferenceFrame[FrameT]` and therefore carries two inherited fields:

    - `base_frame : FrameT` — the ambient reference frame relative to which the curve frame is defined.
    - `xop : AbstractTransform` — the transform operator from the base frame to this frame.

    `AbstractParallelTransportFrame` is **not instantiable directly**; concrete subclasses (e.g. `FrenetSerretFrame`, `BishopFrame`) must be `@final`.

    Because `AbstractParallelTransportFrame` IS-A `AbstractTransformedReferenceFrame`, the generic `frame_transition` dispatches registered for `AbstractTransformedReferenceFrame` apply automatically. No additional frame-transition dispatches are needed for concrete curve-frame subclasses.

(curveframes-sw-frenet-transform)=

!!! info `FrenetSerretTransform`

    A `@final` subclass of `coordinax.transforms.AbstractTransform` representing a $\tau$-dependent rigid-body curve-frame transform.

    Fields:

    - `location : Callable[[Any], Any]` — $\tau \mapsto \boldsymbol{\gamma}(\tau)$, the curve position.
    - `tangent : Callable[[Any], Any]` — $\tau \mapsto \mathbf{T}(\tau)$, the unit tangent vector.
    - `normal : Callable[[Any], Any]` — $\tau \mapsto \mathbf{N}(\tau)$, the unit normal vector.
    - `binormal : Callable[[Any], Any]` — $\tau \mapsto \mathbf{B}(\tau)$, the unit binormal vector.
    - `curve : Callable[[Any], Any]` — the original curve callable, stored for inverse reconstruction.
    - `tau_unit : unxt.AbstractUnit` — the physical unit of the curve parameter $\tau$.

    All primary fields (`location`, `tangent`, `normal`, `binormal`) are **lazy**: they are $\tau$-dependent callables, not pre-evaluated arrays. Evaluating a field at a concrete $\tau$ value triggers JAX computation.

    The `curve` field stores the original curve callable passed to `from_curve`. For a forward transform, `location is curve` (identity). For an inverse transform, `location` is a derived closure and `location is not curve`.

    Constructors:

    - `from_curve(curve, /, tau_unit="s")` — Constructs the full Frenet–Serret transform from a curve callable using JAX automatic differentiation. `curve` is a function $\tau \mapsto \text{Quantity}[(3,)]$. `tau_unit` (default `"s"`) specifies the unit of $\tau$ for differentiation. First and second derivatives are obtained via `unxt.experimental.jacfwd(curve, units=(tau_unit,))`, which correctly tracks physical units through forward-mode AD. The unit normal $\mathbf{N}$ is computed via Gram–Schmidt orthogonalisation: project out the tangent component from $\boldsymbol{\gamma}''$, then normalise. Sets `curve = curve` and `tau_unit = u.unit(tau_unit)`.
    - `from_(curve)` — Plum multiple-dispatch constructor. Dispatches on `Callable` and delegates to `from_curve(curve)` with default `tau_unit="s"`.

    `inverse` property:

    - Returns a new `FrenetSerretTransform` representing the inverse rigid-body transform, computed lazily from the stored callable fields.
    - **Forward case** (`location is curve`): the inverse fields are computed as closures over `self.location`, `self.tangent`, `self.normal`, `self.binormal`. The returned instance carries the same `curve` and `tau_unit`.
    - **Inverse case** (`location is not curve`): the double-inverse is detected via identity comparison. Instead of accumulating another layer of closures, `inverse` reconstructs the forward transform cleanly by calling `from_curve(self.curve, tau_unit=self.tau_unit)`.
    - This guarantees a **two-step cycle**: forward ↔ inverse, with no closure chain accumulation regardless of how many times `.inverse` is called.

    JAX compatibility:

    - All fields are pure-function closures; the transform is a valid JAX PyTree (via Equinox).
    - `location`, `tangent`, `normal`, `binormal` (and their inverses) are compatible with `jax.jit` and `jax.vmap`.
    - Fields should operate on **scalar** $\tau$ values; batching is achieved via `jax.vmap`.

    `act` dispatches:

    - `act(op: FrenetSerretTransform, tau, x: AbstractQuantity, chart, rep)` — evaluates the frame fields at $\tau$ and applies the uniform formula $R(\tau)(x - \gamma(\tau))$. Works for both forward and inverse instances.
    - `act(op: FrenetSerretTransform, tau, x: CDict, chart, rep)` — extracts the component dictionary into a Quantity array, applies the transform, and repacks into a CDict.

(curveframes-sw-frenet-frame)=

!!! info `FrenetSerretFrame`

    A `@final` subclass of `AbstractParallelTransportFrame[FrameT]` representing a Frenet–Serret curve-attached reference frame.

    Fields (all inherited):

    - `base_frame : FrameT` — the ambient reference frame (e.g. `Alice()`).
    - `xop : FrenetSerretTransform` — the $\tau$-dependent rigid-body transform from the base frame to the curve frame.

    The `xop` field is constrained to `FrenetSerretTransform`. At evaluation time, the evolution parameter $\tau$ is passed via `act(op, tau, x)`, not stored on the frame.

    Constructors:

    - `FrenetSerretFrame(base_frame, xop)` — direct construction from a base frame and a pre-built `FrenetSerretTransform`.
    - `from_curve(base_frame, curve, /, tau_unit="s")` — convenience constructor that calls `FrenetSerretTransform.from_curve(curve, tau_unit=tau_unit)` and wraps the result.

    Frame transitions:

    - Handled entirely by the generic `AbstractTransformedReferenceFrame` dispatches in `coordinax.frames`.
    - `frame_transition(base, fs_frame)` returns `(base → base_frame) | xop`.
    - `frame_transition(fs_frame, base)` returns `xop.inverse | (base_frame → base)`.
    - `frame_transition(fs_frame_1, fs_frame_2)` composes through both base frames.

    Usage pattern:

    ```python
    import jax.numpy as jnp
    import unxt as u
    import coordinax.frames as cxf
    import coordinax.transforms as cxfm
    import coordinax.curveframes as cxfc


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

!!! info `BishopTransform`

    A `@final` subclass of `coordinax.transforms.AbstractTransform` representing a $\tau$-dependent rotation-minimising curve-frame transform.

    Fields:

    - `location : Callable[[Any], Any]` — $\tau \mapsto \boldsymbol{\gamma}(\tau)$, the curve position.
    - `tangent : Callable[[Any], Any]` — $\tau \mapsto \mathbf{T}(\tau)$, the unit tangent vector.
    - `normal1 : Callable[[Any], Any]` — $\tau \mapsto \mathbf{U}_1(\tau)$, the first normal vector (parallel-transported).
    - `normal2 : Callable[[Any], Any]` — $\tau \mapsto \mathbf{U}_2(\tau)$, the second normal vector ($\mathbf{T} \times \mathbf{U}_1$).
    - `curve : Callable[[Any], Any]` — the original curve callable, stored for inverse reconstruction.
    - `tau_unit : unxt.AbstractUnit` — the physical unit of $\tau$.
    - `tau_0 : unxt.AbstractQuantity` — the reference parameter value at which the initial frame is defined.
    - `initial_normal : jax.Array | None` — the initial $\mathbf{U}_{1,0}$ (dimensionless 3-vector).  Stored for double-inverse reconstruction.  `None` when auto-chosen.

    All primary fields are **lazy** callables.  The `normal1` and `normal2` fields internally solve the parallel-transport ODE from `tau_0` to the requested $\tau$ using `jax.experimental.ode.odeint`.

    The `curve` field stores the original curve callable.  For a forward transform, `location is curve`.  For an inverse, `location is not curve`.

    Constructors:

    - `from_curve(curve, /, tau_unit="s", *, tau_0=None, initial_normal=None)` — Constructs the Bishop transform from a curve callable.  `curve` is a function $\tau \mapsto \text{Quantity}[(3,)]$.  `tau_unit` specifies the unit of $\tau$.  `tau_0` (default `Q(0.0, tau_unit)`) is the reference parameter.  `initial_normal` (default `None`) is a dimensionless 3-vector for $\mathbf{U}_{1,0}$; when `None`, one is auto-chosen via Gram–Schmidt.  The unit tangent $\mathbf{T}$ is computed via `unxt.experimental.jacfwd`.  The parallel-transport ODE is solved via `jax.experimental.ode.odeint`.
    - `from_(curve)` — Plum multiple-dispatch constructor.  Dispatches on `Callable` and delegates to `from_curve(curve)` with defaults.

    `inverse` property:

    - Returns a new `BishopTransform` representing the inverse rigid-body transform.
    - **Forward case** (`location is curve`): inverse fields are closures over the forward fields.
    - **Inverse case** (`location is not curve`): double-inverse detected; reconstructs forward via `from_curve(self.curve, tau_unit=self.tau_unit, tau_0=self.tau_0, initial_normal=self.initial_normal)`.
    - Guarantees a **two-step cycle** with no closure accumulation.

    JAX compatibility:

    - All fields are pure-function closures; the transform is a valid JAX PyTree.
    - Compatible with `jax.jit` and `jax.vmap`.
    - Fields operate on **scalar** $\tau$; batching via `jax.vmap`.

    `act` dispatches:

    - `act(op: BishopTransform, tau, x: AbstractQuantity, chart, rep)` — same uniform formula as FrenetSerretTransform: $R(\tau)(x - \gamma(\tau))$ where $R = [\mathbf{T};\,\mathbf{U}_1;\,\mathbf{U}_2]$.
    - `act(op: BishopTransform, tau, x: CDict, chart, rep)` — extract → transform → repack.

(curveframes-sw-bishop-frame)=

!!! info `BishopFrame`

    A `@final` subclass of `AbstractParallelTransportFrame[FrameT]` representing a Bishop (rotation-minimising) curve-attached reference frame.

    Fields (all inherited):

    - `base_frame : FrameT` — the ambient reference frame (e.g. `Alice()`).
    - `xop : BishopTransform` — the $\tau$-dependent rotation-minimising transform from the base frame to the curve frame.

    The `xop` field is constrained to `BishopTransform`.

    Constructors:

    - `BishopFrame(base_frame, xop)` — direct construction.
    - `from_curve(base_frame, curve, /, tau_unit="s", *, tau_0=None, initial_normal=None)` — convenience constructor that calls `BishopTransform.from_curve(...)` and wraps the result.

    Frame transitions:

    - Handled by the generic `AbstractParallelTransportFrame` dispatches (same as FrenetSerretFrame).

    Usage pattern:

    ```python
    import jax.numpy as jnp
    import unxt as u
    import coordinax.frames as cxf
    import coordinax.transforms as cxfm
    import coordinax.curveframes as cxfc


    def curve(tau: u.Q) -> u.Q:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "m")


    b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), curve)
    op = cxf.frame_transition(cxf.Alice(), b_frame)
    tau = u.Q(0.0, "s")
    p_ambient = u.Q(jnp.array([1.0, 0.0, 0.0]), "m")
    p_curve = cxfm.act(op, tau, p_ambient)
    ```
