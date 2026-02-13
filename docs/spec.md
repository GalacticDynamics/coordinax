# Coordinax Core Specification

This document is the **normative** specification for the mathematical and
software design of the `coordinax` coordinate and vector system. It defines a
low-level, mathematically correct framework—**Charts**, **Metrics**, **Frames**,
**Embeddings**, and **Maps**—and a high-level user API built on top of them --
**Vector**, **vconvert**, **PointedVector**, **Reference Frames**,
**Coordinate** .

The goals are:

1. **Correctness-first foundation**: rigorous definitions of points,
   tangent/cotangent objects, and their transformation laws.
2. **Ergonomic high-level API**: common tasks should not expose low-level
   details.
3. **Extensibility via multiple dispatch**: new charts/metrics/embeddings/roles
   are added by registration, not by editing core logic.
4. **JAX compatibility**: dispatch resolves on static Python objects; numerical
   kernels are pure and traceable.

## Table of Contents

- [Coordinax Core Specification](#coordinax-core-specification)
  - [Table of Contents](#table-of-contents)
  - [Package and API structure](#package-and-api-structure)
    - [High-level API (default user entry points)](#high-level-api-default-user-entry-points)
    - [Low-level API (normative mathematical kernels)](#low-level-api-normative-mathematical-kernels)
    - [Naming conventions and terminology](#naming-conventions-and-terminology)
    - [Dependency direction (normative)](#dependency-direction-normative)
  - [Mathematical Background and Conventions](#mathematical-background-and-conventions)
    - [Affine spaces vs. vector spaces](#affine-spaces-vs-vector-spaces)
    - [Tangent and cotangent spaces](#tangent-and-cotangent-spaces)
    - [Charts, bases, and orthonormal physical components](#charts-bases-and-orthonormal-physical-components)
    - [Embedded manifolds](#embedded-manifolds)
      - [Definition](#definition)
      - [Charts on embedded manifolds](#charts-on-embedded-manifolds)
      - [Tangent spaces and pushforwards](#tangent-spaces-and-pushforwards)
- [\\iota\_{\*p}(v)](#iota_pv) - [Induced metric](#induced-metric)
- [g_p(u, v)](#g_pu-v) -
  [Orthonormal frames on embedded manifolds](#orthonormal-frames-on-embedded-manifolds) -
  [Projection and tangent recovery](#projection-and-tangent-recovery) -
  [Cotangent spaces (pullback)](#cotangent-spaces-pullback) -
  [Affine structure and limitations](#affine-structure-and-limitations) -
  [Role of embeddings in this specification](#role-of-embeddings-in-this-specification)
  - [Cartesian products](#cartesian-products)
    - [Product manifolds](#product-manifolds)
    - [Tangent spaces of product manifolds](#tangent-spaces-of-product-manifolds)
    - [Cotangent spaces of product manifolds](#cotangent-spaces-of-product-manifolds)
- [\\langle \\alpha, v \\rangle](#langle-alpha-v-rangle) -
  [Product charts](#product-charts)
- [\\varphi](#varphi)
- [\\varphi(p_1,\\dots,p_k)](#varphip_1dotsp_k) -
  [Jacobians and transformation structure](#jacobians-and-transformation-structure)
- [\\frac{\\partial u}{\\partial q}(q)](#fracpartial-upartial-qq) -
  [Product metrics](#product-metrics)
- [g_p(v,w)](#g_pvw) -
  [Orthonormal frames on product manifolds](#orthonormal-frames-on-product-manifolds) -
  [Affine structure and differences of points](#affine-structure-and-differences-of-points) -
  [Role of Cartesian products in this specification](#role-of-cartesian-products-in-this-specification)
  - [Low-level concepts](#low-level-concepts)
    - [Charts](#charts)
      - [`AbstractFixedComponentsChart`](#abstractfixedcomponentschart)
        - [`AbstractFixedComponentsChart` is therefore the **preferred base class** wherever applicable](#abstractfixedcomponentschart-is-therefore-the-preferred-base-class-wherever-applicable)
      - [`cartesian_chart`](#cartesian_chart)
        - [Normative semantics](#normative-semantics)
        - [Implementation guidance (dispatch)](#implementation-guidance-dispatch)
        - [Recommended usage patterns](#recommended-usage-patterns)
          - [Do not use `cartesian_chart` as a proxy for metric selection; `metric_of` is the source of truth for geometry](#do-not-use-cartesian_chart-as-a-proxy-for-metric-selection-metric_of-is-the-source-of-truth-for-geometry)
    - [Cartesian products](#cartesian-products-1)
      - [Mathematical definition (normative)](#mathematical-definition-normative)
- [\\frac{\\partial u}{\\partial q}(q)](#fracpartial-upartial-qq-1) -
  [Product metrics and frames (normative)](#product-metrics-and-frames-normative) -
  [`AbstractCartesianProductChart`](#abstractcartesianproductchart)
- [\\mathrm{components}!\\left(\\prod_i R_i\\right)](#mathrmcomponentsleftprod_i-r_iright) -
  [Transform rules for products (normative)](#transform-rules-for-products-normative)
- [\\mathrm{cartesian_chart}!\\left(\\prod_i R_i\\right)](#mathrmcartesian_chartleftprod_i-r_iright) -
  [`CartesianProductChart`](#cartesianproductchart) -
  [`SpaceTimeCT` as a Cartesian product](#spacetimect-as-a-cartesian-product)
  - [Metrics](#metrics)
  - [Frames](#frames)
    - [Frame operations (normative)](#frame-operations-normative)
  - [Reference frames and operators](#reference-frames-and-operators)
    - [Mathematical model](#mathematical-model)
    - [ReferenceFrame API (conceptual)](#referenceframe-api-conceptual)
    - [Operator API (normative, dispatch-first)](#operator-api-normative-dispatch-first)
      - [Conceptual interface](#conceptual-interface)
      - [Dispatch surface (normative)](#dispatch-surface-normative)
      - [Role-aware semantics](#role-aware-semantics)
      - [Role-specialized primitive operators (normative)](#role-specialized-primitive-operators-normative)
        - [Domain rules (normative)](#domain-rules-normative)
        - [Boost](#boost)
        - [Worked example (normative semantics)](#worked-example-normative-semantics)
      - [Anchoring rules](#anchoring-rules)
      - [Chart discipline](#chart-discipline)
      - [Composition](#composition)
    - [Implementation guidance (normative)](#implementation-guidance-normative)
  - [Rotation operator: role-dependent mathematical action (normative)](#rotation-operator-role-dependent-mathematical-action-normative)
    - [Action on `Point` (normative)](#action-on-point-normative)
    - [Action on `PhysDisp` (physical displacement) (normative)](#action-on-physdisp-physical-displacement-normative)
    - [Action on `PhysVel` (physical velocity) (normative)](#action-on-physvel-physical-velocity-normative)
    - [Action on `PhysAcc` (physical acceleration) (normative)](#action-on-physacc-physical-acceleration-normative)
    - [Summary table (normative)](#summary-table-normative)
    - [Normative implications for implementation](#normative-implications-for-implementation)
  - [Embeddings](#embeddings)
  - [Roles and geometric meaning](#roles-and-geometric-meaning)
    - [Core roles (current)](#core-roles-current)
      - [Role class hierarchy (normative)](#role-class-hierarchy-normative)
    - [Additional roles (planned)](#additional-roles-planned)
    - [Role → transformation law (normative)](#role--transformation-law-normative)
    - [Coercions and anchoring (`as_disp`, `at=`)](#coercions-and-anchoring-as_disp-at)
  - [Core functional API](#core-functional-api)
    - [Point transforms](#point-transforms)
    - [Physical tangent transforms](#physical-tangent-transforms)
    - [Coordinate-basis tangent roles (normative)](#coordinate-basis-tangent-roles-normative)
    - [Coordinate-derivative transforms](#coordinate-derivative-transforms)
    - [Cotangent transforms](#cotangent-transforms)
    - [Physicalize / Coordinateize](#physicalize--coordinateize)
    - [Raise / Lower](#raise--lower)
  - [High-level API](#high-level-api)
    - [CsDict conventions](#csdict-conventions)
    - [Vector](#vector)
      - [Vector operator implementation via Quax](#vector-operator-implementation-via-quax)
      - [Subtraction and cross-chart operations (normative)](#subtraction-and-cross-chart-operations-normative)
      - [`Vector.from_` for `Quantity` inputs (normative)](#vectorfrom_-for-quantity-inputs-normative)
        - [Explicit role constructor for `Quantity`](#explicit-role-constructor-for-quantity)
      - [`Vector + Quantity` (normative convenience)](#vector--quantity-normative-convenience)
    - [PointedVector](#pointedvector)
  - [Multiple dispatch and JAX compatibility](#multiple-dispatch-and-jax-compatibility)
  - [Design invariants](#design-invariants)

---

## Package and API structure

Coordinax is organized into **low-level** mathematical primitives and
**high-level** user-facing containers. This split is intentional and normative:
low-level modules define mathematically precise kernels and registration points,
while high-level modules provide ergonomic objects that delegate all semantics
downward.

### High-level API (default user entry points)

These modules provide the objects most users interact with directly:

- `coordinax` (top-level re-exports)
  - `Vector`, `PointedVector`, `Coordinate`
  - `vconvert`, `as_disp`
- `coordinax.objects`
  - high-level container implementations and ergonomic constructors
- `coordinax.charts`
  - chart definitions for common coordinate systems to be used in `vconvert`.

Normative guidance:

- User documentation and examples should prefer the top-level re-exports.
- High-level objects must _not_ re-implement geometry; they must delegate to
  low-level functional APIs (`point_transform`, `physical_tangent_transform`,
  `apply_op`, etc.).

Example:

```
import coordinax.charts as cxc
from coordinax.objs import Vector

p = Vector.from_({"x": x, "y": y, "z": z})
v = p.vconvert(cxc.sph3d)
r = v["r"]
```

### Low-level API (normative mathematical kernels)

These modules define the mathematically precise building blocks of the system.

- `coordinax.charts` (import as `cxc`)
  - Chart definitions (`AbstractChart`, `AbstractFixedComponentsChart`,
    `AbstractCartesianProductChart`, `CartesianProductChart`, concrete charts)
  - Canonical chart utilities (`cartesian_chart`)
- `coordinax.roles` (import as `cxr`)
  - Role flags and role semantics (`Point`, `PhysDisp`, `PhysVel`, `PhysAcc`,
    etc.)
- `coordinax.metrics` (import as `cxm`)
  - Metrics (`AbstractMetric`, `EuclideanMetric`, `MinkowskyMetric`, etc.)
  - metric-dependent operations (`metric_of`, `raise_index`, `lower_index`)
- `coordinax.embeddings` (import as `cxe`)
  - Embedded manifolds (`EmbeddedManifold`)
  - maps (`embed_point`, `project_point`, `embed_tangent`, `project_tangent`)
- `coordinax.ops` (import as `cxo`)
  - Operator objects (`Operator`, `Pipe`, `Translate`, etc.)
  - application function `apply_op`
- `coordinax.frames` (import as `cxf`)
  - Reference frames (Alice/Bob-style frames)
  - Construction of reference-frame operators via `frame_transform_op`
  - Extension-point re-export of select frame libraries (e.g. `ICRS` and
    `Galactocentric` from `coordinax-astro`)

Normative guidance:

- All low-level functions must be:
  - pure,
  - side-effect free,
  - extended exclusively via multiple dispatch,
  - dispatching only on **static Python objects** (charts, roles, operators).
- Numerical kernels must be JAX-traceable and written in terms of scalar
  component operations; batching is achieved via jax function (e.g. `jax.vmap` /
  `jax.lax.scan`, `jax.numpy.vectorize`).

### Naming conventions and terminology

- The term **frame** is used in two distinct but non-overlapping senses:
  1. **Physical frames**: orthonormal tangent frames defined by
     `frame_cart(chart, at=...)`.
  2. **Reference frames**: observer-dependent frames (`ICRS`, `Galactocentric`,
     `Alice`/`Bob`) that induce time-dependent operators.
- The module split above exists specifically to keep these concepts separate and
  prevent semantic confusion.

### Dependency direction (normative)

Dependencies must flow strictly in the following direction:

```
charts / roles / metrics / embeddings
            ↓
        functional APIs
            ↓
      operators / frames
            ↓
   Vector / PointedVector / Coordinate
```

- Low-level modules must not import high-level containers.
- High-level containers may import low-level modules.
- Reference-frame logic must be expressed _only_ via operators and `apply_op`.

This structure ensures mathematical correctness, testability, and long-term
maintainability of the coordinax codebase.

## Mathematical Background and Conventions

This section establishes the mathematical foundations and conventions assumed
throughout the rest of this specification. It is normative: all subsequent
design choices, APIs, and transformation rules are defined with respect to the
concepts introduced here.

### Affine spaces vs. vector spaces

We distinguish carefully between **points** and **vectors**.

Let $M$ be a smooth manifold.

- A **point** $p \in M$ represents a location.
- A **vector** does not exist “by itself”; it is an element of a vector space
  attached to a point.

In particular:

- The set of points on $M$ is modeled as an **affine space**, not a vector
  space.
- Differences of points are vectors, but points do not add.

For any $p \in M$, the **tangent space** $T_pM$ is a real vector space.

### Tangent and cotangent spaces

For each point $p \in M$:

- The **tangent space** $T_pM$ is the vector space of derivations at $p$.
- The **cotangent space** $T_p^*M$ is its dual.

In coordinates $(q^1,\dots,q^n)$:

$$
v = v^i \frac{\partial}{\partial q^i}, \qquad
\alpha = \alpha_i\,dq^i.
$$

Tangent vectors and cotangent vectors transform differently (pushforward vs
pullback). Coordinax supports tangent objects now and specifies the cotangent
API here for planned implementation.

### Charts, bases, and orthonormal physical components

A **chart** is a local diffeomorphism

$$
\varphi : U \subset M \to \mathbb{R}^n,
$$

assigning coordinates $q=(q^1,\dots,q^n)$ to points $p\in U$.

Coordinate bases are generally **not orthonormal**. A **metric** $g$ defines

$$
g_{ij}(q) = g\!\left(\frac{\partial}{\partial q^i}, \frac{\partial}{\partial q^j}\right).
$$

From $g$, one may construct an **orthonormal frame** $\{\hat e_a(q)\}$
satisfying $g(\hat e_a,\hat e_b)=\delta_{ab}$.

Coordinax distinguishes:

- **physical tangent components**: e.g. the speed $v_\phi$ [km/s] in a spherical
  chart, with uniform physical units; expressed in an orthonormal frame, vs.
- **coordinate derivatives**: $dq^i/dt$ with heterogeneous units; not physical
  components.

This distinction is central to the role system and conversion functions.

### Embedded manifolds

Many coordinate systems of interest are not intrinsically Euclidean but are
naturally realized as **embedded submanifolds** of a higher-dimensional
Euclidean space. Examples include spheres, tori, constrained surfaces, and
curvilinear coordinate systems defined via smooth embeddings.

This section fixes the mathematical framework for embedded manifolds and the
induced geometric structures used throughout this specification.

#### Definition

An **embedded manifold** is a smooth manifold $M$ together with a smooth
embedding

$$
\iota : M \hookrightarrow \mathbb{R}^N,
$$

where $\iota$ is:

- smooth,
- injective,
- an immersion (its differential has full rank everywhere),
- a homeomorphism onto its image.

The ambient space $\mathbb{R}^N$ is equipped with a fixed canonical Euclidean
metric.

The embedding $\iota$ realizes $M$ as a smooth $n$-dimensional submanifold of
$\mathbb{R}^N$, where $n = \dim M \le N$.

#### Charts on embedded manifolds

An embedded manifold still admits **intrinsic charts**:

$$
\varphi : U \subset M \to \mathbb{R}^n,
$$

which parametrize points on $M$. The embedding induces a map from chart
coordinates to ambient coordinates:

$$
x(q) = \iota\!\bigl(\varphi^{-1}(q)\bigr) \in \mathbb{R}^N.
$$

In Coordinax, an embedded chart is represented by:

- an **intrinsic chart** on $M$,
- an **ambient chart** on $\mathbb{R}^N$ (typically Cartesian),
- and explicit maps between them.

#### Tangent spaces and pushforwards

At a point $p \in M$, the tangent space $T_pM$ is an $n$-dimensional vector
space. The embedding induces a **pushforward** (differential)

$$
\iota_{*p} : T_pM \to T_{\iota(p)}\mathbb{R}^N \cong \mathbb{R}^N,
$$

which is injective.

In coordinates, if $q \in \mathbb{R}^n$ are intrinsic coordinates and
$x(q) \in \mathbb{R}^N$ are ambient coordinates, the pushforward is given by the
Jacobian:

$$
\iota_{*p}(v)
=
J(q)\,v,
\qquad
J(q) = \frac{\partial x}{\partial q}(q),
$$

where $J(q) \in \mathbb{R}^{N \times n}$ has full column rank.

Thus:

- intrinsic tangent vectors are mapped to ambient vectors tangent to the
  embedded submanifold,
- ambient vectors not lying in the image of $J(q)$ are not tangent to $M$.

#### Induced metric

The embedding induces a **pullback metric** on $M$ from the ambient Euclidean
metric:

$$
g_p(u, v)
=
\langle \iota_{*p}(u), \iota_{*p}(v) \rangle_{\mathbb{R}^N}.
$$

In coordinates, the metric matrix is

$$
g(q) = J(q)^{\mathsf T} J(q),
$$

a symmetric positive-definite $n \times n$ matrix.

This metric is the **canonical geometry** of the embedded manifold unless
explicitly overridden.

#### Orthonormal frames on embedded manifolds

Physical tangent components are expressed in **orthonormal frames**.

For an embedded manifold, an orthonormal frame at $p$ may be constructed by:

1. computing $J(q) = \partial x / \partial q$,
2. orthonormalizing its columns (e.g. via QR or SVD),
3. yielding a matrix

$$
B(p) \in \mathbb{R}^{N \times n},
$$

whose columns form an orthonormal basis of the tangent space embedded in the
ambient space.

Normatively:

- $B(p)$ has orthonormal columns,
- $B(p)^{\mathsf T} B(p) = I_n$,
- $B(p) B(p)^{\mathsf T}$ is the orthogonal projector onto $T_{\iota(p)}M$.

This matrix is the return value of `frame_cart` for embedded charts.

#### Projection and tangent recovery

Given an ambient vector $w \in \mathbb{R}^N$, its projection onto the tangent
space is:

$$
\Pi_p(w) = B(p) B(p)^{\mathsf T} w.
$$

To recover intrinsic tangent components from ambient components, one applies:

$$
v = B(p)^{\mathsf T} w.
$$

This defines the inverse of the pushforward **on tangent vectors only**.

#### Cotangent spaces (pullback)

Cotangent vectors $\alpha \in T_p^*M$ correspond to ambient covectors whose
restriction to tangent directions agrees with $\alpha$.

Under the embedding, the pullback of an ambient covector $\tilde\alpha$ is:

$$
\alpha(v) = \tilde\alpha(\iota_{*p} v).
$$

In coordinates, cotangent transforms involve the transpose Jacobian:

$$
\alpha_i = J(q)_{i a}\, \tilde\alpha_a.
$$

Explicit cotangent support is planned but not yet fully implemented.

#### Affine structure and limitations

An embedded manifold generally **does not inherit an affine structure**, even if
the ambient space is affine.

Consequences:

- Points on $M$ cannot be added or subtracted intrinsically.
- Displacements (`PhysDisp`) must live in tangent spaces and require anchoring.
- Operations like `Point + PhysDisp` are only defined via exponential-map-like
  constructions, which are **not assumed** unless explicitly specified.

Therefore, embedded manifolds impose stricter requirements on arithmetic and
require explicit base points.

#### Role of embeddings in this specification

Embedded manifolds are used to:

- define curved coordinate systems with canonical geometry,
- induce metrics and frames from ambient Euclidean space,
- support physical tangent vectors with uniform units,
- cleanly separate intrinsic geometry from ambient representation.

All embedding-related APIs (`embed_point`, `project_point`, `embed_tangent`,
`project_tangent`, `frame_cart`) must implement the mathematical laws fixed in
this section.

No high-level object may bypass these rules.

### Cartesian products

Many geometric configuration spaces arise as **Cartesian products of
manifolds**. This section fixes the mathematical meaning of product manifolds
and the induced structures on tangent and cotangent spaces. These conventions
are normative for all later sections of this specification.

#### Product manifolds

Let $(M_i)_{i=1}^k$ be smooth manifolds with dimensions $\dim M_i = n_i$. Their
**Cartesian product** is the manifold

$$
M \;=\; \prod_{i=1}^k M_i
\;=\;
\{(p_1, \dots, p_k) \mid p_i \in M_i\}.
$$

$M$ is a smooth manifold of dimension

$$
\dim M = \sum_{i=1}^k n_i.
$$

Each projection map $\pi_i : M \to M_i$, defined by $\pi_i(p_1,\dots,p_k)=p_i$,
is smooth.

Cartesian products model situations in which degrees of freedom are
**geometrically independent**, such as:

- time × space,
- configuration × momentum,
- sky position × distance,
- repeated copies of the same manifold.

#### Tangent spaces of product manifolds

The tangent bundle of a Cartesian product splits canonically as a direct sum:

$$
T_{(p_1,\dots,p_k)} M
\;\cong\;
\bigoplus_{i=1}^k T_{p_i} M_i.
$$

Thus any tangent vector $v \in T_p M$ admits a unique decomposition

$$
v = (v_1, \dots, v_k),
\qquad
v_i \in T_{p_i} M_i.
$$

This decomposition is **canonical**: it does not depend on coordinates, metrics,
or embeddings.

#### Cotangent spaces of product manifolds

Dually, the cotangent space decomposes as

$$
T^*_{(p_1,\dots,p_k)} M
\;\cong\;
\bigoplus_{i=1}^k T^*_{p_i} M_i.
$$

Covectors decompose factorwise, and pairings with tangent vectors satisfy

$$
\langle \alpha, v \rangle
=
\sum_{i=1}^k \langle \alpha_i, v_i \rangle,
\qquad
\alpha_i \in T^*_{p_i} M_i.
$$

#### Product charts

If each factor $M_i$ admits a chart
$\varphi_i : U_i \subset M_i \to \mathbb{R}^{n_i}$, the **product chart**

$$
\varphi
=
\prod_{i=1}^k \varphi_i
:
\prod_i U_i
\to
\mathbb{R}^{n_1+\cdots+n_k}
$$

is defined by concatenation:

$$
\varphi(p_1,\dots,p_k)
=
\bigl(
\varphi_1(p_1),
\varphi_2(p_2),
\dots,
\varphi_k(p_k)
\bigr).
$$

Coordinate indices are grouped by factor; there is no canonical interleaving.

#### Jacobians and transformation structure

If each factor undergoes a coordinate transformation $u_i = f_i(q_i)$, then the
induced product transformation

$$
u = f(q) = (f_1(q_1), \dots, f_k(q_k))
$$

has a **block-diagonal Jacobian**:

$$
\frac{\partial u}{\partial q}(q)
=
\begin{pmatrix}
\frac{\partial f_1}{\partial q_1} & 0 & \cdots & 0 \\
0 & \frac{\partial f_2}{\partial q_2} & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \frac{\partial f_k}{\partial q_k}
\end{pmatrix}.
$$

Consequently:

- point transformations act factorwise,
- pushforwards on tangent vectors are block diagonal,
- pullbacks on cotangent vectors are block diagonal.

No coupling between factors occurs unless explicitly introduced by additional
structure.

#### Product metrics

If each factor $(M_i, g_i)$ is equipped with a (possibly pseudo-Riemannian)
metric, the **canonical product metric** on $M$ is

$$
g = g_1 \oplus \cdots \oplus g_k,
$$

defined by

$$
g_p(v,w)
=
\sum_{i=1}^k g_i(v_i, w_i),
\qquad
v=(v_1,\dots,v_k),\; w=(w_1,\dots,w_k).
$$

In coordinates, the metric matrix is block diagonal with factor metric blocks.

Unless explicitly specified otherwise, Cartesian products are assumed to carry
this product metric.

#### Orthonormal frames on product manifolds

If each factor admits an orthonormal frame $\{ \hat e^{(i)}_a \}$ on $T M_i$,
then the product manifold admits the canonical orthonormal frame obtained by
concatenation of the factor frames.

Equivalently, frame matrices on product manifolds are block diagonal.

This implies that **physical tangent components decompose factorwise** and that
no mixing occurs between independent factors at the level of orthonormal
components.

#### Affine structure and differences of points

If some factors are affine manifolds (e.g. Euclidean space, time), their product
inherits a partial affine structure.

In particular:

- points may be translated independently in affine factors,
- differences of points decompose factorwise into tangent vectors,
- tangent vectors inherit independent geometric meaning per factor.

This distinction is essential for later sections on **roles**, especially the
difference between `Point` and `PhysDisp`.

#### Role of Cartesian products in this specification

Cartesian products provide the mathematical foundation for:

- space–time constructions,
- phase spaces and repeated factors,
- clean separation of transformation logic,
- block-structured Jacobians, metrics, and frames.

All later APIs describing product charts, metrics, frames, embeddings, and
operators must reduce to the identities fixed in this section.

---

## Low-level concepts

Low-level concepts are explicit and mathematically precise. The high-level API
builds on these while hiding them in common workflows.

### Charts

A **Chart** is the structural descriptor of coordinates used to represent
**points**. It defines:

- component names and order,
- per-component physical dimensions,
- optional parameters that affect semantics (e.g. an embedding radius).

Charts are **stateless**, **immutable**, and used for dispatch.

Conceptual interface:

```
class AbstractChart:
    components: tuple[str, ...]
    ndim: int
    coord_dimensions: tuple[str, ...]  # e.g. length/angle/etc
```

---

#### `AbstractFixedComponentsChart`

Many charts have **statically known** component schemas (names and dimensions)
at the class level, e.g. `Cart3D`, `Cylindrical3D`, `Spherical3D`,
`LonLatSpherical3D`. For these, Coordinax provides:

- `AbstractFixedComponentsChart`: a subclass of `AbstractChart` intended for
  charts whose `components` and `coord_dimensions` are fixed by the chart class
  itself.

Normative intent:

- If a chart’s component schema is known at definition time (class-level
  constants), it **should** inherit from `AbstractFixedComponentsChart`.
- If a chart’s component schema depends on runtime parameters (e.g. wraps
  another chart, varies component set, depends on configuration), it **should
  not** inherit from `AbstractFixedComponentsChart` and should instead implement
  `components`/`coord_dimensions` dynamically (e.g. product charts like
  `SpaceTimeCT`).

Conceptual interface:

```
class AbstractFixedComponentsChart(AbstractChart):
    # Implemented via class-level constants / typing parameters.
    # Instances are still stateless and immutable.
    components: tuple[str, ...]         # fixed for the class
    coord_dimensions: tuple[str, ...]   # fixed for the class
```

This class exists primarily to:

- make chart definitions concise and less error-prone,
- enable stronger static typing (fixed keys / dimensions),
- support faster, simpler validation of `CsDict` keys and per-component
  dimensions.

##### `AbstractFixedComponentsChart` is therefore the **preferred base class** wherever applicable

Charts provide (structurally) a canonical Cartesian chart of the same ambient
space when applicable:

```
cartesian_chart(chart: AbstractChart) -> AbstractChart
```

---

#### `cartesian_chart`

`cartesian_chart(chart)` returns a **canonical ambient Euclidean chart** used as
a reference coordinate system for:

- implementing some `point_transform` rules via an intermediate canonical chart,
- expressing physical frames via `frame_cart` (which is defined _in Cartesian
  ambient components_),
- defining the ambient chart for `EmbeddedChart` implementations,
- interoperability and “default” Euclidean exports.

This function is **not** replaced by `metric_of`:

- `metric_of(chart)` specifies the inner product on tangent spaces (geometry),
- `cartesian_chart(chart)` specifies a canonical coordinate choice for an
  ambient Euclidean space (representation).

##### Normative semantics

- If `chart` is a chart on Euclidean $\mathbb{R}^n$ (even if non-Cartesian, e.g.
  spherical), then `cartesian_chart(chart)` **must** return the canonical
  Cartesian chart on the _same_ ambient space (e.g. `cart3d` for 3D Euclidean
  charts).
- If `chart` is a wrapper/product chart whose factors include Euclidean
  components, `cartesian_chart(chart)` **should** act **factor-wise**, e.g.:
  - `SpaceTimeEuclidean(spatial_kind)` maps to
    `SpaceTimeEuclidean(cartesian_chart(spatial_kind))`,
  - a generic product chart maps to the product of the factors’
    `cartesian_chart` results.
- If `chart` is an `EmbeddedChart`, then `cartesian_chart(chart)` **must**
  return the canonical Cartesian chart of its ambient space, i.e.
  `cartesian_chart(chart.ambient_kind)`. (The embedding’s ambient is the
  relevant Euclidean reference.)
- If `chart` does **not** admit a canonical Euclidean ambient chart (e.g. a
  chart whose ambient is not Euclidean, or for which no canonical Euclidean
  reference is defined), then `cartesian_chart(chart)` **must** raise
  `NotImplementedError`.

##### Implementation guidance (dispatch)

`cartesian_chart` is extended by multiple dispatch on the chart type and must be
a **pure**, **static** function (no dependence on runtime array values):

```
@plum.dispatch
def cartesian_chart(chart: SomeChartType) -> AbstractChart: ...
```

For most Euclidean chart families, a small number of registrations should
suffice:

- the Cartesian chart returns itself,
- non-Cartesian Euclidean charts return the corresponding canonical Cartesian
  chart of the same dimension,
- wrapper/product charts defer to their factors,
- embedded charts defer to their `ambient_kind`.

##### Recommended usage patterns

- `frame_cart(chart, at=...)` remains the primary mechanism for physical tangent
  transforms; `cartesian_chart` provides the canonical ambient coordinate choice
  that makes “Cartesian components” unambiguous.
- Implementations of `point_transform` may (optionally) use `cartesian_chart` to
  reduce the number of direct conversion rules by supporting:
  - `from -> cartesian -> to` where both legs are implemented and numerically
    stable.

###### Do not use `cartesian_chart` as a proxy for metric selection; `metric_of` is the source of truth for geometry

### Cartesian products

Many coordinate systems used in physics and astronomy are naturally expressed as
**products** of independent manifolds or coordinate factors (e.g. time × space,
sky × distance, configuration × parameter). Coordinax models such systems using
**Cartesian product charts**, which provide a principled way to compose charts,
metrics, frames, and transforms while minimizing special-casing.

#### Mathematical definition (normative)

Let $(M_i)_{i=1}^k$ be smooth manifolds with $\dim M_i = n_i$. Their Cartesian
product is the manifold

$$
M \;=\; \prod_{i=1}^k M_i,
$$

with points $p = (p_1,\dots,p_k)$ where $p_i \in M_i$ and total dimension
$\dim M = \sum_i n_i$.

The tangent bundle of a Cartesian product splits canonically:

$$
T_pM \;\cong\; \bigoplus_{i=1}^k T_{p_i} M_i.
$$

If each factor has a chart $\varphi_i : U_i \subset M_i \to \mathbb{R}^{n_i}$,
the product chart

$$
\varphi \;=\; \prod_{i=1}^k \varphi_i : \prod_i U_i \to \mathbb{R}^{n_1+\cdots+n_k}
$$

is defined by concatenation:

$$
\varphi(p_1,\dots,p_k) = \bigl(\varphi_1(p_1),\dots,\varphi_k(p_k)\bigr).
$$

If $u_i = f_i(q_i)$ are factorwise coordinate transitions, then the product
transition $u=f(q)$ has a block-diagonal Jacobian:

$$
\frac{\partial u}{\partial q}(q)
=
\begin{pmatrix}
\frac{\partial u_1}{\partial q_1}(q_1) & 0 & \cdots & 0 \\
0 & \frac{\partial u_2}{\partial q_2}(q_2) & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \frac{\partial u_k}{\partial q_k}(q_k)
\end{pmatrix}.
$$

This block structure is the key reason Cartesian products are useful: many
transform laws become “factorwise” and do not require bespoke wrapper logic.

#### Product metrics and frames (normative)

If each factor $(M_i,g_i)$ is equipped with a metric $g_i$, the canonical
product metric is the direct sum

$$
g \;=\; g_1 \oplus \cdots \oplus g_k,
$$

meaning that for tangent vectors $v=(v_1,\dots,v_k)$ and $w=(w_1,\dots,w_k)$:

$$
g_p(v,w) \;=\; \sum_{i=1}^k (g_i)_{p_i}(v_i,w_i).
$$

In coordinates, the product metric matrix is block diagonal with blocks
$g^{(i)}(q_i)$.

For physical orthonormal frames, if each factor provides a frame matrix
$B_i(p_i)$ in a canonical ambient orthonormal basis, the product frame is the
block diagonal matrix

$$
B(p) \;=\; \mathrm{diag}\bigl(B_1(p_1),\dots,B_k(p_k)\bigr),
$$

with shape $(N\times n)$ where $n=\sum_i n_i$ and $N=\sum_i N_i$ are intrinsic
and ambient orthonormal dimensions, respectively.

#### `AbstractCartesianProductChart`

```python
# A product chart may use *namespaced* component keys.
# Ordinary charts use `str` component keys.
# ComponentsKey is an internal type alias, not part of the public API.
ComponentsKey = str | tuple[str, str]  # ("factor_name", "component_name")


class AbstractCartesianProductChart(AbstractChart):
    """A chart on a Cartesian product manifold with named factors.

    Product charts namespace their component keys as pairs
    `(factor_name, component_name)` to avoid collisions (e.g. phase space).
    """

    factors: tuple[AbstractChart, ...]  # static dispatch keys
    factor_names: tuple[str, ...]  # names aligned with `factors`

    @property
    def ndim(self) -> int: ...
    @property
    def components(self) -> tuple[ComponentsKey, ...]: ...
```

Normative requirements:

- `factor_names` **must** always be provided and have the same length as
  `factors`.
- `components` **must** be **namespaced** keys:
  $$
  \mathrm{components}\!\left(\prod_i R_i\right)
  =
  \bigl((n_i, c)\ \mid\ c \in \mathrm{components}(R_i)\bigr)_{i=1}^k,
  $$
  where `n_i` is the `i`-th factor name and `c` is a component key of factor
  `R_i`. Concretely, each product component key is the 2-tuple
  `(factor_name, component_name)`.
- Namespacing is the **canonical** collision-avoidance mechanism and enables
  repeated factors (e.g. phase space).
- `coord_dimensions` must be the concatenation of factor `coord_dimensions` in
  factor order.

```python
# Example: 6D phase space as a product of two identical Cart3D factors.
# Using tuple keys avoids collisions by construction.
phase6d = CartesianProductChart(
    factors=(cart3d, cart3d),
    factor_names=("q", "p"),
)
assert phase6d.components == (
    ("q", "x"),
    ("q", "y"),
    ("q", "z"),
    ("p", "x"),
    ("p", "y"),
    ("p", "z"),
)
```

#### Transform rules for products (normative)

Product charts must satisfy factorwise transform laws.

**Point transforms:**

Let $R=\prod_i R_i$ and $S=\prod_i S_i$ be product charts on the same product
manifold, and let `p` be a `CsDict` whose keys match `R.components`. Partition
`p` into factor dictionaries `p_i` matching each `R_i`. Then:

$$
\mathrm{point\_transform}(S,R,p)
\;\equiv\;
\bigl(\mathrm{point\_transform}(S_i,R_i,p_i)\bigr)_{i=1}^k,
$$

and the output `CsDict` is the concatenation of factor outputs in `S` order.

Partitioning and merging are **structural** operations on keys:

- If `components` are namespaced pairs `(name, c)`, then the factor dictionary
  for factor `i` uses the factor’s native keys `c` (strings) and is obtained by
  selecting keys with prefix `name` and stripping the prefix.
- The merged product dictionary is formed by re-attaching the prefix `name` to
  each factor key.

These key-level operations are purely syntactic; they do not alter values or
units.

**Cartesian chart:**

If each factor admits a `cartesian_chart`, then:

$$
\mathrm{cartesian\_chart}\!\left(\prod_i R_i\right)
=
\prod_i \mathrm{cartesian\_chart}(R_i).
$$

If any factor does not admit a Cartesian chart, `cartesian_chart(product)` must
raise `NotImplementedError`.

**Metrics and frames:**

- `metric_of(product)` should default to the product metric
  $\bigoplus_i \mathrm{metric\_of}(R_i)$ unless overridden.
- `frame_cart(product, at=...)` should be constructed as a block diagonal of
  factor `frame_cart` matrices, with `at` partitioned factorwise.

These defaults can be overridden by registering a non-product metric or frame
for special coupled geometries, but the product laws are the normative default.

#### `CartesianProductChart`

`CartesianProductChart` is the canonical concrete implementation of
`AbstractCartesianProductChart`.

Conceptual construction:

```
CartesianProductChart(
    factors: tuple[AbstractChart, ...],
    factor_names: tuple[str, ...],
)
```

Guidance:

- `CartesianProductChart` is typically **not** an `AbstractFixedComponentsChart`
  because its `components` depend on the chosen factor instances. However, if a
  project defines a product with statically known factor chart classes (and thus
  statically known component schema), it may introduce a named fixed-components
  subclass.

#### `SpaceTimeCT` as a Cartesian product

`SpaceTimeCT` (a time × space chart wrapper) must be implemented as a Cartesian
product chart:

$$
\mathrm{SpaceTimeCT}(S) \;\equiv\; T \times S,
$$

where $T$ is the canonical 1D time chart `time1d` and $S$ is a spatial chart
(e.g. `cyl3d`, `sph3d`, `loncoslatsph3d`).

Normative code-level requirement:

- `SpaceTimeCT` must subclass `AbstractCartesianProductChart` and expose its
  factors as `(time1d, spatial_chart)`, where `time1d` is the canonical 1D time
  chart.

Conceptual interface:

```python
class SpaceTimeCT(AbstractCartesianProductChart):
    """Time × space product chart with a canonical time factor.

    `SpaceTimeCT(spatial_chart)` always uses the canonical 1D time chart
    `time1d` as its first factor.
    """

    spatial_chart: AbstractChart  # any spatial chart
    factors = (time1d, spatial_chart)
```

Normative behavior:

- `SpaceTimeCT.ndim == time1d.ndim + spatial_chart.ndim`.
- `SpaceTimeCT.components == time1d.components + spatial_chart.components`.
- `point_transform`, `cartesian_chart`, `metric_of`, and `frame_cart` for
  `SpaceTimeCT` must follow the product rules above (factorwise; block
  structure), eliminating bespoke wrapper logic.

**Component keys (special-case ergonomics).** `SpaceTimeCT` is a specialized
product chart whose public `components` are **flat** string keys:

- `SpaceTimeCT.components == ("ct",) + spatial_chart.components`

It does **not** expose namespaced pair keys like `("time","ct")`. This is a
deliberate ergonomic exception because time and space components are naturally
distinct and collision-free in the intended use cases. `SpaceTimeEuclidean`
follows the same flat-key convention.

### Metrics

A **Metric** provides an inner product on tangent spaces.

Conceptual interface:

```
class AbstractMetric:
    def metric_matrix(self, chart: AbstractChart, at: CsDict) -> Array:
        \"\"\"Return g_ij(q) in the coordinate basis of `chart` at `at`.\"\"\"
```

`metric_of(chart)` returns the metric associated with a chart (static dispatch):

```
metric_of(chart: AbstractChart) -> AbstractMetric
```

Normative requirements:

- symmetric matrix of shape `(..., n, n)`,
- pure, traceable by JAX,
- may depend on `at` and chart parameters.

### Frames

In this document, **Frame** refers to a _physical orthonormal frame_ attached to
a chart, used to interpret **physical tangent components** at a base point. For
a chart $R$, define a matrix $B_R(p)$ whose columns are an orthonormal basis of
$T_pM$ expressed in a canonical ambient orthonormal basis (typically Cartesian
components).

Normatively, `frame_cart(chart, at=...)` returns $B_R(p)$.

```
frame_cart(chart: AbstractChart, *, at: CsDict) -> Array
# shape: (..., N, n)
```

- $n$ is the intrinsic dimension of the chart.
- $N$ is the ambient orthonormal dimension (usually equals $n$ for Euclidean
  charts; may be larger for embedded charts).
- Columns of $B_R$ are orthonormal with respect to the ambient metric.

For Euclidean charts where $N=n$, $B_R \in \mathbb{R}^{n\times n}$ and is
orthogonal.

#### Frame operations (normative)

Given a chart $R$ on a manifold $M$ and a base point $p\in M$, let

$$
B_R(p) = \mathrm{frame\_cart}(R,\;at=p)
\in \mathbb{R}^{N\times n}
$$

be the **orthonormal tangent frame matrix** whose columns are an orthonormal
basis of $T_pM$ expressed in a canonical ambient orthonormal basis (typically
Cartesian components). Here $n=\dim M$ and $N$ is the ambient orthonormal
dimension.

Coordinax defines two fundamental linear maps associated with $B_R(p)$:

- **pushforward to ambient Cartesian components**

  $$
  \mathrm{pushforward}\bigl(B_R(p), v_R\bigr)
  \;:=\;
  B_R(p)\,v_R
  \;\in\;\mathbb{R}^{N},
  $$

  where $v_R\in\mathbb{R}^n$ are **physical tangent components** in the
  chart-orthonormal frame at $p$.

- **pullback to chart physical components**
  $$
  \mathrm{pullback}\bigl(g_R,\; B_R(p), v_{\mathrm{cart}}\bigr)
  \;:=\;
  B_R(p)^{\mathsf T}\,v_{\mathrm{cart}}
  \;\in\;\mathbb{R}^{n},
  $$
  where $v_{\mathrm{cart}}\in\mathbb{R}^N$ are ambient orthonormal components
  and $g_R=\mathrm{metric\_of}(R)$.

For Euclidean ambient metrics, $B_R(p)^{\mathsf T}$ is the left-inverse of
$B_R(p)$ because columns of $B_R(p)$ are orthonormal, i.e.
$B_R(p)^{\mathsf
T}B_R(p)=I_n$.

For non-Euclidean ambient metrics (e.g. Minkowski), `pullback` is defined to
include the metric signature so that the returned components remain physically
consistent with the metric; see the metric-specialized `pullback` dispatches.

**Normative use in role transforms.** For physical tangent roles (`PhysDisp`,
`PhysVel`, `PhysAcc`) the canonical way to move between chart components and
ambient Cartesian components is:

$$
v_{\mathrm{cart}} = \mathrm{pushforward}\bigl(B_R(p), v_R\bigr),
\qquad
v_R = \mathrm{pullback}\bigl(g_R, B_R(p), v_{\mathrm{cart}}\bigr).
$$

This is the conceptual basis of `physical_tangent_transform`. Implementations
may either call `physical_tangent_transform` directly or implement the same
logic explicitly via `frame_cart` + `pushforward`/`pullback`, depending on what
is most maintainable in a given operator or conversion rule.

---

### Reference frames and operators

Coordinax also supports **reference frames** in the sense of astronomy/physics
(e.g. ICRS, Galactocentric, “Alice’s frame”, “Bob’s frame”). These are **not**
the same as the physical frames above.

A _reference frame_ is a choice of observer/coordinate system that induces a
(possibly time-dependent) transformation of points and derived objects. In
Coordinax, reference-frame transformations are expressed via **operators** that
act on low-level objects (`CsDict`), intermediate objects (`Vector`,
`PointedVector`), and high-level objects (`Coordinate`).

#### Mathematical model

Let $M$ be the configuration manifold (typically Euclidean space $\mathbb{R}^n$
with a chosen chart family) and let $\tau$ be an affine parameter, usually time.
A reference frame transformation from frame $F$ to frame $G$ is modeled as a
**time-dependent diffeomorphism** (or smooth map) on $M$:

$$
\Phi_{F\to G}(\tau): M \to M.
$$

This map induces:

- a **point transform** on coordinates of $p\in M$:
  $p' = \Phi_{F\to G}(\tau)(p)$,
- a **pushforward** on tangent vectors at $p$:
  $v' = (\Phi_{F\to G}(\tau))_{*p}(v)$,
- and dually a **pullback** on covectors (planned):
  $\alpha' = (\Phi_{F\to G}(\tau))^{*}_p(\alpha)$.

When $M$ is Euclidean and frames are related by a rigid motion, one common form
is:

$$
\Phi_{F\to G}(\tau)(x) = R(\tau)\,x + b(\tau),
$$

with $R(\tau)\in SO(n)$ and translation $b(\tau)\in\mathbb{R}^n$. The
pushforward on physical tangents depends on the intended geometric object:

- For **physical displacement** (`PhysDisp`): $\Delta x' = R(\tau)\,\Delta x$.
- For **physical velocity** (`PhysVel`):
  $v' = R(\tau)\,v + \dot R(\tau)\,x + \dot b(\tau)$ (Galilean kinematics; the
  extra terms encode the time-dependence of the frame).
- For **physical acceleration** (`PhysAcc`): similarly includes
  $\ddot R(\tau)\,x$, $2\dot R(\tau)\,v$, and $\ddot b(\tau)$.

These additional terms are exactly why reference frames are a distinct concept:
they act on _anchored_ objects and can couple points and tangents through
explicit $\tau$-dependence.

The spec does not require that every reference frame be rigid; the abstract
interface allows general smooth maps.

#### ReferenceFrame API (conceptual)

A reference frame object is a **static dispatch key** (like charts/roles). It
does not carry numerical state, but it may carry static parameters (e.g. an
origin frame choice). It supports construction of an operator to another frame:

```
class AbstractReferenceFrame:
    ...
```

The primitive constructor is:

```
frame_transform_op(
    from_frame: AbstractReferenceFrame,
    to_frame: AbstractReferenceFrame,
    *,
    chart: AbstractChart | None = None,
    metric: AbstractMetric | None = None,
) -> Operator
```

- `chart` fixes the chart family in which the operator’s low-level kernels act
  (typically `cartesian_chart(...)` of the relevant space).
- `metric` selects geometry when needed for tangent/cotangent interpretation;
  default is `metric_of(chart)` where applicable.

#### Operator API (normative, dispatch-first)

An `Operator` represents a **time-parameterized geometric action** induced by a
reference-frame transformation. Conceptually, it is a _family of maps_
parameterized by an affine parameter `tau` (usually time).

In Coordinax, an `Operator` is intentionally **thin**: it carries no numerical
state beyond static metadata and delegates all semantics to _multiple-dispatch
kernels_. This design ensures:

- correctness via explicit role- and type-based laws,
- extensibility by registration,
- full compatibility with `jax.jit` (dispatch resolves statically).

##### Conceptual interface

```
class Operator:

    def __call__(self, tau, x, /):
        ...
```

However, **`Operator.__call__` is not the semantic entry point**. Normatively,
`__call__` must delegate to a pure functional API:

```
apply_op(op: Operator, tau, x, /, *, at=None)
```

All behavior is defined by registrations of `apply_op`.

##### Dispatch surface (normative)

`apply_op` must be extended by multiple dispatch on:

- `op` (or equivalently `(op.from_frame, op.to_frame)`),
- the _structural type_ of `x` (`CsDict`, `Vector`, `PointedVector`,
  `Coordinate`),
- and, where relevant, the `role` carried by `x`.

Canonical signatures:

```
apply_op(op: Operator, tau, p: CsDict, /) -> CsDict
apply_op(op: Operator, tau, v: CsDict, /, *, at: CsDict) -> CsDict

apply_op(op: Operator, tau, v: Vector, /, *, at: Vector | None = None) -> Vector
apply_op(op: Operator, tau, fp: PointedVector, /) -> PointedVector
apply_op(op: Operator, tau, coord: Coordinate, /) -> Coordinate
```

**Normative note on `CsDict` (raw dictionary) inputs:** When applying operators
to raw `CsDict` objects (low-level component dictionaries), the caller MUST
provide explicit `role=` and `chart=` keyword arguments to disambiguate the
geometric type. This includes `Boost` applied to `Point`-role dictionaries,
which is allowed and defined per the Boost specification above.

`Operator.__call__` is required only to normalize arguments and forward to
`apply_op`.

##### Role-aware semantics

The role of the input determines _which induced geometric map is applied_:

- `Point`:
  - `apply_op(op, tau, p)` applies the **point map**
    $p' = \Phi_{F\to G}(\tau)(p)$.
- Physical tangent roles (`PhysDisp`, `PhysVel`, `PhysAcc`):
  - `apply_op(op, tau, v, at=p)` applies the **pushforward**
    $(\Phi_{F\to G}(\tau))_{*p}(v)$.
  - If the transformation law depends on the base point or on time derivatives
    (e.g. non-inertial frames), the implementation _must_ use both `tau` and
    `at`.
- Coordinate roles (`CoordDisp`, `CoordVel`, `CoordAcc`):
  - `apply_op(op, tau, v, at=p)` applies the **coordinate-basis pushforward**,
    using Jacobians of the point transform at `p`.
- Other roles (planned):
  - `Covector`: pullback by the inverse Jacobian.

No operator is allowed to ignore role semantics.

---

##### Role-specialized primitive operators (normative)

Reference-frame operators are often built from _primitive_ operations
(translate, rotate, boost, etc.) composed in a pipeline. To remain
mathematically correct and user-friendly, **primitive operators must be
role-specialized**: each primitive declares the role(s) on which it acts.

This spec permits role-specialization in either of two equivalent styles:

1. **Distinct primitive operator types** (e.g. `Translate`, `Boost`), each with
   role-specific `apply_op` registrations; or
2. A **single primitive family** parameterized by a **static role flag** (e.g.
   `Add(..., role=Point)` vs `Add(..., role=PhysVel)`).

Coordinax may choose either implementation, but the _observable semantics_ must
match the rules below.

###### Domain rules (normative)

Let `op` be a role-specialized primitive (either a distinct type like `Boost` or
an `Add(..., role=PhysVel)`).

- If `x` is a `Vector`, then applying `op` is only defined when the vector has
  the matching role:
  - `apply_op(op(role=Point), tau, v: Vector[Point])` is permitted.
  - `apply_op(op(role=PhysVel), tau, v: Vector[Point])` is **not** permitted and
    must raise `TypeError` (do not silently ignore).
- If `x` is a `PointedVector`, then applying `op` must act on **all fields whose
  role matches** the operator’s role:
  - point-like primitives update the base point,
  - tangent-like primitives update the corresponding fibre fields (e.g.
    velocity), using the base point as anchoring when required by the law. If
    the `PointedVector` does not contain any matching field, the operator must
    raise `TypeError`.
- If `x` is a `Coordinate`, `apply_op` must apply to the contained
  `PointedVector` and return a `Coordinate` in the target reference frame.

These rules ensure that pipelines like `Translate | Boost` behave “simply” for
anchored kinematic states, while preventing category errors on bare points.

###### Boost

A **Boost** is a _reference-frame velocity offset_ operator. It is **not**
merely a linear map on tangent vectors; it is a time-parameterized affine
transformation acting on events/points as well as on physical tangent
quantities.

Let $\tau$ be the affine parameter (typically physical time), let
$\Delta v(\tau)$ be the boost velocity field expressed as **physical tangent
components** (units of velocity), and fix an epoch $\tau_0$ (the time at which
the two frames coincide in position).

**Mathematical action by role**

- **Point** (event / position of a point): the boost acts as a time-dependent
  translation

  $$
  p'(\tau) = p(\tau) + \Delta x(\tau),\qquad \Delta x(\tau) = \int_{\tau_0}^{\tau} \Delta v(s)\,ds.
  $$

  In the common **constant boost** case $\Delta v(\tau)=\Delta v_0$ this reduces
  to

  $$
  \Delta x(\tau) = (\tau-\tau_0)\,\Delta v_0,
  \qquad p'(\tau)=p(\tau)+(\tau-\tau_0)\,\Delta v_0.
  $$

  **Implementation rule (normative):** `apply_op(Boost, tau, role=Point, ...)`
  MUST be defined. It MUST require a time-like `tau` when $\Delta v$ is nonzero.
  The point update MUST be implemented via the chart's canonical Cartesian chart
  (or via the product chart's spatial factors) to ensure correctness under
  non-linear point transforms.

- **PhysDisp** (physical displacement; difference of points at fixed $\tau$):
  Galilean boosts leave spatial displacements invariant

  $$
  \Delta p' = \Delta p.
  $$

  **Normative:** `apply_op(Boost, tau, role=PhysDisp, ...)` is the identity.

- **Vel** (physical velocity): add the boost velocity

  $$
  v'(\tau) = v(\tau) + \Delta v(\tau).
  $$

- **Acc** (physical acceleration): add the time derivative of the boost velocity

  $$
  a'(\tau) = a(\tau) + \frac{d}{d\tau}\Delta v(\tau).
  $$

  In particular, for constant $\Delta v$ this is the identity on acceleration.

**Chart conversion rule (normative)**

- The boost field $\Delta v$ is stored in an operator chart `op.chart`. To apply
  it to an input expressed in `chart`, convert $\Delta v$ using
  `physical_tangent_transform(chart, op.chart, dv, at=...)` (base-point required
  when chart conversion is nontrivial).
- For the **Point** rule, the translation $\Delta x(\tau)$ MUST be computed and
  applied in the canonical Cartesian chart of the relevant spatial chart, then
  transformed back with `point_transform`.

**Cartesian-product charts (normative)**

For `AbstractCartesianProductChart` inputs, Boost acts **only on spatial
factor(s)** (e.g. leaving time unchanged). Special spacetime charts (e.g.
`SpaceTimeCT`, `SpaceTimeEuclidean`) may expose un-prefixed time components like
`("t",)`; these MUST remain unchanged under Boost.

###### Worked example (normative semantics)

With role-specialized `Add`:

```
@plum.dispatch
def frame_transform_op(from_frame: Alice, to_frame: Bob, /) -> Operator:
    shift = Add.from_([100_000, 10_000, 0], "km", role=r.Point)   # translate points
    boost = Add.from_([269_813_212.2, 0, 0], "m/s", role=r.PhysVel)   # boost velocities
    return shift | boost
```

- Applying to a **point-only** vector is well-defined _if_ the boost specifies a
  reference epoch and is interpreted as a time-dependent translation:

```
q = Vector.from_([1, 2, 3], "kpc")   # defaults to role=Point
op(tau, q)                           # succeeds: Boost contributes v0*(tau-tau0)
```

- Applying to an anchored state must succeed:

```
base = Vector.from_([1, 2, 3], "kpc")          # Point
vel  = Vector.from_([10, 20, 30], "km/s", role=r.phys_vel)

fp = PointedVector(base=base, velocity=vel)
fp2 = op(tau, fp)
```

When applied to `PointedVector`, the same `Boost` updates both the base point
and the velocity consistently under the chosen kinematic model.

The operator calling convention is always:

```
op(tau, x) -> x_prime
```

and is normatively implemented by delegation to `apply_op(op, tau, x, ...)`.

---

##### Anchoring rules

- If `x` is a `Vector` with a tangent-like role and `at` is required but not
  provided, `apply_op` must raise `TypeError`.
- `PointedVector` and `Coordinate` are _anchored containers_ and therefore
  supply the base point implicitly; their `apply_op` implementations must _not_
  require an explicit `at=` argument.

##### Chart discipline

Reference-frame operators act **within a fixed chart family** (defaulting to
`cartesian_chart(...)`):

- Operators must _not_ implicitly change charts.
- Chart conversion is a separate operation handled by `vconvert`.

This separation is essential to keep geometric meaning and coordinate choice
orthogonal.

##### Composition

Operators are composable at the operator level, not the kernel level.

If `op_fg` maps $F\to G$ and `op_gh$ maps $G\to H$, then:

```
op_fh = compose(op_gh, op_fg)
```

must produce a new `Operator` whose `apply_op` semantics correspond to
$\Phi_{G\to H}(\tau)\circ\Phi_{F\to G}(\tau)$.

Composition must be **static** (no runtime graph construction) so that JAX can
inline through composed operators.

---

#### Implementation guidance (normative)

- `Operator` objects should be **lightweight value objects** (e.g. dataclasses
  or Equinox modules) containing only static identifiers such as
  `(from_frame, to_frame)`.
- All numerical work must live in `apply_op` registrations.
- Implementations should prefer _small, role-specific kernels_ over monolithic
  functions.
- `tau` must always be treated as an explicit input to numeric kernels, even if
  unused by a particular frame family.
- Dispatch must never depend on runtime array values.

This structure mirrors the design of `point_transform`,
`physical_tangent_transform`, and related APIs, and ensures conceptual
uniformity across Coordinax.

---

### Rotation operator: role-dependent mathematical action (normative)

A **rotation operator** represents a (possibly time-dependent) orthogonal
transformation of spatial coordinates. Mathematically, it is defined by a map

$$R(\tau) \in \mathrm{SO}(n),$$

where $\tau$ is an affine parameter (typically time) and $n$ is the spatial
dimension of the chart’s canonical Cartesian representation.

Rotations are **dimensionless**: the rotation matrix $R$ carries no physical
units. Units are carried exclusively by the vectors on which the rotation acts.

The action of `Rotate` depends critically on the **role** of the object being
transformed. This section fixes the normative semantics.

---

#### Action on `Point` (normative)

Let $x(\tau) \in M$ be a point in a Euclidean configuration space with canonical
Cartesian coordinates $\mathbf{x} \in \mathbb{R}^n$.

The action of a rotation on a point is:

$$\mathbf{x}'(\tau) = R(\tau)\,\mathbf{x}(\tau).$$

Normative properties:

- This is a **point map**, not a tangent pushforward.
- No base point (`at`) is required.
- The operation is implemented by:
  1. converting the point to the chart’s canonical Cartesian chart,
  2. applying the matrix multiplication $R(\tau)\mathbf{x}$,
  3. converting back to the original chart.
- For Cartesian-product charts, the rotation applies **only to spatial
  factor(s)** whose Cartesian dimension matches $R$; non-spatial factors (e.g.
  time) are left unchanged.

This corresponds to applying the diffeomorphism
$\Phi(\tau): M \to M, \quad \Phi(\tau)(x) = R(\tau)x.$

---

#### Action on `PhysDisp` (physical displacement) (normative)

A `PhysDisp` represents a **physical displacement** (difference of two points at
the same $\tau$):

$$ \Delta \mathbf{x} = \mathbf{x}\_1 - \mathbf{x}\_2 \in T_p M. $$

Under a rotation:

$$ \Delta \mathbf{x}' = R(\tau)\,\Delta \mathbf{x}. $$

Normative properties:

- `PhysDisp` transforms as a **physical tangent vector**.
- The transformation is **independent of the base point** (but the base point
  may still be required by the API for general manifolds).
- Units (length) are preserved.
- Implementation must use the physical tangent transform law, not
  `point_transform`.

This is the pushforward of the rotation map restricted to tangent vectors.

---

#### Action on `PhysVel` (physical velocity) (normative)

A `PhysVel` represents a physical velocity:

$$ \mathbf{v}(\tau) = \frac{d\mathbf{x}}{d\tau}. $$

Under a general time-dependent rotation $R(\tau)$, the transformed velocity is:

$$
\mathbf{v}'(\tau) = R(\tau)\,\mathbf{v}(\tau) + \dot
R(\tau)\,\mathbf{x}(\tau).
$$

Normative properties:

- Velocity transformation **depends on the base point** $\mathbf{x}(\tau)$.
- Therefore:
  - `apply_op(Rotate, tau, vel, at=point)` **must** use `at`.
  - Applying `Rotate` to a bare `PhysVel` without anchoring must raise
    `TypeError`.
- For **time-independent rotations** ($\dot R = 0$), the law reduces to:
  $\mathbf{v}' = R\,\mathbf{v}.$

This distinction is essential: rotation is not merely a linear map on velocities
when the frame is time-dependent.

---

#### Action on `PhysAcc` (physical acceleration) (normative)

A physical acceleration is:

$$ \mathbf{a}(\tau) = \frac{d^2\mathbf{x}}{d\tau^2}. $$

Under a time-dependent rotation:

$$
\mathbf{a}'(\tau) = R(\tau)\,\mathbf{a}(\tau) + 2\,\dot R(\tau)\,\mathbf{v}(\tau) + \ddot R(\tau)\,\mathbf{x}(\tau).
$$

Normative properties:

- Acceleration transformation depends on:
  - the base point $\mathbf{x}(\tau)$,
  - the velocity $\mathbf{v}(\tau)$,
  - and time derivatives of $R$.
- Therefore:
  - Applying `Rotate` to `PhysAcc` **requires anchoring** and access to the full
    kinematic state (typically via `PointedVector` or `Coordinate`).
  - Applying `Rotate` to a bare `PhysAcc` must raise `TypeError`.

This ensures consistency with Newtonian and relativistic kinematics.

---

#### Summary table (normative)

| Role       | Mathematical object  | Transformation law                                 | Requires `at`                 |
| :--------- | :------------------- | :------------------------------------------------- | :---------------------------- |
| `Point`    | $x \in M$            | $x' = R x$                                         | No                            |
| `PhysDisp` | $\Delta x \in T_p M$ | $\Delta x' = R\,\Delta x$                          | No (Euclidean); Yes (general) |
| `PhysVel`  | $\dot x \in T_p M$   | $\dot x' = R\dot x + \dot R\,x$                    | Yes                           |
| `PhysAcc`  | $\ddot x \in T_p M$  | $\ddot x' = R\ddot x + 2\dot R\dot x + \ddot R\,x$ | Yes                           |

---

#### Normative implications for implementation

1. `Rotate` **must not** be implemented as a single linear map on all roles.
2. Separate `apply_op` dispatches (or equivalent role-specialized kernels)
   **must** exist for:
   - `Point`,
   - physical tangent roles (`PhysDisp`, `PhysVel`, `PhysAcc`).
3. For `PointedVector` / `Coordinate`, `Rotate` must:
   - rotate the base point,
   - rotate velocity and acceleration consistently using the above formulas.
4. Time-dependence of `R` is not optional: even if a particular instance is
   time-independent, the API must be designed to support the general case.

This role-dependent structure is essential for mathematical correctness and is
not an implementation detail.

---

---

### Embeddings

An **Embedding** (as a chart wrapper) equips an intrinsic chart with an ambient
chart and explicit point/tangent maps.

Conceptual wrapper:

```
class EmbeddedChart(AbstractChart):
    chart_kind: AbstractChart      # intrinsic chart on M
    ambient_kind: AbstractChart    # chart on ambient space A
    params: Mapping[str, Any]
```

Normative embedding functions:

```
embed_point(embedded: EmbeddedChart, p_chart: CsDict) -> CsDict
project_point(embedded: EmbeddedChart, p_ambient: CsDict) -> CsDict

embed_tangent(embedded: EmbeddedChart, v_phys_chart: CsDict, *, at: CsDict) -> CsDict
project_tangent(embedded: EmbeddedChart, v_phys_ambient: CsDict, *, at: CsDict) -> CsDict
```

`embed_tangent` implements the pushforward on physical tangent components (using
the embedding Jacobian and/or frames). `project_tangent` projects an ambient
physical vector onto the intrinsic tangent space and expresses it in the
intrinsic physical frame.

---

## Roles and geometric meaning

A **Role** specifies the mathematical type and physical interpretation of a
`Vector`’s component data. Roles determine which conversion law applies and
which algebraic operations are valid.

### Core roles (current)

- `Point`: a point $p \in M$ (affine). Components may have mixed dimensions.
- `PhysDisp`: a physical tangent vector $\Delta p \in T_pM$ with units of
  length. (This role represents a _position difference_ / physical displacement,
  anchored at a base point.)
- `PhysVel`: a physical tangent vector $v\in T_pM$ with units length/time.
- `PhysAcc`: a physical tangent vector $a\in T_pM$ with units length/time$^2$.

`PhysDisp`, `PhysVel`, and `PhysAcc` are all **physical tangent** roles: their
components are expressed in an orthonormal frame, and thus have uniform physical
dimension across components.

#### Role class hierarchy (normative)

- `AbstractRole`: base abstract class for all roles; defines common API such as
  `dimensions()`, and abstract `derivative()` / `antiderivative()`.
- `AbstractPhysRole(AbstractRole)`: abstract base class for physical tangent
  roles; used to group roles that transform via `physical_tangent_transform` and
  require uniform physical dimensions.
- Concrete roles:
  - `Point(AbstractRole)`
  - `PhysDisp(AbstractPhysRole)`
  - `PhysVel(AbstractPhysRole)`
  - `PhysAcc(AbstractPhysRole)`

This hierarchy enables dynamic discovery of physical roles (e.g., for test and
strategy generation) by inspecting subclasses of `AbstractPhysRole`.

### Additional roles (planned)

Additional role families have distinct transformation laws:

- `CoordDeriv`: coordinate derivatives like $dq^i/dt$ in the coordinate basis;
  units may be heterogeneous.
- `Covector`: cotangent vectors $\alpha \in T_p^*M$, transforming by pullback /
  inverse Jacobian.

### Role → transformation law (normative)

- `Point` converts by `point_transform` and does not require `at=`.
- Physical tangent roles (`PhysDisp`, `PhysVel`, `PhysAcc`) convert by
  `physical_tangent_transform` and require `at=`.
- Coordinate-basis roles (`CoordDisp`, `CoordVel`, `CoordAcc`) convert by
  `coord_transform` and require `at=`.
- Cotangent roles (`Covector`) convert by `cotangent_transform` and require
  `at=`.

### Coercions and anchoring (`as_disp`, `at=`)

Physical tangent roles are anchored at a base point $p$ (an element of $T_pM$).
In code this anchoring is carried explicitly by `at=` in operations requiring
tangent/cotangent spaces.

`as_disp(p, origin=o)` is a coercion from a point $p\in M$ to a `PhysDisp`
(physical displacement / position-difference tangent vector) in $T_oM$. In
Euclidean affine spaces it is canonically $p-o$. On a general manifold it
requires an explicit log-map-like choice and is otherwise not defined.

---

## Core functional API

All core operations are **functions** (not methods) and are extended via
multiple dispatch. The high-level API (`Vector.vconvert`, etc.) delegates to
these functions.

### Point transforms

`point_transform` maps coordinates of the _same geometric point_ between charts.

```
point_transform(to_chart: AbstractChart, from_chart: AbstractChart, p: CsDict, /, *, usys=None) -> CsDict
```

Mathematically, if $u=f(q)$ is the chart transition map:

$$
u^a = f^a(q^1,\dots,q^n).
$$

**Value types in CsDict (normative):**

A `CsDict` may contain either:

1. **Quantities** ({class}`unxt.Quantity`): Values with explicit units. No
   additional context needed; the function uses units directly for any
   dimension-dependent operations.
2. **Arrays** (JAX arrays, NumPy arrays, or Python scalars): Dimensionless
   values. For charts where components have physical dimensions (e.g. distance
   in spherical coordinates), the optional `usys` keyword argument specifies the
   unit system in which the array values should be interpreted.

The `usys` parameter is a {class}`unxt.AbstractUnitSystem` that provides default
units for each physical dimension. When `usys` is provided and the input
contains bare arrays, the function interprets those arrays as having units
determined by the chart's `coord_dimensions` and the unit system.

**Examples:**

```python
import coordinax.charts as cxc
import coordinax.transforms as cxt
import unxt as u

# With Quantities (explicit units)
p_qty = {"r": u.Q(1.0, "km"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1.0, "rad")}
p_cart = cxt.point_transform(cxc.cart3d, cxc.sph3d, p_qty)
# Result: {'x': Quantity(..., unit='km'), 'y': ..., 'z': ...}

# With bare arrays (no units)
p_arr = {"r": 5}
p_cart = cxt.point_transform(
    cxc.cart1d, cxc.radial1d, p_arr, usys=u.unitsystems.galactic
)
# Result: {'x': 5}
```

### Physical tangent transforms

{class}`~coordinax.transforms.physical_tangent_transform` maps **physical
tangent components** (orthonormal-frame components) between charts at a base
point.

```
physical_tangent_transform(
    to_chart: AbstractChart, from_chart: AbstractChart, v_phys: CsDict, *, at: CsDict
) -> CsDict
```

Normative implementation uses frames:

1. Construct $B_{\rm from} = \mathrm{frame\_cart}(\mathrm{from\_chart}, at)$.
2. Convert the base point to the target chart:
   `at_to = point_transform(to_chart, from_chart, at)`.
3. Construct $B_{\rm to} = \mathrm{frame\_cart}(\mathrm{to\_chart}, at_to)$.
4. Convert via a canonical ambient orthonormal frame:
   - pack `v_phys` in source order, map to ambient, map back, unpack.

If $N=n$ (square frames):

$$
v_{\rm to} = B_{\rm to}(p)^{\mathsf T}\,B_{\rm from}(p)\,v_{\rm from}.
$$

If embedded ($B \in \mathbb{R}^{N\times n}$ rectangular), projection must be
consistent with `embed_tangent`/`project_tangent`. A normative choice is:

- ambient components: $v_{\rm amb} = B_{\rm from}\,v_{\rm from}$,
- intrinsic components: $v_{\rm to} = B_{\rm to}^+\,v_{\rm amb}$ where $B^+$ is
  the Moore–Penrose pseudoinverse.

### Coordinate-basis tangent roles (normative)

In addition to physical tangent roles, Coordinax supports **coordinate-basis
tangent components**, which represent tangent vectors expressed in the
coordinate basis of a chart.

Let $(M, \varphi)$ be a charted manifold with coordinates
$q = (q^1, \dots, q^n)$ and coordinate basis $\{ \partial / \partial q^i \}$ at
a point $p$.

A tangent vector $v \in T_p M$ may be written as

$$
v = v^i \frac{\partial}{\partial q^i}.
$$

The scalars $v^i$ are **coordinate-basis components**. These generally have
heterogeneous physical units (e.g. radians per second, meters per second).

Coordinax defines the following roles:

- `CoordDisp`: coordinate-basis components of a displacement tangent vector.
- `CoordVel`: coordinate-basis components of a velocity tangent vector.
- `CoordAcc`: coordinate-basis components of an acceleration tangent vector.

These roles are distinct from physical tangent roles (`PhysDisp`, `PhysVel`,
`PhysAcc`), whose components are expressed in an orthonormal physical frame and
therefore have uniform physical units.

Normative clarification:

- `CoordVel` and `CoordAcc` represent the coordinate-basis components of the
  **physical** velocity or acceleration vector in $T_pM$.
- They are **not** raw higher derivatives of coordinates interpreted as tensors.
  Non-tensorial objects such as $d^2 q^i / dt^2$ are not represented by these
  roles.

### Coordinate-derivative transforms

`coord_transform` maps **coordinate-basis tangent components** between charts
using the Jacobian of the point transformation.

```
coord_transform(
    to_chart: AbstractChart,
    from_chart: AbstractChart,
    dqdt: CsDict,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict
```

Mathematical law (normative):

Let $u = f(q)$ be the coordinate transition map between `from_chart` and
`to_chart`. Then at the base point $q$,

$$
\dot u^a = \frac{\partial f^a}{\partial q^i}(q)\,\dot q^i.
$$

This law applies identically to `CoordDisp`, `CoordVel`, and `CoordAcc`; only
the physical dimensions of the components differ.

Normative properties:

- `coord_transform` **requires** a base point `at`, because the Jacobian is
  evaluated at that point.
- Components may have heterogeneous units.
- The transformation is linear in the input components.
- For Cartesian product charts, the Jacobian is block diagonal and the transform
  acts factorwise.

Relationship to physical components:

- Physical tangent transforms (`physical_tangent_transform`) operate on
  orthonormal-frame components.
- Coordinate-basis transforms (`coord_transform`) operate on coordinate-basis
  components.
- Conversion between these representations is handled by `physicalize` /
  `coordinateize`.

Role → transformation law (normative update):

- `Point` uses `point_transform`.
- `PhysDisp`, `PhysVel`, `PhysAcc` use `physical_tangent_transform`.
- `CoordDisp`, `CoordVel`, `CoordAcc` use `coord_transform`.

### Cotangent transforms

`cotangent_transform` maps cotangent components $\alpha \in T_p^*M$ between
charts.

```
cotangent_transform(
    to_chart: AbstractChart, from_chart: AbstractChart, alpha: CsDict, *, at: CsDict
) -> CsDict
```

Normative law (pullback / inverse Jacobian):

$$
\alpha'_a = \alpha_i \frac{\partial q^i}{\partial u^a}(q).
$$

Implementation may compute the Jacobian of `point_transform` and apply an
inverse/linear solve.

### Physicalize / Coordinateize

These functions convert between coordinate derivatives and physical tangent
components at a point.

```
physicalize(chart: AbstractChart, dqdt: CsDict, *, at: CsDict) -> CsDict
coordinateize(chart: AbstractChart, v_phys: CsDict, *, at: CsDict) -> CsDict
```

Normatively, if $E(q)$ maps coordinate-basis components to orthonormal physical
components, then:

$$
v_{\rm phys} = E(q)\,\dot q, \qquad \dot q = E(q)^{-1} v_{\rm phys}.
$$

In classic orthogonal coordinates, $E$ corresponds to applying scale factors
(e.g. $r$, $r\sin\theta$).

### Raise / Lower

Given a metric $g$, raising/lowering relates tangent and cotangent objects.

```
lower_index(chart: AbstractChart, v_coord: CsDict, *, at: CsDict) -> CsDict
raise_index(chart: AbstractChart, alpha: CsDict, *, at: CsDict) -> CsDict
```

For coordinate-basis components:

$$
\alpha_i = g_{ij} v^j, \qquad v^i = g^{ij} \alpha_j.
$$

(Physical covectors, if introduced, require a corresponding orthonormal covector
frame; this is future work.)

---

## High-level API

The high-level API exists to make routine use simple (create vectors, convert
charts, index components) while remaining grounded in the low-level framework.

### CsDict conventions

A `CsDict` is a mapping from component-name to quantity-like array.

- Keys must match `chart.components`.
- Algorithms that pack/unpack must impose ordering via `chart.components` (not
  dict insertion order).

### Vector

A `Vector` bundles:

- `data: CsDict`
- `chart: AbstractChart`
- `role: Role`

The canonical construction is the `Vector(...)` constructor with explicit
`data`, `chart`, and `role`.

```
cart = Vector({"x": x, "y": y, "z": z}, chart=r.cart3d, role=Point)
```

`Vector.from_(...)` is a multiple-dispatch convenience that may infer `chart`
and/or `role`.

```
cart = Vector.from_({"x": x, "y": y, "z": z})  # may infer chart and role
cart = Vector.from_({"x": x, "y": y, "z": z}, chart=cxc.cart3d)  # infer role
... # there are many variants
```

#### Vector operator implementation via Quax

`Vector` operations like `+`, `-`, etc. are implemented using **Quax** multiple
dispatch on JAX primitives, not traditional Python methods. This enables:

1. **Static dispatch**: operations resolve on static Python types before JAX
   tracing
2. **JAX transformations**: operations work correctly under `jit`, `vmap`,
   `grad`
3. **Extensibility**: new types can register handlers for JAX primitives

**Design pattern for binary operations:**

```
@quax.register(jax.lax.add_p)
def add_p_vec_vec(lhs: Vector, rhs: Vector, /) -> Vector:
    """Handle Vector + Vector via jax.lax.add_p."""
    return add(lhs.role, rhs.role, lhs, rhs, at=None)

@quax.register(jax.lax.add_p)
def add_p_vec_qty(lhs: Vector, rhs: Quantity, /) -> Vector:
    """Handle Vector + Quantity by desugaring to Vector + Vector."""
    # Desugar: interpret Quantity as a Vector with appropriate role
    rhs_vec = Vector.from_(rhs)  # or Vector.from_(rhs, role) if needed
    return add(lhs.role, rhs_vec.role, lhs, rhs_vec, at=None)
```

#### Subtraction and cross-chart operations (normative)

Subtraction is role-dependent and must not perform implicit chart conversions
that would be mathematically underdetermined.

**Algebraic meaning (normative):**

- `PhysDisp - PhysDisp -> PhysDisp`, `PhysVel - PhysVel -> PhysVel`,
  `PhysAcc - PhysAcc -> PhysAcc` are vector-space operations in a single tangent
  space $T_pM$.
- `Point - Point -> PhysDisp` is an affine difference that yields a displacement
  (a tangent vector) anchored at the chosen base point.

**Chart discipline (normative):**

- If `lhs.chart == rhs.chart`, component-wise subtraction is permitted _after_
  role-compatibility checks.
- If `lhs.chart != rhs.chart` and the operation involves any **physical
  tangent** role (`PhysDisp`, `PhysVel`, `PhysAcc`) or produces one (e.g.
  `Point - Point -> PhysDisp`), then a base point is required to compare the two
  objects in the same tangent space. Therefore the implementation **must not**
  “convert both sides to Cartesian” unless an explicit base point is available.

Concretely:

- For tangent-like operations (`PhysDisp/PhysVel/PhysAcc`): `rhs` must be
  converted to `lhs.chart` using `vconvert(..., at=base_point)` and only then
  subtracted.
- For `Point - Point`: if charts differ, first convert `rhs` to `lhs.chart` via
  `point_transform` (which does _not_ require `at=`), then perform the affine
  difference in that common chart.

If a required base point is not provided, subtraction must raise
`TypeError`/`ValueError` with an error message that explicitly mentions that
cross-chart tangent operations require `at=` (or an anchored container like
`PointedVector`).

**No implicit Cartesian fallback (normative):**

Code must not silently choose `cartesian_chart(...)` as an intermediate for
subtraction unless doing so is mathematically justified _and_ the necessary base
point information has been provided.

**Key notes:**

- Do **not** implement `Vector.__add__`, `Vector.__sub__`, etc. directly.
- Instead, register handlers on the **JAX primitives** (`jax.lax.add_p`,
  `jax.lax.sub_p`, etc.) using `@quax.register()`.
- For binary operations involving `Quantity` or other types, the dispatch
  handler is responsible for converting to `Vector` using `Vector.from_()` and
  then delegating to the existing multi-dispatch logic (e.g.,
  `add(role_lhs, role_rhs, lhs, rhs, at=)`).
- `Vector` inherits from `quax_blocks.LaxBinaryOpsMixin` (and similar), which
  provides `__add__`, `__sub__`, etc. that automatically dispatch through Quax.

#### `Vector.from_` for `Quantity` inputs (normative)

`Vector.from_(q: Quantity)` infers the appropriate role from the physical
dimension of `q`:

- **Normative rule:** for an `N`-component quantity `q`:
  - **Length dimension** (`u.dimension("length")`): `Vector.from_(q)` **must**
    return a `Vector` with role `Point` (not `PhysDisp`), in the inferred
    canonical Cartesian chart of dimension `N`. This preserves the
    affine/tangent distinction: a bare length-valued coordinate tuple is not, by
    itself, a displacement.
  - **Speed dimension** (`u.dimension("speed")`): `Vector.from_(q)` **must**
    return a `Vector` with role `PhysVel`, representing a velocity vector.
  - **Acceleration dimension** (`u.dimension("acceleration")`):
    `Vector.from_(q)` **must** return a `Vector` with role `PhysAcc`,
    representing an acceleration vector.

Formally:

```
Vector.from_(u.Q([...], "m")) -> Vector[Any, Point, u.Q]      # length → Point
Vector.from_(u.Q([...], "m/s")) -> Vector[Any, PhysVel, u.Q]      # speed → PhysVel
Vector.from_(u.Q([...], "m/s^2")) -> Vector[Any, PhysAcc, u.Q]    # acceleration → PhysAcc
```

This dimension-based inference provides ergonomic construction while maintaining
type safety.

##### Explicit role constructor for `Quantity`

To support ergonomic but unambiguous construction, Coordinax must also provide a
dispatch that accepts an explicit role:

```
Vector.from_(q: u.AbstractQuantity, role: r.AbstractRole, /) -> Vector
```

Normative semantics:

1. Infer the chart from the trailing component axis of `q` (`N = q.shape[-1]`)
   using the same chart inference machinery as `Vector.from_(q)` (typically
   selecting the canonical Cartesian chart `cartNd`).
2. **Validate dimension compatibility** between `q` and `role` using
   `dimension_of(q)`:
   - `Point`: no additional restriction (heterogeneous chart components are
     handled when constructed from mappings; the `Quantity` constructor uses a
     packed vector, so it is taken to represent an `N`-tuple in the inferred
     chart).
   - `PhysDisp`: require `dimension_of(q) == length`.
   - `PhysVel`: require `dimension_of(q) == length/time`.
   - `PhysAcc`: require `dimension_of(q) == length/time^2`.
   - other roles (planned) define their own compatibility rules.
3. Construct and return `Vector(data=..., chart=..., role=role)`.

If the dimension check fails, this overload must raise `UnitConversionError` or
`ValueError` (implementation choice), and the error message must include the
required and actual physical dimensions.

Often in examples -- docstrings and docs -- it is preferable to use
`Vector.from_` over the explicit `Vector(...)` constructor for brevity.

Conversion:

```
sph = cart.vconvert(r.sph3d)
rval = sph["r"]
```

For tangent-like roles:

```
v_sph = v_cart.vconvert(r.sph3d, at=cart_pos)
```

#### `Vector + Quantity` (normative convenience)

`Vector + Quantity` is an ergonomic convenience, but it must preserve geometric
correctness by reducing to role-aware `Vector + Vector` after a **static** role
decision.

Normative rule:

- For any `lhs: Vector` and `q: Quantity`, `lhs + q` **must** be evaluated as:

1. If `dimension_of(q) == length`, interpret `q` as a **physical displacement**
   and construct:

```
rhs = Vector.from_(q, r.phys_disp)
```

2. Otherwise, construct using the default `Quantity` overload:

```
rhs = Vector.from_(q)  # default role is Point
```

3. Finally perform role-aware vector addition:

```
lhs + rhs
```

Notes:

- This rule intentionally treats _length-like_ packed quantities as
  displacements in `+` contexts because that is the overwhelmingly common user
  intent (“translate this point by a displacement”), while leaving
  `Vector.from_(q)` itself conservative (defaulting to `Point`).
- The dimension check (`dimension_of(q) == length`) is static with respect to
  JAX tracing and may be used inside `jit` as an ordinary Python branch.
- The `Vector + Vector` addition path is responsible for enforcing chart
  compatibility and role validity (e.g. `Point + PhysDisp -> Point`,
  `PhysDisp + PhysDisp -> PhysDisp`, and rejecting invalid combinations).

For non-Euclidean charts where addition requires a base point (`at=`),
`Vector + Quantity` must raise an error directing the user to
`Vector.add(..., at=...)` or to an anchored container (e.g. `PointedVector`).

Normative `Vector.vconvert` is a thin wrapper over the functional API:

- uses `point_transform` for `Point`,
- uses `physical_tangent_transform` for physical tangent roles,
- uses `coord_transform` for coordinate-basis tangent roles (`CoordDisp`,
  `CoordVel`, `CoordAcc`),
- uses `cotangent_transform` for `Covector`.

### PointedVector

`PointedVector` binds a base `Point` with one or more anchored fibre vectors at
the same point, reducing the need to pass `at=` repeatedly.

Conceptually:

- $(p, v) \in TM$,
- $(p, \alpha) \in T^*M$.

`PointedVector` may expose convenience methods:

- `.vconvert(...)` that supplies `at=` automatically,
- `.phys_disp`, `.phys_vel`, `.phys_acc`, `.displacement`, etc.

---

## Multiple dispatch and JAX compatibility

All low-level functions are extended using `plum` multiple dispatch (or
equivalent). Dispatch must occur on **static** Python objects
(charts/roles/wrappers) so that under `jax.jit` dispatch resolves before tracing
and is compiled away.

Numerical kernels must be:

- pure,
- JAX-traceable,
- expressed in `jax.numpy` (or compatible primitives),
- accept/return pytrees (typically `CsDict`).

Transform rules are intended to be defined on _scalar component objects_ (the
values in `CsDict`). Performance comes from `jax.jit` compilation and `jax.vmap`
over these scalar rules. Implementations should not require pre-packed shaped
vectors as a primary interface; packing/unpacking is an internal convenience
only.

---

## Design invariants

1. **Stateless charts/roles**: chart and role objects are immutable and
   hashable.
2. **Explicit base points**: operations on tangent/cotangent objects require
   `at=` (or a `PointedVector` supplying it).
3. **Uniform-dimension physical tangents**: physical tangent roles have uniform
   physical dimensions across components.
4. **Coordinate derivatives are separate**: $dq^i/dt$ are not physical tangent
   components.
5. **Extensibility by registration**: new charts and conversions are added by
   registering functional rules.
6. **JAX friendliness**: static dispatch + traceable numeric code.
