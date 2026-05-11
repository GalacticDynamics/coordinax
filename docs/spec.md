# Coordinax Core Specification

This document is the **normative** specification for the mathematical and software design of the `coordinax` coordinate and vector system. It defines a low-level, mathematically correct framework—**Charts**, **Metrics**, **Manifolds**, **Frames**, **Embeddings**, and **Maps**—and a high-level user API built on top of them -- **Point**, **Reference Frames**, **Coordinate**. The goals are:

1. **Correctness-first foundation**: rigorous definitions of points, tangents, their transformations, etc.
2. **Ergonomic high-level API**: common tasks should not expose low-level details.
3. **Extensibility via multiple dispatch**: functional API for existing functions are added by registration, not by editing core logic.
4. **JAX compatibility**: dispatch resolves on static Python objects; numerical kernels are pure and traceable.

## Table of Contents

```{contents}
:depth: 2
```

</br>

---

(math-spec)=

# The Math

In this section we fix the _mathematical objects_ that `coordinax` represents and the _transformation laws_ that determine how component data moves. This is a high-level overview; later sections specify each object and API in detail.

(math-spec-angles-and-distances)=

## Angles and Distances

<details>
<summary>Software Specification</summary>

- [Angles](#software-spec-angles)
- [Distances](#software-spec-distances)

</details>

A **quantity** is a pair $(x, u)$ where $x \in \mathbb{R}$ (or $\mathbb{R}^n$) is a numerical value and $u$ is a unit of measurement. Two quantities are physically equivalent when their values scale inversely with their units; unit conversion is a bijection on $\mathbb{R}$. _(Unit arithmetic is handled by `unxt`.)_

An **angular quantity** is special because it represents an element of the circle $S^1 = \mathbb{R}/2\pi\mathbb{Z}$, not of $\mathbb{R}$. $S^1$ is compact and has no boundary: there is no intrinsically "first" or "last" angle, and the space wraps. To store an angle as a real number one must choose a **branch cut** — a point excluded from the circle used as the interval endpoint. The two standard choices are $[0, 2\pi)$ and $(-\pi, \pi]$; both represent the same $S^1$. The cut is an artifact of representation: arithmetic that crosses it requires explicit wrapping, and the same physical angle has different numerical values under the two conventions.

A **distance quantity** has dimensions of length and, in the strict metric sense, is non-negative: it measures the magnitude of a displacement. A **radial coordinate**, however, can be signed — either because the coordinate system parameterizes a full line ($r \in \mathbb{R}$), or because a negative value is given physical meaning via the antipodal map $(r, \theta, \phi) \equiv (-r,\, \pi - \theta,\, \phi + \pi)$. The distinction between "distance" ($r \geq 0$) and "signed radial coordinate" ($r \in \mathbb{R}$) is a domain constraint on the same dimensional quantity. The point $r = 0$ is a coordinate singularity in spherical and cylindrical systems: the angular coordinates are degenerate there.

</br>

(math-spec-points)=

## Points

A point is an abstract geometric object. In this section we will explore how points can be concretized into coordinates -- e.g. "x", "y", "z" -- in coordinate representations --- e.g. Cartesian --- on manifolds -- e.g. $\mathbb{R}^3$. We will see how these coordinate representations of these points can be changed in the same manifold and across manifolds. We will also explore (time-dependent) actions on these coordinates, e.g. translations, and how to build these actions into changes of reference frames.

(math-spec-smooth-manifolds)=

### Smooth Manifolds

#### Topological Manifolds

A **topological manifold** of dimension $n$ is a topological space $M$ satisfying three axioms:

1. **Hausdorff**: any two distinct points $p, q \in M$ have disjoint open neighbourhoods,
2. **Second-countable**: the topology of $M$ admits a countable basis, and
3. **Locally Euclidean of dimension $n$**: every point $p \in M$ has an open neighbourhood $U_p$ homeomorphic to an open subset of $\mathbb{R}^n$ via a continuous bijection $\varphi_p : U_p \to \mathbb{R}^n$ with continuous inverse.

The locally Euclidean condition is what makes coordinates possible: near any point, the space looks like flat $n$-dimensional Euclidean space, so one can assign $n$ real numbers to each point in a neighbourhood. The Hausdorff condition rules out pathological point identifications, and second-countability ensures the existence of partitions of unity (needed to glue local constructions together globally).

At the topological level one can ask whether maps are **continuous** or **homeomorphisms**, but not whether they are differentiable. Differentiability is an additional structure layered on top.

#### Smooth Manifolds

A **smooth manifold** is a topological manifold $M$ equipped with a **smooth atlas** — a collection of charts $\{(U_\alpha, \varphi_\alpha)\}$ whose domains cover $M$ and whose **transition maps**

$$
\tau_{\alpha\beta} = \varphi_\beta \circ \varphi_\alpha^{-1} :
\varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)
$$

are $C^\infty$ diffeomorphisms wherever chart domains overlap. The unique maximal such atlas defines the **smooth structure**. **Smooth** means that calculus on $M$ is well-defined: one can differentiate functions, define tangent vectors, and integrate differential forms.

A **point** is simply an element $p \in M$.

Most of the important structures of smooth manifolds --- charts, atlases, transition maps, metrics, and embeddings --- are introduced in the sections that follow.

(math-spec-charts)=

### Charts

[Software Spec](#software-spec-charts)

A smooth manifold is defined by its **atlas** $\mathcal{A}$ -- a collection of compatible charts used for representing coordinates. A **chart** is a local diffeomorphism[^diffeomorphism] assigning coordinates $q = (q^1,\dots,q^n)$ to points $p \in U$, where $U$ is the open subset of the manifold on which the coordinate system (chart) is valid. It is notated as the pair

$$
C=(U, \varphi),
$$

where

$$
\varphi : U \subset M \to \mathbb{R}^n,
$$

is the _chart map_ that performs the point to coordinate assignment.

!!! example

    As a simple example, take the 3-dimensional Spherical chart $C_s$ on a 3-dimensional Euclidean manifold $\mathbb{R}^3$. This chart covers the whole manifold, so we can write it simply as

    $$ C_s = (\mathbb{R}^3, \phi_s). $$

    Let $p$ be a point in $\mathbb{R}^3$.
    Applying the chart map $\phi_s$ to $p$ we get a coordinate $q$ with named components $(r,\theta,\phi)$ and physical dimensions of _(length,angle,angle)_.

[^diffeomorphism]: Formally, let M and N be smooth manifolds. A map $\tau : M \to N$ is a diffeomorphism if $\tau$ is bijective and smooth, and $\tau^{-1}$ is smooth. A diffeomorphism preserves dimension, smooth structure, and tangent bundle structures. It doesn't necessarily preserve the metric, distances, angles, or curvature.

### Atlases

An atlas $\mathcal{A}$ is a collection of compatible charts that cover the manifold

$$
\mathcal{A} = \{ (C_\alpha) \}.
$$

An atlas is defined through any collection of (compatible) charts; for example

$$
\mathcal{A} = \{ \text{Cartesian}, \text{Spherical} \}
$$

is a valid atlas for $\mathbb{R}^3$. The **maximal atlas** $\mathcal{A}_{\mathrm{max}}$ is the unique atlas containing every chart compatible with the original collection.

In [Smooth Manifolds](#math-spec-smooth-manifolds) we defined a smooth manifold as a topological space $M$ and a notion of differentiability. Now we can refine this definition, and specify that the maximal smooth atlas $\mathcal{A}_{\mathrm{max}}$ provides this differentiability notion. Thus a smooth manifold is a topological space equipped with a maximal smooth atlas --- formally $(M, \mathcal{A}_{\mathrm{max}})$, though often abbreviated back to just $M$.

(math-spec-transition-maps)=

### Transition Maps

[Software Spec](#software-spec-pt_map)

For charts in the same atlas of a manifold there are **transition maps** $\tau$ between any two charts for points on the manifold. These transition maps are composed of an inverse chart map that takes a coordinate $q$ back to the point $p$, followed by the forward chart map of the second chart that takes $p$ to the new coordinates $q'$. Transition maps are smooth and invertible, and they determine how point components transform between charts.

In mathematical notation, let chart $C_1 = (U, \varphi_1)$ and $C_2 = (V, \varphi_2)$, where $U, V$ have overlapping domains, then the transition map from chart 1 to 2 is

$$
\tau_{C_1\to C_2} \equiv \varphi_2 \circ \varphi_1^{-1} : \varphi_1(U \cap V) \to \varphi_2(U \cap V).
$$

!!! example

    Let Cartesian coordinates have chart $C_C = (\mathbb{R}^3, \varphi_C)$ and spherical coordinates have $C_S = (\mathbb{R}^3, \varphi_S)$. The transition map from Cartesian to spherical coordinates is

    $$
    \tau_{C\to S} \equiv \varphi_S(r, \theta, \phi) \circ \varphi_C^{-1}(x, y, z),
    $$

    which can be expressed numerically as:

    $$
    \tau_{C\to S} = \left( \sqrt(x^2+y^2+z^2), \arccos(z/r), \arctan(y/x) \right)
    $$

(math-spec-transition-maps-jacobian)=

#### Jacobian of the Transition Map

The **Jacobian** of a transition map is the matrix of first partial derivatives of the new coordinates with respect to the old coordinates. For

$$
\tau_{C_1\to C_2}(q) = q', \qquad q = (q^1,\dots,q^n), \quad q' = (\tilde{q}^1,\dots,\tilde{q}^n),
$$

the Jacobian at $q \in \varphi_1(U \cap V)$ is

$$
J^j{}_i(q)
= \frac{\partial \tau_{C_1\to C_2}^j}{\partial q^i}(q)
= \frac{\partial \tilde{q}^j}{\partial q^i}.
$$

Because $\tau_{C_1\to C_2}$ is a diffeomorphism on the overlap, $J(q)$ is invertible at every regular point of the overlap, and

$$
J(\tau_{C_1\to C_2}^{-1}(q')) = J(q)^{-1}
$$

for the inverse transition map.

(math-spec-embedded-manifolds)=

### Embedded Manifolds

The central question is: when does a manifold $N$ sit _well-behaved_ inside a larger manifold $M$? The unit sphere $S^2 \subset \mathbb{R}^3$ is the prototypical example — it does not cross itself, carries no folds or cusps, and its spherical topology is faithfully represented in ambient space. The concept that formalizes this is a **smooth embedding**, built from three conditions that each rule out a distinct pathology.

**Condition 1 — No self-intersections.** A map $f : X \to Y$ is **injective** (one-to-one) if distinct points always map to distinct points: $f(p) = f(q)$ implies $p = q$. Without injectivity, distinct points of $N$ could collapse onto the same point of $M$, making the image self-intersect.

**Condition 2 — No folding of tangent directions.** A smooth map $f : N \to M$ is an **immersion** if its differential $df_p : T_pN \to T_{f(p)}M$ is injective at every point $p \in N$. This is a local condition: the tangent space of $N$ at $p$ maps into $M$ without collapsing any direction, so there are no cusps or folds at $p$. Crucially, an immersion can still cross itself globally.

!!! example

    The curve $f : \mathbb{R} \to \mathbb{R}^2$, $f(t) = (\sin 2t,\, \sin t)$, traces a figure eight. Its derivative $f'(t) = (2\cos 2t,\, \cos t)$ is never zero (if $\cos t = 0$ then $2\cos 2t = \pm 2 \neq 0$), so $f$ is an immersion everywhere. Yet $f(0) = f(\pi) = (0, 0)$: the curve passes through the origin twice, violating injectivity. An immersion is not enough to rule out self-intersections.

**Condition 3 — No topological misbehavior.** A **homeomorphism** $f : X \to Y$ is a bijective continuous map whose inverse is also continuous; it certifies that $X$ and $Y$ are topologically the same space. We need this condition applied to $f$ viewed as a map onto its image. An injective immersion can still fail here: imagine parametrizing an open interval so that one end spirals arbitrarily close to a point already in the image without ever reaching it — the map is injective and smooth, but its inverse is not continuous, and the image is topologically malformed.

When all three conditions hold, $f$ is a **smooth embedding**.

> **Smooth embedding.** A smooth map $\iota : N \hookrightarrow M$ of an $m$-dimensional manifold $N$ into an $n$-dimensional manifold $M$ (with $m < n$) is a **smooth embedding** if it is simultaneously:
>
> - **injective**: distinct points of $N$ map to distinct points of $M$,
> - an **immersion**: the differential $d\iota_p$ is injective at every $p \in N$, and
> - a **homeomorphism onto its image**: $\iota$ is a continuous bijection $N \to \iota(N)$ with continuous inverse.

The **image** of $\iota$ is the subset

$$
\iota(N) = \{ \iota(p) \in M \mid p \in N \} \subset M,
$$

and it is itself an $m$-dimensional manifold. A **submanifold** of $M$ is any subset $S \subset M$ that arises this way: it is a manifold in its own right, and its inclusion map $S \hookrightarrow M$ is a smooth embedding. Together, the three embedding conditions ensure $\iota(N)$ has no self-intersections, no cusps or folds, and carries the same smooth structure as $N$.

!!! note

    For compact $N$, the homeomorphism condition is automatic: any injective immersion from a compact manifold is an embedding. In practice this covers most physical cases ($S^1$, $S^2$, compact level sets of potentials).

!!! example

    The unit sphere $S^2$ embeds in $\mathbb{R}^3$ via the inclusion map $\iota : S^2 \hookrightarrow \mathbb{R}^3$, $p \mapsto p$. In the standard spherical chart $(\theta,\phi) \in (0,\pi)\times(0,2\pi)$ on $S^2$ and the Cartesian chart $(x,y,z)$ on $\mathbb{R}^3$, the coordinate expression of $\iota$ is

    $$
    \iota(\theta, \phi) = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta).
    $$

    The Jacobian of $\iota$ has rank 2 everywhere on the chart domain, confirming it is an immersion; $\iota$ is injective on the chart domain; and $\iota$ is a homeomorphism onto its image (the sphere minus the meridian $\phi=0$). The full image $\iota(S^2) = \{(x,y,z) \mid x^2 + y^2 + z^2 = 1\}$ is a 2-dimensional submanifold of $\mathbb{R}^3$; a single chart does not cover all of $S^2$, but the embedding itself is defined globally on $S^2$.

#### Projections and Embeddings

Two primitive cross-manifold maps are associated with a smooth embedding $\iota : N \hookrightarrow M$, where $n = \dim N \leq m = \dim M$.

The **embed map** carries a point from $N$ into its image in $M$:

$$
\iota : N \to \iota(N) \subset M, \qquad p \mapsto \iota(p).
$$

The **project map** is the left inverse of $\iota$ on the image:

$$
\pi : \iota(N) \subset M \to N, \qquad \pi \circ \iota = \mathrm{id}_N.
$$

Note that $\pi$ is only defined on the image $\iota(N)$, not on all of $M$: because $n < m$, the ambient manifold $M$ has directions transverse to $\iota(N)$, and there is no canonical way to project an arbitrary point of $M$ back to $N$. Together, embed and project are the two irreducible cross-manifold steps; all other coordinate changes on either side are ordinary transition maps within a single atlas.

!!! example

    The two-sphere $S^2$ embeds in $\mathbb{R}^3$ via the unit embedding. In the spherical chart $(\theta, \phi)$ on $S^2$ and the Cartesian chart $(x, y, z)$ on $\mathbb{R}^3$, the coordinate form of $\iota$ is

    $$
    \iota(\theta, \phi) = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta),
    $$

    and $\pi$ is the restriction of $(x,y,z) \mapsto (\arccos z,\ \arctan(y/x))$ to the unit sphere $x^2 + y^2 + z^2 = 1$.

#### Realization Maps

A **realization map** $\rho$ is the coordinate form of a full cross-manifold point transformation between an arbitrary chart $C_N$ on $N$ and an arbitrary chart $C_M$ on $M$. It is common to define the realization map as the composition of transition and embed/project maps, factoring through canonical charts $\bar{C}_N$ on $N$ and $\bar{C}_M$ on $M$ as three sequential steps:

$$
\varphi_{C_N}(U)
    \xrightarrow{\;\tau_N\;}
\varphi_{\bar{C}_N}(U)
    \xrightarrow{\;\iota \ \text{or} \ \pi\;}
\varphi_{\bar{C}_M}(V)
    \xrightarrow{\;\tau_M\;}
\varphi_{C_M}(V),
$$

where

- $\tau_{N \rightarrow \tilde{N}} = \varphi_{\bar{C}_N} \circ \varphi_{C_N}^{-1}$ is the transition map from $C_N$ to the canonical chart on $N$,
- $\iota = \varphi_{\bar{C}_M} \circ \iota \circ \varphi_{\bar{C}_N}^{-1}$ is the coordinate form of the embed map between the two canonical charts (or $\pi$ if a projection), and
- $\tau_{M \rightarrow \tilde{M}} = \varphi_{C_M} \circ \varphi_{\bar{C}_M}^{-1}$ is the transition map from the canonical chart on $M$ to $C_M$.

The full realization embed map is therefore

$$
\tau = \tau_{M \rightarrow \tilde{M}} \circ \iota \circ \tau_{N \rightarrow \tilde{N}}.
$$

The realization project map composes analogously, with $\iota$ replaced by the coordinate form of $\pi$.

### Product Manifolds and Product Charts

Given two smooth manifolds $M_1$ of dimension $n_1$ and $M_2$ of dimension $n_2$, their **product manifold** is the Cartesian product

$$
M = M_1 \times M_2,
$$

equipped with the **product smooth structure**: a point in $M$ is a pair $(p_1, p_2)$ with $p_1 \in M_1$ and $p_2 \in M_2$, and the product manifold has dimension $n_1 + n_2$. The two factors $M_1$ and $M_2$ retain full equal status; neither is a submanifold of the other in the sense of the embedding construction introduced in [Embedded Manifolds](#math-spec-embedded-manifolds).[^product-vs-embedding]

[^product-vs-embedding]: An embedding $\iota : N \hookrightarrow M$ places $N$ as a lower-dimensional submanifold inside a single ambient manifold $M$, with the factors having unequal roles. A product $M_1 \times M_2$ instead combines two manifolds symmetrically into a new manifold of strictly larger dimension in which each factor appears as a coordinate slice, not as a proper submanifold.

A **product chart** on $M_1 \times M_2$ is formed from a chart $C_1 = (U_1,
\varphi_1)$ on $M_1$ and a chart $C_2 = (U_2, \varphi_2)$ on $M_2$ by taking their Cartesian product:

$$
C_1 \times C_2 = \bigl(U_1 \times U_2,\; \varphi_1 \times \varphi_2\bigr),
$$

where the product coordinate map is

$$
(\varphi_1 \times \varphi_2)(p_1, p_2)
= \bigl(\varphi_1(p_1),\, \varphi_2(p_2)\bigr)
\in \mathbb{R}^{n_1} \times \mathbb{R}^{n_2} \cong \mathbb{R}^{n_1 + n_2}.
$$

The **product atlas** on $M_1 \times M_2$ is generated by all such product charts:

$$
\mathcal{A}_{M_1 \times M_2} = \{ C_1 \times C_2 \mid C_1 \in \mathcal{A}_1,\; C_2 \in \mathcal{A}_2 \}.
$$

The key structural property of product charts is that **transition maps on the product factor independently**. Given two product charts $C_1 \times C_2$ and $C_1' \times C_2'$, the transition map between them is exactly the product of the two factor transition maps:

$$
\tau_{(C_1 \times C_2) \to (C_1' \times C_2')}
= \tau_{C_1 \to C_1'} \times \tau_{C_2 \to C_2'}.
$$

In coordinates this means the first $n_1$ components transform by $\tau_{C_1 \to
C_1'}$ and the last $n_2$ components transform by $\tau_{C_2 \to C_2'}$, independently and without mixing. This factorwise law holds because the two factors are geometrically decoupled: a coordinate change on $M_1$ cannot affect coordinates on $M_2$, and vice versa.

!!! example

    _Spacetime._ Newtonian spacetime and the special-relativistic approximation both treat time and space as a product $\mathbb{R}_t \times \mathbb{R}^3$.

!!! example

    _Phase space._ The position-momentum phase space of a particle moving in $\mathbb{R}^3$ is the product manifold $\mathbb{R}^3_q \times \mathbb{R}^3_p$. A product chart in Cartesian coordinates on both factors has components $(x, y, z, p_x, p_y, p_z)$. Changing the position factor from Cartesian to spherical applies $\tau_{C \to S}$ to the first three components and leaves the momentum components unchanged.

</br>

(frame-transforms)=

### Frame Transforms

A **static frame transformation** is a smooth map

$$
F : M \to M
$$

that relates two descriptions of a point on the same manifold. The map $F : M \to M$ is smooth if it is infinitely differentiable in any chart (a major feature for auto-differentiation codes).

There are two ways of thinking about frame transformations:

- an **active transform**, where the transform moves a point, or
- a **passive transform**, where the point is unchanged but the reference frame in which the point is expressed.

These are mathematically equivalent but conceptually distinct perspectives.

_In `coordinax`, frame transforms are active_: operators act directly on points and move them on the manifold.

<!-- Frame Transformations: evolution-parameter-dependent -->

**_Evolution-Parameter-Dependent Frame Transformations_**:

Many physically important frame transformations are not fixed but vary with a smooth **evolution parameter** $\lambda \in \Lambda \subseteq \mathbb{R}$. An **evolution-parameter-dependent frame transformation** is a smooth map

$$
F : \Lambda \times M \to M, \qquad (\lambda, p) \mapsto F_\lambda(p),
$$

where:

- for each fixed $\lambda$, the map $F_\lambda : M \to M$ is a diffeomorphism, and
- $F$ is smooth **jointly** in $(\lambda, p)$ — not merely separately in each argument.

Joint smoothness is the natural condition for auto-differentiation through $\lambda$ (e.g., for computing velocities via `jax.grad`), and it is strictly stronger than smoothness in each argument separately.

The static case $F : M \to M$ is the **special case** in which $F_\lambda = F$ for all $\lambda$.

**Coordinate law for points.** In charts $C = (U, \varphi)$ and $C’ = (U’, \varphi’)$, the coordinate form of the transformation is

$$
q’ = \tau_\lambda(q), \qquad \tau_\lambda = \varphi_{C’} \circ F_\lambda \circ \varphi_{C}^{-1}.
$$

This is structurally identical to the static law — a point-to-point map — but the diffeomorphism $\tau_\lambda$ now depends on $\lambda$.

**Composition.** Given two evolution-parameter-dependent transformations $F$ and $G$ with the **same** parameter type,

$$
(G \circ F)_\lambda(p) = G_\lambda\!\bigl(F_\lambda(p)\bigr).
$$

Both sides evaluate at the same value of $\lambda$; composition does not advance or otherwise alter the parameter. Composing transformations that carry different parameter types requires an explicit reparameterization $\mu \mapsto \lambda(\mu)$ and is not defined automatically.

**Inverse.**

$$
(F^{-1})_\lambda = (F_\lambda)^{-1}.
$$

**Identity.** The identity transformation is trivially static:

$$
F_\lambda = \mathrm{id}_M \quad \forall\, \lambda.
$$

**Group interpretation.** For each fixed $\lambda$, $F_\lambda$ is an element of the relevant transformation group $G$ (e.g., $SO(3)$, $E(3)$). An evolution-parameter-dependent transformation traces a **smooth path** $\lambda \mapsto F_\lambda$ in $G$; it is not itself a group element, but it yields one at every $\lambda$.

</br>

---

<a id="math-spec-tangents"></a>

## Tangents

**_Tangent Vectors_**:

In a chart with local coordinates $q^1,\dots,q^n$, there is a corresponding set of tangent vectors

$$
\left\{ \frac{\partial}{\partial q^1}, \dots, \frac{\partial}{\partial q^n} \right\}_p.
$$

Geometrically, $\partial/\partial q^i|_p$ is the velocity of a coordinate curve $\gamma_i(t)$ obtained by varying only the $i$-th coordinate while holding the others fixed:

$$
\gamma_i(t)
= \varphi^{-1}(q^1,\dots,q^i+t,\dots,q^n),
\qquad
\frac{\partial}{\partial q^i}\bigg|_p
= \dot{\gamma}_i(0).
$$

Any tangent vector at $p$ can therefore be expanded as

$$
v = v^i \frac{\partial}{\partial q^i}\bigg|_p.
$$

Importantly, while we have defined these tangent vectors we do not yet have a way to define a relationship between the components $v^i$, such as whether they are orthogonal. To give geometric meaning to tangent directions, we must add extra structure to the manifold.

**_Tangent Spaces_**:

Let $M$ be a smooth $n$-dimensional manifold and $p \in M$. The **tangent space** at $p$ is the $n$-dimensional real vector space

$$ T_p M $$

whose elements are **tangent vectors** at $p$. Concretely, any smooth curve $\gamma : (-\varepsilon, \varepsilon) \to M$ with $\gamma(0) = p$ has a velocity $\dot\gamma(0) \in T_p M$; the tangent space is spanned by all such velocities. Equivalently, $T_p M$ is the best linear approximation to $M$ near $p$: it captures all instantaneous directions of motion through $p$.

Unlike $M$ itself, $T_p M$ is a genuine vector space: tangent vectors can be added and scaled.

### Metrics on Manifolds

A manifold _without_ a metric is a smooth manifold $M$. It has points, charts, and transition maps, but no notion of distance, angles, or orthogonality. For these, we need additional geometric structure: the **metric**. We can add the **metric**, to obtain a Riemannian manifold $(M, g)$.

A metric $g$ assigns to each point $p \in M$ a symmetric, non-degenerate bilinear form

$$
g_p : T_pM \times T_pM \to \mathbb{R},
$$

varying smoothly with p. In chart coordinates, the metric is represented by the matrix

$$
g_{ij}(q) = g\!\left(\frac{\partial}{\partial q^i},\frac{\partial}{\partial q^j}\right),
$$

evaluated at the base point $p$ with coordinates $q=\varphi(p)$.

Importantly, the metric acts only on tangent spaces; it does not act directly on points. Thus, it equips the manifold with intrinsic geometric meaning beyond smooth structure alone.

Specifically, the metric equips the manifold with notions of:

#### Length of tangent vectors

If

$$
v = v^i \frac{\partial}{\partial q^i}, \quad w = w^j \frac{\partial}{\partial q^j},
$$

then

$$
g_p(v,w) = g_{ij}(q)\, v^i w^j.
$$

In matrix notation,

$$
g_p(v,w) = v^\mathsf{T}\, g(q)\, w.
$$

The norm is then

$$
\|v\|_p^2 = g_p(v,v) = g_{ij}(q)\, v^i v^j.
$$

#### Angle between tangent vectors

The metric also defines the angle between nonzero tangent vectors:

$$
\cos\theta = \frac{g_p(u,v)}{\|u\|_p \, \|v\|_p}.
$$

#### Distance along curves

The metric matrix encodes the infinitesimal line element

$$
ds^2 = g_{ij}(q)\, dq^i dq^j.
$$

and curve length is obtained by integrating these infinitesimal lengths.

#### Geodesics

Geodesics are curves that locally extremize length with respect to the metric.

#### Volume elements

The determinant of the metric, $\det g$, encodes the induced volume element.

#### Raising and lowering indices

The metric identifies $T_p M$ with its dual space $T_p^* M$, enabling index raising and lowering.

#### Diagonal metrics and orthogonal coordinate systems

A metric is **diagonal** at a point $p$ when all off-diagonal entries of the metric matrix vanish:

$$
g_{ij}(p) = 0 \quad \text{for } i \neq j.
$$

Equivalently, the coordinate basis vectors $\partial/\partial q^i|_p$ are **mutually orthogonal** in the metric. A coordinate system whose metric is diagonal at every base point is called an **orthogonal coordinate system**.

**Scale factors.** For a diagonal metric, the diagonal entries $g_{ii}(p)$ are the _squared scale factors_:

$$
h_i(p)^2 = g_{ii}(p).
$$

The scale factors $h_i$ measure how a unit coordinate increment $dq^i$ stretches into actual arc length. The infinitesimal line element simplifies to

$$
ds^2 = \sum_i g_{ii}(q)\,(dq^i)^2 = \sum_i h_i(q)^2\,(dq^i)^2.
$$

The scale factors $h_i$ therefore control both the metric geometry and the relationship between coordinate components and physical (arc-length) components. For orthogonal coordinate systems they are the only information needed to pass between these two representations; for general (non-orthogonal) coordinate systems the analogous role is played by the **vielbein**, obtained via Cholesky factorization of $g$ — see [Physical Basis and Basis Conventions](#physical-basis-and-basis-conventions) below.

### Physical Basis and Basis Conventions

**Orthogonal coordinate systems.** The _coordinate_ basis vectors $\partial/\partial q^i$ are generally not unit vectors. For orthogonal coordinate systems (diagonal metric), they are mutually orthogonal. Normalizing them with the scale factors gives the **physical basis** (also called the **orthonormal frame**):

$$
\hat{e}_i = \frac{1}{h_i(p)}\frac{\partial}{\partial q^i}\bigg|_p.
$$

The components of a tangent vector $v$ in the two bases are related by a simple scaling (no sum on $i$):

$$
\hat{v}^i = h_i(p)\, v^i, \qquad v^i = h_i(p)^{-1}\,\hat{v}^i.
$$

For orthogonal coordinates, these transformations can be written in matrix form. Let $H(p) = \operatorname{diag}(h_1(p),\ldots,h_n(p))$ be the diagonal scale-factor matrix. Then:

$$
\boxed{
\hat{\mathbf{v}} = H(p)\,\mathbf{v}, \qquad \mathbf{v} = H(p)^{-1}\,\hat{\mathbf{v}}.
}
$$

The metric in physical components is the identity by construction: $g(\hat{v}, \hat{w}) = \sum_i \hat{v}^i \hat{w}^i$, whereas in coordinate components it reads $g(v, w) = h_i^2 v^i w^i$ (using diagonality).

!!! example

    _Cartesian coordinates in $\mathbb{R}^3$._ The metric is $g = dx^2 + dy^2 + dz^2$, so $h_x = h_y = h_z = 1$. The coordinate basis equals the physical basis ($\hat{e}_i = \partial_i$), and coordinate components equal physical components: no transformation is needed.

!!! example

    _Spherical coordinates in $\mathbb{R}^3$._ The metric is $g = dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2$, giving scale factors $h_r = 1$, $h_\theta = r$, $h_\phi = r\sin\theta$. The transformation from coordinate to physical components is:

    $$
    \begin{pmatrix} \hat{v}^r \\ \hat{v}^\theta \\ \hat{v}^\phi \end{pmatrix}
    = \begin{pmatrix}
    1 & 0 & 0 \\
    0 & r & 0 \\
    0 & 0 & r\sin\theta
    \end{pmatrix}
    \begin{pmatrix} v^r \\ v^\theta \\ v^\phi \end{pmatrix},
    \qquad
    \begin{pmatrix} v^r \\ v^\theta \\ v^\phi \end{pmatrix}
    = \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1/r & 0 \\
    0 & 0 & 1/(r\sin\theta)
    \end{pmatrix}
    \begin{pmatrix} \hat{v}^r \\ \hat{v}^\theta \\ \hat{v}^\phi \end{pmatrix}.
    $$

**General (non-orthogonal) coordinate systems.** The Cholesky decomposition $g = L\,L^\top$ yields the **vielbein** $E = L^\top$, a unique upper-triangular matrix satisfying:

$$
g = E^\top E.
$$

For any metric (diagonal or not), the vielbein provides the basis-change map to orthonormal-frame components:

$$
\boxed{
\mathbf{v}_{\rm frame} = E \,\mathbf{v}_{\rm coord}, \qquad
\mathbf{v}_{\rm coord} = E^{-1} \,\mathbf{v}_{\rm frame}.
}
$$

For a diagonal metric $g = \operatorname{diag}(h_1^2, \ldots, h_n^2)$, the Cholesky factor is $L = \operatorname{diag}(h_1, \ldots, h_n)$, so $E = H$ and the formula reduces to the orthogonal-coordinate scaling above. For non-diagonal metrics, $E$ is a full upper-triangular matrix, but the transformation remains $O(n^2)$ and numerically stable.

!!! example

    _Non-orthogonal 2-D coordinates._ Consider a 2-D metric $g = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$. The Cholesky factorization gives $L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{pmatrix}$, so $E = \begin{pmatrix} 2 & 1 \\ 0 & \sqrt{2} \end{pmatrix}$. For a coordinate vector $\mathbf{v}_{\rm coord} = (v^1, v^2)^\top$, the frame components are:

    $$
    \begin{pmatrix} v^1_{\rm frame} \\ v^2_{\rm frame} \end{pmatrix}
    = \begin{pmatrix} 2 & 1 \\ 0 & \sqrt{2} \end{pmatrix}
    \begin{pmatrix} v^1 \\ v^2 \end{pmatrix}
    = \begin{pmatrix} 2v^1 + v^2 \\ \sqrt{2}\, v^2 \end{pmatrix}.
    $$

**Basis changes under chart transitions.** When transforming coordinates under a chart transition $\tau : q \to \tilde{q}$ with Jacobian $J^j{}_i$, coordinate components transform as $\tilde{v}^j = J^j{}_i v^i$. The effect on orthonormal-frame components combines the Jacobian with the vielbeins at each chart:

$$
\mathbf{v}'_{\rm frame} = \tilde{E}\,J\,E^{-1}\,\mathbf{v}_{\rm frame} \equiv R\,\mathbf{v}_{\rm frame},
$$

where $R = \tilde{E}\,J\,E^{-1}$ is the **frame-change matrix**. For orthogonal coordinates, this simplifies to $R = \tilde{H}\,J\,H^{-1}$. When the transition is a rigid rotation (so $J \in O(n)$ and both charts are Cartesian), $R$ itself is orthogonal.

</br>

---

## Transformation Groups

Many useful frame transformations form **groups**[^group] under composition. A transformation group is a collection of maps

$$
F : M \to M
$$

closed under composition and inversion. These groups classify the kinds of coordinate and frame transformations that may appear in `coordinax`.

Different groups preserve different geometric structures (smooth structure, affine structure, metric structure, spacetime interval, etc.).

Transformation groups describe classes of maps acting on manifolds that preserve specific geometric structure.

Mathematically, a transformation group acting on a manifold $M$ is a group $G$ together with an action

$$
\tau : G \times M \to M
$$

satisfying

$$
\tau(e, p) = p, \qquad
\tau(g_1 g_2, p) = \tau(g_1, \tau(g_2, p)).
$$

Equivalently, each element $g \in G$ defines a map

$$
\tau_g : M \to M,
$$

and the assignment $g \mapsto \tau_g$ is a group homomorphism

$$
\rho : G \to \mathrm{Diff}(M).
$$

[^group]: A **group** is a set $G$ together with a binary operation $\cdot : G \times G \to G$ satisfying four axioms: (1) **Closure**: $a \cdot b \in G$ for all $a, b \in G$; (2) **Associativity**: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ for all $a, b, c \in G$; (3) **Identity**: there exists $e \in G$ such that $e \cdot a = a \cdot e = a$ for all $a \in G$; (4) **Inverses**: for each $a \in G$ there exists $a^{-1} \in G$ such that $a \cdot a^{-1} = a^{-1} \cdot a = e$.

```text
flowchart TD
    Id["Trivial group {e}"]

    Diff["Diffeomorphism group Diff(M)"]

    AffE["Affine group Aff(R^n) = GL(n) ⋉ R^n"]
    E["Euclidean group E(n) = O(n) ⋉ R^n"]
    O["Orthogonal group O(n)"]
    SO["Special orthogonal group SO(n)"]

    AffM["Affine group of Minkowski spacetime Aff(R^4) = GL(4) ⋉ R^4"]
    P["Poincare group IO(1,3) = O(1,3) ⋉ R^4"]
    Lor["Lorentz group O(1,3)"]
    SOLor["Proper orthochronous Lorentz group SO^+(1,3)"]

    Id --> SO
    Id --> SOLor

    SO --> O
    O --> E
    E --> AffE
    AffE --> Diff

    SOLor --> Lor
    Lor --> P
    P --> AffM
    AffM --> Diff
```

Every transformation group contains the **identity transformation**

$$
\mathrm{id}_M(p) = p
$$

which acts as the neutral element under composition.

### Trivial Group

The **trivial group** contains only the identity transformation:

$$
G = \{ e \}
$$

This group represents the absence of any transformation.

```{admonition} Examples
:class: dropdown

- identity frame transformation
```

### Diffeomorphism Group $\mathrm{Diff}(M)$

A **diffeomorphism** is a smooth, bijective map with a smooth inverse.

$$
F : M \to M
$$

Diffeomorphisms preserve the **smooth structure** of the manifold but need not preserve distances, angles, or straight lines.

The collection of all such maps forms the **diffeomorphism group**:

$$
\mathrm{Diff}(M).
$$

This is the largest transformation group normally considered in differential geometry.

```{admonition} Examples
:class: dropdown

- coordinate system changes
- nonlinear coordinate wraps
- accelerated coordinate systems
- Rindler coordinate transformations
- general smooth reparameterizations
- nonlinear warps
```

### Affine Group $\mathrm{Aff}(\mathbb{R}^n)$

If the manifold has an **affine structure**, transformations may preserve straight lines and parallelism. These transformations form the affine group.

An affine transformation has the form

$$
x \mapsto Ax + b
$$

where $A \in GL(n)$ and $b \in \mathbb{R}^n$.

Affine transformations preserve:

- straight lines
- parallelism
- ratios along lines

but do **not** necessarily preserve distances or angles.

```{admonition} Examples
:class: dropdown

- translations
- linear transformations
- scalings
- shears
- coordinate rescalings
```

### Special Orthogonal Group $SO(n)$

The **special orthogonal group** consists of rotations that preserve orientation and Euclidean distance.

$$
R^T R = I, \quad \det R = 1
$$

These transformations preserve:

- distances
- angles
- orientation

```{admonition} Examples
:class: dropdown

- spatial rotations
- rotation matrices
- rigid rotations of coordinate frames
```

### Orthogonal Group $O(n)$

The orthogonal group extends $SO(n)$ to include **reflections**.

$$
\det R = \pm 1
$$

Transformations preserve distances but may reverse orientation.

```{admonition} Examples
:class: dropdown

- reflections across planes
- mirror symmetry
- rotations combined with reflections
```

### Euclidean Group $E(n)$

The Euclidean group consists of all **distance-preserving transformations of Euclidean space**.

$$
E(n) = O(n) \ltimes \mathbb{R}^n
$$

It combines rotations, reflections, and translations.

These transformations preserve:

- distances
- angles
- rigid body structure

```{admonition} Examples
:class: dropdown

- translations
- rotations
- reflections
- rigid body motions
```

### Lorentz Group $O(1,3)$

In relativistic spacetime, coordinate transformations that preserve the Minkowski interval form the Lorentz group.

$$
s^2 = -c^2 t^2 + x^2 + y^2 + z^2
$$

Lorentz transformations satisfy

$$
\eta_{\alpha\beta}\Lambda^\alpha{}_\mu\Lambda^\beta{}_\nu = \eta_{\mu\nu}
$$

where $\eta$ is the Minkowski metric.

```{admonition} Examples
:class: dropdown

- Lorentz boosts
- spatial rotations in spacetime
- time reversal
- parity transformations
```

### Proper Orthochronous Lorentz Group $SO^+(1,3)$

The physically relevant subgroup of the Lorentz group excludes parity and time reversal.

This subgroup preserves:

- spacetime orientation
- direction of time

```{admonition} Examples
:class: dropdown

- Lorentz boosts
- spatial rotations
```

### Poincaré Group $IO(1,3)$

The **Poincaré group** extends the Lorentz group with spacetime translations.

$$
x'^\mu = \Lambda^\mu{}_{\nu} x^\nu + a^\mu
$$

It is the full group of **isometries of Minkowski spacetime**.

```{admonition} Examples
:class: dropdown

- spacetime translations
- Lorentz boosts
- spacetime rotations
- inertial frame transformations
```

### Minkowski Affine Group

The affine group of Minkowski spacetime consists of transformations

$$
x \mapsto Ax + b
$$

with $A \in GL(4)$.

These transformations preserve affine structure but not necessarily the Minkowski metric.

```{admonition} Examples
:class: dropdown

- linear spacetime transformations
- shears in spacetime coordinates
- coordinate scalings
```

</br>

---

# Packages

```{contents}
:local:
:depth: 1
```

## Main

This package collects all the most-commonly used parts of `coordinax` into a single convenient location.

It should be imported as `import coordinax.main as cx`

A non-exhaustive table of exported objects are:

| Package | Object |
| --- | --- |
| `coordinax.angles` | `AbstractAngle`, `Angle`, `wrap_to` |
| `coordinax.distances` | `AbstractDistance`, `Distance` |
| `coordinax.charts` | `CartesianProductChart`, </br> `cartesian_chart`, `guess_chart`, `cdict`, `pt_map`, `jac_pt_map`, </br> `cart0d`, </br> `cart1d`, `radial1d`, `time1d`, </br> `cart2d`, `polar2d`, </br> `cart3d`, `cyl3d`, `sph3d`, `lonlat_sph3d`, `loncoslat_sph3d`, `math_sph3d`, </br> `cartnd`, </br> `spacetimect` |
| `coordinax.representations` | `cconvert`, `change_basis`, `tangent_map`, </br> `Representation`, `point`, `coord_disp`, `coord_vel`, `coord_acc`, `phys_disp`, `phys_vel`, `phys_acc`, </br> `PointGeometry`, `point_geom`, `TangentGeometry`, `tangent_geom`, </br> `NoBasis`, `no_basis`, `CoordinateBasis`, `coord_basis`, `PhysicalBasis`, `phys_basis`, </br> `Location`, `loc`, `Displacement`, `dpl`, `Velocity`, `vel`, `Acceleration`, `acc`, </br> `guess_geometry_kind`, `guess_semantic_kind`, `guess_rep` |
| `coordinax.vectors` | `Point`, `ToUnitsOptions` |
| `coordinax.manifolds` | `guess_manifold`, `scale_factors`, `angle_between`, </br> `EuclideanManifold`, `Rn`, `EuclideanMetric`, `euclidean3d`, </br> `EmbeddedManifold`, `EmbeddedChart` </br> `twosphere`, `embedded_twosphere`, </br> `CustomManifold`,`CustomAtlas`, |
| `coordinax.transforms` | `act`, `simplify`, `compose`, `materialize_transform`, </br> `AbstractTransform`, `Identity`, `Composed`, `Translate`, `Rotate`, `Reflect`, `Scale`, `Shear`, `identity`, </br> `AbstractTransformGroup`, `IdentityGroup`, `DiffeomorphismGroup`, `AffineGroup`, `EuclideanGroup`, `OrthogonalGroup`, `SpecialOrthogonalGroup`, `PoincareGroup`, `LorentzGroup`, `ProperOrthochronousLorentzGroup` |
| `coordinax.frames` | `frame_transition`, </br> `AbstractReferenceFrame`, `FrameTransformError`, </br> `NoFrame`, `Alice`, `Alex`, `TransformedReferenceFrame` |

</br>

## Internal

The `coordinax.internal` module exposes **semi-public** utilities intended for downstream packages and advanced integrations.

These names are importable and supported for inter-package use, but they are **not** part of the stable top-level `coordinax` API contract. Minor and patch releases may change names, signatures, or behavior in this module without warning. Code that depends on `coordinax.internal` should pin an exact version.

Semi-public API:

- `QuantityMatrix`: heterogeneous 1-D or 2-D quantity container with a per-element unit structure
- `UnitsMatrix`: immutable, hashable wrapper around a numpy object array of `AbstractUnit` elements, aligned with a `QuantityMatrix`; supports tuple-style indexing, iteration, and `to_tuple()`/`to_string()`. **Not** a subclass of `astropy.StructuredUnit`; bidirectional converters to/from `astropy.StructuredUnit` live in `coordinax.interop.astropy`.
- `cdict_units`: extract per-component units from a coordinate dictionary
- `pack_uniform_unit`: stack component data into an array using a shared unit
- `pack_nonuniform_unit`: stack component data into an array while preserving per-component units

</br>

(software-spec-angles)=

## Angles

The `coordinax.angles` module provides the angle-facing scalar API used by charts, vectors, and frame transformations.

!!! info `AbstractAngle`

    Abstract base type for angular quantities. It defines the shared angle interface used by dispatch and typing.

    - `wrap_to` method that calls `coordinax.angles.wrap_to` on the `Angle`.

!!! info `Angle`

     Concrete angular scalar type (value + angular unit), built on `unxt`. Angles represent directions on $S^1$ and do not encode branch-cut convention in the type itself.

    - `wrap_to` method that calls `coordinax.angles.wrap_to` on the `Angle`.

!!! info `wrap_to`

    Functional API for explicit interval wrapping. It remaps an angle into a caller-specified interval (for example $[0, 2\pi)$ or $(-\pi, \pi]$).

</br>

(software-spec-distances)=

## Distances

The `coordinax.distances` module provides the distance-facing scalar API used by coordinate charts and astronomy-oriented conversions.

!!! info `AbstractDistance`

    Abstract base type for distance-like quantities. It defines the shared interface used by dispatch and typing.

    Properties:

    - `distance`: convert to a [`coordinax.distances.Distance`](#software-spec-distance) with length units. This is the canonical distance representation for all distance-like quantities.

(software-spec-distance)=

!!! info `Distance`

    Concrete length quantity type (value + length unit), built on `unxt`, for physical distances.

</br>

(software-spec-charts)=

## Charts

The `coordinax.charts` module provides the chart-facing API for representing points on manifolds with explicit component names, ordering, and physical dimensions, including chart construction, chart selection, and transition maps between compatible coordinate representations.

!!! info `AbstractChart`

    ``AbstractChart`` is the base chart interface: concrete charts are specializations of this type.

    At its core, a chart defines a component schema -- names and order, e.g. _r, theta, phi_ -- and the physical dimensions of each coordinate component -- e.g. _length, angle, angle_.

    Public API:

    - ``components``: a tuple of the component names.
    - ``coord_dimensions``: a tuple of the dimensions (as strings) for points.
    - ``ndim``: number of coordinate components (chart dimensionality).
    - ``cartesian``: the corresponding global Cartesian  chart. This can raise an error if there isn't a global Cartesian chart. Calls `coordinax.charts.cartesian_chart`.
    - ``check_data()``: check that the data is compatible with the chart. Keyword arguments are ``keys`` (default ``True``) to validate key schema, and ``values`` (default ``False``) to validate value dimensions/ranges.

    Notes:

    - Registered to JAX as static using `jax.tree_util.register_static`.
    - Equality is based on matching classes, components, coord_dimensions, and all field values (if any).
    - Implements a `wadler_lindig` ``__pdoc__()`` method which underpins ``__repr__`` and ``__str__`

!!! info `AbstractFixedComponentsChart`

    ``AbstractFixedComponentsChart`` is a subclass of `AbstractChart` that makes it much easier to define a chart with fixed components and dimensions that cannot be changed on chart instantiation.

    ``AbstractFixedComponentsChart`` parses the type annotations to build the ``components`` and ``coord_dimensions`` attributes.

    For example, for a 1-D chart with component ``x``:

    <!-- skip: next -->

    ```python
    class ChartExample(
        AbstractFixedComponentsChart[tuple[Literal["x"]], Literal["Length"]]
    ):
        pass
    ```

!!! info `NoGlobalCartesianChartError`

    Raised when a chart has no global Cartesian representation.

    Some charts represent coordinates on curved manifolds (e.g., 2-sphere) that cannot be globally mapped to a flat Cartesian space without singularities or discontinuities.

### Functional API

!!! info `cartesian_chart`

    Return the canonical Cartesian chart for a chart object.

    `cartesian_chart` is a pure chart-selection operation: it returns a chart object only, and does not transform coordinate data. This function underpins `AbstractChart.cartesian`.

    - For Euclidean standard charts, return the Cartesian chart of matching coordinate dimensionality.
    - Mapping: `Abstract0D -> cart0d`, `Abstract1D -> cart1d`, `Abstract2D -> cart2d`, `Abstract3D -> cart3d`, `AbstractND -> cartnd`.
    - Special-case `Time1D`: `Time1D -> time1d` (time is already canonical in this 1D chart family).
    - For `CartesianProductChart`, apply `cartesian_chart` factorwise and preserve `factor_names`.
    - For `SpaceTimeCT`, apply `cartesian_chart` to the spatial factor and keep the `ct` time factor convention.
    - The operation is idempotent: `cartesian_chart(cartesian_chart(C)) == cartesian_chart(C)`.

    Failure semantics:

    - For intrinsic 2-sphere charts (`AbstractSphericalTwoSphere`), there is no global Cartesian 2D chart. The function raises `NoGlobalCartesianChartError`.

!!! info `guess_chart`

    Infer a chart from lightweight structural information.

    `guess_chart` provides heuristic chart inference for common inputs where full type information is unavailable.

    Dispatches:

    - Input `frozenset[str]` (component names): infer a chart whose `components` set matches exactly. The results are cached; repeated calls with the same key set return the same chart object instance.
    - Input `CDict`/mapping: infer from `frozenset(obj.keys())`.
    - Input array-like or quantity-like with trailing axis length 1, 2, or 3: infer Cartesian chart by that trailing size.
    - Mapping for shaped inputs: trailing `...x1 -> cart1d`, `...x2 -> cart2d`, `...x3 -> cart3d`.

    Failure semantics:

    - If no chart matches a provided key set, raise `ValueError`.
    - Inputs outside registered dispatches (for example trailing axis sizes other than 1/2/3) are not inferred by this API.

    Notes:

    - There is a selection ambiguity. Key-based inference uses component names only. If multiple chart types share the same component names, the result is not uniquely identifiable from keys alone and one matching chart is returned.
    - Inferred chart choice from keys is stable within a process due to caching, but callers should not treat key-only inference as a unique chart identifier when component-name collisions exist.

!!! info `cdict`

    Extract a component dictionary from an object.

    `cdict` decomposes vectors and coordinate data into dictionaries where keys are component names (from a chart) and values are the corresponding scalars or arrays. This is a normalization operation: multiple input types map to a common `CDict` representation.

    Dispatches:

    - Input `CDict` (dict with string keys): returned unchanged (identity).
    - Input `unxt.AbstractQuantity` with last axis size 1, 2, or 3: call `guess_chart` on the quantity to infer chart dimensionality, then apply the chart-based dispatch.
    - Input array-like with chart context: split last axis into named components using `chart.components`. Last axis length MUST match the chart's component count.
    - Input `unxt.AbstractQuantity` with chart context: split last axis into named quantities using `chart.components`. Requires last axis size to match chart.
    - Input `QuantityMatrix` with chart context: extract heterogeneous per-component quantities, one for each chart component.

    Failure semantics:

    - If last axis size does not match the number of chart components, raise `ValueError`.
    - If input type has no registered dispatch, `plum.NotFoundLookupError` is raised.

    Notes:

    - When no chart is explicitly provided, `guess_chart` is used to infer Cartesian chart dimensionality from array/quantity shape. This works for arrays/quantities with trailing axis size 1, 2, or 3 only.
    - Units are preserved: a quantity input returns a `CDict` of quantities.

(software-spec-pt_map)=

!!! info `pt_map`

    Transform point coordinates from one chart to another.

    This function implements the most general point-coordinate map between two compatible chart representations of the same geometric point. It is a point-wise map that preserves the physical location while changing the coordinate description.

    `pt_map` handles both same-manifold chart transitions (the ordinary chart transition map between two charts in the same atlas) and cross-manifold **realization-style** maps (between charts attached to different manifolds when one is a realization of the other, such as an intrinsic chart on an embedded manifold and a chart on its ambient manifold).

    Dispatches:

    - Input ``(p, from_chart, to_chart, usys=None)``: Transform coordinates in `p` from ``from_chart`` to ``to_chart``, representing the same underlying geometric point under a different coordinate representation.
    - Input (from_chart, to_chart, **fixed_kwargs): Partial application returning a callable that accepts `(p, *args, **kw)` and forwards to `pt_map(p, from_chart, to_chart, **fixed_kwargs, **kw)`.
    - Identity transformation: When `to_chart` and `from_chart` are the same chart type (e.g., both `Cart3D`), return `p` unchanged. The output is identical to the input dict.
    - Specific dispatches: Direct transformations exist for many chart pairs on the same manifold (e.g., `Cart3D` ↔ `Cylindrical3D`, `Cart3D` ↔ `Spherical3D`, `Polar2D` ↔ `Cart2D`, `Cart1D` ↔ `Radial1D`).
    - Fallback via Cartesian: If no specific dispatch exists for (from_chart, to_chart), use a two-step intermediate: ``pt_map(p, from_chart, to_chart.cartesian, usys=usys)`` followed by ``pt_map(p_cart, from_chart.cartesian, to_chart, usys=usys)``.
    - Product charts (factorwise): For `AbstractCartesianProductChart` ↔ `AbstractCartesianProductChart`, apply `pt_map` independently to each factor via `split_components` and `merge_components`.
    - Cross-manifold dispatches: Additional dispatches can define realization-style mappings between charts on different manifolds (e.g., an intrinsic chart on an embedded manifold to a chart on the ambient manifold).
    - The `usys` parameter (unit system) is used to interpret non-quantity inputs (e.g., bare floats) in appropriate units. It is optional for unit-ful quantities and required otherwise.

    Failure semantics:

    - If `to_chart` and `from_chart` do not have a dispatch path.
    - If `to_chart` or `from_chart` are product charts but the other is not.
    - If product charts have mismatched factor counts, raise `TypeError`.
    - For non-Quantity inputs without `usys` provided when required for interpretation.

    Notes:

    - This is a position-only transformation.
    - The partial-application dispatch (returning a callable) is useful for currying: `transform_func = pt_map(cart3d, sph3d); result = transform_func(p)`.
    - Product charts (like SpaceTimeCT with a spatial factor) transform by independently transforming each factor's coordinates.
    - Same-atlas chart transitions and cross-manifold realization maps are unified under one function; dispatch resolution selects the appropriate implementation based on the chart types.

(software-spec-tangent-map)=

!!! info `jac_pt_map`

    Compute the Jacobian of the chart transition map at a base point.

    **Dispatches:**

    - `(at: None, /, *fixed_args, usys: AbstractUnitSystem, **fixed_kw)` -> partial
      application. Returns a callable that accepts `(at, *args, **kw)` and forwards to
      `jac_pt_map(at, *fixed_args, *args, **fixed_kw, **kw)`. Requires `usys`.

    - `(from_chart, to_chart, /, *, usys: AbstractUnitSystem)` -> curried partial
      application. Returns `lambda at: jac_pt_map(at, from_chart, to_chart,
      usys=usys)`. Requires `usys`. Useful for repeated Jacobian evaluation at multiple
      base points with a fixed unit system.

    - `(at: Array, from_chart, to_chart, /, *, usys: AbstractUnitSystem)` -> plain
      `Array`. Treats `at` as a flat numeric array (no units), computes
      `jax.jacfwd(pt_map(None, from_chart, to_chart, usys=usys))(at)`, and returns the
      raw array Jacobian. Requires `usys`.

    - `(at: CDict, from_chart, to_chart, /, *, usys: OptUSys = None)` ->
      `Array | QuantityMatrix`. The general dict dispatch. Branches on whether `at`
      values are plain arrays or quantities:

      - **Array-valued** (`is_array=True`): stacks `at` into a plain array via
        `jnp.stack`, then forwards to the `(at: Array, ...)` dispatch. `usys` is
        forwarded and must be an `AbstractUnitSystem` unless a more-specific analytical
        dispatch handles it.

      - **Quantity-valued** (`is_array=False`): packs `at` into a 1-D `QuantityMatrix`
        via `pack_to_qmatrix(at, keys=from_chart.components)`, casts to `float`, then
        computes `J_qq = jax.jacfwd(pt_map_fn)(at_in)`. The jacfwd result is a
        `QuantityMatrix` whose `.value` is itself a `QuantityMatrix` encoding the input
        units, and whose `.unit` encodes the output units. `_repack_q_from_jac` extracts
        both to build the correct 2-D `UnitsMatrix` and returns
        `QuantityMatrix(J_arr, unit=unit_matrix)` of shape `(n_out, n_in)`.

    **`usys` parameter:** required for the `None`-partial, curried, and plain-`Array`
    dispatches. Optional (`None`) for the `CDict` generic dispatch's quantity-valued
    branch, and for all analytical dispatches.

    **Analytical dispatches** (higher precedence than the generic `CDict` fallback;
    `usys` is optional):

    - `Cart2D -> Polar2D`: `Array`, `AbstractQuantity`, and `CDict` overloads.
      The `AbstractQuantity` overload computes closed-form partial derivatives and
      explicitly sets the ∂θ row units to `rad / input_length_unit` (astropy treats rad
      as dimensionless, so the unit must be forced manually).
    - Further analytical pairs (`Polar2D -> Cart2D`, `Cart3D ↔ Sph3D`,
      `Cart3D ↔ Cyl3D`) follow the same pattern via the generic `CDict` dispatch.

    **Failure semantics:**

    - Raises `plum.NotFoundLookupError` when calling the `(at: CDict, ...)` dispatch
      with `is_array=True` values and `usys=None` for a chart pair that has no analytical
      `Array` dispatch (e.g., `cart3d -> sph3d` without a unit system).
    - Raises `ValueError` if `at` keys do not match `from_chart.components` (via
      `check_data`).

### Dimensional Flags

!!! info `AbstractDimensionalFlag`

    ``AbstractDimensionalFlag`` is a mixin used to mark chart dimensionality (for example 0-D, 1-D, 2-D, 3-D, 6-D, and N-D categories).

    - These classes cannot be instantiated and MUST be combined with concrete subclasses of `AbstractChart`.
    - Registered to JAX as static using `jax.tree_util.register_static`.

!!! info `DIMENSIONAL_FLAGS`

    ``DIMENSIONAL_FLAGS`` is the registry of dimensional flags, populated as subclasses of `AbstractDimensionalFlag` are created.

    Subclasses can pass a class-level kwarg registering the dimensionality. For example:

    <!-- skip: next -->

    ```python
    class Abstract0D(AbstractDimensionalFlag, n=0):
        pass
    ```

!!! info `Abstract0D`

    Marker base class for 0-dimensional charts.

    - Semantics: represents coordinate systems with zero components (no coordinate keys).
    - Dimensional flag: `n = 0` only; variable `n` is not supported.
    - Subclass constraint: subclasses must also be chart subclasses; otherwise `TypeError`.
    - If a subclass tries to set `n` explicitly, raise `NotImplementedError`.

!!! info `Abstract1D`

    Marker base class for 1-dimensional charts.

    - Semantics: represents coordinate systems with one component (one coordinate keys).
    - Dimensional flag: `n = 1` only; variable `n` is not supported.
    - Subclass constraint: subclasses must also be chart subclasses; otherwise `TypeError`.
    - If a subclass tries to set `n` explicitly, raise `NotImplementedError`.

!!! info `Abstract2D`

    Abstract marker for 2-dimensional charts.

    - `Abstract2D` extends `AbstractDimensionalFlag` with `n=2`.
    - Concrete 2-D charts should subclass `Abstract2D`.
    - Does not imply flatness: e.g. the two-sphere uses two angular coordinates on a curved manifold.

!!! info `Abstract3D`

    Abstract marker for 3-dimensional charts.

    - `Abstract3D` extends `AbstractDimensionalFlag` with `n=3`.
    - Concrete 3-D charts should subclass `Abstract3D`.

!!! info `Abstract6D`

    Abstract marker for 6-dimensional charts.

    - `Abstract6D` extends `AbstractDimensionalFlag` with `n=6`.
    - Concrete 6-D charts should subclass `Abstract6D`.

!!! info `AbstractND`

    Abstract marker for N-dimensional charts.

    - `AbstractND` extends `AbstractDimensionalFlag` with `n=6`.
    - Concrete N-D charts should subclass `AbstractND`.

### Standard Euclidean Charts

#### 0-D Charts

!!! info `Cart0D` and `cart0d`

    Canonical concrete 0-D Cartesian chart and its instance.

    - `Cart0D` is the final concrete chart type for 0-dimensional coordinates.
    - It has no components and no component dimensions (`components=()`, `dimensions=()`).
    - `cart0d` is its pre-defined instance.
    - As the canonical 0D Cartesian chart, it is the target of `cartesian_chart(Abstract0D)`.

#### 1-D Charts

!!! info `Cart1D` and `cart1d`

    Canonical concrete 1-D Cartesian chart and its instance.

    - `Cart1D` is the final concrete chart type for 1-dimensional Cartesian coordinates.
    - Components: `("x",)` with dimension `("length",)`.
    - `cart1d` is its pre-defined instance.
    - As the canonical 1D Cartesian chart, it is the target of `cartesian_chart(Abstract1D)`.

!!! info `Radial1D` and `radial1d`

    Concrete 1-D radial chart and its instance.

    - `Radial1D` is the final concrete chart type for 1-dimensional radial coordinates.
    - Components: `("r",)` with dimension `("length",)`.
    - `radial1d` is its pre-defined instance.
    - Semantically equivalent to `Cart1D` but uses `r` instead of `x`; transition between the two is a pure component rename (`r ↔ x`).
    - Cartesian projection is canonical: `cartesian_chart(Radial1D) -> cart1d`.

!!! info `Time1D` and `time1d`

    Concrete 1-D time chart and its pre-defined instance.

    - `Time1D` is the final concrete chart type for 1-dimensional time coordinates.
    - Components: `("t",)` with dimension `("time",)`.
    - `time1d` is its pre-defined `Time1D()` instance.
    - `Time1D` is its own Cartesian chart: `cartesian_chart(Time1D) -> time1d`.

#### 2-D Charts

!!! info `Cart2D` and `cart2d`

    Canonical concrete 2-D Cartesian chart and its pre-defined instance.

    - `Cart2D` is the final concrete chart type for 2-dimensional Cartesian coordinates.
    - Components: `("x", "y")` with dimensions `("length", "length")`.
    - `cart2d` is its pre-defined `Cart2D()` instance.
    - As the canonical 2-D Cartesian chart, it is the target of `cartesian_chart(Abstract2D)`.

!!! info `Polar2D` and `polar2d`

    Concrete 2-D polar chart and its pre-defined instance.

    - `Polar2D` is the final concrete chart type for 2-dimensional polar coordinates.
    - Components: `("r", "theta")` with dimensions `("length", "angle")`.
    - `polar2d` is its pre-defined `Polar2D()` instance.
    - Direct transitions registered: `Polar2D ↔ Cart2D` (via `r`, `theta` ↔ `x`, `y` trig formulas).
    - Cartesian projection is canonical: `cartesian_chart(Polar2D) -> cart2d`.

#### 3-D Charts

!!! info `Cart3D` and `cart3d`

    Canonical concrete 3-D Cartesian chart and its pre-defined instance.

    - `Cart3D` is the final concrete chart type for 3-dimensional Cartesian coordinates.
    - Components: `("x", "y", "z")` with dimensions `("length", "length", "length")`.
    - `cart3d` is its pre-defined `Cart3D(M=Rn(3))` instance.
    - As the canonical 3-D Cartesian chart, it is the target of `cartesian_chart(Abstract3D)`.

!!! info `Cylindrical3D` and `cyl3d`

    Concrete 3-D cylindrical chart and its pre-defined instance.

    - `Cylindrical3D` is the final concrete chart type for cylindrical coordinates.
    - Components: `("rho", "phi", "z")` with dimensions `("length", "angle", "length")`.
    - `cyl3d` is its pre-defined `Cylindrical3D()` instance.
    - Direct transitions registered: `Cylindrical3D ↔ Cart3D`.
    - Cartesian projection is canonical: `cartesian_chart(Cylindrical3D) -> cart3d`.

!!! info `AbstractSpherical3D`

    Abstract chart for 3-D spherical-family charts.

    - `AbstractSpherical3D` is a non-final intermediate base for all 3-D spherical charts.
    - Concrete spherical charts (`Spherical3D`, `LonLatSpherical3D`, `LonCosLatSpherical3D`, `MathSpherical3D`, `ProlateSpheroidal3D`) all subclass this.
    - Shares the `Abstract3D` Cartesian projection: `cartesian_chart(AbstractSpherical3D) -> cart3d`.

!!! info `Spherical3D` and `sph3d`

    Physics-convention 3-D spherical chart and its pre-defined instance.

    - `Spherical3D` is the final concrete chart type for physics-convention spherical coordinates $(r, \theta, \phi)$.
    - Components: `("r", "theta", "phi")` with dimensions `("length", "angle", "angle")`.
    - `theta` is the polar (colatitude) angle, $\theta \in [0, \pi]$; `phi` is the azimuth.
    - Embedding: $x = r\sin\theta\cos\phi$, $y = r\sin\theta\sin\phi$, $z = r\cos\theta$.
    - `check_data` enforces `theta` in valid polar range.
    - `sph3d` is its pre-defined `Spherical3D()` instance.

!!! info `LonLatSpherical3D` and `lonlat_sph3d`

    Longitude-latitude spherical chart and its pre-defined instance.

    - `LonLatSpherical3D` is the final concrete chart type for lon/lat/distance coordinates.
    - Components: `("lon", "lat", "distance")` with dimensions `("angle", "angle", "length")`.
    - `lat` is the latitude in $[-\pi/2, \pi/2]$; `lon` is the azimuth.
    - Relation to `Spherical3D`: $\mathrm{lat} = \pi/2 - \theta$, $\mathrm{lon} = \phi$.
    - `check_data` enforces `lat` in $[-90°, 90°]$.
    - `lonlat_sph3d` is its pre-defined `LonLatSpherical3D()` instance.

!!! info `LonCosLatSpherical3D` and `loncoslat_sph3d`

    Longitude-cos(latitude) spherical chart and its pre-defined instance.

    - `LonCosLatSpherical3D` is the final concrete chart type for $(\mathrm{lon}\cos\mathrm{lat}, \mathrm{lat}, r)$ coordinates.
    - Components: `("lon_coslat", "lat", "distance")` with dimensions `("angle", "angle", "length")`.
    - The `lon_coslat` component improves numerical behaviour near the poles where $\cos(\mathrm{lat}) \to 0$.
    - `check_data` enforces `lat` in $[-90°, 90°]$.
    - `loncoslat_sph3d` is its pre-defined `LonCosLatSpherical3D()` instance.

!!! info `MathSpherical3D` and `math_sph3d`

    Math-convention 3-D spherical chart and its pre-defined instance.

    - `MathSpherical3D` is the final concrete chart type for math-convention spherical coordinates $(r, \theta, \phi)$.
    - Components: `("r", "theta", "phi")` with dimensions `("length", "angle", "angle")`.
    - In this convention `phi` is the polar angle (from $z$-axis, $\phi \in [0°, 180°]$) and `theta` is the azimuth — the reverse of `Spherical3D`.
    - `check_data` enforces `phi` in $[0°, 180°]$.
    - `math_sph3d` is its pre-defined `MathSpherical3D()` instance.

!!! info `ProlateSpheroidal3D`

    Prolate spheroidal chart with focal parameter.

    - `ProlateSpheroidal3D` is the final concrete chart type for prolate spheroidal coordinates $(\mu, \nu, \phi)$.
    - Components: `("mu", "nu", "phi")` with dimensions `("area", "area", "angle")`.
    - Carries a required field `Delta` (focal half-length, a static `Quantity["length"]` with `Delta > 0`).
    - Validity constraints: $\mu \ge \Delta^2$, $|\nu| \le \Delta^2$.
    - Unlike many other charts, `ProlateSpheroidal3D` instances are not interchangeable: two instances with different `Delta` are distinct charts.
    - No pre-defined instance (requires `Delta` parameter).
    - Cartesian projection is canonical: `cartesian_chart(ProlateSpheroidal3D) -> cart3d`.

#### N-D Charts

!!! info `CartND` and `cartnd`

    Canonical concrete N-D Cartesian chart and its pre-defined instance.

    - `CartND` is the final concrete chart type for Cartesian coordinates in arbitrary dimension.
    - Components: `("q",)` with dimension `("length",)`, where `q` stores the N Cartesian components as a single length-valued array.
    - `cartnd` is its pre-defined `CartND()` instance.
    - `CartND` is already Cartesian: `cartesian_chart(CartND) -> cartnd`.

### Cartesian Product Charts

!!! info `AbstractCartesianProductChart`

    Abstract base class for Cartesian product charts.

    A Cartesian product chart represents a finite ordered tuple of factor charts: $$ M = \prod_i M_i $$ with factorwise coordinate transformation laws.

    Public API:

    - `factors: tuple[AbstractChart, ...]`: ordered factor charts.
    - `factor_names: tuple[str, ...]`: ordered factor names aligned with `factors` (same length; unique in concrete implementations).
    - `ndim`: total dimension, computed as `sum(f.ndim for f in factors)`.
    - `components` (default behavior): dot-delimited namespaced keys `("name0.c0", ..., "nameN.cK")`.
      - Flat-key product charts are an explicit specialization (`AbstractFlatCartesianProductChart`), used by `SpaceTimeCT`.
    - `coord_dimensions`: concatenation of factor coordinate dimensions in factor order.
    - `split_components(p)`: partitions a `CDict` into one per factor, stripping factor prefixes.
    - `merge_components(parts)`: re-attaches prefixes and merges factor dictionaries.

    Transform semantics:

    - Registered `pt_map` between two product charts is factorwise: split input by factors, transform each factor pair, then merge.
    - If factor counts differ, `pt_map` raises `TypeError`.
    - No automatic product↔non-product transform is defined; those paths raise `NotImplementedError` unless explicit dispatches are added.

!!! info `CartesianProductChart`

    Concrete namespaced Cartesian product chart.

    `CartesianProductChart` is the final implementation of
    `AbstractCartesianProductChart` for general products using dot-delimited
    keys.

    Construction:

    - Parameters:
      - `factors`: tuple of factor charts.
      - `factor_names`: tuple of names for each factor.
    - Post-init validation:
      - `len(factors) == len(factor_names)`, otherwise `ValueError`.
      - `factor_names` are unique, otherwise `ValueError`.

    Component model:

    - Keys are namespaced by factor name:
      e.g. factors `(cart3d, cart3d)` and names `("q", "p")` produce `("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")`.
    - This avoids key collisions for repeated factor chart types.

    Cartesian projection:

    - `cartesian_chart(CartesianProductChart(...))` is applied factorwise.
    - Returns the same object if every factor is already Cartesian.
    - Otherwise returns a new `CartesianProductChart` with transformed factors and unchanged `factor_names`.

    Transformation behavior:

    - Inherits factorwise `pt_map` behavior from `AbstractCartesianProductChart` dispatches.

!!! info `SpaceTimeCT` and `spacetimect`

    Concrete flat-key spacetime product chart with Minkowski-oriented convention.

    `SpaceTimeCT` is a final `AbstractFlatCartesianProductChart` representing:
    $$
    \text{SpaceTimeCT(spatial\_chart)} \equiv \text{time1d} \times \text{spatial\_chart}
    $$
    with components `(ct, *spatial_components)`.

    API:

    - `spatial_chart`: spatial factor chart (default `cart3d`).
    - `c`: speed of light parameter (default `299792.458 km/s`).
    - `factors` is always `(time1d, spatial_chart)`.
    - `factor_names` is always `("time", "space")`.
    - `time_chart` is always `time1d` (not user-selectable).
    - `split_components(p)` returns `({"ct": ...}, {spatial keys...})`.
    - `merge_components(parts)` merges those two dicts directly.

    Component and dimension convention:

    - Overrides default product keys to use **flat** keys:
      - components: `("ct", *spatial_chart.components)`
      - dimensions: `("length", *spatial_chart.coord_dimensions)`
    - `ct` is the chart-level time coordinate key used by this representation.

    Cartesian projection:

    - `cartesian_chart(SpaceTimeCT(...))` maps only the spatial factor to its Cartesian chart.
    - Returns the same object when spatial factor is already Cartesian; otherwise returns a replaced `SpaceTimeCT`.

    Transformation behavior:

    - Uses the generic product-chart factorwise `pt_map` dispatch.
    - In practice, this preserves `ct` while transforming only spatial components when changing the spatial chart.

### Two-Sphere Charts

!!! info `AbstractSphericalTwoSphere`

    Abstract base class for intrinsic 2-sphere charts.

    - `AbstractSphericalTwoSphere` subclasses `Abstract2D` and represents charts on the curved unit sphere surface.
    - Components are purely angular (no radial component); these charts parameterize points on \(S^2\), not \(\mathbb{R}^2\).
    - There is **no global Cartesian 2D chart** for this manifold family.
    - `cartesian_chart(AbstractSphericalTwoSphere)` raises `NoGlobalCartesianChartError`.
    - For Euclidean embedding behavior, use a 3D embedding chart workflow (not intrinsic 2-sphere Cartesianization).

!!! info `SphericalTwoSphere` and `sph2`

    Physics-convention intrinsic 2-sphere chart and its pre-defined instance.

    - `SphericalTwoSphere` is the final chart type with components `("theta", "phi")`, dimensions `("angle", "angle")`.
    - Convention: `theta` is polar/colatitude \([0,\pi]\), `phi` is azimuth.
    - `check_data` enforces valid polar range for `theta` (`0°..180°`).
    - `sph2` is the pre-defined `SphericalTwoSphere()` instance.
    - Registered direct transitions:
      - `SphericalTwoSphere <-> LonLatSphericalTwoSphere`
      - `SphericalTwoSphere <-> LonCosLatSphericalTwoSphere`
      - `SphericalTwoSphere <-> MathSphericalTwoSphere`

!!! info `LonLatSphericalTwoSphere` and `lonlat_sph2`

    Longitude-latitude intrinsic 2-sphere chart and its pre-defined instance.

    - `LonLatSphericalTwoSphere` is the final chart type with components `("lon", "lat")`, dimensions `("angle", "angle")`.
    - `lat` is constrained to \([-90^\circ, 90^\circ]\); `lon` is azimuth/longitude.
    - `check_data` enforces latitude range.
    - `lonlat_sph2` is the pre-defined `LonLatSphericalTwoSphere()` instance.
    - Relation to physics spherical chart:
      - `lat = pi/2 - theta`
      - `lon = phi`
    - Registered direct transitions with `SphericalTwoSphere` implement these formulas.

!!! info `LonCosLatSphericalTwoSphere` and `loncoslat_sph2`

    Longitude-cos(latitude) intrinsic 2-sphere chart and its pre-defined instance.

    - `LonCosLatSphericalTwoSphere` is the final chart type with components `("lon_coslat", "lat")`, dimensions `("angle", "angle")`.
    - `lat` is constrained to \([-90^\circ, 90^\circ]\).
    - `lon_coslat` stores longitude weighted by `cos(lat)` to improve behavior near poles.
    - `check_data` enforces latitude range.
    - `loncoslat_sph2` is the pre-defined `LonCosLatSphericalTwoSphere()` instance.
    - Registered direct transitions with `SphericalTwoSphere`:
      - forward: `lat = pi/2 - theta`, `lon_coslat = phi * cos(lat)`
      - inverse: `theta = pi/2 - lat`, `phi = lon_coslat / cos(lat)`

!!! info `MathSphericalTwoSphere` and `math_sph2`

    Math-convention intrinsic 2-sphere chart and its pre-defined instance.

    - `MathSphericalTwoSphere` is the final chart type with components `("theta", "phi")`, dimensions `("angle", "angle")`.
    - Convention differs from physics spherical chart:
      - `theta` is azimuth
      - `phi` is polar angle
    - `check_data` enforces polar range for `phi` (`0°..180°`).
    - `math_sph2` is the pre-defined `MathSphericalTwoSphere()` instance.
    - Registered direct transitions with `SphericalTwoSphere` are component swaps:
      - physics -> math: `theta <- phi`, `phi <- theta`
      - math -> physics: `theta <- phi`, `phi <- theta`

### Poincare Charts

!!! info `PoincarePolar6D` and `poincarepolar6d`

    Concrete 6-D Poincare-polar chart and its pre-defined instance.

    - `PoincarePolar6D` is a final concrete chart type in the 6-D chart family.
    - Components (ordered):
      `("rho", "pp_phi", "z", "dt_rho", "dt_pp_phi", "dt_z")`.
    - Coordinate dimensions (ordered):
      `("length", "length / time**0.5", "length", "speed", "length / time**1.5", "speed")`.
    - `poincarepolar6d` is the pre-defined `PoincarePolar6D()` instance.
    - Transition behavior currently registered in-core:
      identity transform only (`pt_map(p, PoincarePolar6D, PoincarePolar6D) -> p`).
    - No dedicated Cartesian projection dispatch is currently defined for this chart family.

</br>

(software-spec-representations)=

## Representations

Building off transition maps and the other transformation laws we introduce a generalization over transformation laws that streamlines software implementation.

A `Representation` specifies _what kind of geometric object_ component data is meant to represent, independently of _which chart_ is used to write down the underlying coordinates. In `coordinax`, a representation is structured from three pieces:

1. a **geometry kind** ([`AbstractGeometry`](#software-spec-abstractgeometry)), which identifies the geometric type of the object (for example, a point, or tangent or cotangent object),
2. a **basis** ([`AbstractBasis`](#software-spec-abstractsbasis)), which identifies the basis in which components are expressed when such a choice is meaningful, and
3. a **semantic kind** ([`AbstractSemanticKind`](#software-spec-abstractsemantickind)), which identifies the physical or mathematical meaning attached to that geometric type. Formally, one may view a representation as a triple

$$
R = (K, B, S),
$$

where $K$ is the geometry kind, $B$ is the basis choice, and $S$ is the semantic kind.

A representation is therefore **not** the same thing as a chart: the chart determines the coordinate system, while the representation determines the transformation law and geometric interpretation of the data.

!!! example

    For **points**, the representation determines that the transformation law is a [transition map](#math-spec-transition-maps).

!!! info `Representation`

    ``Representation``.

    Arguments:

    - ``geom_kind``: ``AbstractGometry``
    - ``basis``: ``AbstractBasis``
    - ``semantic_kind``: ``AbstractSemanticKind``

    <!-- skip: next -->

    ```python
    @jax.tree_util.register_static
    @dataclass
    class Representation:
        geom_kind: AbstractGometry
        basis: AbstractBasis
        semantic_kind: AbstractSemanticKind
    ```

(software-spec-instances)=

!!! info Pre-defined Representations

    `Representation` instances for standard use cases.

    |     Name     |   `geom_kind`  |    `basis`    | `semantic_kind` |
    |--------------|----------------|---------------|-----------------|
    | `point`      | [`point_geom`](#software-spec-point-geometry)   | [`no_basis`](#software-spec-no_basis)    |      [`loc`](#software-spec-location)      |
    | `coord_disp` | [`tangent_geom`](#software-spec-tangent-geometry) | [`coord_basis`](#software-spec-coordinatebasis) |      [`dpl`](#software-spec-displacement)      |
    | `coord_vel`  | [`tangent_geom`](#software-spec-tangent-geometry) | [`coord_basis`](#software-spec-coordinatebasis) |      [`vel`](#software-spec-velocity)      |
    | `coord_acc`  | [`tangent_geom`](#software-spec-tangent-geometry) | [`coord_basis`](#software-spec-coordinatebasis) |      [`acc`](#software-spec-acceleration)      |
    | `phys_disp`  | [`tangent_geom`](#software-spec-tangent-geometry) | [`phys_basis`](#software-spec-physicalbasis)  |      [`dpl`](#software-spec-displacement)      |
    | `phys_vel`   | [`tangent_geom`](#software-spec-tangent-geometry) | [`phys_basis`](#software-spec-physicalbasis)  |      [`vel`](#software-spec-velocity)      |
    | `phys_acc`   | [`tangent_geom`](#software-spec-tangent-geometry) | [`phys_basis`](#software-spec-physicalbasis)  |      [`acc`](#software-spec-acceleration)      |

</br>

### Functional API

!!! info `cconvert`

    `cconvert` is the highest-level part of `coordinax`'s coordinate transformation machinery.
    It abstracts over all the specific transformation machinery:

    - [`coordinax.charts.pt_map`](#software-spec-pt_map)

    Consequently, it uses (/needs) more structure to distinguish between transformations.

    Registered Dispatches (with $f$ final and $i$ initial):

    - $(C_{f}, R_{f}, C_{i}, R_{i})$ -> $(C_{f}, K_{f}, R_{f}, C_{i}, K_{i}, R_{i})$ general redispatch to use the geometric type in the transformation selection.
    - $(C_{f}, \text{PointGeometry}, R_{f}, C_{i}, \text{PointGeometry}, R_{i})$ redispatch to use the transition map $\varphi_{C_{f}} \circ \varphi_{C_{i}}^{-1}$. $R_{i,f}$ are checked that [B=NoBasis()](#software-spec-no_basis), [S=Location()](#software-spec-location)

(software-spec-guess_rep)=

!!! info `guess_rep`

    Infer the full representation triple from lightweight structural information.

    Dispatches:

    - `Representation` -> identity.
    - `PointGeometry` -> returns `point`.
    - `u.AbstractDimension | u.AbstractQuantity | CDict` -> infer via `guess_geometry_kind`, then redispatch on geometry kind.
    - `(Any, AbstractChart)` -> infer via `guess_geometry_kind(obj, chart)`, then redispatch on geometry kind.

    Failure semantics:

    - Inherits all failure semantics from `guess_geometry_kind`.

!!! info `tangent_map`

    Transform a tangent vector from one chart to another.

    **Signature:**

    ```text
    tangent_map(v, from_chart, from_geom, from_rep, to_chart, to_geom, to_rep, /, *, at) -> CDict
    ```

    A 4-argument shorthand form is also supported:

    ```text
    tangent_map(v, from_chart, from_rep, to_chart, /, *, at) -> CDict
    ```

    **Arguments:**

    - `v`: `CDict` — tangent vector components in `from_chart` with basis `from_rep.basis`.
    - `from_chart`: source chart.
    - `from_geom`: source geometry (e.g. `TangentGeometry`).
    - `from_rep`: source `Representation` (must have `TangentGeometry`).
    - `to_chart`: target chart.
    - `to_geom`: target geometry.
    - `to_rep`: target `Representation`.
    - `at`: `CDict` — base point in `from_chart` coordinates at which the tangent space is attached. Required for non-Cartesian charts (since the Jacobian depends on the base point).

    **Semantics by basis:**

    - **`CoordinateBasis`**: delegates to `jac_pt_map(at, from_chart, to_chart)` to obtain the Jacobian $J^j{}_i = \partial\tilde{q}^j/\partial q^i$, then applies $\tilde{v}^j = J^j{}_i v^i$.
    - **`PhysicalBasis`**: fetch the orthonormal frame matrices $B_{\rm from}$ (columns = physical basis vectors in Cartesian) and $B_{\rm to}$ via `frame_cart`, compute $R = B_{\rm to}^T B_{\rm from}$, apply $\hat{v}' = R \hat{v}$.

    **Same-chart optimisation:** when `from_chart is to_chart`, returns `v` unchanged.

    **`cconvert` integration:** `cconvert` dispatches to `tangent_map` when the source representation has `TangentGeometry`, passing `at` through the `at` keyword argument.

    **Same-chart basis conversion:** `change_basis(v, chart, from_basis, to_basis, /, *, at)` changes tangent component conventions without changing charts. In v1 it is defined only for Cartesian charts and `CoordinateBasis` $

</br>

### Geometric Kind

A **geometry kind** identifies the mathematical type of geometric object that component data represents. Importantly, the geometry kind is **independent of the coordinate chart** used to represent the components. The same geometric object may be written in any compatible chart.

Geometric objects arise from geometric structures associated with a manifold $M$. A geometry kind associates a set of coordinates with one such structure, and thereby specifies what sort of object the coordinates are describing. Each geometry kind determines the **coordinate transformation law** for components.

- **Points**: a point is an element $$p \in M .$$ If $q_i$ and $q_f$ are two coordinate descriptions of the same point, then the transformation law is $$
q_f = \tau(q_i) .$$ An important example is the **point geometry**.

(software-spec-abstractgeometry)=

!!! info `AbstractGeometry`

    Abstract base class for geometric kind. `AbstractGeometry` identifies the geometric type of represented data, independent of chart and basis. It answers: "what geometric object do these components represent?" (for example, a point). In the representation triple $R = (K, B, S)$, `AbstractGeometry` is the `K` component.

    Transformation-law role:

    - Geometric kind determines the abstract coordinate-change law for
        components.
    - Point-kind data transforms by chart transition maps.
    - Future tangent/cotangent kinds transform by Jacobian pushforward / pullback laws, respectively.

    Distinctions:

    - Not a chart: charts define coordinate systems, component names, and domains.
    - Not a semantic kind: semantics encode interpretation (for example, location, velocity) within a fixed geometric kind.
    - Not a basis: basis selection is tracked separately and may be trivial for affine objects.

    Notes:

    - `AbstractGeometry` is a static dispatch object category (no runtime numerical payload).
    - Concrete subclasses should represent immutable geometric categories.

!!! info `PointGeometry` and `point_geom`

    Concrete geometric kind for manifold points, and its canonical instance.

    - `PointGeometry` is the final concrete subclass of `AbstractGeometry` for point-like data.
    - It encodes that components represent a point $p \in M$ (an affine object), not a vector in a tangent/cotangent space.
    - Point coordinates therefore transform by the ordinary chart transition map (`pt_map` / `pt_map` point behavior), with the represented geometric object unchanged.

    Affine semantics:

    - Point data is location data: points do not add as vectors.
    - Any vector-space operations require a separate geometric kind (for example tangent/cotangent kinds), not `PointGeometry`.

    API instance:

    - `point_geom` is the pre-defined canonical `PointGeometry()` instance used by the default point representation `point = Representation(point_geom, no_basis, loc)`.

(software-spec-tangent-geometry)=

!!! info `TangentGeometry` and `tangent_geom`

    Concrete geometric kind for tangent vectors, and its canonical instance.

    - `TangentGeometry` is the final concrete subclass of `AbstractGeometry` for tangent-vector data.
    - It encodes that components represent a tangent vector $v \in T_p M$ (a vector in the tangent space at a base point $p$), not an affine point.
    - Tangent vector coordinates transform by the **Jacobian pushforward** under chart changes: $\tilde{v}^j = J^j{}_i v^i$.

    Tangent semantics:

    - Requires a **basis** specification (e.g. `CoordinateBasis` or `PhysicalBasis`) to determine component convention.
    - Requires a **base point** `at` for non-Cartesian chart changes (since $J$ depends on the evaluation point).
    - Supports three semantic kinds: `Displacement`, `Velocity`, `Acceleration` — all with identical transformation laws.

    API instance:

    - `tangent_geom` is the pre-defined canonical `TangentGeometry()` instance.

(software-spec-guess_geometry_kind)=

!!! info `guess_geometry_kind`

    Infer geometry kind from lightweight structural information.

    Dispatches:

    - `AbstractGeometry` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_GEOM_MAP` (`"length"`, `"angle"` -> `PointGeometry`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; discard `"angle"` when mixed; infer from remainder.
    - `(CDict, AbstractChart)` -> validate key schema via `check_data`, then redispatch on `CDict`.
    - `(CDict, ProlateSpheroidal3D)` -> special-case: `{area, angle}` -> `PointGeometry`.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Multiple non-angle dimensions after discarding -> `ValueError`.
    - Unregistered dimension -> `ValueError`.
    - Prolate spheroidal: dimensions not `{area, angle}` -> `ValueError`.

!!! info `PointGeometry` and `point_geom`

    Concrete geometric kind for manifold points, and its canonical instance.

    - `PointGeometry` is the final concrete subclass of `AbstractGeometry` for point-like data.
    - It encodes that components represent a point $p \in M$ (an affine object), not a vector in a tangent/cotangent space.
    - Point coordinates therefore transform by the ordinary chart transition map (`pt_map` / `pt_map` point behavior), with the represented geometric object unchanged.

    Affine semantics:

    - Point data is location data: points do not add as vectors.
    - Any vector-space operations require a separate geometric kind (for example tangent/cotangent kinds), not `PointGeometry`.

    API instance:

    - `point_geom` is the pre-defined canonical `PointGeometry()` instance used by the default point representation `point = Representation(point_geom, no_basis, loc)`.

</br>

### Basis

Many geometric objects live in vector spaces. In such spaces, component values depend on the choice of basis.

A **basis** $B$ for a vector space $V$ is a linearly independent set

$$
{ e_a }_{a=1}^{\dim V}
$$

such that any vector $v \in V$ can be written uniquely as

$$
v = v^a e_a .
$$

The coefficients $v^a$ are the **components of the vector in the basis** $B$.

However, not all geometric objects require a basis specification.

- Points are affine objects and do not belong to a vector space. Their coordinates are chart values, not vector components.

(software-spec-abstractsbasis)=

!!! info `AbstractBasis`

    Abstract base class for basis kind.

    - `AbstractBasis` identifies the component basis in which represented data is written, when basis choice is meaningful.
    - In the representation triple $R = (K, B, S)$, `AbstractBasis` is the `B` component.
    - It answers: "in what basis are these components expressed?"

    Role in representation:

    - Basis is orthogonal to chart, geometry kind, and semantic kind.
    - Chart defines coordinates and coordinate domains.
    - Geometry kind defines the geometric object and transformation law class.
    - Semantic kind defines interpretation (for example location or velocity).

    Basis-law role:

    - For affine point data, basis choice is trivial (`NoBasis`).

    Notes:

    - `AbstractBasis` is a static dispatch object category.
    - Concrete subclasses should represent immutable basis categories.

(software-spec-guess_basis_kind)=

!!! info `guess_basis_kind`

    Infer basis kind from lightweight structural information.

    Dispatches:

    - `AbstractBasis` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_BASIS_MAP` (`"length"`, `"angle"` -> `NoBasis`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; infer from remainder.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Unregistered dimension -> `ValueError`.

(software-spec-no_basis)=

!!! info `NoBasis` and `no_basis`

    Concrete no-basis kind, and its canonical instance.

    - `NoBasis` is the final concrete subclass of `AbstractBasis` used when represented data is not expressed in a basis-dependent linear space.
    - Canonical use: point data (`PointGeometry`), where coordinates describe locations on a manifold rather than vector components in $T_pM$.

    Semantics:

    - `NoBasis` does **not** mean "no coordinates".
    - It means basis choice is not part of the representation semantics for the object kind.
    - Coordinate changes for point data remain chart transition maps, not basis changes.

    API instance:

    - `no_basis` is the pre-defined canonical `NoBasis()` instance.
    - It is used in the default point representation `point = Representation(point_geom, no_basis, loc)`.

(software-spec-abstractlinearbasis)=

!!! info `AbstractLinearBasis`

    Abstract marker base class for bases in linear (vector) spaces.

    - `AbstractLinearBasis` is the abstract parent of all basis kinds applicable to tangent vectors (elements of $T_p M$).
    - Concrete subclasses specify which basis convention is used for the components.
    - Not applicable to point data (`PointGeometry` uses `NoBasis`).

(software-spec-coordinatebasis)=

!!! info `CoordinateBasis` and `coord_basis`

    The holonomic coordinate basis $\{\partial_i\} = \{\partial/\partial q^i\}$, and its canonical instance.

    - `CoordinateBasis` is the concrete basis kind for tangent components expressed in the coordinate (holonomic) basis.
    - Components $v^i$ transform under chart changes by the Jacobian $J^j{}_i = \partial\tilde{q}^j/\partial q^i$ computed via `jax.jacfwd`.
    - This is the natural output of differentiation (e.g. `jax.grad`, `jax.jacfwd`) of coordinate-valued functions.

    API instance:

    - `coord_basis` is the pre-defined canonical `CoordinateBasis()` instance.

(software-spec-physicalbasis)=

!!! info `PhysicalBasis` and `phys_basis`

    The orthonormal physical basis $\{\hat{e}_i\} = \{\partial_i / h_i\}$, and its canonical instance.

    - `PhysicalBasis` is the concrete basis kind for tangent components expressed in the orthonormal (physical) frame.
    - Components $\hat{v}^i$ transform under chart changes via the orthonormal-frame rotation matrix $R = B_{\rm to}^T B_{\rm from}$ where $B$ is the frame matrix of physical basis vectors expressed in Cartesian coordinates.
    - Physical components have consistent physical dimensions (e.g. speed in m/s, not speed/length).

    API instance:

    - `phys_basis` is the pre-defined canonical `PhysicalBasis()` instance.

</br>

### Semantic Kind

A **semantic kind** specifies the **interpretation** attached to a geometric object.

While the geometry kind determines the mathematical space of the object, the semantic kind distinguishes _how that object is interpreted physically_.

Examples include:

- **Location**: a point interpreted as the position of a particle or object.

Semantic kinds do **not** change coordinate transformation laws.

Separating semantics from geometry provides two advantages:

1. Correct transformation laws -- transformation behavior depends only on geometry kind and basis, not on semantics.
2. Clear interpretation -- different semantic kinds distinguish objects that share the same mathematical type but represent different physical quantities.

(software-spec-abstractsemantickind)=

!!! info `AbstractSemanticKind`

    Abstract base class for semantic kind.

    - `AbstractSemanticKind` identifies the interpretation attached to represented data.
    - In the representation triple $R = (K, B, S)$, `AbstractSemanticKind` is the `S` component.
    - It answers: "what does this represented object mean?"

    Role in representation:

    - Semantic kind is orthogonal to chart, geometry kind, and basis.
    - Chart defines coordinate system and component domains.
    - Geometry kind defines the geometric object and transformation-law class.
    - Basis defines component basis when basis choice is meaningful.

    Semantic-law role:

    - Semantic kind refines interpretation within a fixed geometric kind and basis without changing coordinate transformation laws.

    Notes:

    - `AbstractSemanticKind` is a static dispatch object category (no runtime numerical payload).
    - Concrete subclasses should represent immutable semantic categories.

(software-spec-guess_semantic_kind)=

!!! info `guess_semantic_kind`

    Infer semantic kind from lightweight structural information.

    Dispatches:

    - `AbstractSemanticKind` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_SEMANTICS_MAP` (`"length"`, `"angle"` -> `Location`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; discard `"angle"` when mixed; infer from remainder.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Multiple non-angle dimensions after discarding -> `ValueError`.
    - Unregistered dimension -> `ValueError`.

(software-spec-location)=

!!! info `Location` and `loc`

    Concrete location semantic kind, and its canonical instance.

    - `Location` is the final concrete subclass of `AbstractSemanticKind` used to denote "where" data: coordinates interpreted as position on a manifold.
    - Canonical pairing in current `coordinax`: point geometry with no basis, i.e. `(PointGeometry, NoBasis, Location)`.

    Semantics:

    - `Location` records interpretation, not chart choice.
    - It does not alter the underlying point transformation law: point coordinates still transform by chart transition maps.
    - It distinguishes location-like point data from other possible semantics introduced for non-point geometric kinds.

    API instance:

    - `loc` is the pre-defined canonical `Location()` instance.
    - It is used in the default point representation `point = Representation(point_geom, no_basis, loc)`.

(software-spec-abstracttangentsemantickind)=

!!! info `AbstractTangentSemanticKind`

    Abstract marker base class for semantic kinds of tangent vectors.

    - All tangent-vector semantic kinds (Displacement, Velocity, Acceleration) share the same Jacobian transformation law.
    - Semantic kind is used for role-aware dispatch in frame transformations (e.g. `Boost` acts on `Velocity`, is identity on `Displacement`).

    **`order` class variable** (`ClassVar[int]`):

    Every concrete subclass must define `order`, an integer encoding its position in the time-derivative chain:

    | Class          | `order` |
    |----------------|---------|
    | `Displacement` |   0     |
    | `Velocity`     |   1     |
    | `Acceleration` |   2     |

    `order` is the key used in the internal order registry.

    **`__init_subclass__`**:

    On class creation, each concrete subclass is automatically registered in the internal order registry at its `order`. Raises `TypeError` if the chosen `order` is already occupied by a *different* class (same-named reconstructions from `@dataclasses.dataclass(slots=True)` are allowed).

    **`derivative() -> AbstractTangentSemanticKind`**:

    Returns a fresh instance of the class registered at `self.order + 1`. Raises `ValueError` if no class is registered at that order.

    **`antiderivative() -> AbstractTangentSemanticKind`**:

    Returns a fresh instance of the class registered at `self.order - 1`. Raises `ValueError` if no class is registered at that order. This is open for extension: defining an `Absement` subclass at `order = -1` automatically makes `Displacement().antiderivative()` return `Absement()`.

    **`derivative()` and `antiderivative()` are mutual inverses** on the interior of the registered ladder:
    - `kind.derivative().antiderivative() == kind` for all kinds that are not the top of the ladder.
    - `kind.antiderivative().derivative() == kind` for all kinds that are not the bottom of the ladder.

(software-spec-displacement)=

!!! info `Displacement` and `dpl`

    Semantic kind for a finite (or infinitesimal) position difference, and its canonical instance.

    - `Displacement` is a concrete subclass of `AbstractTangentSemanticKind`.
    - Represents $\Delta q = q_2 - q_1$, an element of the tangent space in the limit, or a finite difference.
    - Under Galilean boosts: `Displacement` is **invariant** — boost does not change displacements.
    - `order = 0`.
    - `derivative()` returns the `vel` instance directly.
    - `antiderivative()` uses the base-class internal-registry lookup; raises `ValueError` unless a class at order -1 (e.g. `Absement`) is registered.

    API instance:

    - `dpl` is the pre-defined canonical `Displacement()` instance.

(software-spec-velocity)=

!!! info `Velocity` and `vel`

    Semantic kind for time-derivative of position, and its canonical instance.

    - `Velocity` is a concrete subclass of `AbstractTangentSemanticKind`.
    - Represents $\dot{q} = dq/dt$, a genuine element of $T_p M$.
    - Under Galilean boosts: **shifts** by the boost velocity $\Delta v$.
    - `order = 1`.
    - `derivative()` returns the `acc` instance directly.
    - `antiderivative()` returns the `dpl` instance directly.

    API instance:

    - `vel` is the pre-defined canonical `Velocity()` instance.

(software-spec-acceleration)=

!!! info `Acceleration` and `acc`

    Semantic kind for time-derivative of velocity, and its canonical instance.

    - `Acceleration` is a concrete subclass of `AbstractTangentSemanticKind`.
    - Represents $\ddot{q} = d^2q/dt^2$, also an element of $T_p M$.
    - Under Galilean boosts with constant $\Delta v$: **invariant** (since $\dot{\Delta v} = 0$).
    - `order = 2`.
    - `derivative()` uses the base-class ladder lookup; raises `ValueError` unless a class at order 3 (e.g. `Jerk`) is registered.
    - `antiderivative()` returns the `vel` instance directly.

    API instance:

    - `acc` is the pre-defined canonical `Acceleration()` instance.

</br>

(software-spec-vectors)=

## Vectors

!!! info `AbstractVector`

    Methods \& Properties:

    - ``from_()``: multiple dispatch constructor of vector-like objects from arguments.
    - ``uconvert``: convert the vector to the given units.
      - ``(V, *args, **kwargs) -> uconvert(*args, V, **kwargs)`` redispatch
      - ``(V, u.AbstractUnitSystem) -> uconvert(u.AbstractUnitSystem, V)`` redispatch
    - ``__array_namespace__``: the array API namespace -- `quax.numpy`. This delegates to `quax` primitives.
    - ``__eq__``: equality check, based on type equality and `quax` equality primitive.
    - ``copy()``: call `dataclass.replace`.
    - ``flatten()``: flatten the vector.
    - ``ravel()``: return a flattened vector.
    - ``reshape(*shape)``: return a reshaped vector
    - ``round(decimals)``: return a rounded vector.
    - ``to_device(device)``: move the vector to a new device
    - ``__repr__()``: string representation through `wadler_lindig`.
    - ``__str__()``: string representation through `wadler_lindig`.
    - ``is_like()``: check if the object is a `AbstractVector` object.

    Abstract Methods \& Properties:

    - ``shape``: the shape of the vector. Abstract method.
    - ``__getitem__(slice)``: slice the vector.
    - ``astype(dtype, **kw)`` : cast the vector to a new dtype.

    Not Supported:

    - ``materialise``: for materialising the vector for `quax`.
    - ``__complex__``
    - ``__float__``
    - ``__index__``
    - ``__int__``
    - ``__setitem__`` : vectors are immutable.
    - ``__hash__()``: hash the vector by its field items. In general this raises an error.

!!! info `Point`

    Arguments:

    - ``data``: the data for each component.
    - ``chart``: the chart of the vector, e.g. ``cxc.cart3d``.
    - ``rep``: the `coordinax.representations.Representation`, e.g. `cxr.point`.
    - ``frame``: the reference frame, e.g. ``cxf.alice``. Optional; defaults to
      ``cxf.noframe`` when not provided.

    Methods \& Properties:

    - ``__getitem__()``
    - ``__pdoc__()``
    - ``cconvert()`` — chart conversion; preserves ``frame``.
    - ``to_frame(toframe, t=None) -> Point`` — frame transform; applies the
      frame transition ``self.frame -> toframe`` to the data and returns a new
      ``Point`` with the updated data and ``frame=toframe``.
    - ``aval()``
    - ``shape``
    - ``norm()``

    `from_` Constructor Dispatches:

    - ``(Point,) -> Point``
    - ``(dict,C,R) -> Point``
    - ``(dict,C) -> (dict,C,R) -> Point``
    - ``(dict,) -> (dict,C,R) -> Point``
    - ``(Q,C,R) -> (dict,C,R) -> Point``
    - ``(Q,C) -> (dict,C,R) -> Point``
    - ``(Q,) -> (dict,C) -> (dict,C,R) -> Point``
    - ``(Q,R) -> (dict,C,R) -> Point``
    - ``(Array,unit) -> (Q,) -> ... -> Point``
    - ``(Array,unit,C) -> ... -> Point``
    - ``(Array,unit,C,R) -> ... -> Point``
    - ``(Point, frame) -> Point`` — identity on data, replaces frame.
    - ``(Vector, frame) -> Point`` — wraps vector data with given frame.
    - ``(obj, chart, rep, manifold, frame) -> Point``
    - ``(obj, chart, rep, frame) -> Point``
    - ``(obj, chart, frame) -> Point``
    - ``(obj, frame) -> Point``
    - ``(Array, unit, frame) -> Point``

!!! info `ToUnitsOptions`

    Used for `unxt.uconvert` dispatches.

</br>

(software-spec-manifolds)=

## Manifolds

The `coordinax.manifolds` module, typically imported as `import coordinax.manifolds as cxm`, provides the manifold hierarchy used by `coordinax`. Concrete manifold classes attach an atlas and, where appropriate, a Riemannian (or pseudo-Riemannian) metric to smooth manifolds.

A **metric** on a manifold $M$ is a smooth assignment of a symmetric, non-degenerate bilinear form to each tangent space:

$$g_p : T_pM \times T_pM \to \mathbb{R}, \quad p \in M.$$

In a local chart with coordinates $q = (q^1, \ldots, q^n)$, the metric is encoded by the **metric matrix**

$$g_{ij}(q) = g_p\!\left(\frac{\partial}{\partial q^i}, \frac{\partial}{\partial q^j}\right).$$

!!! info `AbstractTopologicalManifold`

    `AbstractTopologicalManifold` is the root base class for all manifold objects in `coordinax`. It encodes only the topological layer: intrinsic dimension and chart membership. Subclasses that attach a smooth atlas and metric structure implement the full smooth-manifold interface.

    **Invariant**: `AbstractTopologicalManifold` is a pure structural descriptor — it carries no numerical point data. Instances are registered as JAX static nodes and appear as compile-time metadata inside JIT-compiled functions.

    Public API:

    - `ndim`: intrinsic dimension $n$ of the manifold.
    - `has_chart(chart) -> bool`: return `True` if `chart` belongs to this manifold's atlas.
    - `check_chart(chart)`: assert chart membership; raise `ValueError` if not supported.

    Notes:

    - Registered to JAX as static via `jax.tree_util.register_static`.
    - Implements a `wadler_lindig` `__pdoc__()` method underpinning `__repr__` and `__str__`.
    - The class hierarchy follows the abstract-final pattern: `AbstractTopologicalManifold` → concrete final classes (e.g. `EuclideanManifold`, `HyperSphericalManifold`); no intermediate layers.

!!! info `NoManifold` and `no_manifold`

    `NoManifold` is a degenerate placeholder manifold with no charts and no geometry. It serves as a sentinel value when a manifold object is required by the API but none has been specified by the user.

    - `ndim == -1` (sentinel for "no manifold specified").
    - `has_chart(chart)` always returns `False`.

    `no_manifold` is the canonical module-level instance of `NoManifold`. It should be used in preference to constructing `NoManifold()` directly, since `NoManifold` carries no state and a shared instance is cheaper.

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> cxm.no_manifold
    NoManifold(ndim=-1)

    >>> cxm.no_manifold.ndim
    -1

    >>> import coordinax.charts as cxc
    >>> cxm.no_manifold.has_chart(cxc.cart3d)
    False
    ```

(software-spec-guess-manifold)=

!!! info `guess_manifold`

    Infer or pass-through a manifold from various input types via multiple dispatch.

    `guess_manifold` is a dispatched function that converts various representational forms into an explicit manifold object. It provides multiple implementations for different input types.

    **Dispatch signatures:**

    - `guess_manifold(manifold: AbstractManifold) -> AbstractManifold`
      Returns the input manifold unchanged.

    - `guess_manifold(point: CDict) -> AbstractManifold`
      Infers the manifold from a point represented as a mapping (coordinate dictionary). First infers the chart via `cxc.guess_chart()`, then redispatches on that chart.

    - `guess_manifold(chart: AbstractChart) -> AbstractManifold`
      Infers the manifold from a chart. Domain-specific implementations exist:
      - `guess_manifold(atlas: EuclideanAtlas) -> EuclideanManifold`
      - `guess_manifold(atlas: HyperSphericalAtlas) -> HyperSphericalManifold`
      - Chart-specific versions for `Cart0D`, `Cart1D`, `Cart2D`, `Cart3D`, `Radial1D`, `Polar2D`, spherical chart types, etc.

    **Purpose:**

    `guess_manifold` enables writing code that is polymorphic over manifold representations: users can pass a point, a chart, an atlas, or a manifold directly, and the function will normalize to an explicit manifold object. This is useful in high-level APIs where users may provide coordinate data without specifying the manifold structure.

    **Examples**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> # From a manifold (pass-through)
    >>> M = cxm.EuclideanManifold(3)
    >>> cxm.guess_manifold(M) is M
    True

    >>> # From a point (mapping)
    >>> cxm.guess_manifold({"x": 1.0, "y": 2.0, "z": 3.0})
    Rn(3)

    >>> # From a chart
    >>> cxm.guess_manifold(cxc.cart3d)
    Rn(3)

    >>> # From an atlas
    >>> cxm.guess_manifold(cxm.EuclideanAtlas(2))
    Rn(2)

    >>> # Spherical manifold inference
    >>> cxm.guess_manifold(cxc.sph2)
    HyperSphericalManifold(ndim=2)

    >>> cxm.guess_manifold({"theta": 1.0, "phi": 0.5})
    HyperSphericalManifold(ndim=2)
    ```

    **Notes:**

    - This function is primarily used internally and in high-level convenience APIs.
    - When the input is a manifold, it is returned unchanged without validation.
    - When inferring from a point (CDict), the chart inference is performed by `cxc.guess_chart()`, which uses its own dispatch logic.

(software-spec-scale-factors)=

!!! info `scale_factors`

    Return the diagonal entries of the metric matrix.

    `scale_factors` is a dispatched function that returns the vector

    $$
    (g_{11}(p), \ldots, g_{nn}(p))
    $$

    for a metric evaluated in a chart at the base point $p$. This API returns the diagonal metric entries themselves, not the coordinate-basis lengths $\sqrt{g_{ii}(p)}$.

    **Signature:**

    ```
    cxm.scale_factors(chart, /, *, at, usys=None)
    ```

    Or via convenience wrappers on metric and manifold objects:

    ```
    metric.scale_factors(chart, at=at, usys=usys)
    manifold.scale_factors(chart, at=at, usys=usys)
    ```

    **Arguments:**

    - `metric_or_manifold`: an `AbstractMetric` instance (metric-level call) or an `AbstractManifold` instance (manifold-level call). When a manifold is passed, `scale_factors` delegates to `manifold.metric`.
    - `chart`: the coordinate chart in which the metric is expressed.
    - `at` (keyword): the base point $p$ in chart coordinates at which the metric is evaluated.
    - `usys` (keyword, optional): unit system forwarded to metric evaluation when needed.

    **Return:**

    - Always a 1-D `QuantityMatrix` of length `ndim`.
    - A `QuantityMatrix` is used even when the diagonal entries are dimensionless, because different coordinate directions may carry different units.

    **Dispatch behavior:**

    - Generic metric dispatch: evaluate `metric.metric_matrix(chart, at=at, usys=usys)` and return `QuantityMatrix.diag()` on the result. If the metric matrix is array-valued, it is first promoted to a dimensionless `QuantityMatrix` and then diagonalized.
    - Manifold dispatch: resolve to `scale_factors(manifold.metric, chart, at=at, usys=usys)`.
    - `EuclideanMetric` specialization: compute the diagonal more efficiently than forming the full metric matrix. In Cartesian charts this returns a vector of ones directly; in non-Cartesian Euclidean charts it uses the chart-to-Cartesian Jacobian and computes only the squared column norms needed for the diagonal entries.

    **Position dependence:**

    - For flat metrics (for example `EuclideanMetric` in Cartesian coordinates), the result may be position-independent numerically, though `at` remains part of the API.
    - For curved or curvilinear cases, the returned diagonal entries depend on the supplied base point.

    **Examples**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {
    ...     "r": u.Q(2.0, "km"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }

    >>> gdiag = cxm.scale_factors(cxc.sph3d, at=at)
    >>> gdiag.shape
    (3,)
    >>> gdiag.unit.to_string()
    '(, km2 / rad2, km2 / rad2)'

    >>> metric = cxm.HyperSphericalMetric(2)
    >>> at_s2 = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at_s2).value
    Array([1., 1.], dtype=float64)
    ```

    **Notes:**

    - `AbstractMetric.scale_factors` and `AbstractManifold.scale_factors` are thin wrappers over `cxm.scale_factors`.
    - The name `scale_factors` in the software API follows the library convention for metric diagonal entries, even though some mathematical texts reserve “scale factor” for $\sqrt{g_{ii}}$.

(software-spec-angle-between)=

!!! info `angle_between`

    Return the metric angle between two nonzero tangent vectors.

    `angle_between` is a dispatched function that evaluates the standard
    Riemannian angle formula at a base point $p$:

    $$
    \cos\theta = \frac{g_p(u, v)}{\|u\|_p\,\|v\|_p},
    \qquad
    \|u\|_p^2 = g_p(u, u),
    \qquad
    \|v\|_p^2 = g_p(v, v).
    $$

    The input component dictionaries represent **tangent-vector components in
    the coordinate basis of `chart`**, not point-role coordinates. The base
    point `at` specifies where the metric is evaluated. Even for flat metrics,
    `at` remains part of the public API for consistency with curvilinear and
    embedded-manifold cases.

    **Signature:**

    ```
    cxm.angle_between(metric_or_manifold, chart, u, v, /, *, at, usys=None)
    ```

    Or via convenience wrappers on metric and manifold objects:

    ```
    metric.angle_between(chart, u, v, at=at, usys=usys)
    manifold.angle_between(chart, u, v, at=at, usys=usys)
    ```

    **Arguments:**

    - `metric_or_manifold`: an `AbstractMetric` instance (metric-level call) or an `AbstractManifold` instance (manifold-level call). When a manifold is passed, `angle_between` delegates to `manifold.metric`.
    - `chart`: the coordinate chart in whose basis the tangent-vector components are expressed.
    - `u`, `v`: `CDict` tangent-vector components keyed by `chart.components`.
    - `at` (keyword): the base point $p$ in chart coordinates at which the metric is evaluated.
    - `usys` (keyword, optional): unit system forwarded to metric evaluation when needed.

    **Return:**

    - Returns an angular quantity in radians, typically a `coordinax.angles.Angle`.
    - For supported positive-definite metrics, the result lies in $[0, \pi]$.

    **Dispatch behavior:**

    - Generic metric dispatch: evaluate `metric.metric_matrix(chart, at=at, usys=usys)`, compute the bilinear forms `u^T g v`, `u^T g u`, and `v^T g v`, then return `arccos(...)` of the normalized inner product.
    - Manifold dispatch: resolve to `angle_between(manifold.metric, chart, u, v, at=at, usys=usys)`.
    - The implementation supports full symmetric metric matrices; it is not restricted to diagonal metrics.

    **Failure semantics:**

    - If either input tangent vector has zero norm, raise `ValueError`.
    - If the chart keys do not match `chart.components`, validation may raise `ValueError`.
    - In v1, pseudo-Riemannian / indefinite metrics are unsupported by this API and raise `NotImplementedError`.

    **Examples**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EuclideanManifold(2)
    >>> at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> uvec = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
    >>> vvec = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m")}

    >>> cxm.angle_between(M, cxc.cart2d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    >>> at_sph = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> u_tan = {"r": u.Q(0.0, "m"), "theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> v_tan = {"r": u.Q(0.0, "m"), "theta": u.Angle(0.0, "rad"), "phi": u.Angle(1.0, "rad")}
    >>> cxm.angle_between(cxm.EuclideanMetric(3), cxc.sph3d, u_tan, v_tan, at=at_sph)
    Angle(1.57079633, 'rad')
    ```

    **Notes:**

    - This API is for tangent-space geometry. Point-role coordinates should first be converted into a tangent/displacement representation if that is the intended meaning.
    - The angle is defined intrinsically by the metric at the supplied base point and is therefore chart-invariant under valid coordinate changes.

(software-spec-abstractatlas)=

!!! info `AbstractAtlas`

    An atlas is the collection of compatible charts that defines the smooth structure of a manifold. If $ \mathcal{A} = \{(U_\alpha, \varphi_\alpha)\} $ is an atlas on a topological manifold $M$, then the pair $ (M, \mathcal{A}) $ is a smooth manifold. In `coordinax`, the atlas object is the software representation of this smooth structure: it specifies which charts are valid coordinate descriptions of the manifold.

    The atlas is therefore responsible for chart compatibility, but it does **not** itself implement the numerical coordinate formulas for transforming data between charts. Those formulas live at the chart and transformation-function level.

    **Core attribute:**

    - `ndim`: the dimension of the manifold covered by the atlas. This means that every chart supported by the atlas must have the same coordinate dimension: $$ \dim M = \text{atlas.ndim}. $$

    **Methods:**

    - ``default_chart()``: returns a canonical or preferred chart in the atlas. This does **not** mean that the default chart is mathematically preferred in any intrinsic sense; it is simply the atlas-level convention chosen by the library for ergonomic defaults.

    - ``has_chart(chart)``: returns whether a given chart belongs to the atlas. It is the fundamental membership test for the atlas. It answers whether `chart` is an allowed coordinate system for points on the manifold associated with this atlas. This is a structural question, not a numerical one: it does not evaluate coordinates, only chart compatibility.
      In particular:

      - a Euclidean atlas supports Euclidean coordinate charts of the correct dimension,
      - a two-sphere atlas supports intrinsic two-sphere charts,
      - charts from incompatible manifolds must return `False`.

      Atlases redirect ``__contains__`` to ``has_chart()``, enabling

      ```
      chart in atlas
      ```

    - ``pt_map(...)``: checks atlas compatibility and then delegates to the ordinary chart-level point transition machinery.

    **Examples**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> atlas = cxm.EuclideanAtlas(3)
    >>> atlas.ndim
    3
    >>> atlas.default_chart()
    Cart3D(M=Rn(3))

    >>> import coordinax.charts as cxc
    >>> cxc.cart3d in atlas
    True
    >>> cxc.cyl3d in atlas
    True
    >>> cxc.cart2d in atlas
    False

    >>> atlas2 = cxm.EuclideanAtlas(2)
    >>> x = {"x": 1.0, "y": 1.0}
    >>> atlas2.pt_map(x, cxc.cart2d, cxc.polar2d)
    {'r': Array(1.41421356, dtype=float64, ...), 'theta': Array(0.78539816, dtype=float64, ...)}
    ```

    In this call the atlas first verifies that both charts belong to the same atlas before delegating to the registered point transition map implementation.

(software-spec-abstractmetric)=

!!! info `AbstractMetric`

    `AbstractMetric` is the abstract base for metric tensors used by manifold objects.

    A metric assigns a symmetric, non-degenerate bilinear form to each tangent space:

    $$
    g_p : T_pM \times T_pM \to \mathbb{R}, \quad p \in M.
    $$

    In local coordinates $q = (q^1, \ldots, q^n)$, this is represented by the metric matrix
    $$g_{ij}(q).$$

    **Immutability and JAX-static requirements:**

    - All metric classes are immutable frozen dataclasses.
    - All metric classes are registered with `@jax.tree_util.register_static`.
    - Metric instances therefore flatten as static PyTree nodes (no dynamic leaves).

    **Core API contract:**

    - `signature: tuple[int, ...]` (abstract property): encodes metric signature signs in coordinate order and has length equal to the metric dimension.
    - `ndim: int`: defined as `len(signature)`.
    - `metric_matrix(chart, /, *, at, usys=None) -> QuantityMatrix | Array` (abstract method): returns the metric matrix expressed in `chart`, evaluated at base point `at`.

    **Metric matrix requirements:**

    - Shape is `(ndim, ndim)`.
    - Matrix is symmetric.
    - The return type follows input/unit context.
    - Quantity-valued evaluation returns `QuantityMatrix`.
    - Array-valued evaluation may return `Array`.

    **Behavioral guarantees:**

    - Position-independent metrics (for example Minkowski in canonical coordinates) may ignore `at` numerically, but must still satisfy the same interface.
    - Position-dependent metrics (for example induced or hyperspherical metrics) evaluate `metric_matrix` at the supplied base point.
    - Chart compatibility is a manifold-level concern; metrics assume callers provide charts and points compatible with the surrounding manifold contract.

    **Subclassing requirements:**

    - Implement `signature` and `metric_matrix`.
    - Preserve immutability and static PyTree behavior.
    - Remain JAX-transform compatible (`jit`, `vmap`) as pure functions of inputs.

    See the [Metrics](#software-spec-metrics) section for concrete metric families and formulas.

    **Methods:**

    - `scale_factors(chart, /, *, at, usys=None)`: convenience wrapper around [`cxm.scale_factors`](#software-spec-scale-factors). Returns the 1-D `QuantityMatrix` of diagonal metric entries in `chart` at base point `at`.
    - `is_diagonal(chart, /, *, at, usys=None) -> Bool[Array, ""]`: returns `True` if all off-diagonal entries of the metric matrix vanish at `at`, checked numerically via `jnp.allclose`. This is a **point-specific, numerical** check. For a **structural, global** guarantee on a metric's diagonal chart domain (typically orthogonal charts), use `isinstance(metric, AbstractDiagonalMetric)`.
    - `cholesky(chart, /, *, at, usys=None) -> QuantityMatrix | Array`: returns the lower-triangular Cholesky factor $L$ satisfying $g = L\,L^\top$. The vielbein is $E = L^\top$; see [Diagonal metrics and orthogonal coordinate systems](#diagonal-metrics-and-orthogonal-coordinate-systems) for the relationship to physical-basis components. Returns a `QuantityMatrix` when the metric matrix carries units, otherwise a plain `Array`. Element $L_{ij}$ carries unit $\sqrt{u_{ij}}$ where $u_{ij}$ is the unit of $g_{ij}$.

(software-spec-abstractdiagonalmetric)=

!!! info `AbstractDiagonalMetric`

    `AbstractDiagonalMetric` is a **structural marker** subclass of `AbstractMetric` for metrics whose matrix is diagonal at **every** base point on their diagonal chart domain.

    A metric is diagonal when all off-diagonal entries vanish globally:

    $$
    g_{ij}(p) = 0 \quad \text{for } i \neq j, \quad \forall\, p \in M.
    $$

    The coordinate basis is orthogonal everywhere, and the diagonal entries are the squared scale factors $h_i^2 = g_{ii}$. See [Diagonal metrics and orthogonal coordinate systems](#diagonal-metrics-and-orthogonal-coordinate-systems) for the mathematical background.

    **Structural guarantee vs. point check.** `AbstractDiagonalMetric` makes a **global, type-level** promise on the metric's diagonal chart domain (typically orthogonal charts): the metric is diagonal at every valid base point. This is strictly stronger than the point-wise `is_diagonal()` test on `AbstractMetric`, which inspects the matrix numerically at a specific `at`. Consequently, `AbstractDiagonalMetric.is_diagonal()` **unconditionally returns `True`** without evaluating the metric matrix.

    **Atlas compatibility vs. diagonality.** Atlas/manifold chart compatibility (`has_chart`) is a broader structural criterion and does not by itself imply orthogonality or diagonal metric form.

    **No new abstract members.** `AbstractDiagonalMetric` inherits the full `AbstractMetric` interface and adds no new abstract methods. Subclasses must still implement:

    - `signature` (property): tuple of $\pm 1$ of length `ndim`.
    - `metric_matrix(chart, /, *, at, usys=None)` (method): **must** return a diagonal `QuantityMatrix` (or plain `Array`) — all off-diagonal entries numerically zero.

    **Dispatch optimization.** Because the diagonal structure is guaranteed statically, dispatch implementations (e.g., for `scale_factors`) can read diagonal entries directly without constructing or inspecting the full $n \times n$ matrix.

    **Immutability and JAX-static requirements:** same as `AbstractMetric` — frozen dataclasses registered with `@jax.tree_util.register_static`.

    **Concrete subclasses:**

    - [`EuclideanMetric`](#software-spec-euclideanmetric): flat Riemannian metric on $\mathbb{R}^n$; identity in Cartesian charts, computed via Jacobian pullback in curvilinear charts.
    - [`MinkowskiMetric`](#software-spec-minkowskimetric): Lorentzian metric $\eta = \operatorname{diag}(-1, 1, 1, 1)$ on Minkowski spacetime; diagonal in the canonical Cartesian spacetime chart.
    - [`HyperSphericalMetric`](#software-spec-hypersphericalmetric): round metric on $S^{n-1}$; diagonal entries follow the cumulative-sine rule $g_{kk} = \prod_{j < k}\sin^2\!\theta_j$.

(software-spec-abstractdiagonalmetric)=

!!! info `AbstractDiagonalMetric`

    `AbstractDiagonalMetric` is an abstract subclass of `AbstractMetric` for metrics whose matrix is diagonal at every base point in every compatible chart.

    A metric is **diagonal** (equivalently, the coordinate chart is an **orthogonal coordinate system**) when all off-diagonal entries of the metric matrix vanish:

    $$g_{ij}(p) = 0 \quad \text{for } i \neq j \quad \forall\, p \in U.$$

    The coordinate basis vectors $\partial/\partial q^i$ are mutually orthogonal. The diagonal entries $g_{ii}(p)$ give the squared scale factors

    $$h_i(p)^2 = g_{ii}(p),$$

    and the infinitesimal line element simplifies to

    $$ds^2 = \sum_i g_{ii}(q)\,(dq^i)^2.$$

    **Role: structural marker, not behavioral interface.**

    `AbstractDiagonalMetric` adds no new abstract methods beyond those of `AbstractMetric`.  Its sole purpose is to **declare** that `metric_matrix` will always return a diagonal matrix.  This allows:

    - Dispatch specialisations that compute `scale_factors` more efficiently
      (e.g., extracting only the diagonal of $g$, or using squared Jacobian
      column norms instead of the full $J^\top J$).
    - Type-level distinction between orthogonal and general metrics in
      multiple-dispatch registrations.

    **Subclassing contract:**

    Subclasses must implement the two abstract members inherited from
    `AbstractMetric`:

    - `signature` (abstract property): a tuple of $\pm 1$ of length `ndim`
      encoding the metric signature in coordinate order.  Positive entries
      are Riemannian (space-like); a ``-1`` entry is pseudo-Riemannian
      (time-like).
    - `metric_matrix(chart, /, *, at, usys=None)` (abstract method): must
      return a diagonal `QuantityMatrix` (or plain `Array`) of shape
      `(ndim, ndim)` with all off-diagonal entries exactly zero.

    All other behavioral requirements of `AbstractMetric` also apply:
    immutability (frozen dataclass), static JAX PyTree registration, and
    `jit`/`vmap` compatibility.

    **Relationship to `AbstractMetric.is_diagonal`:**

    `AbstractMetric.is_diagonal(chart, at=at)` inspects the metric matrix at
    a **specific base point** and returns a `bool`.
    `AbstractDiagonalMetric` makes this an unconditional **structural
    promise** across all base points: instances are always diagonal,
    regardless of which chart or point is supplied.

    **Concrete subclasses (built-in):**

    | Class | Manifold | Diagonal in |
    |-------|----------|-------------|
    | [`EuclideanMetric`](#software-spec-euclideanmetric) | $\mathbb{R}^n$ | Cartesian charts; orthogonal curvilinear charts via $g = J^\top J$ |
    | [`HyperSphericalMetric`](#software-spec-hypersphericalmetric) | $S^{n-1}$ | Intrinsic hyperspherical chart; cumulative-sine rule $g_{kk} = \prod_{j<k}\sin^2\!\theta_j$ |
    | [`MinkowskiMetric`](#software-spec-minkowskimetric) | $\mathbb{R}^{1,3}$ | Canonical Cartesian spacetime chart $\eta = \operatorname{diag}(-1,1,1,1)$ |

    **Example**

    ```pycon
    >>> from coordinax._src.manifolds.diagonal import AbstractDiagonalMetric
    >>> import coordinax.manifolds as cxm

    >>> isinstance(cxm.EuclideanMetric(3), AbstractDiagonalMetric)
    True

    >>> isinstance(cxm.MinkowskiMetric(), AbstractDiagonalMetric)
    True

    >>> import unxt as u
    >>> isinstance(
    ...     cxm.InducedMetric(
    ...         cxm.TwoSphereIn3D(radius=u.Q(1.0, "m")),
    ...         cxm.EuclideanMetric(3),
    ...     ),
    ...     AbstractDiagonalMetric,
    ... )
    False
    ```

(software-spec-abstractmanifold)=

!!! info `AbstractManifold`

    `AbstractManifold` defines the core interface for manifolds. A smooth manifold object represents the mathematical pair
    $$
    (M, \mathcal{A})
    $$
    In other words, the manifold object owns the primary geometric structures used by the library:

    - the [**atlas** $\mathcal{A}$](#software-spec-abstractatlas), which determines the smooth structure and the set of compatible charts

    The manifold object is responsible for enforcing the compatibility of these structures and for providing a small set of geometry‑level operations that depend on the manifold rather than on a specific chart.

    **Core attributes:**

    Every manifold provides one structural attribute:

    - `atlas`: an instance of `AbstractAtlas` describing the compatible charts.

    The manifold dimension is determined by its atlas:

    $$
    \dim M = \text{atlas.ndim}.
    $$

    Atlas-related Methods \& Attributes:

    - ``has_chart(chart)``: determines whether a chart may be used to represent coordinates on it. This calls the atlas ``has_chart(chart)`` method.
    - ``check_chart(chart)``: raises an error if the chart is not supported by the manifold's atlas.
    - ``default_chart``: the manifold's default chart, which is the atlas's default chart.

    Manifold‑level coordinate operations:

    The manifold provides thin wrappers around coordinate transformations that ensure atlas compatibility before delegating to chart‑level machinery.

    - ``pt_map(...)`` performs chart transitions while checking that both charts belong to the manifold.
    - ``scale_factors(chart, /, *, at, usys=None)``: convenience wrapper that delegates to the manifold metric. Returns the 1-D `QuantityMatrix` of diagonal metric entries in `chart` at base point `at`. See the [`scale_factors` functional API section](#software-spec-scale-factors) for full semantics.

    Pre-defined manifolds:

    - Euclidean [`EuclideanManifold`](#software-spec-euclideanmanifold)
    - Two-sphere [`HyperSphericalManifold`](#software-spec-twospheremanifold)
    - Minkowski [`MinkowskiManifold`](#software-spec-minkowskimanifold)
    - Custom [`CustomManifold`](#software-spec-custommanifold)

### Euclidean Manifolds

!!! info `EuclideanAtlas`

    `EuclideanAtlas` is the atlas for Euclidean space $\mathbb{R}^n$. It defines the set of coordinate charts that represent smooth coordinate systems on flat Euclidean space.

    Formally, the atlas corresponds to the collection

    $$
    \mathcal{A}_E = \{ C \mid C \text{ is a Euclidean chart on } \mathbb{R}^n \}.
    $$

    The atlas does not implement coordinate transformations itself; it only determines **which charts belong to the Euclidean smooth structure**. Numerical coordinate formulas are implemented by the chart transition system.

    Construction:

    ```
    EuclideanAtlas(ndim: int)
    ```

    where `ndim` is the dimension of the Euclidean space.

    Supported charts:

    A chart belongs to a `EuclideanAtlas(n)` when:

    1. the chart represents coordinates on Euclidean space, and
    2. the chart dimension equals `n`.

    Examples include

    - Cartesian charts (`CartND`, `Cart1D`, `Cart2D`, `Cart3D`)
    - polar or cylindrical charts (`Polar2D`, `Cylindrical3D`)
    - spherical coordinate charts (`Spherical3D`, `LonLatSpherical3D`)
    - radial charts (`Radial1D`)

    Charts whose intrinsic manifold is not Euclidean (for example intrinsic two-sphere charts) are rejected.

    Default chart:

    The default chart is the canonical Cartesian chart of the same
    dimension:

    | dimension | default chart |
    |-----------|---------------|
    | 0         | `Cart0D()`    |
    | 1         | `Cart1D()`    |
    | 2         | `Cart2D()`    |
    | 3         | `Cart3D(M=Rn(3))`    |
    | otherwise | `CartND()`    |

    ### Chart registration

    Charts become members of the Euclidean atlas through **chart registration**.

    When a chart class is defined for Euclidean coordinates, it registers itself with the Euclidean atlas for the appropriate dimension. This allows the atlas to determine membership without hard-coding chart lists.

    In other words, chart membership is determined structurally by the chart definition rather than by manual enumeration.

    ### Example

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> A = cxm.EuclideanAtlas(3)

    >>> A.default_chart()
    Cart3D(M=Rn(3))

    >>> cxc.cart3d in A
    True

    >>> cxc.cyl3d in A
    True

    >>> cxc.cart2d in A
    False
    ```

(software-spec-euclideanmetric)=

!!! info `EuclideanMetric`

    `EuclideanMetric` is the flat Riemannian metric on $\mathbb{R}^n$.

    In Cartesian coordinates, the metric matrix is the identity:

    $$
    g = I_n.
    $$

    In any non-Cartesian chart, the metric is computed by pullback through the chart-to-Cartesian Jacobian:

    $$
    g_{ij} = (J^T J)_{ij} = \sum_k \frac{\partial x^k}{\partial q^i}\frac{\partial x^k}{\partial q^j}.
    $$

    Construction:

    ```text
    EuclideanMetric(ndim: int)
    ```

    Semantics:

    - `signature = (1,) * ndim`.
    - `metric_matrix(chart, /, *, at, usys=None)` returns a `QuantityMatrix`.
    - For Cartesian charts, `metric_matrix` returns a dimensionless identity matrix of shape `(ndim, ndim)`.
    - For compatible non-Cartesian charts, `metric_matrix` is computed as `J^T J`, where `J = jac_pt_map(at, chart, chart.cartesian, usys=usys)`.
    - This pullback is diagonal exactly for orthogonal charts. Therefore the `AbstractDiagonalMetric` interpretation of `EuclideanMetric` is scoped to that orthogonal chart domain; atlas compatibility alone does not guarantee diagonality.
    - If a chart has no global Cartesian sibling, the current implementation falls back to a dimensionless identity matrix.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.EuclideanMetric(3)
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> m.metric_matrix(cxc.cart3d, at=at)
    QuantityMatrix([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], '((, , ), (, , ), (, , ))')
    >>> m.signature
    (1, 1, 1)
    ```

(software-spec-euclideanmanifold)=

!!! info `EuclideanManifold`

    `EuclideanManifold` represents flat Euclidean space $\mathbb{R}^n$ with its
    standard smooth structure:

    $$
    (\mathbb{R}^n, \mathcal{A}_E).
    $$

    It is the canonical manifold used for ordinary flat coordinate systems (Cartesian, polar, cylindrical, spherical) when expressed in dimension $n$.

    A `EuclideanManifold(n)` provides both Euclidean smooth and metric structure:

    - `atlas = EuclideanAtlas(n)`
    - `metric = EuclideanMetric(n)`

    with manifold dimension $ \dim M = n. $

    Construction:

    ```text
    EuclideanManifold(ndim: int)
    ```

    **Alias**: `Rn` is an alias for `EuclideanManifold`. Instances print using the `Rn` name rather than `EuclideanManifold` by default (e.g. `repr(EuclideanManifold(3))` yields `Rn(3)`). Pass `alias=False` to `__pdoc__` to get the full class name instead (e.g. `EuclideanManifold(3)`).

    The metric object is attached at construction and is available as `M.metric`.

    The default chart is the canonical Cartesian chart of the same dimension provided by the atlas. For example, `EuclideanManifold(2).default_chart == Cart2D()`.

    Coordinate operations (`pt_map`) are inherited from `AbstractManifold`. These methods verify that the charts belong to the Euclidean atlas before delegating to the chart-level implementations.

    **Example**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> M = cxm.EuclideanManifold(3)
    >>> M.ndim
    3
    >>> M.default_chart
    Cart3D(M=Rn(3))
    >>> M.metric.signature
    (1, 1, 1)
    >>> M.has_chart(cxc.cart3d)
    True
    >>> M.has_chart(cxc.cart2d)
    False

    >>> x = {"x": 1.0, "y": 1.0, "z": 1.0}
    >>> M.pt_map(x, cxc.cart3d, cxc.sph3d)
    {'r': Array(1.73205081, dtype=float64, ...),
     'theta': Array(0.95531662, dtype=float64),
     'phi': Array(0.78539816, dtype=float64, ...)}
    ```

### Hyper-Spherical Manifolds

!!! info `HyperSphericalAtlas`

    `HyperSphericalAtlas` is the atlas for the intrinsic two-sphere $S^2$. It defines which charts are valid coordinate systems on the sphere surface.

    Formally,

    $$
    \mathcal{A}_{S^2} = \{ C \mid C \text{ is an intrinsic chart on } S^2 \}.
    $$

    The atlas determines chart membership only; numerical coordinate transformations are handled by the chart transition system.

    - `ndim = 2`
    - `default_chart() -> SphericalTwoSphere()`

    Supported charts include:

    - `SphericalTwoSphere`
    - `LonLatSphericalTwoSphere`
    - `LonCosLatSphericalTwoSphere`
    - `MathSphericalTwoSphere`

    Charts on other manifolds, such as `Cart2D` or Euclidean 3-space charts, are not members of this atlas.

    As with `EuclideanAtlas`, membership is determined by chart registration rather than by hard-coded enumeration using the ``register`` class method.

(software-spec-hypersphericalmetric)=

!!! info `HyperSphericalMetric`

    `HyperSphericalMetric` is the round Riemannian metric on the unit sphere in hyperspherical coordinates.

    For $S^2$ with chart $(\theta, \phi)$, the metric is

    $$
    g = \begin{pmatrix}
    1 & 0 \\
    0 & \sin^2\theta
    \end{pmatrix}.
    $$

    More generally, diagonal entries follow the cumulative-sine rule:

    $$
    g_{kk} = \prod_{j=0}^{k-1} \sin^2(\theta_j),
    $$

    with $g_{00}=1$.

    Construction:

    ```text
    HyperSphericalMetric(ndim: int)
    ```

    Semantics:

    - `signature = (1,) * ndim`.
    - `metric_matrix(chart, /, *, at, usys=None)` returns either a plain array or a `QuantityMatrix`, depending on whether inputs are unitful.
    - Angular inputs are interpreted in radians by default, or via `usys["angle"]` when a unit system is provided.
    - The returned metric is diagonal in the intrinsic hyperspherical chart basis.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.HyperSphericalMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> m.metric_matrix(cxc.sph2, at=at)
    Array([[1., 0.],
           [0., 1.]], dtype=float64)
    >>> m.signature
    (1, 1)
    ```

(software-spec-twospheremanifold)=

!!! info `HyperSphericalManifold`

    `HyperSphericalManifold` represents the smooth two-sphere $S^2$:

    $$
    (S^2, \mathcal{A}_{S^2}).
    $$

    The manifold describes the curved surface of a sphere with fixed radius and intrinsic geometry. It is commonly used for angular coordinate systems such as longitude–latitude parameterizations.

    Construction:

    ```text
    HyperSphericalManifold(ndim: int = 2)
    ```

    Structure:

    - `atlas = HyperSphericalAtlas()`
    - `metric = HyperSphericalMetric(ndim)`

    The intrinsic dimension is $ \dim S^2 = 2$.

    The metric object is attached at construction time and is available as `M.metric`.

    The atlas provides the canonical spherical chart

    ```text
    SphericalTwoSphere()
    ```

    which uses components `(theta, phi)` with the physics convention `theta` = polar (colatitude) and `phi` = azimuth.

    Chart compatibility:

    The manifold accepts only intrinsic two-sphere charts; planar charts such as `Cart2D` are **not valid** coordinate systems for this manifold.

    Coordinate operations:

    `HyperSphericalManifold` inherits the coordinate transformation API from `AbstractManifold`:

    - `pt_map(...)`

    These methods first verify atlas compatibility before delegating to the chart-level transformation system.

    **Example**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> M = cxm.HyperSphericalManifold()
    >>> M.ndim
    2

    >>> M.default_chart
    SphericalTwoSphere()

    >>> M.metric.signature
    (1, 1)

    >>> M.has_chart(cxc.sph2)
    True

    >>> M.has_chart(cxc.cart2d)
    False
    ```

### Minkowski Manifolds

(software-spec-minkowskiatlas)=

!!! info `MinkowskiAtlas`

    Atlas for Minkowski spacetime $\mathbb{R}^{1,3}$.

    Construction:

    - MinkowskiAtlas() has fixed dimension `ndim = 4`.

    Membership semantics:

    A chart belongs to this atlas iff:

    1. the chart has `ndim == 4`, and
    2. its chart class is registered in the atlas eligibility set (built-in registration includes SpaceTimeCT).

    Built-in chart family:

    - SpaceTimeCT charts with any compatible spatial factor chart, for example:
      - SpaceTimeCT(cart3d)
      - SpaceTimeCT(sph3d)
      - SpaceTimeCT(cyl3d)

    Default chart:

    - default_chart() returns SpaceTimeCT() with Cartesian spatial part.

    Registration API:

    - register(chart_class) adds a chart class to the set of chart classes accepted by membership checks.

    Notes:

    - This atlas defines chart compatibility only.
    - Metric formulas are specified by MinkowskiMetric.

(software-spec-minkowskimetric)=

!!! info `MinkowskiMetric`

    `MinkowskiMetric` is the Lorentzian pseudo-Riemannian metric on Minkowski spacetime.

    In canonical spacetime Cartesian coordinates $(ct, x, y, z)$:

    $$
    \eta = \operatorname{diag}(-1, 1, 1, 1).
    $$

    For general `SpaceTimeCT` charts, the metric is computed by pullback:

    $$
    g = J^T \eta J,
    $$

    where $J$ is the Jacobian of the map from the chosen spacetime chart to its canonical Cartesian counterpart.

    Construction:

    ```text
    MinkowskiMetric()
    ```

    Semantics:

    - `signature = (-1, 1, 1, 1)`.
    - `ndim = 4`.
    - `metric_matrix(chart, /, *, at, usys=None)` returns a `QuantityMatrix`.
    - In canonical Cartesian spacetime coordinates, `metric_matrix` returns `diag(-1, 1, 1, 1)`.
    - In other compatible `SpaceTimeCT` charts, `metric_matrix` returns `J^T η J`.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.MinkowskiMetric()
    >>> at = {
    ...     "ct": jnp.array(0.0),
    ...     "x": jnp.array(0.0),
    ...     "y": jnp.array(0.0),
    ...     "z": jnp.array(0.0),
    ... }
    >>> m.metric_matrix(cxc.spacetimect, at=at).value
    Array([[-1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float64)
    >>> m.signature
    (-1, 1, 1, 1)
    ```

(software-spec-minkowskimanifold)=

!!! info `MinkowskiManifold`

    Minkowski spacetime manifold $( \mathbb{R}^{1,3}, \eta)$ with Lorentzian metric signature $(-1, 1, 1, 1)$.

    Structure:

    - `ndim = 4`
    - `atlas = MinkowskiAtlas(4)`
    - `metric = MinkowskiMetric()`
    - `default_chart = atlas.default_chart()`

    Construction:

    ```text
    MinkowskiManifold(ndim: int = 4)
    ```

    The metric object is attached at construction time and is available as `M.metric`.

    Chart compatibility:

    - has_chart(chart) delegates to the atlas membership rule.
    - Accepted charts are the atlas-supported 4D SpaceTimeCT chart family.
    - Non-spacetime charts such as cart3d are not members.

    Coordinate operations:

    Inherits manifold-level wrappers from AbstractManifold:

    - `pt_map(...)`

    These operations first enforce atlas compatibility, then delegate to chart-level transition machinery.

    Pre-defined instance:

    - minkowski4d is the canonical pre-built MinkowskiManifold() instance.

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> M = cxm.MinkowskiManifold()
    >>> M.metric.signature
    (-1, 1, 1, 1)
    ```

### Custom Manifolds

(software-spec-customatlas)=

!!! info `CustomAtlas`

    `CustomAtlas` is an explicit atlas implementation where chart membership is controlled by user-provided chart classes.

    Construction:

    ```text
    CustomAtlas(
        charts: tuple[type[AbstractChart], ...],
        chart_default: AbstractChart,
    )
    ```

    Semantics:

    - The atlas dimension is defined by `chart_default.ndim`.
    - A chart is supported iff:
        1. its class is in `charts`, and
        2. its `ndim` equals the atlas dimension.
    - `charts` is an ordered tuple of unique chart classes.
    - The default chart class must be present in `charts`.
    - Every registered chart class must be zero-argument constructible and have dimensionality equal to the atlas dimension.

    Unlike `EuclideanAtlas`, `CustomAtlas` does not use automatic chart registration or fallback chart-family membership. Membership is explicit and local to the atlas instance.

    **Example**

    ```pycon
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> A = cxm.CustomAtlas(
    ...     charts=(cxc.Cart2D, cxc.Polar2D),
    ...     chart_default=cxc.cart2d,
    ... )

    >>> A.ndim
    2

    >>> cxc.cart2d in A
    True
    >>> cxc.polar2d in A
    True
    >>> cxc.cart3d in A
    False
    ```

(software-spec-custommetric)=

!!! info `CustomMetric`

    `CustomMetric` is a concrete metric implementation for user-defined manifolds.

    Construction:

    ```text
    CustomMetric(
        metric_matrix: Callable[[AbstractChart], QuantityMatrix | Array],
        signature: tuple[int, ...],
    )
    ```

    Semantics:

    - `metric_matrix(chart, /, *, at, usys=None)` is supplied by the caller and must satisfy the [`AbstractMetric`](#software-spec-abstractmetric) contract.
    - `signature` is the metric signature as a tuple of `+1` and `-1` entries.
    - `ndim = len(signature)`.
    - `CustomMetric` is immutable and registered as a static JAX PyTree, matching the behavior required of all concrete metric types.

    This type exists so users can define metrics for custom manifolds without subclassing `AbstractMetric`.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> def flat_3d(chart, /, *, at):
    ...     return jnp.eye(3)
    ...

    >>> metric = cxm.CustomMetric(metric_matrix=flat_3d, signature=(1, 1, 1))
    >>> metric.signature
    (1, 1, 1)
    >>> metric.ndim
    3
    ```

(software-spec-custommanifold)=

!!! info `CustomManifold`

    `CustomManifold` represents a smooth manifold with a caller-supplied explicit atlas:

    $$
    (M, \mathcal{A}_{\mathrm{custom}}).
    $$

    Construction:

    ```text
    CustomManifold(atlas: CustomAtlas, metric: AbstractMetric)
    ```

    The manifold is intentionally thin: it forwards chart-membership checks, default-chart selection, and point transition wrappers to the provided atlas, while storing an explicit metric object for geometric computations.

    - `ndim = atlas.ndim`
    - `default_chart = atlas.default_chart()`
    - `has_chart(chart) = atlas.has_chart(chart)`
    - `metric` is the caller-supplied metric object.
    - `atlas.ndim` and `metric.ndim` must match; otherwise construction raises `ValueError`.

    Coordinate operations (`pt_map`) are inherited from `AbstractManifold` and therefore enforce atlas compatibility before delegating to chart-level machinery.

    ```pycon
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> A = cxm.CustomAtlas(
    ...     charts=(cxc.Cart2D, cxc.Polar2D),
    ...     chart_default=cxc.cart2d,
    ... )
    >>> M = cxm.CustomManifold(A, cxm.EuclideanMetric(2))

    >>> M.ndim
    2
    >>> M.default_chart
    Cart2D()
    >>> M.metric.signature
    (1, 1)

    >>> M.has_chart(cxc.cart2d)
    True
    >>> M.has_chart(cxc.polar2d)
    True
    >>> M.has_chart(cxc.cart3d)
    False
    ```

### Product Manifolds

(software-spec-cartesianproductatlas)=

!!! info `CartesianProductAtlas`

    Atlas for a Cartesian product manifold $M = M_1 \times M_2 \times \cdots \times M_k$.

    Construction:

    ```text
    CartesianProductAtlas(
        factors: tuple[AbstractAtlas, ...],
        factor_names: tuple[str, ...],
    )
    ```

    The atlas is formed as a Cartesian product of the factor atlases.

    Structure:

    - `ndim = sum(factor.ndim for factor in factors)`
    - `factors` is an ordered tuple of constituent atlases.
    - `factor_names` is an ordered tuple of unique string names for each factor, used as keys for indexing.

    Membership semantics:

    A chart belongs to this atlas iff:

    1. it is a {class}`~coordinax.charts.CartesianProductChart`, and
    2. each factor chart in the product chart belongs to the corresponding factor atlas.

    Default chart:

    - `default_chart()` returns a {class}`~coordinax.charts.CartesianProductChart` constructed from the default charts of each factor atlas.

    Factor access:

    - `atlas[name]` returns the factor atlas corresponding to the given name.
    - Factor names must be unique.

    **Example**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> atlas = cxm.CartesianProductAtlas(
    ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
    ...     factor_names=("S2", "R1"),
    ... )
    >>> atlas.ndim
    3

    >>> default = atlas.default_chart()
    >>> default
    CartesianProductChart(
        factors=(SphericalTwoSphere(), Cart1D()), factor_names=('S2', 'R1')
    )

    >>> atlas["S2"]
    HyperSphericalAtlas(ndim=2)

    >>> product_chart = cxc.CartesianProductChart(
    ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1")
    ... )
    >>> atlas.has_chart(product_chart)
    True

    >>> atlas.has_chart(cxc.sph2)
    False
    ```

(software-spec-cartesianproductmetric)=

!!! info `CartesianProductMetric`

    `CartesianProductMetric` is the canonical metric on a Cartesian product manifold.

    For factor manifolds $(M_i, g_i)$, the product manifold
    $$
    M = M_1 \times M_2 \times \cdots \times M_k
    $$
    carries the metric
    $$
    g_{(p_1,\ldots,p_k)}((v_1,\ldots,v_k),(w_1,\ldots,w_k))
    = \sum_{i=1}^k g_i(v_i, w_i).
    $$

    Construction:

    ```text
    CartesianProductMetric(factors: tuple[AbstractMetric, ...])
    ```

    Semantics:

    - `signature` is the concatenation of factor signatures in product order.
    - `ndim = sum(metric.ndim for metric in factors)`.
    - `metric_matrix(chart, /, *, at, usys=None)` requires a product chart and returns a block-diagonal matrix with one block per factor metric.
    - Each block is the factor metric matrix evaluated at the corresponding factor point extracted from `at` using product-chart factor splitting.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> M = cxm.CartesianProductManifold(
    ...     factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
    ...     factor_names=("S2", "R1"),
    ... )
    >>> metric = M.metric
    >>> metric.signature
    (1, 1, 1)

    >>> chart = cxc.CartesianProductChart(
    ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1")
    ... )
    >>> at = {
    ...     "S2.theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "S2.phi": u.Angle(0.0, "rad"),
    ...     "R1.x": u.Q(1.0, "m"),
    ... }
    >>> g = metric.metric_matrix(chart, at=at)
    >>> g.shape
    (3, 3)
    ```

(software-spec-cartesianproductmanifold)=

!!! info `CartesianProductManifold`

    Manifold defined as a Cartesian product of other manifolds: $M = M_1 \times M_2 \times \cdots \times M_k$.

    Given smooth manifolds $M_1, \ldots, M_k$ with intrinsic dimensions $n_1, \ldots, n_k$, the product manifold $M$ has:

    - Total intrinsic dimension: $\dim(M) = n_1 + n_2 + \cdots + n_k$.
    - **Smooth structure:** The atlas consists of Cartesian product charts $C_1 \times C_2 \times \cdots \times C_k$ where each $C_i$ is from the atlas of $M_i$.
    - **Transition maps:** Factor component-wise. If $\tau_i : C_i^\alpha \to C_i^\beta$ is a transition map on $M_i$, the product transition map is $\tau(p_1, \ldots, p_k) = (\tau_1(p_1), \ldots, \tau_k(p_k))$.

    Construction:

    ```text
    CartesianProductManifold(
        factors: tuple[AbstractManifold, ...],
        factor_names: tuple[str, ...],
    )
    ```

    Structure:

    - `ndim = sum(factor.ndim for factor in factors)`
    - `atlas = CartesianProductAtlas(...)` formed from factor atlases.
    - `metric = CartesianProductMetric(...)` formed from factor metrics.
    - `default_chart = atlas.default_chart()`
    - Factor names must be unique and are used as keys when accessing the product atlas.

    Chart membership:

    - `has_chart(chart)` returns true iff the chart is a {class}`~coordinax.charts.CartesianProductChart` whose factor charts belong to the corresponding factor atlases.

    Coordinate operations:

    Inherits manifold-level wrappers from `AbstractManifold`:

    - `pt_map(...)`

    All operations enforce atlas compatibility and then delegate to chart-level transition machinery.

    **Example**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> M = cxm.CartesianProductManifold(
    ...     factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
    ...     factor_names=("S2", "R1"),
    ... )
    >>> M.ndim
    3

    >>> M.default_chart
    CartesianProductChart(
        factors=(SphericalTwoSphere(), Cart1D()), factor_names=('S2', 'R1')
    )

    >>> M.atlas["S2"]
    HyperSphericalAtlas(ndim=2)

    >>> product_chart = cxc.CartesianProductChart(
    ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1")
    ... )
    >>> M.has_chart(product_chart)
    True

    >>> M.has_chart(cxc.sph2)
    False
    ```

### Embedded Manifolds

(software-spec-abstractembeddingmap)=

!!! info `AbstractEmbeddingMap`

    Abstract base class representing a smooth embedding of an intrinsic manifold into an ambient manifold.

    An embedding is a smooth injective map $$\iota : M \hookrightarrow N$$ from an intrinsic manifold $M$ (the lower-dimensional embedded space) to an ambient manifold $N$ (the higher-dimensional containing space).

    Structure:

    - `intrinsic: AbstractChart` — the chart on the intrinsic manifold
    - `ambient: AbstractChart` — the chart on the ambient manifold

    Abstract interface:

    - `embed(point: CDict, *, usys=None) -> CDict` — maps intrinsic coordinates to ambient coordinates
    - `project(point: CDict, *, usys=None) -> CDict` — maps ambient coordinates back to intrinsic coordinates (inverse or local projection)

    Concrete subclasses (like `CustomEmbeddingMap` or domain-specific embeddings such as `TwoSphereIn3D`) are responsible for implementing the coordinate-level transformation logic.

    Subclasses must implement both `embed()` and `project()` methods. The signatures must preserve the signature property of the ambient chart when appropriate, and correctly handle coordinate transformations with optional unit system support.

    **Example use case**

    An embedding of the 2-sphere $S^2$ into 3D Cartesian space with fixed radius $R$:

    - `intrinsic`: $(\theta, \phi)$ spherical coordinates on $S^2$
    - `ambient`: $(x, y, z)$ Cartesian coordinates in $\mathbb{R}^3$
    - `embed`: $(\theta, \phi) \mapsto (R\sin\theta\cos\phi, R\sin\theta\sin\phi, R\cos\theta)$
    - `project`: drop radial component, recover $(\theta, \phi)$

(software-spec-customembeddingmap)=

!!! info `CustomEmbeddingMap`

    Concrete embedding map defined by user-provided `embed` and `project` functions.

    Construction:

    ```text
    CustomEmbeddingMap(
        intrinsic: AbstractChart,
        ambient: AbstractChart,
        embed_fn: callable,
        project_fn: callable,
    )
    ```

    The `embed_fn` and `project_fn` are callables with signature:

    ```text
    (point: CDict, *, usys: OptUSys = None) -> CDict
    ```

    Semantics:

    - `intrinsic` and `ambient` define the chart types involved in the embedding.
    - `embed_fn` transforms coordinates from intrinsic to ambient space.
    - `project_fn` transforms coordinates from ambient back to intrinsic space.
    - Both functions must respect optional unit system (`usys`) arguments.

    Notes:

    - `CustomEmbeddingMap` allows users to define embeddings without subclassing `AbstractEmbeddingMap`.
    - The embedding functions are user-provided; coordinax does not validate mathematical correctness (e.g., smoothness, injectivity, or consistency between embed/project).
    - This is a thin wrapper ideal for simple coordinate transformations or testing.

    **Example**

    ```pycon
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> # Define a custom embedding of R into R^2 (line in plane)
    >>> def embed_fn(point, *, usys=None):
    ...     return {"x": point["t"], "y": point["t"] * 2}
    ...

    >>> def project_fn(point, *, usys=None):
    ...     return {"t": point["x"]}
    ...

    >>> embed_map = cxm.CustomEmbeddingMap(
    ...     intrinsic=cxc.Cart1D(),
    ...     ambient=cxc.Cart2D(),
    ...     embed_fn=embed_fn,
    ...     project_fn=project_fn,
    ... )
    >>> embed_map
    CustomEmbeddingMap(intrinsic=Cart1D(), ambient=Cart2D(), ...)
    ```

(software-spec-inducedmetric)=

!!! info `InducedMetric`

    `InducedMetric` is the pullback metric on an embedded manifold.

    Given an embedding $\iota : N \hookrightarrow M$, `InducedMetric` constructs the intrinsic metric on $N$ from the ambient metric on $M$ by pullback:

    $$
    g_N = \iota^* g_M,
    $$

    or, in local coordinates,

    $$
    (g_N)_{ij} = (J^T G J)_{ij},
    $$

    where $J = \partial \iota / \partial q$ is the Jacobian of the embedding map and $G$ is the ambient metric evaluated at the embedded point.

    Construction:

    ```text
    InducedMetric(
        embed_map: AbstractEmbeddingMap,
        ambient_metric: AbstractMetric,
    )
    ```

    Semantics:

    - `embed_map` defines the intrinsic and ambient charts used to compute the pullback.
    - `ambient_metric` is evaluated at the embedded point `embed_map.embed(at, usys=usys)`.
    - `metric_matrix(chart, /, *, at, usys=None)` computes the embedding Jacobian and returns `J^T G J` as a `QuantityMatrix`.
    - The current implementation ignores the `chart` argument numerically and computes the induced metric in the embedding map's intrinsic chart.
    - `signature = (1,) * embed_map.intrinsic.ndim`.
    - `ndim = embed_map.intrinsic.ndim`.

    This metric is the canonical metric exposed by [`EmbeddedManifold`](#software-spec-embeddedmanifold).

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> embed_map = cxm.TwoSphereIn3D(radius=u.Q(1.0, "km"))
    >>> metric = cxm.InducedMetric(embed_map, cxm.EuclideanMetric(3))
    >>> at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
    >>> metric.metric_matrix(cxc.sph2, at=at)
    QuantityMatrix([[1., 0.],
                    [0., 1.]], '((km2 / rad2, km2 / rad2), (km2 / rad2, km2 / rad2))')
    >>> metric.signature
    (1, 1)
    ```

(software-spec-embeddedmanifold)=

!!! info `EmbeddedManifold`

    Manifold structure for a smooth embedding of an intrinsic manifold into an ambient manifold.

    An embedded manifold is a triple $(M, N, \iota)$ where:

    - $M$ is the **intrinsic manifold** (the space being embedded)
    - $N$ is the **ambient manifold** (the containing space)
    - $\iota: M \to N$ is a smooth **embedding map**

    Construction:

    ```text
    EmbeddedManifold(
        intrinsic: AbstractManifold,
        ambient: AbstractManifold,
        embed_map: AbstractEmbeddingMap,
    )
    ```

    Structure:

    - `intrinsic: AbstractManifold` — the embedded manifold structure
    - `ambient: AbstractManifold` — the ambient manifold structure
    - `embed_map: AbstractEmbeddingMap` — the smooth embedding defining coordinates transformation
    - `atlas = intrinsic.atlas` — uses the intrinsic manifold's atlas
    - `ndim = intrinsic.ndim` — the dimension is that of the intrinsic manifold
    - `metric = InducedMetric(embed_map, ambient.metric)` — the metric is derived from the embedding and the ambient manifold metric, not passed separately at construction time

    Embedding API:

    - `embed(intrinsic_point, from_intrinsic_chart, to_ambient_chart, *, usys=None) -> CDict`
      Transforms a point from intrinsic coordinates (in a specified chart) to ambient coordinates (in a specified chart).

    - `project(ambient_point, from_ambient_chart, to_intrinsic_chart, *, usys=None) -> CDict`
      Transforms a point from ambient coordinates (in a specified chart) to intrinsic coordinates (in a specified chart).

    Manifold API:

    - `has_chart(chart)` — delegates to `intrinsic.has_chart(chart)`
    - `default_chart` — inherited from intrinsic manifold
    - `metric` — computed property returning the induced pullback metric from the ambient manifold

    Chart membership:

    - Only charts supported by the intrinsic atlas are valid members.
    - Ambient charts are validated at embedding/projection time, not during atlas membership checks.

    Coordinate operations:

    - `pt_embed()` and `pt_project()` are the primary high-level operations.
    - These functions perform chart transitions in both intrinsic and ambient spaces around the embedding.
    - Usage: `pt_embed(intrinsic_point, manifold)` or `pt_embed(intrinsic_point, from_chart, to_chart, manifold)`

    Notes:

    - Realize/unrealize to Cartesian coordinates are not yet implemented (marked TODO).
    - The embedding map encodes both the transition between intrinsic and ambient chart types and any embedding parameters (e.g., radius).
    - Because the metric is induced from the ambient manifold, updating the ambient manifold changes the embedded manifold's geometric metric semantics.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> # Create a 2-sphere embedded in R^3 with radius 2 km
    >>> manifold = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.HyperSphericalManifold(),
    ...     ambient=cxm.EuclideanManifold(3),
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")),
    ... )
    >>> manifold.metric.signature
    (1, 1)

    >>> # Embed a point from spherical (theta, phi) to Cartesian (x, y, z)
    >>> p_int = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> p_amb = cxm.pt_embed(p_int, manifold)
    >>> p_amb
    {'x': Q(2., 'km'), 'y': Q(0., 'km'), 'z': Q(0., 'km')}

    >>> # Project back from ambient to intrinsic
    >>> p_int_recovered = cxm.pt_project(p_amb, manifold)
    >>> p_int_recovered
    {'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}
    ```

(software-spec-embeddedchart)=

!!! info `EmbeddedChart`

    Convenience wrapper chart that pairs an intrinsic chart with an embedding map to define a chart on an embedded manifold.

    `EmbeddedChart` is a lighter-weight alternative to `EmbeddedManifold` for cases where only chart-level coordinate transformations (embedding/projection) are needed.

    Construction:

    ```text
    EmbeddedChart(embed_map: AbstractEmbeddingMap)
    ```

    The intrinsic and ambient charts are derived from the embedding map:

    - `intrinsic: AbstractChart` — retrieved from `embed_map.intrinsic`
    - `ambient: AbstractChart` — retrieved from `embed_map.ambient`

    Chart API:

    - `components` — inherited from intrinsic chart
    - `coord_dimensions` — inherited from intrinsic chart
    - `cartesian` — the ambient chart's Cartesian realization

    Embedding convenience methods:

    - `embed(point) -> CDict` — wraps embedding via the `embed_map`
    - `project(point) -> CDict` — wraps projection via the `embed_map`

    Notes:

    - `EmbeddedChart` delegates dimension and component information to the intrinsic chart.
    - Cartesian realization is delegated through the ambient chart.
    - Explicit chart-level embedding/projection may be preferable to the full `EmbeddedManifold` API when working with points in a single context.

    **Example**

    ```pycon
    >>> import jax.numpy as jnp
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    >>> # Wrap a 2-sphere embedding in a chart
    >>> chart = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))

    >>> # Use as a normal chart with embedded coordinates
    >>> p_int = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> p_amb = cxm.pt_embed(p_int, chart)
    >>> p_amb
    {'r': Q(2., 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    >>> # Chart properties are from the intrinsic chart
    >>> chart.components
    ('theta', 'phi')
    ```

</br>

<a id="software-spec-transforms"></a>

## Transforms

The canonical transformation API is exposed by `coordinax.transforms`, which is typically imported as `import coordinax.transforms as cxfm`.

Frame objects in `coordinax.frames` depend on `coordinax.transforms` for operator definitions; frame transitions are constructed in `coordinax.frames` and returned as `cxfm.AbstractTransform` instances.

### Transformation Groups

Transformation groups classify families of coordinate transformations that preserve particular geometric structures. In _coordinax_, these are represented by subclasses of `AbstractTransformGroup`.

A transformation-group class identifies the **structural category** of a transformation (for example affine, orthogonal, or Lorentz). These classes do **not** represent concrete group elements. Instead they are used to

- classify transformations,
- constrain which transformations are valid for a given manifold,
- support dispatch when constructing or applying frame transformations.

The currently supported transformation-group hierarchy is:

```text
flowchart TD
    Diff["DiffeomorphismGroup"]
    Aff["AffineGroup"]
    E["EuclideanGroup"]
    O["OrthogonalGroup"]
    SO["SpecialOrthogonalGroup"]
    Lor["LorentzGroup"]
    LorP["ProperOrthochronousLorentzGroup"]
    Point["PoincareGroup"]
    Id["IdentityGroup"]

    Diff --> Aff
    Aff --> E
    Aff --> O
    O --> SO
    O --> Lor
    Lor --> LorP
    Diff --> Point
```

Each group corresponds to a set of transformations preserving a particular geometric structure.

(software-spec-identitygroup)=

!!! info `IdentityGroup`

    The trivial transformation group containing only the identity map.

    Its single element acts as

    $$
    p \mapsto p
    $$

    for every point $p$ on the manifold.

(software-spec-diffeomorphismgroup)=

!!! info `DiffeomorphismGroup`

    The group of smooth invertible self-maps of a manifold.

    Its elements are **diffeomorphisms**

    $$
    f : M \to M
    $$

    such that both $f$ and $f^{-1}$ are smooth.

    This is the largest natural transformation group associated with a smooth manifold.

(software-spec-affinegroup)=

!!! info `AffineGroup`

    The group of affine transformations of an affine space.

    In coordinates, an affine transformation takes the form

    $$
    x \mapsto A x + b
    $$

    where $A$ is an invertible linear map and $b$ is a translation vector.

    Affine transformations preserve affine combinations and parallelism.

(software-spec-euclideangroup)=

!!! info `EuclideanGroup`

    The group of Euclidean isometries of Euclidean space.

    Its elements preserve the Euclidean metric

    $$
    d(f(x), f(y)) = d(x, y)
    $$

    for all points $x, y$.

    In coordinates these correspond to **rigid motions**, including

    - translations
    - rotations
    - reflections

(software-spec-orthogonalgroup)=

!!! info `OrthogonalGroup`

    The group of orthogonal linear transformations.

    These transformations preserve a quadratic form and fix the origin. In Euclidean space they satisfy

    $$
    Q^{\mathsf T} Q = I.
    $$

    Elements correspond to rotations and reflections about the origin.

(software-spec-specialorthogonalgroup)=

!!! info `SpecialOrthogonalGroup`

    The subgroup of the orthogonal group with determinant

    $$
    \det(Q) = +1.
    $$

    These transformations preserve both the inner product and the orientation of space. In Euclidean space they correspond to **rotations**.

(software-spec-lorentzgroup)=

!!! info `LorentzGroup`

    The group of linear isometries of Minkowski spacetime.

    Its elements preserve the Minkowski bilinear form

    $$
    \eta(v, w).
    $$

    Equivalently, matrices in this group satisfy

    $$
    \Lambda^{\mathsf T} \eta \Lambda = \eta.
    $$

(software-spec-properorthochronouslorentzgroup)=

!!! info `ProperOrthochronousLorentzGroup`

    The identity component of the Lorentz group.

    These transformations preserve

    - spatial orientation
    - time orientation

    and are continuously connected to the identity transformation.

(software-spec-poincaregroup)=

!!! info `PoincareGroup`

    The group of isometries of Minkowski spacetime.

    It is the semidirect product

    $$
    \mathbb{R}^{1,3} \rtimes O(1,3)
    $$

    consisting of spacetime translations combined with Lorentz transformations.

    The Poincaré group preserves the Minkowski metric and therefore the spacetime interval.

</br>

### Concrete Transforms

(software-spec-transforms-identity)=

!!! info `Identity`

    The identity transformation, which acts as the identity map on all representations.

    $$
    I : p \mapsto p
    $$

(software-spec-transforms-translate)=

!!! info `Translate`

    A **Translate** is a transformation that adds a constant displacement to position components while leaving other representations unchanged.

    **Mathematical definition**:

    The transform shifts all points by a constant displacement vector:

    $$
    F(p) = p + a .
    $$

    In Cartesian coordinates this is $ x’ = x + a .$

    A time-dependent translation replaces the constant $a$ with a smooth curve $a(t)$:

    $$
    F_t(p) = p + a(t).
    $$

    **Fields:**

    - `delta : CDict | Callable[[tau], CDict]` — the position offset $\Delta x$. If callable, evaluated at the time parameter `tau`.
    - `chart : AbstractChart` — the chart in which `delta` is expressed (static).
    - `right_add : bool` (default `True`) — whether to compute $x + \Delta x$ (``True``) or $\Delta x + x$ (``False``).

    **Inverse:**

    ```text
    translate.inverse == Translate(-delta, chart)
    ```

    **Composition:** Two `Translate` instances with the same chart combine by adding their `delta` values:

    ```text
    Translate(delta1) + Translate(delta2) == Translate(delta1 + delta2)
    ```

(software-spec-transforms-rotate)=

!!! info `Rotate`

    A **Rotate** is a transformation that applies a linear orthogonal map to position components while leaving other representations unchanged.

    **Mathematical definition**:

    A rotation is a linear transformation preserving orientation and distances in Euclidean space. In $\mathbb{R}^n$, rotations are represented by orthogonal matrices with unit determinant:

    $$
    R^T R = I, \quad \det R = 1 .
    $$

    Rotations form the special orthogonal group $ SO(n).$ A time-dependent rotation replaces the fixed matrix $R$ with a smooth path $R(t) \in SO(n)$:

    $$
    F_t(p) = R(t)\, p.
    $$

    **Fields:**

    - `matrix : CDict | Callable[[tau], CDict]` — the rotation matrix $Q$. If callable, evaluated at the time parameter `tau`.
    - `chart : AbstractChart` — the chart in which `matrix` is expressed (static).

    **Inverse:**

    ```text
    rotate.inverse == Rotate(matrix.T, chart)
    ```

    **Composition:** Two `Rotate` instances with the same chart combine by matrix multiplication of their `matrix` fields:

    ```text
    Rotate(Q1) + Rotate(Q2) == Rotate(Q2 @ Q1)
    ```

!!! info `Reflect`

    A **Reflect** is a transformation that applies a linear orthogonal map with determinant -1 to position components while leaving other representations unchanged.

    **Mathematical definition**:

    A reflection is a linear transformation that reverses orientation across a hyperplane. Reflections preserve distances but have determinant -1.

    In Euclidean space, reflection across the hyperplane orthogonal to a nonzero normal vector $n$ is represented by the Householder matrix

    $$
    H_n = I - 2 \hat{n}\hat{n}^T,
    $$

    where $\hat{n} = n / \|n\|$. The corresponding transformation law is

    $$
    F(p) = H_n p.
    $$

    This matrix is orthogonal, symmetric, and involutive:

    $$
    H_n^T H_n = I, \qquad H_n^T = H_n, \qquad H_n^2 = I.
    $$

    In `coordinax`, the `Reflect` transform denotes exactly this hyperplane reflection semantics.

    Together with rotations, reflections generate the orthogonal group $ O(n). $

    **Fields:**

    - `matrix : CDict | Callable[[tau], CDict]` — the reflection matrix $Q$. If callable, evaluated at the time parameter `tau`.
    - `chart : AbstractChart` — the chart in which `matrix` is expressed (static).

    **Inverse:**

    ```text
    reflect.inverse == Reflect(matrix.T, chart)
    ```

    **Composition:** Two `Reflect` instances with the same chart combine by matrix multiplication of their `matrix` fields:

    ```text
    Reflect(Q1) + Reflect(Q2) == Reflect(Q2 @ Q1)
    ```

(software-spec-transforms-scaling)=

!!! info `Scale`

    A **Scale** is a transformation that applies a linear scaling to position components while leaving other representations unchanged.

    **Mathematical definition**:

    A scaling is a linear map that rescales coordinate magnitudes along one or more axes. In Cartesian coordinates with diagonal scale factors $s = (s_1, \ldots, s_n)$:

    $$
    F(p) = S p, \qquad S = \operatorname{diag}(s_1, \ldots, s_n).
    $$

    Uniform scaling uses $s_1 = \cdots = s_n = s$; anisotropic scaling allows different factors per axis. Invertibility requires $s_i \neq 0$ for all $i$.

    In `coordinax`, the `Scale` transform denotes this diagonal linear scaling semantics.

    **Fields:**

    - `factor : float | Callable[[tau], float]` — the scaling factor $s$. If callable, evaluated at the time parameter `tau`.
    - `chart : AbstractChart` — the chart in which `factor` is expressed (static).

    **Inverse:**

    ```text
    scaling.inverse == Scaling(1 / factor, chart)
    ```

    **Composition:** Two `Scaling` instances with the same chart combine by multiplying their `factor` values:

    ```text
    Scaling(s1) + Scaling(s2) == Scaling(s1 * s2)
    ```

!!! info `Shear`

    A **Shear** is a transformation that applies a linear shear map to position components while leaving other representations unchanged.

    **Mathematical definition**:

    A shear is a linear affine transform that preserves parallelism while shifting coordinates along one axis proportionally to another. In matrix form:

    $$
    F(p) = H p,
    $$

    where $H$ is an invertible shear matrix (typically with ones on the diagonal and one or more off-diagonal shear coefficients).

    In `coordinax`, the `Shear` transform denotes this purely spatial linear shear semantics.

    **Fields:**

    - `factor : float | Callable[[tau], float]` — the shear factor $k$. If callable, evaluated at the time parameter `tau`.
    - `chart : AbstractChart` — the chart in which `factor` is expressed (static).

    **Inverse:**

    ```text
    shear.inverse == Shear(-factor, chart)
    ```

    **Composition:** Two `Shear` instances with the same chart combine by adding their `factor` values:

    ```text
    Shear(k1) + Shear(k2) == Shear(k1 + k2)
    ```

</br>

(software-spec-frames)=

## Frames

The canonical frame API is exposed by `coordinax.frames`, which is typically imported as `import coordinax.frames as cxf`.

A **reference frame** is an abstract label used to identify a coordinate description of points. Frame objects do not store coordinates; they classify which frame is in use.

**Frame transitions** (`frame_transition`) map a pair of frames to the `AbstractTransform` that converts from one to the other. This transform is then applied via `act`.

(software-spec-abstractreferenceframe)=

!!! info `AbstractReferenceFrame`

    Abstract base class for reference frames.

(software-spec-alice)=

!!! info `Alice` and `Alex`

    Two example reference frames for testing and illustration.

    - `frame_transition(Alice, Alice)` → `Identity()`
    - `frame_transition(Alex, Alex)` → `Identity()`
    - `frame_transition(Alice, Alex)` → a `Translate | Rotate` composition.
