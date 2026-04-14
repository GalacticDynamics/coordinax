# Coordinax Astro Specification

This document is the **normative** specification for the mathematical and software design of `coordinax.astro`. The goals are:

1. **Correctness-first foundation**: rigorous definitions of points, tangents, their transformations, etc.
2. **Ergonomic high-level API**: common tasks should not expose low-level details.
3. **Extensibility via multiple dispatch**: functional API for existing functions are added by registration, not by editing core logic.
4. **JAX compatibility**: dispatch resolves on static Python objects; numerical kernels are pure and traceable.

## Table of Contents

```{contents}
:depth: 2
```

---

# The Math

## Distance Moduli

The **distance modulus** $\mu$ is a logarithmic measure of distance used in astronomy. It relates a celestial object's apparent magnitude $m$ and absolute magnitude $M$:

$$
\mu \;=\; m - M.
$$

Given a physical distance $d$ (in parsecs) the distance modulus is

$$
\mu \;=\; 5\,\log_{10}\!\bigl(d/\text{pc}\bigr) - 5,
$$

or equivalently

$$
\mu \;=\; 5\,\log_{10}\!\bigl(d/10\,\text{pc}\bigr).
$$

Inverting the relation gives the distance in parsecs:

$$
d \;=\; 10^{\,1 + \mu/5}\;\text{pc}.
$$

$\mu$ is expressed in **magnitudes** (`mag`), a dimensionless logarithmic unit. All constructors and conversions in `coordinax.astro` enforce this unit constraint.

## Parallaxes

**Trigonometric parallax** $p$ is the apparent angular shift of a source caused by the observer's orbital motion. It provides a direct, geometric measurement of distance.

Let $b$ denote the baseline length (conventionally $b = 1\,\text{AU}$, the Earth–Sun distance). For a source at distance $d$, the parallax angle $p$ is defined by

$$
\tan p \;=\; \frac{b}{d}.
$$

Because astronomical parallaxes are small ($p \ll 1$), the small-angle approximation $\tan p \approx p$ is often used, but `coordinax.astro` retains the exact $\tan$ form.

Inverting the definition:

$$
d \;=\; \frac{b}{\tan p} \;=\; \frac{1\,\text{AU}}{\tan p}.
$$

The result carries length units; the implementation preserves the natural unit of the quotient (AU) and lets the caller convert.

Given a distance $d$ in parsecs:

$$
p \;=\; \arctan\!\Bigl(\frac{1\,\text{AU}}{d}\Bigr).
$$

By definition of the parsec, a source at $d = 1\,\text{pc}$ has a parallax of $1''$ (one arcsecond).

Combining with the distance-modulus formula:

$$
\mu \;=\; 5\,\log_{10}\!\Bigl(\frac{1\,\text{AU}}{\tan(p)\;\text{pc}}\Bigr) - 5.
$$

- $p$ must have **angular** dimensions.
- $p$ must be **non-negative** (negative parallax is unphysical); this is checked at construction time by default.
- The baseline length $b = 1\,\text{AU}$ is a module-level constant (`parallax_base_length`).

---

# Packages

```{contents}
:local:
:depth: 1
```

## `coordinax.astro`

!!! info `DistanceModulus`

    Magnitude-space distance representation used in astronomical workflows and conversions.

!!! info `Parallax`

    Angular distance proxy represented in angular units and linked to distance conversion flows.

## `coordinax.hypothesis.astro`

!!! info `distance_moduli`

    `hypothesis` strategy for generating `coordinax.astro.DistanceModulus` instances.

!!! info `parallaxes`

    `hypothesis` strategy for generating `coordinax.astro.Parallax` instances.
