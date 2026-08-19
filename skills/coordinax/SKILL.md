---
name: coordinax
description: Use when writing, reviewing, or debugging code that imports coordinax (`cx.Point`, `cx.Coordinate`, `cx.Tangent`, `cx.Angle`, `cx.Distance`, `coordinax.charts`, `coordinax.frames`, `coordinax.manifolds`, `coordinax.transforms`), or that converts coordinates between charts, frames, or bases in JAX. Also use when `pt_map`/`cconvert` raises `KeyError` or `NoGlobalCartesianChartError`, when two vectors that "should" be equal compare `False`, when a `Distance` or `Angle` silently becomes a plain `Quantity`, when a `Tangent` operation complains about a missing `at=` anchor, when a metric or chart function breaks on batched input, or when coordinax code is unexpectedly slow under `jax.jit`.
---

# Using Coordinax Effectively

`coordinax` is differential geometry as data structures, in JAX. A position is not an array of three numbers — it is numbers **plus** a chart (which coordinate system) **plus** a representation (what kind of geometric object) **plus** a frame (from whose perspective). Every conversion in the library is a function of that metadata, which is why the metadata is worth understanding before the API.

Objects are `equinox.Module` pytrees and `quax.ArrayValue`s, so they flow through `jit`, `vmap`, and `grad`, and through ordinary-looking `jax.numpy` code.

Checked against coordinax on `main` (2026-08), with unxt 2.0.2, quax 0.4.x, plum 2.8+, Python >=3.12. Docs: <https://coordinax.readthedocs.io>. The normative definitions live in [`docs/spec.md`](../../docs/spec.md); this skill is the practical layer on top of it.

**Read the [`quax` agent skill](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md) for anything about dispatch, `quax.register`, or the quaxify boundary**, and the [`quaxed` skill](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md) for the `quaxed.numpy` fallback hazard. This skill does not restate either.

## The four layers

Almost every confusion with this library is a confusion between two of these:

| Layer | What it is | Carries data? |
| --- | --- | --- |
| **Chart** | Component names + physical dimensions: `Cart3D` is `(x, y, z)` lengths | No |
| **Representation** | The triple (geometry kind, basis, semantic kind) — e.g. `point`, `coord_vel` | No |
| **Vector** | Data + chart + representation + frame: `Point`, `Tangent`, `Coordinate` | Yes |
| **Manifold** | The space the chart charts, plus its metric: `Rn(3)`, `S2` | No |

A chart is **not** a representation. `Cart3D` tells you the components are `x, y, z` in metres; it does not tell you whether they are a position, a velocity, or a displacement — that is the representation, and it is what selects the transformation law.

Two consequences that catch people:

- `pt_map` works on **raw dicts** and only knows about charts. `cconvert` works on **vectors** and is representation-aware. They are not interchangeable.
- Converting a `Point` needs only the chart pair. Converting a `Tangent` needs a base point too, because the Jacobian is evaluated somewhere.

## Quick start

Three levels of API, cheapest first. Pick the lowest one that carries the metadata you actually need.

### Level 1 — dicts and charts

```pycon
>>> import coordinax as cx
>>> import unxt as u

>>> q = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
>>> cx.pt_map(q, cx.cart3d, cx.sph3d)
{'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'), 'phi': Q(1.10714872, 'rad')}

```

`cdict` and `carray` move between the packed and unpacked forms:

```pycon
>>> import jax.numpy as jnp

>>> cx.cdict(jnp.array([1.0, 2.0, 3.0]), cx.cart3d)
{'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64), 'z': Array(3., dtype=float64)}

>>> cx.carray(q, cx.cart3d)
QM([1., 2., 3.], '(km, km, km)')

```

### Level 2 — `Point`

```pycon
>>> p = cx.Point.from_([1.0, 2.0, 3.0], "m")
>>> print(p)
<Point: chart=Cart3D (x, y, z) [m]
    [1. 2. 3.]>

>>> print(p.cconvert(cx.sph3d))
<Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
    [3.742 0.641 1.107]>

```

### Level 3 — `Coordinate`

A base `Point` plus named `Tangent` fibres. One `cconvert` moves the whole bundle, pushing each fibre forward with the Jacobian at the base:

```pycon
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr

>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d,
...     cxr.coord_vel,
... )
>>> pv = cx.Coordinate(point=p, velocity=vel)
>>> pv_sph = pv.cconvert(cx.sph3d)
>>> pv_sph.point.chart
Spherical3D(M=Rn(3))

```

### Choosing a level

| Use | When |
| --- | --- |
| dict + `pt_map` | Hot inner loops; you already know what the numbers mean. No safety net. |
| `Point` / `Tangent` | You want the chart and frame checked and carried for you. |
| `Coordinate` | Position and its derivatives must stay consistent through every chart/frame change. |

All three are equally fast **inside** `jit` — see [Performance](#performance). The difference is what the type system enforces, and what a pytree boundary costs.

## Tangents have an anchor

A tangent vector lives in the tangent space _at a point_. The library enforces this: chart conversion of a `Tangent` requires `at=`, and the anchor must be in the same chart as the tangent.

```pycon
>>> base = cx.Point.from_([1.0, 0.0, 0.0], "m")
>>> v = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d,
...     cxr.coord_vel,
... )
>>> print(v.cconvert(cx.sph3d, at=base))
<Tangent: chart=Spherical3D (r[m / s], theta[rad / s], phi[rad / s])
    [ 1. -0.  0.]>

```

The same rule governs arithmetic. Adding two tangents is only meaningful when they are anchored at the same point and expressed in the same chart — mixing charts is a category error, not a conversion the library will silently perform for you. If you need position-and-velocity to travel together, use `Coordinate` rather than carrying a loose `Point` and `Tangent` and hoping they stay in sync.

### `coord_basis` vs `phys_basis` — the expensive mistake

A tangent's components depend on the basis, and in a curvilinear chart the two choices differ by the chart's scale factors:

| Basis | Components |
| --- | --- |
| `coord_basis` (`coord_vel`) | Coordinate basis $\partial_i$ — `theta` component is in `rad/s` |
| `phys_basis` (`phys_vel`) | Orthonormal physical frame — every component is in `m/s` |

Reading `coord_vel` components as if they were physical velocities is a silent factor-of-$r$ (or $r\sin\theta$) error that no type check will catch, because both are valid `Tangent`s. Convert explicitly, at a base point:

```pycon
>>> p_sph = cx.Point.from_(
...     {"r": u.Q(2.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.0, "rad")}, cxc.sph3d
... )
>>> v_sph = cx.Tangent.from_(
...     {"r": u.Q(0.0, "m/s"), "theta": u.Q(1.0, "rad/s"), "phi": u.Q(0.0, "rad/s")},
...     cxc.sph3d,
...     cxr.coord_vel,
... )
>>> v_phys = cxr.change_basis(v_sph, cxr.phys_basis, at=p_sph)
>>> print(v_phys)
<Tangent: chart=Spherical3D (r, theta, phi) [m / s]
    [0. 2. 0.]>

```

The `theta` component went from `1 rad/s` to `2 m/s`: the scale factor $r = 2\,$m. `change_basis` also promotes a `Point` to a `Displacement` tangent when you pass it a point (the numbers are unchanged; only the geometric interpretation is).

## `==` is strict; `equivalent` is geometric

`==` compares type, chart, frame, **and** data. It short-circuits to `False` when the metadata differs, rather than raising on mismatched component names:

```pycon
>>> p_sph = p.cconvert(cx.sph3d)
>>> p == p_sph
Array(False, dtype=bool)

>>> cx.equivalent(p, p_sph)
Array(True, dtype=bool)

```

`equivalent` is the coordinate-free relation: invariant to chart and to component units, but **frame-strict** — two vectors in different frames denote different physical points and are never equivalent. Use `==` only when you mean "same object, same coordinates"; use `equivalent` for "same point in space".

## `Angle` and `Distance` are not plain Quantities

They are constrained quantities: `Angle` carries wrapping semantics on $S^1$; `Distance` is non-negative. Operations that cannot preserve the constraint **degrade to a plain `Quantity` by design** — this is not a bug to work around:

```pycon
>>> d = cx.Distance(10.0, "kpc")
>>> d + cx.Distance(1.0, "kpc")  # closed: sign is a theorem
Distance(11., 'kpc')

>>> -d  # not closed: degrades
Q(-10., 'kpc')

>>> d - cx.Distance(20.0, "kpc")  # not closed: degrades
Q(-10., 'kpc')

```

If a downstream function annotates `Distance` and you feed it the result of a subtraction, that is where the failure will surface — reconstruct explicitly rather than loosening the annotation. Prefer `u.Q` over `u.Quantity` when building plain quantities, matching the rest of the codebase.

## Performance

Two rules, and they account for nearly all of the difference.

**1. Charts and representations are static.** They are `register_static` pytrees. Close over them; never pass one as a traced argument or route it through `static_argnums` in a hot loop. `pt_map` has a curried form for exactly this:

```pycon
>>> import jax

>>> usys = u.unitsystems.si
>>> c2s = jax.jit(jax.vmap(cx.pt_map(cx.cart3d, cx.sph3d, usys=usys)))
>>> c2s(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
Array([[3.74165739, 0.64052231, 1.10714872],
       [8.77496439, 0.81788856, 0.89605538]], dtype=float64)

```

Closed over this way, the coordinax version costs the same as hand-written `jnp` — the objects exist only inside the trace. Passing the charts in as arguments instead is dramatically slower. See [`docs/guides/perf.md`](../../docs/guides/perf.md) for the measurements.

**2. Keep pytrees off the jit boundary.** `Point`/`Coordinate` are pytrees; flattening and unflattening them is paid per call, not per element. Structure hot code as raw arrays in, raw arrays out, and build the objects _inside_.

**3. Write scalar, batch with `vmap`.** The library is scalar-first by design: functions are written for a single point and you batch them yourself. Reaching for a reshape because a function "should take an (N, 3) array" is the wrong move — `vmap` it.

## Frames

A frame is the observer. `noframe` is the default, and frame changes are a separate axis from chart changes:

```pycon
>>> import coordinax.frames as cxf

>>> p_alice = cx.Point.from_([1.0, 2.0, 3.0], "km", cxf.alice)
>>> p_alice.to_frame(cxf.alice) is p_alice  # identity is a no-op
True

>>> p_alice.cconvert(cx.sph3d).frame  # cconvert preserves the frame
Alice()

```

`frame_transition(a, b)` builds and fuses the operator chain between two frames. Astronomy frames (ICRS, Galactic, Galactocentric, ...) need the `[astro]` extra and register themselves through entry points, so import order does not matter. Time-dependent frames take `t=`:

```
p.to_frame(bar_frame, t=u.Q(500.0, "Myr"))
```

## Manifolds and metrics

Metrics answer the geometric questions — lengths, angles, distances — and they all need to know the chart, and usually the point:

```pycon
>>> import coordinax.manifolds as cxm

>>> cxm.S2, cxm.S2.metric
(HyperSphericalManifold(ndim=2), RoundMetric(ndim=2))

>>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
>>> north = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.0, "rad")}
>>> east = {"theta": u.Angle(0.0, "rad"), "phi": u.Angle(1.0, "rad")}
>>> cxm.S2.angle_between(cxc.sph2, north, east, at=at)
Angle(1.57079633, 'rad')

```

Note the argument orders differ: `angle_between(chart, u, v, at=...)` but `norm(v, chart, at=...)`. Check the signature rather than assuming.

Distance between two points comes in two flavours, and picking the wrong one is a real error rather than a rounding difference:

```pycon
>>> a = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
>>> b = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.pi / 2, "rad")}

>>> cxm.geodesic_distance(cxm.S2, cxc.sph2, a, b)  # along the manifold
Q(1.57079633, 'rad')

>>> cxm.chord_distance(cxm.S2, cxc.sph2, a, b)  # straight through the ambient
Q(1.41421356, '')

```

Causal verbs for Lorentzian metrics (`interval`, `proper_time`, `proper_distance`, `rapidity_between`, `causal_character`) live in `coordinax.manifolds.lorentzian` and are gated on the metric type, so they are unavailable — rather than wrong — on a Riemannian manifold.

Not every chart has a global Cartesian chart, and asking for one is an error rather than a silent approximation:

```pycon
>>> cxc.cartesian_chart(cxc.sph2)
Traceback (most recent call last):
    ...
coordinax._src.exceptions.NoGlobalCartesianChartError: SphericalTwoSphere has no global Cartesian representation. ...

```

That is the point of the type: $S^2$ is not $\mathbb{R}^2$. Use an `EmbeddedChart` when you need ambient coordinates.

## Extending coordinax from outside

Extend the dispatch API, not the internals. `coordinaxs.api` exists precisely so downstream packages have a stable surface to register against; nothing should import from `coordinax._src`.

- **A new chart**: subclass the right abstract chart, then register transition maps. In practice you register to and from an existing chart and let composition route the rest — but check the route does not cycle.
- **A new frame**: register it through the entry point mechanism so import order stays irrelevant (`coordinaxs.astro` is the reference implementation).
- **A new array-ish type**: that is a `quax` question — see the quax skill.
- **Register on your own types.** A dispatch annotated with a coordinax abstract base competes with coordinax's own methods and will be ambiguous rather than additive.
- **Never parametrize a generic in a plum signature.** Write `chart: AbstractChart` with `# type: ignore[type-arg]`, not `AbstractChart[Any, Any, Any]` — the parametrized form breaks plum's matching _and_ disables its method cache for the whole function, including everyone else's registered methods.
- Registration happens on import, so the registering module must actually be imported. Check it landed with `len(some_function.methods)`.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `KeyError: 'y'` from `pt_map` | The dict does not have the source chart's components. `pt_map` trusts the chart you pass; check it matches the keys. |
| `NoGlobalCartesianChartError` | The chart's manifold has no global Cartesian cover ($S^2$, ...). Use an `EmbeddedChart`, not a workaround. |
| `TypeError: Tangent requires a TangentGeometry representation` | `Tangent.from_(x, "m")` guessed `point_geom` from the length dimension. Pass a rate unit, or pass the representation explicitly. |
| `NotFoundLookupError` naming a chart/metric/rep | No dispatch for that type combination. Read the "closest candidates" list — usually an argument-order or a chart-vs-dict mismatch. |
| `AmbiguousLookupError` | Two dispatches match equally. Usually a downstream method annotated on a coordinax abstract base; narrow it to your own type. |
| Two vectors that should match compare `False` | `==` is strict on chart and frame. Use `equivalent` for the geometric question. |
| A `Distance`/`Angle` came back as a plain `Q` | Deliberate degradation: the operation could not preserve the constraint. Reconstruct explicitly if you need the constrained type back. |
| Velocity components are off by a factor of `r` or `r*sin(theta)` | `coord_basis` vs `phys_basis`. Convert with `change_basis(..., at=point)`. |
| Shape/broadcast error inside a metric or chart function | Scalar-first code given batched input. `vmap` the call; if it is a library function, that is a batch-safety bug worth reporting. |
| Correct but ~100x slower than expected | A chart/representation is crossing the jit boundary as a traced arg, or pytrees are crossing per call. Close over the static objects. |
| `jnp.<f>` returns a bare `Array`, stripping the coordinax type | The quaxed fallback, not a coordinax defect. See the quaxed skill. |
| Astro frames missing (`cxf.ICRS` absent) | `coordinaxs.astro` not installed. `pip install "coordinax[astro]"`. |

## Version notes

Coordinax is pre-1.0 and moves; several recent renames are ones prior knowledge will get wrong. Current spellings:

| Old / wrong | Current |
| --- | --- |
| `prolong` | `coordinax.transforms.act_jet` |
| `Parametric` | `coordinax.transforms.TimeDep` |
| `materialize_transform` | `coordinax.transforms.evaluate_at` |
| Group markers in `coordinax.transforms` | `coordinax.transforms.groups` (`EuclideanGroup`, `LorentzGroup`, ...) |
| Callable operator parameters | `coordinax.transforms.builders` (`RotationAboutAxis`, `UniformTranslation`) |
| `separation` | `coordinax.manifolds.geodesic_distance` (with `chord_distance` as the ambient sibling) |
| Causal verbs on `coordinax.manifolds` | `coordinax.manifolds.lorentzian` |
| `AbstractAtlas.default_chart_for` | Removed |
| `coordinax.quantity_matrix` | Removed — use `unxts.linalg` |

Requires unxt >=2.0.2 (plus `unxts.linalg`) and Python >=3.12. The distribution split matters when reading imports: `coordinax` is a regular package, while the optional sub-distributions (`coordinaxs.api`, `coordinaxs.astro`, `coordinaxs.curveframes`, `coordinaxs.hypothesis`, `coordinaxs.interop.astropy`) live in the separate `coordinaxs` PEP 420 namespace. Minimum supported dependencies follow [SPEC 0](https://scientific-python.org/specs/spec-0000/).
