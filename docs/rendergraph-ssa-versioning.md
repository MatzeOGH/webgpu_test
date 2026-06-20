# SSA-style resource versioning in compile()

## Context
`compile()` phase 1 used to track **one producer per resource id, last-writer-wins**: a single
`producer[id]` table, then a sweep adding a RAW edge from each reader to that one producer. Two
passes writing the same resource (a depth prepass + a main pass, ping-pong, accumulation) got **no
edge between them** (WAW unmodeled → undefined order → corruption), and a write after a read of the
same resource got no edge either (WAR unmodeled → the read could be clobbered before it ran). Both
gaps were listed in `RenderGraphsTODO.md`.

This change makes a resource carry an implicit **version**: each write starts a new version, each
read binds to the current one, and the three hazards (RAW/WAW/WAR) fall out of a single walk.

## Design — implicit, declaration-order SSA
No public API change. `ResourceHandle` stays `{ uint32_t id }`; every `GraphBuilder` method keeps its
signature; no call site changes. "Version" is purely a `compile()`-internal concept — the **writing
pass pointer is the version identity**, so no renaming/numbering and (because a flat pass list has no
control-flow merges) no phi nodes. We deliberately did *not* use Myth-Engine-style explicit
handle-threading (`write()` returns a new handle): it would change the API and every call site for no
benefit here.

The sweep walks `m_passes` in list order (== declaration order, since `list_append` appends at the
tail) and each pass's `accesses[]` in call order. Per resource id it keeps two pieces of scratch
state (both from the per-frame **scratch arena** — `scratch_alloc` / `scratch_make`, reclaimed via
`defer { reset_scratch(); }` — not `calloc`):

- `currentProducer[id]` — the pass holding the current version.
- `pendingReaders[id]` — passes that read the current version and haven't been retired by a newer write.

A read seen before any writer simply binds to "no producer" and emits no edge; detecting that authoring
error is **not** the sweep's job — it's done by a separate post-cull pass (below) over the final schedule.

```
on write(id):
    if currentProducer[id] and != self:  add_dependency(self -> currentProducer[id])   # WAW
    for r in pendingReaders[id]: if r != self: add_dependency(self -> r)                # WAR
    currentProducer[id] = self;  pendingReaders[id] = empty                             # new version
on read(id):
    if currentProducer[id] not null and != self: add_dependency(self -> currentProducer[id])  # RAW
    pendingReaders[id] += self                       # (read before any writer: no edge -- flagged later)
```

The sweep is factored into a shared helper `sweep_resource_versions(...)` templated on a single
`onEdge(dependent, dep, id, kind)` callback (with a `HazardKind { RAW, WAW, WAR }` enum). `compile()`
maps every `onEdge` to the existing `add_dependency` (whose dedup collapses duplicate `(dependent, dep)`
pairs, so the id/kind are ignored there); `debug_print_mermaid()` reuses the *same* helper to draw
labelled edges, so the dump can never drift from the real graph. Both self-guards are load-bearing:
read-then-write of one handle in a single pass would otherwise WAR-self-edge; write-then-write would
WAW-self-edge.

**Trade-off:** the old phase 1 was declaration-order-*independent* (a reader could be declared before
its producer). The new model requires producers declared before consumers, and enforces it: after
dead-node culling, a **post-cull pass walks the final schedule in execution order** and flags any read
of a **transient** resource that no earlier surviving pass has produced. Such a read would sample
uninitialized contents (its writer was declared after it, or culled), so **`compile()` prints the
offending pass/resource and returns `false`** (no `assert`/`abort` — behavior is identical in debug and
release; the caller must skip `realize()`/`execute()` that frame). Two exemptions prevent false
positives: **imported** resources (their value comes from outside the graph) and resources with **no
writer at all** (e.g. a host-uploaded uniform buffer filled via `wgpuQueueWriteBuffer`).

Why this and not order-independence? The access lists carry only `{pass, id, read|write}`. For a
single-writer resource that uniquely determines every binding, but for a **multi-writer** resource it
does not — swapping two writers leaves the access lists byte-identical yet inverts which version each
reader sees, so declaration order is the only available disambiguator under this API. Truly
order-independent multi-writer would need author-supplied explicit version labels (deferred).

**Bonus — acyclic by construction:** every edge points from a later-visited pass back to a
strictly-earlier one (the slot/reader-list only ever hold already-visited passes). So `compile()`'s
own adjacency can never contain a cycle, which is why `RenderGraphsTODO.md`'s cycle-detection item is
now moot for this code path.

## Implementation
- `src/RenderGraph.cpp` — the versioning sweep lives in the shared `sweep_resource_versions()` helper
  (`currentProducer` / `pendingReaders`, from the scratch arena; `pendingReaders` nodes reuse
  `NodeAdjacency` via `scratch_make<>()`). It emits hazard edges only — it does **not** track early reads.
- The early-read check is a **post-phase-2 pass** in `compile()`. Over the culled, execution-ordered
  `m_passes` it walks accesses in order keeping three scratch bitmaps over `next_id`: `hasWriter[id]`
  (some surviving pass writes it), `produced[id]` (written so far in execution order), and `imported[id]`.
  A read of `id` is an error iff `!imported[id] && hasWriter[id] && !produced[id]` (writes mark `produced`
  as they are walked, so a pass's own earlier write satisfies its later read). On any error it returns
  `false` before phase 3, so the caller never realizes/executes a misordered graph; `compile()`'s return
  type is `bool`. This lives in `compile()`, not the sweep, because it must see the *final* schedule
  (surviving passes only) to be culling-correct and to catch every surviving reader, not just the first.
- The whole post-cull pass is gated by **`RG_VALIDATE`** (defaults to `0` under `NDEBUG`, `1` otherwise —
  the same strip-in-release behaviour as `assert`, but a separate macro so the OOM-style "assert vanishes"
  trap doesn't apply elsewhere). When off, the per-frame walk is compiled out entirely and `compile()`
  always returns `true` — a shipping build assumes author-valid graphs. Predefine `RG_VALIDATE=1` to keep
  it in a release build, or `=0` to drop it from a debug build.
- `debug_print_mermaid()` reuses the **same** `sweep_resource_versions` helper, emitting one labelled
  edge per discovered hazard (RAW unlabelled; WAW/WAR tagged), so the dump matches the real graph exactly
  — it does **not** keep a separate last-writer-wins table.

## Smoke test
`src/RenderGraph_main.cpp` gained a depth-buffer multi-writer case. Note: `execute()` auto-wires a
`DepthStencilAttachment` into the render pass for **graphics** passes, which would then require a
depth-enabled pipeline — so the depth *writers* are no-op `PassKind::Transfer` passes (execute() skips
attachment wiring for those), and the *reads* are `sampled()` / `depth_stencil_read_only()` on existing
passes (reads are never auto-wired). This exercises the versioning without touching any pipeline. A real
renderer would z-prepass / draw geometry in those passes.

- `depth` resource (`Depth32Float`, relative to the swapchain), declared each frame.
- `depth.prepass` (Transfer): writes depth v1.
- `scene`: `sampled(depth)` — reads v1.
- `lighting` (Transfer): `depth_stencil_read_only(depth)` — reads v1 *read-only* (an `attachment-read`,
  not a write, so no false WAW vs `scene`), and writes `sceneColor` v2 to stay reachable from a sink.
- `depth.main` (Transfer): writes depth v2 — WAW to prepass, WAR to scene's *and* lighting's reads of v1.
- `compose` (glow) / `present` (no-glow): `sampled(depth)` — read v2.

`ubo` (a uniform buffer filled host-side via `wgpuQueueWriteBuffer`) is *read but never written* by any
pass; the post-cull check's `hasWriter` exemption keeps it from being flagged as an early read.

## Verification
Build the standalone smoke test with `build_rg.bat` (compiles `src/RenderGraph.cpp`, whose tail
`#include`s `RenderGraph_main.cpp`, and runs `rg.exe`). Observed, glow ON:

```
execution order (glow ON, representative): depth.prepass scene lighting depth.main sobel blur compose

deps depth.prepass <-                                  (first writer of depth)
deps scene         <- depth.prepass                    RAW (depth v1)
deps lighting      <- depth.prepass, scene             RAW (depth v1, read-only) + WAW (sceneColor)
deps depth.main    <- depth.prepass, scene, lighting   WAW (<-prepass) + WAR (<-scene, <-lighting)
deps sobel         <- lighting                         RAW (sceneColor v2);  ubo read -> no writer, no edge
deps blur          <- sobel                            RAW (sobelOut)
deps compose       <- lighting, blur, depth.main       RAW (sceneColor v2, blurOut, depth v2)
```

All three hazards (RAW/WAW/WAR) are present and the depth chain stays a clean 2-version graph despite
three readers; no early-read error fires (every transient read is produced before it, and `ubo` /
read-only depth are correctly exempt); the app renders with no GPU validation errors. To see the error
fire, declare a read of a transient before its writer (e.g. add `b.sampled(depth)` to a pass declared
*before* `depth.prepass`): `compile()` prints
`error: pass "…" reads resource "depth" before any pass writes it …` and returns `false`, and the
frame loop skips that frame.

## Future work / deferred
- **Per-version edge labels in the dump** — `debug_print_mermaid()` already shares the sweep and labels
  edges by hazard kind + resource name (the shared-helper refactor is done); the only missing piece is
  annotating the resource *version* (`#1`/`#2`) on each edge.
- **Explicit version labels** for order-independent multi-writer — let a read/write name the version it
  produces/consumes so ping-pong/accumulation no longer depend on declaration order (the residual the
  fatal-error rule above cannot remove).
- **Cross-frame temporal resources** (TAA/accumulation) — orthogonal; the graph is rebuilt every
  frame, so a history buffer is a *persistence* problem for the planned `GraphResourceCache`, not an
  intra-frame versioning one. Import frame N's final resource into frame N+1 as version 0.
