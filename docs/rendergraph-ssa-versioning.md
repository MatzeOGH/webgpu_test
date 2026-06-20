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
state (plus a debug flag):

- `currentProducer[id]` — the pass holding the current version.
- `pendingReaders[id]` — passes that read the current version and haven't been retired by a newer write.

```
on write(id):
    if currentProducer[id] and != self:  add_dependency(self -> currentProducer[id])   # WAW
    for r in pendingReaders[id]: if r != self: add_dependency(self -> r)                # WAR
    currentProducer[id] = self;  pendingReaders[id] = empty                             # new version
on read(id):
    if currentProducer[id] is null:  earlyUnresolvedRead[id] = true
    elif currentProducer[id] != self: add_dependency(self -> currentProducer[id])       # RAW
    pendingReaders[id] += self
```

All three hazard kinds call the existing `add_dependency` (whose dedup collapses duplicate
`(dependent, dep)` pairs), so phase 1 needs no hazard-kind tagging and stays inline — no templated
helper, no enum. Both self-guards are load-bearing: read-then-write of one handle in a single pass
would otherwise WAR-self-edge; write-then-write would WAW-self-edge.

**Trade-off:** the old phase 1 was declaration-order-*independent* (a reader could be declared before
its producer). The new model requires producers declared before consumers. A read seen before *any*
writer of a resource that is written later is almost certainly an authoring bug, so it emits a
non-fatal `printf` warning (not `assert`, which is stripped in release — see the OOM note in
`RenderGraphsTODO.md`).

**Bonus — acyclic by construction:** every edge points from a later-visited pass back to a
strictly-earlier one (the slot/reader-list only ever hold already-visited passes). So `compile()`'s
own adjacency can never contain a cycle, which is why `RenderGraphsTODO.md`'s cycle-detection item is
now moot for this code path.

## Implementation
- `src/RenderGraph.cpp`, `compile()` phase 1 — replaced the `producer`-table block with the single
  versioning sweep above (`currentProducer` / `pendingReaders` / `earlyUnresolvedRead`, all calloc'd
  over `next_id` like the existing `imported`/`byId` tables; `pendingReaders` nodes reuse
  `NodeAdjacency` via `m_allocator->make<>()`, the same path `add_dependency` uses). Phase 2 (topo
  sort) and phase 3 (usage/size) are generic over `adjacency`/`accesses` and were untouched.
- `debug_print_mermaid()` was **left as-is** — it keeps its own approximate last-writer-wins table,
  so it under-reports multi-writer graphs (it draws reads against the *last* writer and omits WAW
  edges). Acceptable for a debug aid; superseded by the Future-work dump.

## Smoke test
`src/RenderGraph_main.cpp` gained a depth-buffer multi-writer case. Note: `execute()` auto-wires a
`DepthStencilAttachment` into the render pass for **graphics** passes, which would then require a
depth-enabled pipeline — so the two depth *writers* are no-op `PassKind::Transfer` passes (execute()
skips attachment wiring for those), and the *reads* are `sampled()` on existing passes (reads are
never auto-wired). This exercises the versioning without touching any pipeline. A real renderer would
z-prepass / draw geometry in those passes.

- `depth` resource (`Depth32Float`, relative to the swapchain), declared each frame.
- `depth.prepass` (Transfer): writes depth v1.
- `scene`: gains `sampled(depth)` — reads v1.
- `depth.main` (Transfer): writes depth v2.
- `compose` (glow) / `present` (no-glow): gain `sampled(depth)` — read v2.

## Verification
Build the standalone smoke test with `build_rg.bat` (compiles `src/RenderGraph.cpp`, whose tail
`#include`s `RenderGraph_main.cpp`, and runs `rg.exe`). Observed, glow ON:

```
execution order: depth.prepass scene depth.main sobel blur compose

deps depth.prepass <-                       (first writer)
deps scene         <- depth.prepass          RAW
deps depth.main    <- scene depth.prepass    WAR (<-scene) + WAW (<-prepass)
deps sobel         <- scene                   unchanged
deps blur          <- sobel                   unchanged
deps compose       <- depth.main blur scene   RAW (<-depth.main)
```

All three hazards present; the existing single-writer edges are byte-identical to before (no
regression); no early-read warning (every read is declared after its producer); the app renders with
no GPU validation errors. To see the warning fire, declare a read before its writer (e.g. comment out
`scene`'s/`depth.prepass`'s depth write).

## Future work / deferred
- **Hazard-labeled graph dump** like the chat diagram — pass nodes, edges labeled by hazard kind and
  resource version (`#1`/`#2`). It would extract the phase-1 sweep into a shared helper templated on
  an `onEdge(dependent, dep, id, kind)` callback so `compile()` and the dump share one source of
  truth, and supersede the current approximate `debug_print_mermaid`. (Safe to run post-`compile()`
  over the sorted, culled list: surviving same-resource passes keep their relative order and culled
  passes drop from nodes and edges together.)
- **Cross-frame temporal resources** (TAA/accumulation) — orthogonal; the graph is rebuilt every
  frame, so a history buffer is a *persistence* problem for the planned `GraphResourceCache`, not an
  intra-frame versioning one. Import frame N's final resource into frame N+1 as version 0.
