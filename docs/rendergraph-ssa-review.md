# RenderGraph SSA implementation — review findings

Review of the SSA-style resource versioning in `src/RenderGraph.cpp` (`compile()` /
`sweep_resource_versions`) and the design doc `docs/rendergraph-ssa-versioning.md`.

## How this was produced
Each finding below was **traced against the actual code and adversarially verified** (an independent pass
tried to refute it). Items that didn't survive verification are listed under *Confirmed non-issues*.
Severity is the post-verification severity, which sometimes differs from the first-pass guess.

## Reading the table
- **Severity** — `high` / `medium` / `low`, judged by blast radius + silence.
- **Status** — `real` (genuine defect) or `non-issue`; `acknowledged` means already noted in a ponytail
  comment or `RenderGraphsTODO.md`.
- **Latent** — most high-severity items cannot fire *today* because the graph is rebuilt every frame
  (fresh arena + `next_id`) and the smoke test declares producers before consumers. They activate the
  moment usage deviates.

---

## A. Order-dependence (the headline concern)

### A1 — Consumer-before-producer emits an inverted WAR edge instead of RAW
- **Severity:** high · **Status:** real, acknowledged (doc:45-49) · **Location:** `RenderGraph.cpp:356-373`, `:380-382`, `:441-452`
- **Mechanism:** `sweep_resource_versions` binds a read to `currentProducer[id]` at the moment the read is
  visited. If a consumer is declared before the producer, the read sees `currentProducer==null`, becomes
  an `earlyReader`, and gets **no edge** to the producer. When the writer is visited later, the WAR loop
  fires `onEdge(writer, reader)` → `add_dependency(writer, reader)` — i.e. the **writer is ordered after
  the reader**. The intended RAW edge is never emitted; the dependency is inverted.
- **Verified trace:** declare order `Reader{sampled X}`, `Writer{storage_write X}`, X internal & consumed
  downstream. Reader → `earlyReader[X]=Reader`, pushed to `pendingReaders[X]`, no edge. Writer → WAR loop
  sees `pendingReaders[X]={Reader}` → `add_dependency(Writer, Reader)`. `topo_visit` emits Reader then
  Writer. At `execute()` the reader samples X before it is written. The end-of-sweep warning *does* fire,
  but it's only a `printf` at `:448`; compile still "succeeds" with the wrong order. (If instead the
  *reader* is the sink, the producer is unreachable and dead-code-eliminated entirely — even worse.)
- **This is the issue the planned fix targets** (`compile()` → `false`).

### A2 — Read declared between two writers binds to the earlier version, with NO diagnostic
- **Severity:** high · **Status:** real, **not** acknowledged · **Location:** `RenderGraph.cpp:367-369`, `:380-382`
- **Mechanism:** the early-read warning fires only when *no* producer existed at read time
  (`earlyReader[id] && currentProducer[id]`). A read visited *after* writer v1 but *before* writer v2
  takes the RAW branch, binds to v1, and never touches `earlyReader` — so no warning. If the author meant
  v2, they silently get v1.
- **Verified trace:** order `prepass(write)`, `R(read)`, `main(write)`. R binds RAW→prepass (v1),
  `earlyReader` stays null. `main` writes v2. End-of-sweep check is false → no warning. Graph compiles
  cleanly with R reading v1.
- **Note:** inherent to declaration-order versioning. Only author-supplied explicit version labels could
  catch it — out of scope for the chosen "keep the API" direction; documented as a known limitation.

### A3 — Multi-writer resources are fundamentally order-defined
- **Severity:** high · **Status:** real, acknowledged (doc:14-21) · **Location:** `RenderGraph.cpp:359-365`; `RenderGraph_main.cpp:353-373`
- **Mechanism:** for a resource with ≥2 writers, the version identity is the writing pass and the sweep
  walks pure declaration order, so which write a reader sees is decided **solely** by `add_pass` order.
  Nothing in the access lists (`{pass, id, read|write}`) carries it.
- **Verified trace:** current order `prepass, scene, depth.main` → scene reads v1, compose reads v2.
  Swap to `depth.main, scene, prepass` → scene now reads `depth.main`'s output, compose reads `prepass`'s
  — the read→version map inverts. **No warning, no cycle, byte-identical access lists.** This is the proof
  that declaration order is the only disambiguating signal under the current API.

### A4 — Editing the pass list re-points multi-writer reads
- **Severity:** medium · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:366-372`
- **Mechanism:** moving a reader across a writer boundary (or inserting a write before an existing reader)
  silently rebinds the reader to a different existing version. Both versions exist, so `currentProducer`
  is non-null → no early-read warning.
- **Verified trace:** writers W1,W2, reader Rd. Order `W1,W2,Rd` → Rd binds v2. Move Rd between W1 and W2 →
  Rd binds v1 + an extra WAR W2→Rd. Same passes, one moved, different data flow, no diagnostic. Same root
  cause as A2.

### A5 — Diagnostic gaps in the early-read warning
- **Severity:** low · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:368`, `:380-382`, `:441-452`
- **Two sub-issues:** (a) only the *first* early-reader per resource is recorded (`if (!earlyReader[id])`),
  so additional readers-before-writer are misordered without being named; (b) the warning runs in phase 1
  *before* culling, so it can fire for a read-before-write between two passes that phase 2 then drops
  (spurious noise about non-executing passes). Edge generation is unaffected in both; purely diagnostic.
- **Note:** the planned fix's "filter violations to post-cull survivors" closes sub-issue (b).

### A6 — `debug_print_mermaid()` inherits inverted edges and suppresses the warning
- **Severity:** low · **Status:** real, acknowledged-ish · **Location:** `RenderGraph.cpp:544-568`, `:563`
- **Mechanism:** the dump reuses `sweep_resource_versions` (good — it can't drift from the real graph) but
  passes a no-op `onEarlyRead`. So in a consumer-before-producer situation it draws the WAR-labelled edge
  with no hint it's the symptom of a misorder — in the very tool you'd reach for to debug an order bug.
  `compile()` still warns to stdout, so impact is minor.

---

## B. Correctness (non-order)

### B1 — `compile()` is not idempotent (second call empties the graph)
- **Severity:** medium (mechanism high, but unreachable today) · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:404-405`, `:478`; flag `:158`
- **Mechanism:** `topo_visit` uses `PassNode::placed` as its visited marker and nothing resets it. A
  *second* `compile()` finds every pass `placed==true`, so every `topo_visit` early-returns, `count`
  stays 0, and `:478` sets `m_passes = nullptr`. The graph silently loses every pass.
- **Verified:** phase 1 is re-run-safe (`add_dependency` dedups) and phase 3 memoizes; only phase 2
  corrupts, totally. **Cannot fire under the current driver** — `create_render_graph()` makes fresh nodes
  every frame and `compile()` is called exactly once per graph. It would bite the build-once/re-execute
  workflow the header advertises (`RenderGraph.h:152-154`). Fix: reset `placed` (+ clear `adjacency`) at
  the top of `compile()`. **Out of scope** per the agreed plan (one compile per frame).

### B2 — `resolve_size` has no cycle guard → stack overflow
- **Severity:** high · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:426-433`
- **Mechanism:** recursive `relativeTo` walk with only an "already-resolved" memo (`resolved.width != 0`)
  and an Absolute base case. The memo is written *after* the recursive call, so a node currently on the
  stack still reads `width==0` → a self/mutual `relativeTo` recurses forever → crash.
- **Verified:** `A.relativeTo=B`, `B.relativeTo=A`, both Relative → infinite recursion (self-loop
  identical). `create_image` copies `desc.relativeTo` with **zero validation**; phase 3 resolves every
  texture unconditionally. Distinct from the `topo_visit` cycle item (that graph is acyclic by
  construction; the `relativeTo` graph has no such guarantee). Fix: in-progress tombstone / visited flag,
  report a cycle. **Out of scope** per the agreed plan.

### B3 — Compute dispatch derives from full swapchain size, not resolved size
- **Severity:** medium · **Status:** real, acknowledged (TODO scale=0.5) · **Location:** `RenderGraph_main.cpp:387`
- **Mechanism:** `groupsX/Y = (cfg.width/height + 7) / 8` use the swapchain extent, but `sobel.out`/
  `blur.out` are `Relative` to `sceneColor`. At scale 1.0 they coincide; at any other scale the dispatch
  grid doesn't match the resource written. This is the operational half of the `scale=0.5` TODO (the
  float→uint truncation at `:432`, finding B6, is a sub-pixel red herring by comparison). Fix: compute
  groups post-`compile()` from `rg->node(handle)->resolved`. **Out of scope.**

### B4 — Buffer copy usage gap
- **Severity:** medium · **Status:** real, acknowledged (ponytail `:510`) · **Location:** `RenderGraph.cpp:511-512`
- **Mechanism:** `CopySrc`/`CopyDst` unconditionally OR into `texUsage`. A graph-created **buffer** used as
  a copy target therefore never gets `WGPUBufferUsage_CopyDst`; `realize()` sees `bufUsage==0` and skips
  it (`:625`), leaving `r->buffer==null` → the copy targets a null buffer at execute. Hard-fails the
  instant buffer-to-buffer / buffer↔texture copy is used through the graph. **Out of scope.**

### B5 — `DepthStencilAttachment` is unconditionally a write
- **Severity:** medium · **Status:** real, partially acknowledged (`RenderGraph.h:48`) · **Location:** `RenderGraph.cpp:213-219`, `:356-365`
- **Mechanism:** there's no read-only depth path; `depth_stencil()` always records a write. So a pass that
  only depth-*tests* against an existing buffer is modeled as a writer: it starts a new version, retires
  the real readers, emits a spurious WAW to the previous depth writer, and becomes the "producer" a later
  sampler binds to — shadowing the true z-prepass. Usage bits are unaffected (RenderAttachment either
  way); only dependency edges are wrong. Today's executor is sequential so the only live effect is
  wrong-producer attribution influencing topo order. **Out of scope.**

### B6 — `resolve_size` numeric / sentinel issues (cluster)
- **Severity:** low (each) · **Status:** real · **Location:** `RenderGraph.cpp:428-432`
- (a) float→uint truncates instead of rounding (sub-pixel; even dims unaffected). (b) the
  `resolved.width != 0` memo sentinel collides with a legitimately-zero size, so such a node is re-walked
  every call (perf, not correctness). (c) a dangling/zero `relativeTo` (`{id=0}` default) silently yields
  a `{0,0,1}` texture → WebGPU rejects it at create time, but no graph-level hint. (d) the Relative path
  hardcodes `depthOrArrayLayers=1`, so a Relative texture can't inherit array/3D depth from its base. Fix
  for (b)/(c): a dedicated `resolved` flag instead of overloading width, + a warning on null/zero base.
  **Out of scope.**

### B7 — `node()` is O(n), called per-access in `execute()` and per pass-body lookup
- **Severity:** low · **Status:** real, acknowledged (ponytail `:570`) · **Location:** `RenderGraph.cpp:572-577`, `:667`, `:587-600`
- **Mechanism:** linear list walk per call; `compile()` builds an id→node table (`byId`, `:492`) but frees
  it. Negligible for the smoke test (≈6 passes), but quadratic-ish per frame at scale. Fix: retain an
  id→node table on the graph after `compile()`. **Out of scope.**

### B8 — Default-handle (id==0) accesses create phantom edges through slot 0
- **Severity:** low · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:354-373`, `:720-727`
- **Mechanism:** `push_access` has no validity check; a default `ResourceHandle{}` has `id==0`. The sweep
  arrays are sized `next_id` so slot 0 is processed normally — two passes that both pass a null handle get
  linked by a phantom RAW/WAW/WAR through slot 0. The end-of-sweep scan starts at `id=1`, so the typo is
  also undiagnosed. Bounded impact (only adds an ordering edge between two equally-buggy passes). Fix:
  skip `id==0` in the sweep, or validate in `push_access`. **Out of scope.**

### B9 — `execute()` attachment caps drop silently
- **Severity:** low · **Status:** real, not acknowledged · **Location:** `RenderGraph.cpp:660-686`
- **Mechanism:** `color[8]` + `nc < 8` silently ignores a 9th color attachment (it still influenced
  compile's edges/usage, so the resource is created and ordered but never wired). Depth: each
  `DepthStencilAttachment` access overwrites `depth` (last wins, no count guard); stencil load/store ops
  are always `Undefined` and unrepresentable by the API → a combined depth-stencil format would fail
  WebGPU validation. Latent (smoke test is single-target / depth-only). **Out of scope.**

---

## C. Culling

### C1 — Side-effect-only passes are silently culled
- **Severity:** high · **Status:** real, acknowledged (TODO `force_keep`) · **Location:** `RenderGraph.cpp:413-419`, `:470-478`
- **Mechanism:** `is_sink` roots the topo DFS only from passes that write an **imported** resource. A pass
  whose only effect is on a graph-owned resource — GPU→CPU readback (`copy_dst` into a graph buffer the
  caller maps), occlusion/timestamp query resolve, indirect-draw-arg generation — is never reached and is
  dropped at `:476-478` with no warning. The body never runs; the caller's mapped buffer is stale → wrong
  CPU-side data, far from the graph.
- **Verified:** `create_buffer` never sets `imported`, so such a pass can never be a sink writer. Bites the
  first readback/query pass anyone adds. (Nuance: an indirect-args producer survives *if* the consumer
  declares `storage_read(args)` and is itself live — the clean repro is a readback with no in-graph
  reader.) Fix: a `force_keep`/`has_side_effect` root + a warning when a declared pass is culled.
  **Out of scope.**

### C2 — `push_access` silently drops the 17th+ access
- **Severity:** high · **Status:** real, partially acknowledged (ponytail `:726`) · **Location:** `RenderGraph.cpp:720-727`
- **Mechanism:** `accessCount < kMaxAccess` (16) with no else. Both the sweep and `is_sink` iterate
  `accesses[0..accessCount)`, so a dropped access is invisible. Two catastrophic outcomes: (a) if the
  dropped access is the `b.color(swapchain)` write that makes a pass a sink → `is_sink` false →
  `m_passes=nullptr` → blank frame; (b) if it's an intermediate read/write → the RAW/WAW/WAR edge is never
  made → topo may run consumer before producer → GPU reads stale data. Reachable for a heavy
  deferred-lighting pass (many G-buffer/shadow/atlas reads + the swapchain write). Fix: assert/grow.
  **Out of scope.**

---

## D. Memory / hardening

### D1 — Arena OOM null-derefs across many sites (release builds)
- **Severity:** medium · **Status:** real, partially acknowledged (TODO 7/11) · **Location:** `RenderGraph.cpp:40-44` + call sites; `RenderGraph.h:181-183`
- **Mechanism:** `alloc_raw` guards OOM with `assert(false)` + `return nullptr`; `assert` is stripped under
  NDEBUG, so release silently returns null. `make<T>()`/`copy_string` propagate null, but no caller checks
  (`create_*`/`import_*`, `add_dependency`, the sweep). Worst: `store_exec` (`RenderGraph.h:181-183`) does
  an unchecked placement-new into possibly-null. Latent (1 MB arena vs a handful of passes). **Relevant to
  the planned fix:** the new error must be fatal in *all* builds — hence `compile()` returns `bool` rather
  than relying on `assert`. **Otherwise out of scope.**

### D2 — Unchecked `calloc` + per-frame heap churn
- **Severity:** low · **Status:** real, acknowledged (TODO 12) · **Location:** `RenderGraph.cpp:349-351`, `:459`, `:468`, `:492`
- **Mechanism:** six `calloc`/`free` pairs every frame (three in the sweep, three across phases), none
  null-checked — exactly the heap thrash the arena exists to avoid. `debug_print_mermaid()` additionally
  allocates `pendingReaders` link nodes into the *live* arena. Fix: a scratch arena. **Out of scope.**

---

## E. Documentation drift — `docs/rendergraph-ssa-versioning.md`
The doc describes the **pre-refactor** code: the sweep was extracted into the shared
`sweep_resource_versions` template that *both* `compile()` and `debug_print_mermaid()` now use. To correct
(part of the planned doc update):
- **md:62-64** — "`debug_print_mermaid()` was left as-is … approximate last-writer-wins table … omits WAW
  edges." False; it shares the exact sweep and emits RAW/WAW/WAR.
- **md:101-106** — "Hazard-labeled graph dump" listed under *Future work*; already implemented (shared
  templated helper with `onEdge(dependent, dep, id, kind)`). (The per-version `#1/#2` label is the only
  unbuilt part.)
- **md:40-43** — "phase 1 … stays inline — no templated helper, no enum." There is a `HazardKind` enum
  (`RenderGraph.cpp:332`) and the templated helper (`:342`); phase 1 is a lambda passed to it.
- **md:22-37** — "two pieces of scratch state (plus a debug flag)" / `earlyUnresolvedRead[id] = true`. The
  code keeps **three** arrays and `earlyReader` is a `PassNode*` (not a bool) — load-bearing, it names the
  pass in the warning. Also the early-read warning is emitted by the caller via an `onEarlyRead` callback,
  not inline in the sweep.

---

## Confirmed non-issues (verified, no action)
- **`next_id` array sizing + `id=1` loop start** — correct. Handles are `1..next_id-1`; arrays sized
  `next_id` cover them; the diagnostic loop correctly skips invalid id 0.
- **"Reordering breaks acyclic-by-construction via a 2-cycle"** — refuted. A read can only bind to an
  earlier-visited pass, so a mutual cycle cannot be encoded into adjacency; it stays acyclic. Reordering
  produces a *different valid* program, not a cyclic one.
- **`topo_visit` lacking cycle detection** — moot. The only writer of `adjacency` is the sweep, which emits
  only later→earlier edges (acyclic by construction). And the claimed "stack overflow" is wrong anyway:
  `placed` is set on entry, so a hypothetical cycle would terminate with a wrong order, not overflow.
  Already a documented `+2 line` deferral (TODO 10).

---

## Triage summary

| # | Finding | Sev | Status | In planned fix? |
|---|---|---|---|---|
| A1 | Consumer-before-producer → inverted WAR | high | real, ackd | **Yes** (`compile()`→false) |
| A2 | Read between two writers binds v1 silently | high | real | No — needs explicit versions (deferred) |
| A3 | Multi-writer is order-defined | high | real, ackd | No — by design |
| A4 | Editing list re-points multi-writer reads | med | real | No — same root as A2 |
| A5 | Early-read diagnostic gaps | low | real | Partial (post-cull filter) |
| A6 | Mermaid dump hides misorder | low | real | No |
| B1 | `compile()` not idempotent | med | real | No — can't fire (one compile/frame) |
| B2 | `resolve_size` no cycle guard | high | real | No — deferred |
| B3 | Dispatch from swapchain not resolved size | med | real, ackd | No — deferred |
| B4 | Buffer copy usage gap | med | real, ackd | No — deferred |
| B5 | Depth attachment always a write | med | real, ackd | No — deferred |
| B6 | `resolve_size` numeric/sentinel cluster | low | real | No — deferred |
| B7 | `node()` O(n) | low | real, ackd | No — deferred |
| B8 | id==0 phantom edges | low | real | No — deferred |
| B9 | `execute()` attachment caps | low | real | No — deferred |
| C1 | Side-effect passes culled | high | real, ackd | No — deferred (`force_keep`) |
| C2 | `push_access` drops 17th access | high | real, ackd | No — deferred |
| D1 | Arena OOM null-deref | med | real, ackd | Indirect (fix avoids `assert`) |
| D2 | `calloc` churn / unchecked | low | real, ackd | No — deferred |
| E  | Doc drift (4 stale claims) | — | real | **Yes** (doc update) |
