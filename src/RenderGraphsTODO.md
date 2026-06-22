
# RenderGraph TODO

merged from RenderGraphsTODO.md + docs/rendergraph-state.md, sorted by priority

## Critical (active bugs)

[] fix GraphAllocator OOM path: assert() is stripped in release builds, so alloc_raw returns nullptr and callers (e.g. begin_pass) null-deref instead of failing gracefully
[] scaling does not work in the smoke test: when using a scale of 0.5 for relative size the rendering produces errors

## High Priority

[] force_keep flag on passes / explicit mark_output(ResourceHandle) -- culling currently only roots at passes writing an *imported* resource, so side-effect-only passes (readback/debug/profiling/indirect-arg-gen) and non-imported outputs get silently dropped
[] kMaxAccess=16 per pass and the 1MB arena are fixed with no growth path; add an assert or grow-on-demand -- currently silently drops accesses past the cap
[] review/critique the scratch_alloc implementation -- scrutinize the two-sided arena arithmetic (alignment round-down, unsigned-underflow guards, the capacity-scratchUsed vs capacity-used boundary checks in alloc_raw/scratch_alloc_raw), confirm no front/scratch overlap edge case, and sanity-check the per-scope defer reset_scratch() pattern in compile()/sweep_resource_versions()

## Medium Priority

[] resource type validation: surface ResourceNode::Kind (Texture/Buffer) at the API layer and reject mismatched access (e.g. vertex_buffer() on a texture handle, sampled() on a buffer)
[] MSAA support: sampleCount on TextureDesc + resolve attachment wiring in execute() (realize() currently hardcodes sampleCount=1, mipLevelCount=1)
[] add instrumentation to measure compile() time

## Low Priority

[] transient resource aliasing: once lifetimes are known, let multiple logical resources share one physical allocation to cut VRAM use
[] formalize usage-flag derivation: pull the AccessType -> WGPUTextureUsage/WGPUBufferUsage switch (compile() phase 3) into named helpers (texture_usage(AccessType), buffer_usage(AccessType)) for reuse in realize()/import validation/debugging
[] detect cycles in compile() and assert/report instead of silently producing a wrong execution order -- note: the SSA versioning sweep only emits backward edges (later- to earlier-declared pass), so compile()'s OWN adjacency is acyclic by construction; this only matters again if some future code path populates adjacency outside that sweep

## Review follow-ups (TAA / pools)

[] transient pool stale content: TransientResourcePool::acquire() recycles a texture still holding the previous tenant's pixels, so a pooled transient no longer gets Dawn's implicit zero-clear. Safe today (every transient's first write is LoadOp_Clear; sky's LoadOp_Load follows lighting's clear of the same node), but a future LoadOp_Load or partial/RMW write on a transient would read another resource's leftovers -- clear-on-acquire or document+enforce the no-load invariant before adding within-frame aliasing
[] dedup pool texture create/destroy: PersistentResourcePool, TransientResourcePool, and realize()'s direct path each repeat the same {size,format,dim,usage} compare + WGPUTextureDescriptor/CreateTexture/CreateView pair + destroy(); fold into make_pooled_texture()/desc_matches() helpers -- only the ping-pong-vs-eviction policy legitimately differs
[] comment brevity pass on the temporal/TAA additions: run the avoid-ai-writing skill over the kTaaFs preamble (RenderGraph_main.cpp), the TemporalImage/create_temporal_image doc (RenderGraph.h), and the PersistentResourcePool header comment -- they read long/AI-ish against the codebase's terse style (CLAUDE.md "Be brief")

## Future / Long-Term

[] async compute: QueueType (Graphics/Compute/Transfer) -- WebGPU abstracts queues heavily today, but avoid baking in single-queue/linear-execution assumptions
[] pass merging: merge compatible passes to cut encoder overhead and attachment transitions
[] barrier optimization: analyze consecutive usages, merge compatible states, skip redundant transitions
[] memory budgeting: track transient memory usage, peak allocation, alias opportunities

## Done

[x] subresource support: mip level / array layer selection -- TextureDesc.mipLevelCount (threaded through realize() + both pools, replacing the hardcoded 1) and per-access baseMip/baseLayer on the GraphBuilder calls (color/depth_stencil/sampled/storage_*). execute() builds a single-(baseMip,baseLayer) 2D attachment view from the texture (imported keeps its registered view); pass bodies build their own sampled/storage views via ctx.texture() + wgpuTextureCreateView (mip view for bloom, Cube view for cubemaps -- no graph-side abstraction). in_pass_accesses_conflict() is now subresource-aware: disjoint (mip,layer) never conflicts, so sampling mip i while rendering mip i+1 of one texture in one pass is allowed (the bloom downsample). Hazards/versioning stay WHOLE-RESOURCE on purpose (a chain serializes via RAW on the shared handle; independent subresource passes over-order harmlessly since the graph topo-sorts, not barriers) -- go per-(mip,layer) only if that ever costs. Demos in RenderGraph_main.cpp: B = bloom mip-chain (extract -> downsample -> additive upsample -> composite), C = cubemap (6 per-layer face clears -> Cube-view skybox sample). Cube needs no creation-time textureBindingViewDimension on Dawn native. Remaining: cube-face/shadow-atlas depth subresources are the same path; mip-chain auto-gen helper left to user code
[x] resource lifetime tracking (firstUse/lastUse per resource) -- compile() phase 3 records, over the post-cull execution order, the first/last pass index touching each TRANSIENT resource (imported excluded); stored on ResourceNode with a kNoPass sentinel (untouched/dead). debug_print_lifetimes() dumps them as a mermaid Gantt (one bar per resource over pass order). prerequisite for the transient aliasing item under Low Priority
[x] immediate (declaration-time) usage validation in GraphBuilder::use() -- asserts at the exact b.xxx() call site (one frame above) when a resource is aliased read+write in one pass (e.g. sampled + storage_write, the named case) or written more than once (the graph can't order two writes in an atomic pass). StorageRead+StorageWrite (RMW: a read_write storage binding) and read+read are exempt -- see in_pass_accesses_conflict(). Gated by RG_VALIDATE, compiled out in release like the def-before-use check. The cross-pass def-before-use check is KEPT: it catches reader-before-writer over the culled schedule, which a per-pass builder check structurally cannot see
[x] DepthStencilReadOnly access type -- read-only depth (lighting reading a prepass's depth) is now AccessType::DepthStencilReadOnly: classified as a read (attachment-read), sets depthReadOnly in execute(), no false write hazard. Also fixed buffer copy_src/copy_dst usage (were texture-only) as part of a full AccessType->WGPU-usage spec audit
[x] AccessType::Vertex/Index/Indirect + matching WGPUBufferUsage bits, so RG-managed vertex/index/indirect-args buffers are possible
[x] pass culling -- compile() phase 2 does DFS from sinks (passes writing an imported OR temporal/persistent resource); unreached passes are dropped for free
[x] PersistentResourcePool PoC + first-class temporal/history resources -- create_temporal_image(name, desc) returns a TemporalImage{curr, prev}: a ping-pong pair (two contiguous-id ResourceNodes). Write `.curr`, read `.prev`. The pool (RenderGraph.cpp) owns the two physical textures per resource across the per-frame arena reset, keyed by name content, and swaps which slot backs curr/prev every frame -- so last frame's "current" is this frame's "previous" with no manual ping-pong or caller-owned textures. Rotation is keyed to a per-frame frameToken (bumped in create_render_graph), so declaring the same resource twice in one frame doesn't double-rotate. realize() unions usage over the two layers then binds each rotated slot; release_resources() leaves pool textures alone (freed on resize/pool destroy); writing `.curr` roots culling and reading `.prev` is exempt from the def-before-use check (both treated like imported). Writing `.prev` is an authoring error reported by compile() under RG_VALIDATE. Originally modeled on the REAC2023 COLOR_TARGET_CREATE_TEMPORAL(name,...,N) + indexed layer access, then deliberately narrowed to ping-pong only (no N-deep / temporal_prev) for simplicity -- reinstate a layers count + indexed accessor if checkerboard reconstruction or frame-gen (reads 2 frames back) ever needs it. Demo: press T in the smoke test (temporal accumulation; smears under motion since there are no motion vectors). Remaining: non-temporal persistent resources, camera-cut null-reset
[x] unified access declaration: GraphBuilder::use(handle, AccessType, ...) is now the one primitive; color()/depth_stencil()/sampled()/storage_read()/storage_write()/uniform()/copy_src()/copy_dst()/vertex_buffer()/index_buffer()/indirect_buffer() are thin wrappers over it
[x] add a scratch arena to replace the calloc() calls in compile() -- GraphAllocator grew a top-down scratch_alloc<T>()/scratch_make<T>()/reset_scratch() region (mirrors the existing bottom-up used/alloc<T>()/make<T>()), used by sweep_resource_versions()'s id-tables + pendingReaders chain and compile()'s phase-2/phase-3 id-tables; each scope pairs its scratch allocations with one `defer { alloc->reset_scratch(); }` (AlpUtils.h) instead of std::free
