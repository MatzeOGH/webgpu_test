
# RenderGraph TODO

merged from RenderGraphsTODO.md + docs/rendergraph-state.md, sorted by priority

## Critical (active bugs)

[] fix GraphAllocator OOM path: assert() is stripped in release builds, so alloc_raw returns nullptr and callers (e.g. begin_pass) null-deref instead of failing gracefully
[] scaling does not work in the smoke test: when using a scale of 0.5 for relative size the rendering produces errors

## High Priority

[] force_keep flag on passes / explicit mark_output(ResourceHandle) -- culling currently only roots at passes writing an *imported* resource, so side-effect-only passes (readback/debug/profiling/indirect-arg-gen) and non-imported outputs get silently dropped
[] kMaxAccess=16 per pass and the 1MB arena are fixed with no growth path; add an assert or grow-on-demand -- currently silently drops accesses past the cap
[] implement a PoC of GraphResourceCache -- struct already exists (RenderGraph.cpp) but is just an empty vector, not wired into realize(), so resources are still recreated every frame
[] review/critique the scratch_alloc implementation -- scrutinize the two-sided arena arithmetic (alignment round-down, unsigned-underflow guards, the capacity-scratchUsed vs capacity-used boundary checks in alloc_raw/scratch_alloc_raw), confirm no front/scratch overlap edge case, and sanity-check the per-scope defer reset_scratch() pattern in compile()/sweep_resource_versions()

## Medium Priority

[] resource type validation: surface ResourceNode::Kind (Texture/Buffer) at the API layer and reject mismatched access (e.g. vertex_buffer() on a texture handle, sampled() on a buffer)
[] MSAA support: sampleCount on TextureDesc + resolve attachment wiring in execute() (realize() currently hardcodes sampleCount=1, mipLevelCount=1)
[] add instrumentation to measure compile() time

## Low Priority

[] subresource support: mip level / array layer / cube face selection (realize() always creates the default full-resource view) -- needed for mip-chain gen, shadow atlases, cubemaps, temporal pyramids
[] transient resource aliasing: once lifetimes are known, let multiple logical resources share one physical allocation to cut VRAM use
[] formalize usage-flag derivation: pull the AccessType -> WGPUTextureUsage/WGPUBufferUsage switch (compile() phase 3) into named helpers (texture_usage(AccessType), buffer_usage(AccessType)) for reuse in realize()/import validation/debugging
[] detect cycles in compile() and assert/report instead of silently producing a wrong execution order -- note: the SSA versioning sweep only emits backward edges (later- to earlier-declared pass), so compile()'s OWN adjacency is acyclic by construction; this only matters again if some future code path populates adjacency outside that sweep

## Future / Long-Term

[] async compute: QueueType (Graphics/Compute/Transfer) -- WebGPU abstracts queues heavily today, but avoid baking in single-queue/linear-execution assumptions
[] pass merging: merge compatible passes to cut encoder overhead and attachment transitions
[] barrier optimization: analyze consecutive usages, merge compatible states, skip redundant transitions
[] memory budgeting: track transient memory usage, peak allocation, alias opportunities

## Done

[x] resource lifetime tracking (firstUse/lastUse per resource) -- compile() phase 3 records, over the post-cull execution order, the first/last pass index touching each TRANSIENT resource (imported excluded); stored on ResourceNode with a kNoPass sentinel (untouched/dead). debug_print_lifetimes() dumps them as a mermaid Gantt (one bar per resource over pass order). prerequisite for the transient aliasing item under Low Priority
[x] immediate (declaration-time) usage validation in GraphBuilder::use() -- asserts at the exact b.xxx() call site (one frame above) when a resource is aliased read+write in one pass (e.g. sampled + storage_write, the named case) or written more than once (the graph can't order two writes in an atomic pass). StorageRead+StorageWrite (RMW: a read_write storage binding) and read+read are exempt -- see in_pass_accesses_conflict(). Gated by RG_VALIDATE, compiled out in release like the def-before-use check. The cross-pass def-before-use check is KEPT: it catches reader-before-writer over the culled schedule, which a per-pass builder check structurally cannot see
[x] DepthStencilReadOnly access type -- read-only depth (lighting reading a prepass's depth) is now AccessType::DepthStencilReadOnly: classified as a read (attachment-read), sets depthReadOnly in execute(), no false write hazard. Also fixed buffer copy_src/copy_dst usage (were texture-only) as part of a full AccessType->WGPU-usage spec audit
[x] AccessType::Vertex/Index/Indirect + matching WGPUBufferUsage bits, so RG-managed vertex/index/indirect-args buffers are possible
[x] pass culling -- compile() phase 2 does DFS from sinks (passes writing an imported resource); unreached passes are dropped for free
[x] unified access declaration: GraphBuilder::use(handle, AccessType, ...) is now the one primitive; color()/depth_stencil()/sampled()/storage_read()/storage_write()/uniform()/copy_src()/copy_dst()/vertex_buffer()/index_buffer()/indirect_buffer() are thin wrappers over it
[x] add a scratch arena to replace the calloc() calls in compile() -- GraphAllocator grew a top-down scratch_alloc<T>()/scratch_make<T>()/reset_scratch() region (mirrors the existing bottom-up used/alloc<T>()/make<T>()), used by sweep_resource_versions()'s id-tables + pendingReaders chain and compile()'s phase-2/phase-3 id-tables; each scope pairs its scratch allocations with one `defer { alloc->reset_scratch(); }` (AlpUtils.h) instead of std::free
