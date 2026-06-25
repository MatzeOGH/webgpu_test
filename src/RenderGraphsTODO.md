# RenderGraph TODO

## Critical (active bugs)

[x] fix GraphAllocator OOM path: assert() is stripped in release builds, so alloc_raw returns nullptr and callers (e.g. begin_pass) null-deref instead of failing gracefully -- DONE (P4): always-on loud printf in alloc_raw/scratch_alloc_raw before the stripped assert; store_exec null-checks alloc_exec and leaves exec_fn null so execute() skips the pass; the 3 PassContext resolvers (+size) null-guard via rg_resolve_miss and return {}
[x] scaling does not work in the smoke test: when using a scale of 0.5 for relative size the rendering produces errors -- DONE (P3): resolve_size now rounds (+0.5f) before the uint32 cast so an odd dim at 0.5 lands on the right pixel (1281->641); proven by making the SSAO ao texture Relative 0.5 and driving its dispatch from ctx.size(ao). foreground-verified clean

## High Priority

[] force_keep flag on passes / explicit mark_output(ResourceHandle) -- culling currently only roots at passes writing an *imported or temporal* resource, so side-effect-only passes (readback/debug/profiling/indirect-arg-gen) and non-external outputs get silently dropped
[] persistent non-temporal resources: a pool-backed resource that survives the per-frame teardown but is NOT ping-pong rotated -- written once (or every N frames) and read for many frames after. unblocks compute-once/read-many bakes (BRDF LUT, prefiltered/irradiance IBL, SH, atlases) and reduced-cadence GI; today PersistentResourcePool only does 2-slot temporal history, so everything else must be imported caller-owned. add create_persistent_image (1-slot, no rotation, exempt from def-before-use like temporal)
[] kMaxAccess=16 per pass and the 1MB arena are fixed with no growth path; add an assert or grow-on-demand -- currently silently drops accesses past the cap
[] review/critique the scratch_alloc implementation -- scrutinize the two-sided arena arithmetic (alignment round-down, unsigned-underflow guards, the capacity-scratchUsed vs capacity-used boundary checks in alloc_raw/scratch_alloc_raw), confirm no front/scratch overlap edge case, and sanity-check the per-scope defer reset_scratch() pattern in compile()/sweep_resource_versions()

## Medium Priority

[] resource type validation: surface ResourceNode::Kind (Texture/Buffer) at the API layer and reject mismatched access (e.g. vertex_buffer() on a texture handle, sampled() on a buffer)
[x] MSAA support: sampleCount on TextureDesc (threaded through both pools + realize()) + resolve attachment wiring (b.resolve() -> AccessType::ResolveAttachment -> WGPURenderPassColorAttachment.resolveTarget). depth/stencil attachment now also carries stencil load/store/clear ops. see docs/rendergraph-attachments.md
[] add instrumentation to measure compile() time
[] camera-cut null-reset for temporal/persistent resources: on a camera cut (or first frame) the history read is stale garbage -- a way to flag a temporal resource's prev as invalid so the pass clears/ignores it instead of smearing. see docs/rendergraph-null-textures.md

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
