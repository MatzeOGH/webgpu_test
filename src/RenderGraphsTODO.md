
[] add AccessType::Vertex/Index/Indirect + matching WGPUBufferUsage bits, so RG-managed vertex/index/indirect-args buffers are possible
[] add a force_keep flag on passes so culling doesn't silently drop side-effect-only passes (readback/debug/profiling/indirect-arg-gen) that don't write an imported resource
[] implement a PoC of GraphResourceCache to not recreate all reasorces each frame.
[] scaling does not work in the smoke test: when using a scale of 0.5 for relative size the rendering produces errors
[] add instrumentation to mesure compile time of render graph
[] fix GraphAllocator OOM path: assert() is stripped in release builds, so alloc_raw returns nullptr and callers (e.g. begin_pass) null-deref instead of failing gracefully
[] add MSAA support: sampleCount on TextureDesc, resolve attachment wiring in execute() (realize() currently hardcodes sampleCount=1/mipLevelCount=1)
[] add mip level / array layer / cube view selection (realize() always creates the default full-resource view) -- needed for mip-chain gen, shadow atlases, cubemaps
[] detect cycles in compile() and assert/report instead of silently producing a wrong execution order -- note: the SSA versioning sweep only emits backward edges (later- to earlier-declared pass), so compile()'s OWN adjacency is acyclic by construction; this only matters again if some future code path populates adjacency outside that sweep
[] kMaxAccess=16 per pass and the 1MB arena are fixed with no growth path; add an assert or grow-on-demand
[] add a scratch arena to replace the calloc calls
