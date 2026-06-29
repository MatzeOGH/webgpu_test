

// RenderGraph PoC
// The goal is to define a immidiate mode render graph system
// without thrashing the memory.
// RULE: dont abstract the webgpu api

// Major Refactor before v0
// calling add_{resource} to the rendergraph gives to opportunity to call it after compile by mistake where the graph
// is in read mode onaly and it is not allowed to add new passes or resources
// - 


#pragma once

#ifndef RENDERGRAPH_H
#define RENDERGRAPH_H

#include <cstdint>
#include <new>
#include <utility>
#include <type_traits>
#include <assert.h>

#include <webgpu/webgpu.h>

// Create a string view form a string literal
#define WEBGPU_STR(str) WGPUStringView{ .data = "" str, .length = sizeof("" str) - 1 }

namespace RG
{

enum struct PassKind : uint8_t
{
    None = 0,   // invalide!
    Graphics,
    Compute,
    Transfer
};

enum struct SizeKind : uint8_t
{
    Absolute,
    Relative
};

//
enum struct ResourceKind : uint8_t {
    None,       // invalide!
    Texture,    // tages `ResourceHandle` as a texture
    Buffer,     // tags `ResourceHandle` as a buffer
};


// Resource Handle
struct ResourceHandle
{
    uint32_t id{}; // internal index 
    ResourceKind kind{}; // type tag for error checking
    // TODO: maybe add the `ResourceUsage` as a field to the `ResourceHandle`
};


// a ping-pong temporal (history) resource: two physical textures or buffers rotate each frame.
// write `curr`, read `prev`. returned by create_temporal_image / create_temporal_buffer
struct TemporalResource
{
    ResourceHandle curr;   // this frame's WRITE target
    ResourceHandle prev;   // last frame's result, READ-only this frame
};

struct ResourceNode;
struct PassNode;
struct RenderGraph; // PassContext holds a back-pointer for resource lookup

// live during execute(): the encoder a pass body records into + resolved-resource lookup.
// the real thing that replaces the old forward-declared stub.
struct PassContext
{
    WGPUCommandEncoder encoder{};               // always set
    WGPURenderPassEncoder render{};             // set for Graphics passes
    WGPUComputePassEncoder compute{};           // set for Compute passes
    WGPUQueue queue{};                          // needed for copy ops
    RenderGraph* graph{};
    PassNode* pass{};                           // the pass being recorded; lets view(h) build the subresource the access declared

    WGPUTextureView view(ResourceHandle h) const;   // resolved view
    WGPUTexture texture(ResourceHandle h) const;    // resolved texture (copies need the texture, not a view)
    WGPUBuffer buffer(ResourceHandle h) const;  // resolved buffer
    WGPUExtent3D texture_size(ResourceHandle h) const;  // resolved extent -- derive a compute dispatch / scissor from a relative-sized target
    uint32_t buffer_size(ResourceHandle h) const;
};

struct PassBuilder
{
    // color attachment. Only legal for render passes
    void color(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, WGPUColor clear = {0, 0, 0, 1}, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // depth stencil attachment. for a depth+stencil format pass the stencil load/store/clear too (a
    // depth-only format leaves them Undefined/0); the bound pipeline's depthStencil state drives the actual
    // stencil test/write -- the graph only carries the attachment's load/store/clear ops.
    void depth_stencil(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, float clearDepth = 1.0f, uint32_t baseMip = 0, uint32_t baseLayer = 0, WGPULoadOp stencilLoad = WGPULoadOp_Undefined, WGPUStoreOp stencilStore = WGPUStoreOp_Undefined, uint32_t stencilClear = 0);
    // depth stencil attachment, read-only (depth/stencil test, no write; e.g. lighting depth-testing
    // a prepass depth). no load/store/clear: WebGPU requires depthLoadOp/StoreOp Undefined when read-only.
    void depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // MSAA resolve target for the preceding color() (positional: call right after the color() it resolves).
    // the target must be single-sample, same format + size as that multisample color; Dawn validates. the
    // target may be imported (e.g. resolve a multisample color straight into the swapchain).
    void resolve(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // sampled resouces
    void sampled(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    void storage_read(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    void storage_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // in-place read+write of one storage resource in a single pass: the API mirror of WGSL
    // `var<storage, read_write>`. records the StorageRead+StorageWrite pair (the only read+write pairing
    // legal in one pass) so the graph orders the pass against other producers/readers but not against
    // itself. own-slot / atomic use only -- the in-dispatch race is the shader's to handle.
    void storage_read_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // uniform
    void uniform(ResourceHandle handle);
    // transfer (copy) source / destination
    void copy_src(ResourceHandle handle);
    void copy_dst(ResourceHandle handle);
    // buffer-only: vertex/index/indirect-args
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    // mark this pass as the initializer for a persistent or temporal resource: it runs only when the
    // target needs (re)baking, then compile() skips it. `hash` is a digest of the settings the baked
    // content depends on: the pass runs while the target is unrealized (first frame, or after the pool
    // evicts/recreates it) OR while `hash` differs from the hash last baked into it -- so changing a
    // setting re-bakes, a steady setting bakes once. `hash == 0` (the default) never changes => pure
    // bake-once. the baked result persists in the pool, so readers bind to it with no in-graph writer
    // (legal for a persistent/external resource). declare the write to `target` as usual
    // (color()/storage_write()); this only gates. `target` must be pool-backed (persistent or temporal
    // .curr); one target per pass, write only it. for temporal resources this fills .curr on a camera
    // cut / first frame; combine with the history hash on create_temporal_image for full invalidation.
    void initialize(ResourceHandle target, uint64_t hash = 0);

    // keep this pass even when nothing in the graph reads its output and it writes no imported/persistent
    // resource: marks it an extra cull root so compile() never drops it (and keeps what it depends on).
    // for side-effect-only passes -- GPU->CPU readback, timestamp/profiling resolve, indirect-arg gen
    // consumed outside the graph. not an access (records no hazard/usage), just a marker.
    void force_keep();

    PassNode* m_new_pass{};
};
struct GraphAllocator; // internal allocator: bump arena + the two resource pools (persistent + transient)

struct TextureDesc
{
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    //
    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    ResourceHandle relativeTo{};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;   // depthOrArrayLayers = array/cube layers (6 for a cube)
    uint32_t mipLevelCount = 1;                    // > 1 for a mip chain (downsample pyramid, mip generation); per-mip size is implicit
    uint32_t sampleCount = 1;                      // > 1 = MSAA (multisampled attachment); must be 1 or 4 ( WebGPU 1.0 limit)
    // ponytail: sampleCount > 1 implies a 2D, mipLevelCount-1, non-storage texture and every attachment in
    // a pass must share it; not enforced here -- Dawn validates at texture/render-pass creation.
    // ponytail: hazards stay whole-resource. a mip chain still serializes right (each step RAW-depends on
    // the shared handle); independent subresource passes just over-order, which is free here (the graph
    // topo-sorts, it doesn't emit barriers). go per-(mip,layer) only if that over-ordering ever costs.
};

struct BufferDesc
{
    uint64_t size{};
};

// IMPORTANT: the RenderGraph is a transient object and should never be stored longer than `release_resources()` is called
struct RenderGraph
{

    // a per-frame transient GPU texture the graph owns: realize() backs it
    ResourceHandle create_image(WGPUStringView name, const TextureDesc& desc);
    // Declares an external buffer the graph can use. i.e. swapchain texture
    ResourceHandle importe_image(WGPUStringView name, WGPUTextureView view, WGPUExtent3D size);
    // a per-frame transient GPU buffer the graph owns: realize() backs it
    ResourceHandle create_buffer(WGPUStringView name, const BufferDesc& desc);
    // Declares an external buffer the graph can use.
    ResourceHandle import_buffer(WGPUStringView name, WGPUBuffer buffer);

    // Temporal (history) texture: a ping-pong pair the PersistentResourcePool rotates each frame, so
    // this frame's `.curr` becomes next frame's `.prev` for free -- no manual ping-pong or caller-owned
    // textures. Write `.curr`, read `.prev`. Survives the per-frame teardown (realize()/release_resources()
    // defer to the pool the allocator owns). Writing `.prev` is an authoring
    // error (it backs a future frame's curr): compile() reports it under RG_VALIDATE.
    // `hash`: nonzero = camera-cut invalidation. on mismatch the pool destroys and recreates both
    // physical layers (Dawn zeros them), clearing stale .prev. 0 (default) = no invalidation.
    // ponytail: ping-pong only. N-deep history reinstate a `layers` count if ever needed.
    TemporalResource create_temporal_image(WGPUStringView name, const TextureDesc& desc, uint64_t hash = 0);

    // Temporal (history) BUFFER: the GPU-buffer twin of create_temporal_image -- two physical buffers
    // the PersistentResourcePool ping-pongs each frame, same contract (write `.curr`, read `.prev`;
    // writing `.prev` is an authoring error, reported under RG_VALIDATE). `hash`: same camera-cut
    // invalidation as create_temporal_image (nonzero = destroy+recreate on mismatch, 0 = off).
    TemporalResource create_temporal_buffer(WGPUStringView name, const BufferDesc& desc, uint64_t hash = 0);

    // Persistent (cross-frame) SINGLE GPU buffer the graph owns: one pool-backed buffer (no ping-pong),
    // survives the per-frame teardown, auto-evicted once no pass declares it. Read AND write it in one pass
    // (var<storage, read_write>): the graph models that as the StorageRead+StorageWrite RMW pair and does
    // not self-order the pass. Use for in-place own-slot state -- accumulators, atomic counters, append /
    // free-lists; for cross-element reads (neighbour / previous-frame), use create_temporal_buffer instead.
    // The graph owns lifetime + usage + cross-pass ordering only; making the in-dispatch RMW race-free is
    // the shader's job (own-slot discipline / atomics). GPU-authored only (no host-upload affordance).
    // ponytail: in-place persistent state is correct only because this is a single-queue, no-frames-in-flight
    // loop (frame N's submit completes before N+1 reads). Add explicit sync if frames-in-flight ever land.
    ResourceHandle create_persistent_buffer(WGPUStringView name, const BufferDesc& desc);

    // Persistent (cross-frame) SINGLE texture the graph owns: the image twin of create_persistent_buffer.
    // One pool-backed texture (no ping-pong), survives the per-frame teardown, auto-evicted once no pass
    // declares it. For a precomputed/baked resource written once then sampled every frame -- IBL / env map,
    // BRDF LUT, prefiltered specular. Declare it every frame (to read it), and fill it with a pass marked
    // PassBuilder::initialize(handle, hash): that bake runs only when the texture is freshly (re)created
    // (first frame, eviction, or a descriptor/resize change -- the pool clears its `baked` flag) or the
    // settings `hash` changes, not every frame.
    ResourceHandle create_persistent_image(WGPUStringView name, const TextureDesc& desc);


    // CAVEAT: pass declaration order matters (implicit SSA versioning, def-before-use): declare a
    // resource's WRITER before any pass that reads it. Each write starts a new version; each read
    // binds to the latest version declared so far. Passes that share no resource may be declared in
    // any order. Reading a TRANSIENT resource before any pass writes it is an authoring error:
    // compile() prints the offending pass/resource and returns false (do NOT realize()/execute() that
    // frame) instead of silently scheduling the reader first against uninitialized contents (e.g. a
    // depth prepass writing depth must be added before a scene pass that samples it). Reading an
    // IMPORTED resource's pre-supplied value before an in-graph write is legal (a normal WAR).
    // Order-independent declaration would need multi-pass analysis or explicit versions; out of scope.
    template<typename BuilderFn, typename ExecuteFn>
    void add_pass(WGPUStringView name, PassKind kind, BuilderFn&& setup, ExecuteFn&& executeFn)
    {
        assert(name.length != 0 && "must have name");
        assert(kind != PassKind::None);
        PassBuilder builder = begin_pass(name, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    // returns false if the graph has an ordering error (see the add_pass CAVEAT above): a pass reads a
    // transient resource before any pass writes it. messages are printed; on false skip this frame.
    // that check is a dev aid compiled out in release (NDEBUG) like assert; see RG_VALIDATE in the
    // .cpp; a release build skips the per-frame walk and compile() always returns true.
    // IMPORTANT: TODO: if compile failes all cached resources as well as temporal and anything else gets perged.
    // enableAlias: opt in to transient memory aliasing (phase 4) -- pack disjoint-lifetime, same-signature
    // transients onto shared physical objects to cut peak VRAM. default off = byte-identical to before.
    bool compile(bool enableAlias = false);

    // create GPU resources from the usage + size that compile() worked out
    void realize(WGPUDevice device);
    // record the compiled passes into a caller-owned encoder (caller submits + presents).
    // enableProfiling: opt in to per-pass GPU timestamp queries (needs the device's TimestampQuery
    // feature). Off = byte-identical to before. Pair with collect_gpu_timings() after the submit.
    void execute(WGPUCommandEncoder encoder, WGPUQueue queue, bool enableProfiling = false);
    // kick the async read-back of the last execute()'s GPU timestamps. Call AFTER queue submit; results
    // surface a couple frames later via the instance's event pump. No-op if profiling was off that frame.
    void collect_gpu_timings();
    // release graph-created textures/views/buffers (imported resources left alone)
    void release_resources();
    // No data members: state lives in a .cpp-private RenderGraphStorage (see RenderGraph.cpp).

private:
    // type-erase the execute callback into allocator-owned memory; the trampoline is a
    // captureless lambda (-> plain fn-pointer), so no extra named symbol leaks onto the struct
    template<class F> void store_exec(PassBuilder& b, F&& f){
        using D = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<D>,
            "execute callback must be trivially destructible (arena frees without dtor); capture handles/ids by value");
        void* m = alloc_exec(sizeof(D), alignof(D));
        if (!m) return;   // arena OOM (alloc_exec already announced it): leave exec_fn null -> execute() skips this pass
        ::new (m) D(std::forward<F>(f));
        set_exec(b, m, [](void* o, PassContext& c){ (*static_cast<D*>(o))(c); });
    }
    void* alloc_exec(size_t size, size_t align);                                       // forwards to GraphAllocator
    void  set_exec(PassBuilder& builder, void* obj, void(*fn)(void*, PassContext&));  // writes obj+fn onto PassNode

    // .cpp shims add_pass uses to alloc/append a PassNode across the header/.cpp boundary
    // (PassNode/GraphAllocator are incomplete here); not part of the public API.
    PassBuilder begin_pass(WGPUStringView name, PassKind kind);
    void end_pass(PassBuilder& builder);
};

// Creates an instance of the `GraphAllocator` with a given areana size default to to 1MB
GraphAllocator* create_allocator(size_t arenaSize = 1u << 20);

void destroy_allocator(GraphAllocator* allocator);


RenderGraph* create_render_graph(GraphAllocator* allocator);

// debug: dump the graph as a Mermaid flowchart to stdout (passes = nodes, resources = edges)
void debug_print_mermaid(RenderGraph* rg);

// debug: dump transient resource lifetimes as a Mermaid Gantt (one bar per resource over pass order)
void debug_print_lifetimes(RenderGraph* rg);

}// RG
#endif // RENDERGRAPH_H
