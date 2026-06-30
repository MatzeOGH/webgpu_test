

// Immediate-mode render graph for WebGPU. Declare resources and passes per frame; compile
// schedules them; realize + execute records commands. No GPU-API abstraction. Pass bodies
// call wgpu* directly. Graph handles ordering and resource lifetime only.


#pragma once

#ifndef RENDERGRAPH_H
#define RENDERGRAPH_H

#include <cstdint>
#include <new>
#include <utility>
#include <type_traits>
#include <assert.h>
#include <string_view>

#include <webgpu/webgpu.h>

// Create a string view form a string literal
#define WEBGPU_STR(str) WGPUStringView{ .data = "" str, .length = sizeof("" str) - 1 }

namespace RG
{

enum struct PassKind : uint8_t
{
    None = 0,
    Graphics,
    Compute,
    Transfer
};

enum struct SizeKind : uint8_t
{
    Absolute,
    Relative
};


enum struct ResourceKind : uint8_t {
    None,
    Texture,
    Buffer,
};

// helper for const char* to fnv1a hash
inline constexpr uint64_t fnv1a(const char* s) {
    uint64_t hash = 14695981039346656037ULL;
    while (*s) {
        hash ^= static_cast<uint8_t>(*s++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// `ResourceId` used for passes, textures and buffers
// Careful! the string internals are not owned 
struct ResourceId
{
    uint64_t value{};
    WGPUStringView name{};
};

// `ResourceId` literal
constexpr ResourceId operator"" _rid(const char* s, size_t l)
{
    return ResourceId(fnv1a(s), WGPUStringView{ .data = s, .length = l });
}

constexpr bool operator==(ResourceId a, ResourceId b)
{
    return a.value == b.value &&
        a.name.length == b.name.length && 
        // slow 
        std::string_view(a.name.data, a.name.length) ==
        std::string_view(b.name.data, b.name.length);
}

struct ResourceHandle
{
    uint32_t id{};
    ResourceKind kind{};
};


// Ping-pong history pair. Two GPU objects rotate each frame: write .curr, read .prev.
struct TemporalResource
{
    ResourceHandle curr; // used for writes
    ResourceHandle prev; // used for reads
};

struct ResourceNode;
struct PassNode;
struct RenderGraph;

// Error message chain for reporting errors to the client
struct ErrorMessage {
    WGPUStringView message;
    ErrorMessage* next{};
};

// Pass body gets this during execute(). Resolves handles to GPU objects and holds the encoders.
struct PassContext
{
    WGPUCommandEncoder encoder{};
    WGPURenderPassEncoder render{};     // set for Graphics passes
    WGPUComputePassEncoder compute{};   // set for Compute passes
    WGPUQueue queue{};
    RenderGraph* graph{};
    PassNode* pass{};

    // Returns a texture view for the given resource. 
    WGPUTextureView view(ResourceHandle h) const;
    WGPUTexture texture(ResourceHandle h) const;
    WGPUBuffer buffer(ResourceHandle h) const;
    WGPUExtent3D texture_size(ResourceHandle h) const;
    uint32_t buffer_size(ResourceHandle h) const;
};

struct PassBuilder
{
    // Color attachment (render passes only).
    void color(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, WGPUColor clear = {0, 0, 0, 1}, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Depth-stencil attachment. For combined formats pass stencil load/store/clear too; the bound
    // pipeline's depthStencil state drives the actual test/write, the graph only carries ops.
    void depth_stencil(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, float clearDepth = 1.0f, uint32_t baseMip = 0, uint32_t baseLayer = 0, WGPULoadOp stencilLoad = WGPULoadOp_Undefined, WGPUStoreOp stencilStore = WGPUStoreOp_Undefined, uint32_t stencilClear = 0);
    // Read-only depth-stencil attachment (test only, no write). WebGPU requires Undefined load/store.
    void depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // MSAA resolve target. Call right after the color() it resolves. Must be single-sample with same
    // format and size. Dawn validates. Can be imported (e.g. resolve into the swapchain).
    void resolve(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Sampled (read-only in shader).
    void sampled(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    void storage_read(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    void storage_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Storage read+write on one resource in a single pass. Maps to WGSL var<storage, read_write>.
    // In-dispatch race is the shader's responsibility (own-slot / atomics only).
    void storage_read_write(ResourceHandle handle, uint32_t baseMip = 0, uint32_t baseLayer = 0);
    // Uniform buffer.
    void uniform(ResourceHandle handle);
    // Transfer (copy) source / destination.
    void copy_src(ResourceHandle handle);
    void copy_dst(ResourceHandle handle);
    // Buffer-only: vertex, index, indirect args.
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    // Mark this pass as the initializer for a pool-backed resource (persistent or temporal .curr).
    // Runs only when the target needs baking: first frame, after pool eviction/recreation, or hash
    // mismatch versus the last bake. hash=0 (default) bakes once then skips permanently. Declare
    // the write (color()/storage_write()) normally; initialize() only gates execution. One target
    // per pass, write only that target.
    void initialize(ResourceHandle target, uint64_t hash = 0);

    // Prevent this pass from being culled even when no pass reads its output and it writes no
    // imported/persistent resource. For side-effect-only passes: GPU readback, profiling resolve,
    // indirect-arg gen consumed outside the graph. Not an access, no hazard/usage recorded.
    void force_keep();

    PassNode* m_new_pass{};
};
struct GraphAllocator;

struct TextureDesc
{
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    ResourceHandle relativeTo{};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;
    uint32_t mipLevelCount = 1;
    uint32_t sampleCount = 1;  // 1 or 4 (WebGPU 1.0 limit)
};

struct BufferDesc
{
    uint64_t size{};
};

// RenderGraph is frame-scoped. Destroy via release_resources() before next frame's create.
struct RenderGraph
{
    // Per-frame transient GPU texture. realize() backs it.
    ResourceHandle create_image(ResourceId id, const TextureDesc& desc);
    // Import an external texture (e.g. swapchain). Caller owns lifetime.
    ResourceHandle importe_image(ResourceId id, WGPUTextureView view, WGPUExtent3D size);
    // Per-frame transient GPU buffer. realize() backs it.
    ResourceHandle create_buffer(ResourceId id, const BufferDesc& desc);
    // Import an external buffer. Caller owns lifetime.
    ResourceHandle import_buffer(ResourceId id, WGPUBuffer buffer);

    // Ping-pong texture pair. Write .curr, read .prev. Pool rotates each frame. hash != 0 enables
    // camera-cut invalidation: mismatch destroys+recreates both layers. ping-pong only (not N-deep).
    TemporalResource create_temporal_image(ResourceId id, const TextureDesc& desc, uint64_t hash = 0);

    // Ping-pong buffer pair. Same contract as create_temporal_image.
    TemporalResource create_temporal_buffer(ResourceId id, const BufferDesc& desc, uint64_t hash = 0);

    // Cross-frame persistent buffer. Pool-backed, auto-evicted when no pass declares it.
    // Storage read+write in one pass works (the graph models the RMW pair, no self-ordering).
    // RMW race is the shader's responsibility (own-slot / atomics). GPU-authored only.
    // Safe because single-queue, no frames-in-flight. Add sync if that changes.
    ResourceHandle create_persistent_buffer(ResourceId id, const BufferDesc& desc);

    // Cross-frame persistent texture. Same lifetime as create_persistent_buffer. Pair with
    // PassBuilder::initialize() for a bake-once, sample-every-frame pattern (IBL, BRDF LUT).
    ResourceHandle create_persistent_image(ResourceId id, const TextureDesc& desc);

    // Declare a pass. Declaration order defines SSA versions: declare a resource's writer before
    // its readers. Reading a transient resource before any writer is an authoring error: compile()
    // returns false. Imported resources read before an in-graph write is legal (normal WAR).
    // Order-independent declaration is not supported.
    template<typename BuilderFn, typename ExecuteFn>
    void add_pass(ResourceId id, PassKind kind, BuilderFn&& setup, ExecuteFn&& executeFn)
    {
        assert(id.name.length != 0 && "must have name");
        assert(kind != PassKind::None);
        PassBuilder builder = begin_pass(id, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    // Build the DAG, cull dead passes, topo-sort, accumulate usage. Returns false on ordering error
    // (read-before-write of a transient).
    // enableAlias: pack disjoint-lifetime transients onto shared GPU objects to cut peak VRAM.
    void compile(bool enableAlias);

    // Record passes into a caller-owned encoder. enableProfiling: per-pass GPU timestamps (needs
    // TimestampQuery feature). Pair with collect_gpu_timings() after submit.
    void execute(WGPUDevice device, WGPUCommandEncoder encoder, WGPUQueue queue, bool enableProfiling = false);
    // Kick async GPU timestamp readback. Call after queue submit. Results arrive via the instance
    // event pump a few frames later. No-op if profiling was off.
    void collect_gpu_timings();

    // Query for errors. is null when no errors where found
    // TODO: add error messages
    ErrorMessage* getErrors();

private:
    // Type-erase execute callback into allocator-owned memory.
    template<class F> void store_exec(PassBuilder& b, F&& f){
        using D = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<D>,
            "execute callback must be trivially destructible (arena frees without dtor); capture handles/ids by value");
        void* m = alloc_exec(sizeof(D), alignof(D));
        assert(m && "Faild to alloc! GraphAllocator out of memory!");
        if (!m) return; // arena OOM: skip this pass (exec fn null)
        ::new (m) D(std::forward<F>(f));
        set_exec(b, m, [](void* o, PassContext& c){ (*static_cast<D*>(o))(c); });
    }
    void* alloc_exec(size_t size, size_t align);
    void  set_exec(PassBuilder& builder, void* obj, void(*fn)(void*, PassContext&));

    // Internal: allocates+appends a PassNode. Not public API.
    PassBuilder begin_pass(ResourceId id, PassKind kind);
    void end_pass(PassBuilder& builder);
};

// Create the `GraphAllocator`
// arenaSize defaults to 1 MB for now
GraphAllocator* create_allocator(size_t arenaSize = 1u << 20);

// Destroys a `GraphAllocator`
void destroy_allocator(GraphAllocator* allocator);

// Create render graph that is ready for recording.
// Don't ever store the in instance of the `RenderGraph` 
// Calling this invalidates the old `RenderGraph`
RenderGraph* create_render_graph(GraphAllocator* allocator);

// Dump the compiled graph as a Mermaid flowchart to stdout.
void debug_print_mermaid(RenderGraph* rg);

// Dump transient resource lifetimes as a Mermaid Gantt chart.
void debug_print_lifetimes(RenderGraph* rg);

}// RG
#endif // RENDERGRAPH_H
