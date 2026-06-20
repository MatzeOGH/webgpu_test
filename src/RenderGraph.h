

// RenderGraph PoC
// The goal is to define a immidiate mode render graph system
// without thrashing the memory.
// RULE: dont abstract the webgpu api

#pragma once

#ifndef RENDERGRAPH_H
#define RENDERGRAPH_H

#include <cstdint>
#include <new>
#include <utility>
#include <type_traits>

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

struct ResourceHandle
{
    uint32_t id{};
};

// how a pass touches a resource -> read/write (hazards) and WGPU usage flags.
// comment column = WebGPU internal usage (usage-scope class) -> hazard -> usage bit.
enum struct AccessType : uint8_t
{
    ColorAttachment,         // attachment      write   tex RenderAttachment
    DepthStencilAttachment,  // attachment      write   tex RenderAttachment  (depth/stencil test + write)
    DepthStencilReadOnly,    // attachment-read read    tex RenderAttachment  (test only, depthReadOnly; no write hazard)
    Sampled,                 // constant        read    tex TextureBinding
    StorageRead,             // storage-read    read    tex StorageBinding / buf Storage
    StorageWrite,            // storage         write   tex StorageBinding / buf Storage
    Uniform,                 // constant        read    buf Uniform (+CopyDst host-upload affordance)
    CopySrc,                 // copy            read    tex/buf CopySrc
    CopyDst,                 // copy            write   tex/buf CopyDst
    Vertex,                  // input           read    buf Vertex
    Index,                   // input           read    buf Index
    Indirect,                // input           read    buf Indirect
};

struct ResourceAccess
{
    ResourceHandle handle{};
    AccessType     type{};

    // attachment-only (ColorAttachment / DepthStencilAttachment); ignored for other access types.
    WGPULoadOp  loadOp{};
    WGPUStoreOp storeOp{};
    WGPUColor   clearColor{};
    float       clearDepth{};
};

struct ResourceNode;
struct PassNode;
struct RenderGraph; // PassContext holds a back-pointer for resource lookup

// live during execute(): the encoder a pass body records into + resolved-resource lookup.
// the real thing that replaces the old forward-declared stub.
struct PassContext
{
    WGPUCommandEncoder encoder{};  // always set
    WGPURenderPassEncoder render{};   // set for Graphics passes
    WGPUComputePassEncoder compute{};  // set for Compute passes
    WGPUQueue queue{};
    RenderGraph* graph{};

    WGPUTextureView view(ResourceHandle h) const;   // resolved view
    WGPUTexture texture(ResourceHandle h) const;    // resolved texture (copies need the texture, not a view)
    WGPUBuffer buffer(ResourceHandle h) const;  // resolved buffer
};

struct GraphBuilder
{
    // single primitive every helper below is a thin wrapper over. load/store/clear/clearDepth only
    // matter for the two attachment AccessTypes; leave them defaulted for every other access.
    void use(ResourceHandle handle, AccessType type,
             WGPULoadOp load = WGPULoadOp_Undefined, WGPUStoreOp store = WGPUStoreOp_Undefined,
             WGPUColor clear = {}, float clearDepth = {});

    // color attachment
    void color(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, WGPUColor clear = {0, 0, 0, 1});
    // depth stencil attachment
    void depth_stencil(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, float clearDepth = 1.0f);
    // depth stencil attachment, read-only (depth/stencil test, no write -- e.g. lighting depth-testing
    // a prepass depth). no load/store/clear: WebGPU requires depthLoadOp/StoreOp Undefined when read-only.
    void depth_stencil_read_only(ResourceHandle handle);
    // sampled resouces
    void sampled(ResourceHandle handle);
    void storage_read(ResourceHandle handle);
    void storage_write(ResourceHandle handle);
    // uniform
    void uniform(ResourceHandle handle);
    // transfer (copy) source / destination
    void copy_src(ResourceHandle handle);
    void copy_dst(ResourceHandle handle);
    // buffer-only: vertex/index/indirect-args
    void vertex_buffer(ResourceHandle handle);
    void index_buffer(ResourceHandle handle);
    void indirect_buffer(ResourceHandle handle);

    PassNode* m_new_pass{};
};
struct GraphAllocator; // internal allocator
struct GraphResourceCache; // holds the graphs resources over multiple frames.

struct TextureDesc
{
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    //
    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f, scaleY = 1.0f;
    ResourceHandle relativeTo{};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;
};

struct BufferDesc
{
    uint64_t size{};
};

struct RenderGraph
{

    //
    ResourceHandle create_image(WGPUStringView name, const TextureDesc& desc);
    ResourceHandle importe_image(WGPUStringView name, WGPUTextureView view, WGPUExtent3D size);
    ResourceHandle create_buffer(WGPUStringView name, const BufferDesc& desc);
    ResourceHandle import_buffer(WGPUStringView name, WGPUBuffer buffer);


    // CAVEAT -- pass declaration order matters (implicit SSA versioning, def-before-use): declare a
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
        GraphBuilder builder = begin_pass(name, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    // returns false if the graph has an ordering error (see the add_pass CAVEAT above): a pass reads a
    // transient resource before any pass writes it. messages are printed; on false skip this frame.
    // that check is a dev aid compiled out in release (NDEBUG) like assert -- see RG_VALIDATE in the
    // .cpp; a release build skips the per-frame walk and compile() always returns true.
    bool compile();

    // create GPU resources from the usage + size that compile() worked out
    void realize(WGPUDevice device);
    // record the compiled passes into a caller-owned encoder (caller submits + presents)
    void execute(WGPUCommandEncoder encoder, WGPUQueue queue);
    // release graph-created textures/views/buffers (imported resources left alone)
    void release_resources();
    // resolve a handle to its node (linear walk; see ceiling note in .cpp)
    ResourceNode* node(ResourceHandle h);

    GraphAllocator* m_allocator{};
    GraphResourceCache* cache{};
    ResourceNode* m_resouces{};
    PassNode* m_passes{};       // after compile(): in execution order (toposorted)
    WGPUDevice m_device{};      // set by realize(); resources are created against it
    uint32_t next_id = 1; // 0 = invalid handle

private:
    // type-erase the execute callback into allocator-owned memory; the trampoline is a
    // captureless lambda (-> plain fn-pointer), so no extra named symbol leaks onto the struct
    template<class F> void store_exec(GraphBuilder& b, F&& f){
        using D = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<D>,
            "execute callback must be trivially destructible (arena frees without dtor); capture handles/ids by value");
        void* m = alloc_exec(sizeof(D), alignof(D));
        ::new (m) D(std::forward<F>(f));
        set_exec(b, m, [](void* o, PassContext& c){ (*static_cast<D*>(o))(c); });
    }
    void* alloc_exec(size_t size, size_t align);                                       // forwards to GraphAllocator
    void  set_exec(GraphBuilder& builder, void* obj, void(*fn)(void*, PassContext&));  // writes obj+fn onto PassNode

    // .cpp shims add_pass uses to alloc/append a PassNode across the header/.cpp boundary
    // (PassNode/GraphAllocator are incomplete here); not part of the public API.
    GraphBuilder begin_pass(WGPUStringView name, PassKind kind);
    void end_pass(GraphBuilder& builder);
};

GraphAllocator* create_allocator();
GraphResourceCache* create_cache();
RenderGraph* create_render_graph(GraphAllocator* allocator, GraphResourceCache* cache);

// debug: dump the graph as a Mermaid flowchart to stdout (passes = nodes, resources = edges)
void debug_print_mermaid(RenderGraph* rg);

}// RG
#endif // RENDERGRAPH_H
