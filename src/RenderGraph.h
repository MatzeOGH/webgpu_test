

// RenderGraph
// The goal is to define a immidiate mode render graph system
// without thrashing the memory.
// RULE: dont abstract webgpu api

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

// how a pass touches a resource. type -> read/write (deps) and -> WGPU usage flags
enum struct AccessType : uint8_t
{
    ColorAttachment,         // write
    DepthStencilAttachment,  // read+write (treat as write for deps for now)
    Sampled,                 // read
    StorageRead,             // read
    StorageWrite,            // write
    Uniform,
    CopySrc,                 // read  (transfer source)
    CopyDst,                 // write (transfer destination)
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
    // color attachment
    void color(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, WGPUColor clear = {0, 0, 0, 1});
    // depth stencil attachment
    void depth_stencil(ResourceHandle handle, WGPULoadOp load = WGPULoadOp_Clear, WGPUStoreOp store = WGPUStoreOp_Store, float clearDepth = 1.0f);
    // sampled resouces
    void sampled(ResourceHandle handle);
    void storage_read(ResourceHandle handle);
    void storage_write(ResourceHandle handle);
    // uniform
    void uniform(ResourceHandle handle);
    // transfer (copy) source / destination
    void copy_src(ResourceHandle handle);
    void copy_dst(ResourceHandle handle);

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


    template<typename BuilderFn, typename ExecuteFn>
    void add_pass(WGPUStringView name, PassKind kind, BuilderFn&& setup, ExecuteFn&& executeFn)
    {
        GraphBuilder builder = begin_pass(name, kind);
        setup(builder);
        store_exec(builder, std::forward<ExecuteFn>(executeFn));
        end_pass(builder);
    }

    void compile();

    // create GPU resources from the usage + size that compile() worked out
    void realize(WGPUDevice device);
    // record the compiled passes into a caller-owned encoder (caller submits + presents)
    void execute(WGPUCommandEncoder encoder, WGPUQueue queue);
    // release graph-created textures/views/buffers (imported resources left alone)
    void release_resources();
    // re-point an imported resource at this frame's GPU handle (e.g. the swapchain view, which the
    // surface hands out fresh every frame) so a graph can be built/compiled once and re-executed.
    void update_imported_view(ResourceHandle h, WGPUTextureView view, WGPUExtent3D size, WGPUTexture texture = nullptr);
    // resolve a handle to its node (linear walk; see ceiling note in .cpp)
    ResourceNode* node(ResourceHandle h);

    // debug: dump the graph as a Mermaid flowchart to stdout (passes = nodes, resources = edges)
    void debug_print_mermaid();

    // Marks the beginning of the declaration of a render pass
    GraphBuilder begin_pass(WGPUStringView name, PassKind kind);

    // Marks the end of the render pass
    void end_pass(GraphBuilder& builder);

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
};

GraphAllocator* create_allocator();
GraphResourceCache* create_cache();
RenderGraph* create_render_graph(GraphAllocator* allocator, GraphResourceCache* cache);


}// RG
#endif // RENDERGRAPH_H
