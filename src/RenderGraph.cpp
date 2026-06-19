#include "RenderGraph.h"
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>


namespace RG{



// arena allocator to not fragment the heap during graph construction
// TODO: implement scratch arena for all resouces that life inside of the rendergraph
struct GraphAllocator
{
    // Pointer to the base memory block
    uint8_t* base{};
    // Offset to the next free byte
    size_t used{};
    // Total capacity in bytes
    size_t capacity{};

    // alignment must be a power of two
    static constexpr size_t align_up(size_t value, size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // Raw aligned allocation for type-erased payloads.
    void* alloc_raw(size_t size, size_t align)
    {
        const size_t offset = align_up(used, align);

        if (offset + size > capacity)
        {
            assert(false && "GraphAllocator OOM");
            return nullptr;
        }

        void* p = base + offset;
        used = offset + size;
        return p;
    }

    // Allocate + construct
    template<typename T, typename... Args>
    T* make(Args&&... args)
    {
        void* m = alloc_raw(sizeof(T), alignof(T));
        return m ? ::new (m) T(std::forward<Args>(args)...) : nullptr;
    }

    // Allocate zeroed POD storage
    template<typename T>
    T* alloc(size_t count = 1)
    {
        void* m = alloc_raw(sizeof(T) * count, alignof(T));

        if (m)
            std::memset(m, 0, sizeof(T) * count);

        return static_cast<T*>(m);
    }

    // Copy a string view into allocator-owned storage.
    // Result is always null-terminated.
    WGPUStringView copy_string(WGPUStringView s)
    {
        const size_t len = (s.length == WGPU_STRLEN)
                ? (s.data ? std::strlen(s.data) : 0)
                : s.length;

        char* buf = alloc<char>(len + 1);
        if (!buf)
            return {};

        if (len)
            std::memcpy(buf, s.data, len);

        buf[len] = '\0';

        return WGPUStringView{ buf, len };
    }

    void reset()
    {
        used = 0;
    }
};

struct GraphResourceCache
{
    std::vector<ResourceNode> cachedResources;
};

// internal resouceNode of an image or buffer
// structured as a intrusive linked list for memory resouce
// fat stuct style
struct ResourceNode
{
    ResourceHandle handle{};
    WGPUStringView name{};
    enum struct Kind { Texture, Buffer} kind{};

    // the resource is managed from outside te render graph. i.e: swapchain
    bool imported{};

    // texture fields
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;

    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f, scaleY = 1.0f;
    ResourceHandle relativeToHandle{};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;

    // buffer fields
    uint64_t bufferSize{};

    // realized / registered GPU handles
    WGPUTexture      texture{};                       // created: the texture object backing `view`
    WGPUTextureView  view{};                         // imported: the registered swapchain view
    WGPUBuffer       buffer{};                        // imported: the registered buffer
    WGPUExtent3D     resolved = WGPU_EXTENT_3D_INIT;  // imported: registered size (base for future Relative resolution)
    WGPUTextureUsage texUsage{};                      // accumulated in compile() from the access list
    WGPUBufferUsage  bufUsage{};                      //   "       — WebGPU needs these at create time

    ResourceNode* next{}; // ptr to the next resouce node of the render graph
};

struct NodeAdjacency;

// internal passNode, intrusive linked list + inline access array, same fat-struct
// style as ResourceNode so the builder records accesses without per-access allocation
struct PassNode
{
    WGPUStringView name{};
    PassKind    kind{};

    // type-erased execute callback: stored by add_pass, invoked by the future execute phase
    void* exec_obj{};
    void (*exec_fn)(void*, PassContext&){};

    // ponytail: fixed inline array, no alloc; bump N or add a spill list if a pass needs more
    static constexpr uint32_t kMaxAccess = 16;
    ResourceAccess accesses[kMaxAccess];
    uint32_t accessCount{};

    // inline adjacency list
    NodeAdjacency* adjacency{};

    bool placed{}; // topo sort: already emitted into execution order

    PassNode* next{}; // ptr to the next pass node of the render graph
};

struct NodeAdjacency
{
    PassNode* pass{};
    NodeAdjacency* next{};
};

GraphAllocator* create_allocator(){
    GraphAllocator* allocator = new GraphAllocator;
    size_t capacity = 1u << 20;// alloc 1 MB
    allocator->base = (uint8_t*)malloc(capacity);
    allocator->used = 0u;
    allocator->capacity = capacity;
    return allocator;
}

GraphResourceCache* create_cache()
{
    GraphResourceCache* cache = new GraphResourceCache;
    return cache;
}

RenderGraph* create_render_graph(GraphAllocator* allocator, GraphResourceCache* cache)
{
    allocator->reset();
    RenderGraph* rg = allocator->make<RenderGraph>();
    rg->m_allocator = allocator;
    rg->cache = cache;
    return rg;
}


// appends newNode to the end of the intrusive list whose head is *head.
// works for any node with a `next` pointer (ResourceNode, PassNode)
template<typename T>
static void list_append(T** head, T* newNode)
{
    if (*head == nullptr) {
        *head = newNode;
        return;
    }

    T* current = *head;
    while (current->next) {
        current = current->next;
    }

    current->next = newNode;
}


static bool access_is_write(AccessType t)
{
    return t == AccessType::ColorAttachment
        || t == AccessType::DepthStencilAttachment
        || t == AccessType::StorageWrite
        || t == AccessType::CopyDst;
}


ResourceHandle RenderGraph::create_image(WGPUStringView name, const TextureDesc& desc)
{
    ResourceNode* resouce = m_allocator->make<ResourceNode>();

    resouce->handle = { next_id++ };
    resouce->name = m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Texture;
    resouce->dimension = desc.dimension;
    resouce->format = desc.format;
    resouce->sizeKind = desc.sizeKind;
    resouce->scaleX = desc.scaleX;
    resouce->scaleY = desc.scaleY;
    resouce->relativeToHandle = desc.relativeTo;
    resouce->absolute = desc.absolute;

    list_append(&m_resouces, resouce);

    return resouce->handle;
}


ResourceHandle RenderGraph::create_buffer(WGPUStringView name, const BufferDesc& desc)
{
    ResourceNode* resouce = m_allocator->make<ResourceNode>();

    resouce->handle = { next_id++ };
    resouce->name = m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Buffer;

    resouce->bufferSize = desc.size;

    list_append(&m_resouces, resouce);

    return resouce->handle;
}


// imported resources are managed outside the graph (swapchain, etc). they carry no desc;
// the graph only needs the `imported` flag so passes that write them count as sinks (compile()).
ResourceHandle RenderGraph::importe_image(WGPUStringView name, WGPUTextureView view, WGPUExtent3D size)
{
    ResourceNode* resouce = m_allocator->make<ResourceNode>();

    resouce->handle = { next_id++ };
    resouce->name = m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Texture;
    resouce->imported = true;
    resouce->view = view;
    resouce->resolved = size;

    list_append(&m_resouces, resouce);

    return resouce->handle;
}


ResourceHandle RenderGraph::import_buffer(WGPUStringView name, WGPUBuffer buffer)
{
    ResourceNode* resouce = m_allocator->make<ResourceNode>();

    resouce->handle = { next_id++ };
    resouce->name = m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Buffer;
    resouce->imported = true;
    resouce->buffer = buffer;

    list_append(&m_resouces, resouce);

    return resouce->handle;
}


GraphBuilder RenderGraph::begin_pass(WGPUStringView name, PassKind kind)
{
    PassNode* pass = m_allocator->make<PassNode>();
    pass->name = m_allocator->copy_string(name);
    pass->kind = kind;

    GraphBuilder builder;
    builder.m_new_pass = pass;
    return builder;
}

void RenderGraph::end_pass(GraphBuilder& builder)
{
    list_append(&m_passes, builder.m_new_pass);
}

void* RenderGraph::alloc_exec(size_t size, size_t align)
{
    return m_allocator->alloc_raw(size, align);
}

void RenderGraph::set_exec(GraphBuilder& builder, void* obj, void(*fn)(void*, PassContext&))
{
    builder.m_new_pass->exec_obj = obj;
    builder.m_new_pass->exec_fn  = fn;
}

// records that pass `p` depends on pass `dep` (dedup; p->adjacency = predecessors)
static void add_dependency(GraphAllocator* alloc, PassNode* p, PassNode* dep)
{
    for (NodeAdjacency* a = p->adjacency; a; a = a->next)
        if (a->pass == dep) return;            // already linked
    NodeAdjacency* link = alloc->make<NodeAdjacency>();
    link->pass = dep;
    link->next = p->adjacency;
    p->adjacency = link;                       // prepend
}

// Topological sort via recursive DFS post-order over predecessor (depends-on) edges: a pass is
// appended only after everything it depends on, so `order` comes out deps-first. `placed` doubles
// as the visited marker so shared deps are emitted once.
//
// Why this and not Kahn's algorithm? Kahn is iterative BFS over in-degrees, seeded from sources,
// and visits the whole graph. We seed this DFS from sinks instead (see compile() phase 2), so
// recursion only reaches passes a sink transitively depends on -> dead-node removal falls out for
// free, with no in-degree counters or worklist. The trade-off: Kahn detects cycles for free (it
// emits < N nodes), this does not -- we assume an author-acyclic graph and would need an `onstack`
// flag to catch back-edges (also noted in phase 2).
static void topo_visit(PassNode* p, PassNode** order, uint32_t& count)
{
    if (p->placed) return;
    p->placed = true;
    for (NodeAdjacency* a = p->adjacency; a; a = a->next)
        topo_visit(a->pass, order, count);
    order[count++] = p;          // all deps already placed
}

// a sink/output pass writes at least one imported resource (swapchain, etc).
// these are the only roots that keep a pass alive; anything not reachable from one is dead.
static bool is_sink(PassNode* p, const bool* imported)
{
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (access_is_write(p->accesses[i].type) && imported[p->accesses[i].handle.id])
            return true;
    return false;
}

// resolve one texture's concrete size by walking its relativeTo chain. imported (registered size)
// and Absolute are the base cases; a Relative node multiplies its base's resolved size by its own
// scale, so the scale accumulates down a chain (depth -> colorAttachment -> swapchain). memoized via
// resolved.width (non-zero => already done), which also makes declaration order irrelevant.
// assumes an acyclic relativeTo graph (same author-acyclic assumption as topo_visit).
static WGPUExtent3D resolve_size(ResourceNode* r, ResourceNode** byId)
{
    if (r->resolved.width) return r->resolved;                       // imported, or already walked
    if (r->sizeKind == SizeKind::Absolute) return r->resolved = r->absolute;
    ResourceNode* base = byId[r->relativeToHandle.id];
    WGPUExtent3D b = base ? resolve_size(base, byId) : WGPUExtent3D{};
    return r->resolved = { (uint32_t)(b.width * r->scaleX), (uint32_t)(b.height * r->scaleY), 1 };
}

void RenderGraph::compile()
{
    // phase 1: build adjacency (pass dependency DAG, "depends-on" direction).
    // declaration-order-independent: a reader links to its producer even when the reader was
    // declared first (e.g. sink B sampling A's texture while B is declared before A).
    {
        // producer[id] = the pass that writes resource id; ids are 1..next_id-1. computed over
        // ALL passes first, so the read sweep below can see producers declared later.
        PassNode** producer = (PassNode**)std::calloc(next_id, sizeof(PassNode*));
        for (PassNode* p = m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (access_is_write(p->accesses[i].type))
                    producer[p->accesses[i].handle.id] = p;   // last writer wins (see ceiling note)

        // RAW edges: each pass depends on the producer of every resource it READS.
        for (PassNode* p = m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (!access_is_write(p->accesses[i].type)) {
                    PassNode* w = producer[p->accesses[i].handle.id];
                    if (w && w != p) add_dependency(m_allocator, p, w);   // self-read skipped
                }

        std::free(producer);
        // ponytail: single-producer model — RAW only, declaration order no longer matters. Drops
        // the old order-dependent WAW edge (incompatible with order-independence; a resource with
        // two live writers is undefined here). Ceiling: one writer per resource — add resource
        // versioning if ping-pong / multiple live writers ever appear. WAR still unmodeled.
    }

    // phase 2: dead-node removal + topo sort, fused into one DFS seeded from sinks.
    {
        // sinks = passes writing an imported resource. accesses store only handle.id, so flatten
        // the imported flags into an id-indexed table first (same calloc-over-next_id trick as
        // phase 1's lastWriter).
        bool* imported = (bool*)std::calloc(next_id, sizeof(bool));
        for (ResourceNode* r = m_resouces; r; r = r->next)
            imported[r->handle.id] = r->imported;

        // topo into a transient array, then relink the intrusive list into execution order. The
        // result lives in m_passes itself; the array is just DFS scratch, freed here.
        uint32_t N = 0;
        for (PassNode* p = m_passes; p; p = p->next) ++N;

        PassNode** order = (PassNode**)std::calloc(N, sizeof(PassNode*));
        uint32_t count = 0;
        for (PassNode* p = m_passes; p; p = p->next)
            if (is_sink(p, imported))
                topo_visit(p, order, count);          // only reaches passes that feed a sink

        // relink next-pointers to follow topo order; m_passes is now == execution order, and any
        // pass not reachable from a sink was never emitted -> dead, dropped here for free.
        for (uint32_t i = 0; i + 1 < count; ++i) order[i]->next = order[i + 1];
        if (count) order[count - 1]->next = nullptr;
        m_passes = count ? order[0] : nullptr;

        std::free(order);
        std::free(imported);
        // ponytail: transient array as DFS scratch — can't sort a one-field intrusive list in
        // place (a dep emitted before the driver reaches it clobbers its `next`, dropping
        // disconnected nodes). recursive DFS, no cycle detection: graph is author-acyclic; add an
        // `onstack` flag (+2 lines) only if a cyclic graph ever needs catching.
    }

    // phase 3: frame-independent CPU analysis -> accumulate WGPU usage + resolve concrete sizes.
    // WebGPU requires the usage bit at create time; realize() then only does the device create calls.
    {
        // id->node table, same calloc-over-next_id trick as phases 1/2.
        ResourceNode** byId = (ResourceNode**)std::calloc(next_id, sizeof(ResourceNode*));
        for (ResourceNode* r = m_resouces; r; r = r->next) byId[r->handle.id] = r;

        for (PassNode* p = m_passes; p; p = p->next)          // m_passes == surviving (post-cull) passes
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                ResourceNode* r = byId[p->accesses[i].handle.id];
                if (!r) continue;
                switch (p->accesses[i].type) {
                  case AccessType::ColorAttachment:
                  case AccessType::DepthStencilAttachment: r->texUsage |= WGPUTextureUsage_RenderAttachment; break;
                  case AccessType::Sampled:                r->texUsage |= WGPUTextureUsage_TextureBinding;  break;
                  case AccessType::StorageRead:
                  case AccessType::StorageWrite:
                      // texture vs buffer go to different usage fields (distinct types -> if/else, not ?:)
                      if (r->kind == ResourceNode::Kind::Texture) r->texUsage |= WGPUTextureUsage_StorageBinding;
                      else                                        r->bufUsage |= WGPUBufferUsage_Storage;
                      break;
                  case AccessType::Uniform: r->bufUsage |= WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst; break;
                  // ponytail: copy accesses map to texture usage only; buffer copy not needed yet
                  case AccessType::CopySrc: r->texUsage |= WGPUTextureUsage_CopySrc; break;
                  case AccessType::CopyDst: r->texUsage |= WGPUTextureUsage_CopyDst; break;
                }
            }

        // resolve concrete sizes here (CPU-only -> belongs in compile, not realize) by walking each
        // texture's relativeTo chain. memoized + recursive, so chains and any declaration order work.
        for (ResourceNode* r = m_resouces; r; r = r->next)
            if (r->kind == ResourceNode::Kind::Texture) resolve_size(r, byId);

        std::free(byId);
        // ponytail: usage==0 here == untouched by a live pass -> future realize() skips it = free
        // dead-resource culling. no separate resource liveness list needed.
    }

}

// debug-only: position of `target` in the pass list = its Mermaid node id. O(n) per call.
static uint32_t pass_index(PassNode* head, PassNode* target)
{
    uint32_t i = 0;
    for (PassNode* p = head; p; p = p->next, ++i)
        if (p == target) return i;
    return 0;
}

// dump the graph as a Mermaid flowchart on stdout. passes are nodes; an edge P -->|res| Q means
// pass P writes a resource that pass Q reads. edges are recomputed from the access lists (same
// producer->reader rule as compile() phase 1), so this works before or after compile(). resources
// read-but-unwritten (imported inputs) and written-but-unread (sinks like the swapchain) carry no
// pass->pass edge and don't appear.
void RenderGraph::debug_print_mermaid()
{
    std::printf("flowchart LR\n");

    // node decl: stable id Pi -> pass name, indexed by list position.
    uint32_t idx = 0;
    for (PassNode* p = m_passes; p; p = p->next, ++idx)
        std::printf("  P%u[\"%.*s\"]\n", idx, (int)p->name.length, p->name.data ? p->name.data : "");

    // producer[id] = pass writing resource id (last writer wins, matches compile()'s single-producer model)
    PassNode** producer = (PassNode**)std::calloc(next_id, sizeof(PassNode*));
    for (PassNode* p = m_passes; p; p = p->next)
        for (uint32_t i = 0; i < p->accessCount; ++i)
            if (access_is_write(p->accesses[i].type))
                producer[p->accesses[i].handle.id] = p;

    // one edge per read: link the reader back to the resource's producer, labelled with the resource name
    for (PassNode* p = m_passes; p; p = p->next)
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            const ResourceAccess& a = p->accesses[i];
            if (access_is_write(a.type)) continue;
            PassNode* w = producer[a.handle.id];
            if (!w || w == p) continue;                 // unproduced input / self-read
            ResourceNode* r = node(a.handle);
            WGPUStringView nm = r ? r->name : WGPUStringView{};
            std::printf("  P%u -->|\"%.*s\"| P%u\n",
                        pass_index(m_passes, w), (int)nm.length, nm.data ? nm.data : "",
                        pass_index(m_passes, p));
        }

    std::free(producer);
    std::fflush(stdout);
    // ponytail: pass_index is O(n) so the edge loop is O(n*edges); fine for a debug dump of a
    // handful of passes. names assumed pipe/quote-free (they're identifiers) -> no escaping.
}

// resolve a handle to its node by linear walk of the resource list.
// ponytail: O(n) per lookup; build an id->node table on the graph if pass bodies do many lookups.
ResourceNode* RenderGraph::node(ResourceHandle h)
{
    for (ResourceNode* r = m_resouces; r; r = r->next)
        if (r->handle.id == h.id) return r;
    return nullptr;
}

void RenderGraph::update_imported_view(ResourceHandle h, WGPUTextureView view, WGPUExtent3D size, WGPUTexture texture)
{
    if (ResourceNode* r = node(h)) { r->view = view; r->resolved = size; r->texture = texture; }
    // texture stays caller-owned: release_resources() skips imported nodes, so no double-free.
}

WGPUTextureView PassContext::view(ResourceHandle h) const
{
    ResourceNode* r = graph ? graph->node(h) : nullptr;
    return r ? r->view : nullptr;
}

WGPUTexture PassContext::texture(ResourceHandle h) const
{
    ResourceNode* r = graph ? graph->node(h) : nullptr;
    return r ? r->texture : nullptr;
}

WGPUBuffer PassContext::buffer(ResourceHandle h) const
{
    ResourceNode* r = graph ? graph->node(h) : nullptr;
    return r ? r->buffer : nullptr;
}

// create the GPU resources compile() worked out (size in `resolved`, usage in tex/bufUsage).
// imported resources are caller-owned and skipped; a resource with no accumulated usage was
// untouched by a live pass -> skipped too (the free dead-resource cull compile() phase 3 set up).
void RenderGraph::realize(WGPUDevice device)
{
    m_device = device;
    for (ResourceNode* r = m_resouces; r; r = r->next) {
        if (r->imported) continue;
        if (r->kind == ResourceNode::Kind::Texture) {
            if (!r->texUsage) continue;
            WGPUTextureDescriptor d{
                .label         = r->name,
                .usage         = r->texUsage,
                .dimension     = r->dimension,
                .size          = r->resolved,
                .format        = r->format,
                .mipLevelCount = 1,
                .sampleCount   = 1,
            };
            r->texture = wgpuDeviceCreateTexture(device, &d);
            r->view    = wgpuTextureCreateView(r->texture, nullptr);
        } else {
            if (!r->bufUsage) continue;
            WGPUBufferDescriptor d{
                .label = r->name,
                .usage = r->bufUsage,
                .size  = r->bufferSize,
            };
            r->buffer = wgpuDeviceCreateBuffer(device, &d);
        }
    }
    // ponytail: recreated on every realize(); no caching/aliasing. add a transient pool if
    // per-frame churn ever shows up in a profile.
}

// record the compiled passes (already in execution order) into a caller-owned encoder: open the
// right pass kind, wire the attachments declared in setup, invoke the stored body against a live
// PassContext. caller owns submit + present.
// ponytail: mirrors RenderPassBuilder/ComputePassBuilder in Renderer.cpp; reimplemented inline
// rather than shared because those live in Renderer.cpp (not a header) and this TU is standalone.
void RenderGraph::execute(WGPUCommandEncoder encoder, WGPUQueue queue)
{
    for (PassNode* p = m_passes; p; p = p->next) {
        PassContext ctx{};
        ctx.encoder = encoder;
        ctx.graph = this;
        ctx.queue = queue;

        if (p->kind == PassKind::Compute) {
            WGPUComputePassDescriptor cd{ .label = p->name };
            ctx.compute = wgpuCommandEncoderBeginComputePass(encoder, &cd);
            if (p->exec_fn) p->exec_fn(p->exec_obj, ctx);
            wgpuComputePassEncoderEnd(ctx.compute);
            wgpuComputePassEncoderRelease(ctx.compute);
        }
        else if (p->kind == PassKind::Graphics) {
            // gather declared attachments from the access list -> WebGPU render pass descriptor
            WGPURenderPassColorAttachment color[8]{};
            uint32_t nc = 0;
            WGPURenderPassDepthStencilAttachment depth{};
            bool hasDepth = false;

            for (uint32_t i = 0; i < p->accessCount; ++i) {
                const ResourceAccess& a = p->accesses[i];
                ResourceNode* r = node(a.handle);
                if (!r) continue;
                if (a.type == AccessType::ColorAttachment && nc < 8) {
                    color[nc++] = WGPURenderPassColorAttachment{
                        .view       = r->view,
                        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                        .loadOp     = a.loadOp,
                        .storeOp    = a.storeOp,
                        .clearValue = a.clearColor,
                    };
                } else if (a.type == AccessType::DepthStencilAttachment) {
                    depth = WGPURenderPassDepthStencilAttachment{
                        .view            = r->view,
                        .depthLoadOp     = a.loadOp,
                        .depthStoreOp    = a.storeOp,
                        .depthClearValue = a.clearDepth,
                        // stencil ops left Undefined -> depth-only formats (e.g. Depth32Float)
                    };
                    hasDepth = true;
                }
            }

            WGPURenderPassDescriptor rd{
                .label                  = p->name,
                .colorAttachmentCount   = nc,
                .colorAttachments       = color,
                .depthStencilAttachment = hasDepth ? &depth : nullptr,
            };
            ctx.render = wgpuCommandEncoderBeginRenderPass(encoder, &rd);
            if (p->exec_fn) p->exec_fn(p->exec_obj, ctx);
            wgpuRenderPassEncoderEnd(ctx.render);
            wgpuRenderPassEncoderRelease(ctx.render);
        }
        else { // Transfer / None: body records straight onto the encoder
            if (p->exec_fn) p->exec_fn(p->exec_obj, ctx);
        }
    }
}

// release graph-created GPU handles (imported ones are caller-owned -> left alone). pairs with
// realize(); call once the frame's commands have been submitted.
void RenderGraph::release_resources()
{
    for (ResourceNode* r = m_resouces; r; r = r->next) {
        if (r->imported) continue;
        if (r->view)    { wgpuTextureViewRelease(r->view); r->view    = nullptr; }
        if (r->texture) { wgpuTextureRelease(r->texture);  r->texture = nullptr; }
        if (r->buffer)  { wgpuBufferRelease(r->buffer);    r->buffer  = nullptr; }
    }
}

// records one access on the pass currently being built. load/store/clear are only meaningful for
// attachment accesses; the read/storage helpers below leave them at their (ignored) defaults.
static void push_access(PassNode* p, ResourceHandle h, AccessType t,
                        WGPULoadOp load = {}, WGPUStoreOp store = {},
                        WGPUColor clearColor = {}, float clearDepth = {})
{
    if (p && p->accessCount < PassNode::kMaxAccess)
        p->accesses[p->accessCount++] = { h, t, load, store, clearColor, clearDepth };
    // ponytail: silently drops past kMaxAccess; add assert/grow when a real pass hits it
}

void GraphBuilder::color(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, WGPUColor clear)
{
    push_access(m_new_pass, handle, AccessType::ColorAttachment, load, store, clear);
}

void GraphBuilder::depth_stencil(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, float clearDepth)
{
    push_access(m_new_pass, handle, AccessType::DepthStencilAttachment, load, store, {}, clearDepth);
}

void GraphBuilder::sampled(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::Sampled);
}

void GraphBuilder::storage_read(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::StorageRead);
}

void GraphBuilder::storage_write(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::StorageWrite);
}

void GraphBuilder::uniform(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::Uniform);
}

void GraphBuilder::copy_src(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::CopySrc);
}

void GraphBuilder::copy_dst(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::CopyDst);
}

} // RG


// standalone smoke-test driver, compiled as part of this TU (sees the internal structs above)
#include "RenderGraph_main.cpp"
