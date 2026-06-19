#include "RenderGraph.h"
#include <malloc.h>
#include <memory>
#include <new>
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>


namespace RG{

// arena allocator to not fragment the heap during graph construction
// TODO: implement scratch arena for all resouces that life inside of the rendergraph
struct GraphAllocator
{
    template<typename T>
    inline T* make(){
        return new T;
    }

    // copies a (possibly non-owning) view into allocator-owned storage and returns a view
    // onto the copy. kept null-terminated so .data also works as a plain C string.
    WGPUStringView copy_string(WGPUStringView s){
        size_t len = (s.length == WGPU_STRLEN) ? (s.data ? std::strlen(s.data) : 0) : s.length;
        char* buf = new char[len + 1];
        if (len) std::memcpy(buf, s.data, len);
        buf[len] = '\0';
        return WGPUStringView{ buf, len };
        // ponytail: per-string new[]; swap for a bump arena when make<T> stops being new T
    }

    // raw aligned alloc seam for type-erased payloads (the execute callbacks). same arena
    // story as make<T>/copy_string: per-alloc operator new today, bump pointer later.
    void* alloc_raw(size_t size, size_t align){
        return ::operator new(size, std::align_val_t(align));
    }
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
    uint32_t       accessCount{};

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
    return new GraphAllocator;
}


RenderGraph* create_render_graph(GraphAllocator* allocator)
{
    RenderGraph* rg = allocator->make<RenderGraph>();
    rg->m_allocator = allocator;
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
                  case AccessType::Sampled:                r->texUsage |= WGPUTextureUsage_TextureBinding;   break;
                  case AccessType::StorageRead:
                  case AccessType::StorageWrite:
                      // texture vs buffer go to different usage fields (distinct types -> if/else, not ?:)
                      if (r->kind == ResourceNode::Kind::Texture) r->texUsage |= WGPUTextureUsage_StorageBinding;
                      else                                        r->bufUsage |= WGPUBufferUsage_Storage;
                      break;
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

// records one access on the pass currently being built
static void push_access(PassNode* p, ResourceHandle h, AccessType t)
{
    if (p && p->accessCount < PassNode::kMaxAccess)
        p->accesses[p->accessCount++] = { h, t };
    // ponytail: silently drops past kMaxAccess; add assert/grow when a real pass hits it
}

void GraphBuilder::color(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::ColorAttachment);
}

void GraphBuilder::depth_stencil(ResourceHandle handle)
{
    push_access(m_new_pass, handle, AccessType::DepthStencilAttachment);
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

} // RG


// standalone smoke-test driver, compiled as part of this TU (sees the internal structs above)
#include "RenderGraph_main.cpp"
