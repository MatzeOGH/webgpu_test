#include "RenderGraph.h"
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "AlpUtils.h"

// Graph validation -- the post-cull "reads a resource before any pass writes it" check in compile().
// It is a per-frame development aid; like assert it is compiled OUT in release builds (NDEBUG), so a
// shipping build assumes author-valid graphs and compile() always returns true. Predefine RG_VALIDATE
// to 1/0 to force it on/off independently of NDEBUG.
#ifndef RG_VALIDATE
#  ifdef NDEBUG
#    define RG_VALIDATE 0
#  else
#    define RG_VALIDATE 1
#  endif
#endif


namespace RG{

// arena allocator: bumps from both ends of one buffer. `used` grows up from base[0] for
// permanent per-frame nodes (reset once per frame); `scratchUsed` grows down from base[capacity]
// for compile()-local temporaries, reset per-scope via defer (AlpUtils.h) instead of calloc/free.
struct GraphAllocator
{
    // Pointer to the base memory block
    uint8_t* base{};
    // Offset to the next free byte
    size_t used{};
    // Total capacity in bytes
    size_t capacity{};
    // Offset from the top of the buffer to the next free scratch byte (grows downward).
    size_t scratchUsed{};

    // alignment must be a power of two
    static constexpr size_t align_up(size_t value, size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // Raw aligned allocation for type-erased payloads.
    void* alloc_raw(size_t size, size_t align)
    {
        const size_t offset = align_up(used, align);

        if (offset + size > capacity - scratchUsed)
        {
            assert(false && "GraphAllocator OOM");
            return nullptr;
        }

        void* p = base + offset;
        used = offset + size;
        return p;
    }

    // Raw aligned allocation from the TOP of the buffer, growing scratchUsed down -- short-lived,
    // compile()-local scratch. Pair every call site with `defer { <allocator>->reset_scratch(); };`
    // right after the scratch allocations in that scope.
    void* scratch_alloc_raw(size_t size, size_t align)
    {
        if (size > capacity - scratchUsed)
        {
            assert(false && "GraphAllocator scratch OOM");
            return nullptr;
        }

        size_t rawTop = (capacity - scratchUsed - size) & ~(align - 1);   // round start down to align
        size_t newScratchUsed = capacity - rawTop;

        if (newScratchUsed > capacity - used)   // would cross into the live front/permanent region
        {
            assert(false && "GraphAllocator scratch OOM");
            return nullptr;
        }

        scratchUsed = newScratchUsed;
        return base + rawTop;
    }

    // shared by make<T>/scratch_make<T>: placement-new construct T in raw storage.
    template<typename T, typename... Args>
    static T* construct(void* m, Args&&... args)
    {
        return m ? ::new (m) T(std::forward<Args>(args)...) : nullptr;
    }

    // shared by alloc<T>/scratch_alloc<T>: zero `count` T's worth of raw storage and reinterpret.
    template<typename T>
    static T* zero(void* m, size_t count)
    {
        if (m) std::memset(m, 0, sizeof(T) * count);
        return static_cast<T*>(m);
    }

    // Allocate + construct
    template<typename T, typename... Args>
    T* make(Args&&... args)
    {
        return construct<T>(alloc_raw(sizeof(T), alignof(T)), std::forward<Args>(args)...);
    }

    // Allocate + construct in scratch.
    template<typename T, typename... Args>
    T* scratch_make(Args&&... args)
    {
        return construct<T>(scratch_alloc_raw(sizeof(T), alignof(T)), std::forward<Args>(args)...);
    }

    // Allocate zeroed POD storage
    template<typename T>
    T* alloc(size_t count = 1)
    {
        return zero<T>(alloc_raw(sizeof(T) * count, alignof(T)), count);
    }

    // Allocate zeroed POD scratch storage (calloc replacement).
    template<typename T>
    T* scratch_alloc(size_t count = 1)
    {
        return zero<T>(scratch_alloc_raw(sizeof(T) * count, alignof(T)), count);
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
        scratchUsed = 0;
    }

    void reset_scratch()
    {
        scratchUsed = 0;
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


// the exclusive-write set (WebGPU usage-scope `attachment`/`storage`). everything else is a read
// (`input`/`constant`/`storage-read`/`attachment-read`). NOTE: DepthStencilReadOnly is deliberately
// absent -- read-only depth is `attachment-read`, a read; adding it here would reintroduce the false
// write hazard this distinction exists to remove.
static bool access_is_write(AccessType t)
{
    return t == AccessType::ColorAttachment
        || t == AccessType::DepthStencilAttachment
        || t == AccessType::StorageWrite
        || t == AccessType::CopyDst;
}

#if RG_VALIDATE
// do two accesses to the SAME resource in ONE pass (one usage scope) conflict? read+read never does.
// the lone read+write exception is StorageRead+StorageWrite: that is how the graph spells a read-modify-
// write storage binding (var<storage, read_write>) -- one writable-storage usage, not an alias (the
// "multi-writer chain" test + the sweep's WAR self-guard depend on it). Any other pairing involving a
// write is illegal: a read-only binding aliasing a write (e.g. Sampled+StorageWrite, the named case), or
// two writes the graph can't order within an atomic pass ("multiple unsynchronized writes").
static bool in_pass_accesses_conflict(AccessType a, AccessType b)
{
    if (!access_is_write(a) && !access_is_write(b)) return false;
    if ((a == AccessType::StorageRead  && b == AccessType::StorageWrite) ||
        (a == AccessType::StorageWrite && b == AccessType::StorageRead)) return false;
    return true;
}
#endif


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

enum struct HazardKind : uint8_t { RAW, WAW, WAR };

// One declaration-order sweep with implicit SSA resource versioning, shared by compile() phase 1
// (turns each edge into add_dependency) and debug_print_mermaid() (prints it) so the dump can never
// drift from the real graph. Each write to a resource starts a new "version" (the writing pass IS the
// version identity); each read binds to the current one. Calls onEdge(dependent, dep, id, kind) per
// discovered hazard -- RAW (read -> producer), WAW (write -> prev writer), WAR (write -> readers of
// the version being clobbered); `dependent` is always later in the walk than `dep`. A read seen before
// any writer of its resource simply binds to "no producer" (no edge) -- detecting that authoring error
// is left to compile()'s post-cull pass, which sees the final schedule; the sweep stays edge-only.
template<typename OnEdge>
static void sweep_resource_versions(GraphAllocator* alloc, PassNode* head, uint32_t next_id,
                                    OnEdge&& onEdge)
{
    // per resource id (1..next_id-1): the pass holding the current version, and the readers of that
    // version not yet retired by a newer write.
    PassNode** currentProducer = alloc->scratch_alloc<PassNode*>(next_id);
    NodeAdjacency** pendingReaders  = alloc->scratch_alloc<NodeAdjacency*>(next_id);
    defer { alloc->reset_scratch(); };

    for (PassNode* p = head; p; p = p->next)
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            uint32_t id = p->accesses[i].handle.id;
            if (access_is_write(p->accesses[i].type)) {
                // WAW: order this write after the previous writer. without it two writers of one
                // resource have no edge -> undefined order -> corruption.
                if (currentProducer[id] && currentProducer[id] != p)
                    onEdge(p, currentProducer[id], id, HazardKind::WAW);
                // WAR: order this write after every reader still using the version it clobbers.
                for (NodeAdjacency* r = pendingReaders[id]; r; r = r->next)
                    if (r->pass != p) onEdge(p, r->pass, id, HazardKind::WAR);
                currentProducer[id] = p;        // new version born
                pendingReaders[id]  = nullptr;  // its readers retired (old nodes are arena garbage)
            } else {
                // RAW: this read depends on the producer of the version it sees. a read before any
                // writer binds to "no producer" (no edge) -- compile()'s post-cull pass flags it.
                if (currentProducer[id] && currentProducer[id] != p)
                    onEdge(p, currentProducer[id], id, HazardKind::RAW);
                // register as a pending reader of the current version (for a future write's WAR).
                NodeAdjacency* link = alloc->scratch_make<NodeAdjacency>();   // transient: dead at function return
                link->pass = p; link->next = pendingReaders[id]; pendingReaders[id] = link;
            }
        }

    // both self-guards above are load-bearing: read-then-write of one handle in a single pass would
    // WAR-self-edge; write-then-write would WAW-self-edge. every edge points from a later- to an
    // earlier-visited pass, so adjacency is acyclic by construction (no compile()-made cycles).
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

bool RenderGraph::compile()
{
    // phase 1: build adjacency (pass dependency DAG, "depends-on" direction). The versioning sweep
    // (see sweep_resource_versions) discovers every RAW/WAW/WAR hazard in declaration order; here all
    // three collapse to add_dependency -- its dedup folds multiple hazards between one pass pair into
    // the single ordering edge phase 2 needs, so the resource id and hazard kind are ignored. Reads
    // before any writer get no edge here; the post-cull pass below turns them into errors.
    sweep_resource_versions(m_allocator, m_passes, next_id,
        [&](PassNode* dependent, PassNode* dep, uint32_t /*id*/, HazardKind /*kind*/) {
            add_dependency(m_allocator, dependent, dep);
        });

    // phase 2: dead-node removal + topo sort, fused into one DFS seeded from sinks.
    {
        // sinks = passes writing an imported resource. accesses store only handle.id, so flatten
        // the imported flags into an id-indexed table first (same scratch_alloc-over-next_id trick
        // as phase 1's lastWriter).
        bool* imported = m_allocator->scratch_alloc<bool>(next_id);
        for (ResourceNode* r = m_resouces; r; r = r->next)
            imported[r->handle.id] = r->imported;

        // topo into a transient array, then relink the intrusive list into execution order. The
        // result lives in m_passes itself; the array is just DFS scratch, reclaimed by the
        // deferred reset_scratch() below.
        uint32_t N = 0;
        for (PassNode* p = m_passes; p; p = p->next) ++N;

        PassNode** order = m_allocator->scratch_alloc<PassNode*>(N);
        defer { m_allocator->reset_scratch(); };
        uint32_t count = 0;
        for (PassNode* p = m_passes; p; p = p->next)
            if (is_sink(p, imported))
                topo_visit(p, order, count);          // only reaches passes that feed a sink

        // relink next-pointers to follow topo order; m_passes is now == execution order, and any
        // pass not reachable from a sink was never emitted -> dead, dropped here for free.
        for (uint32_t i = 0; i + 1 < count; ++i) order[i]->next = order[i + 1];
        if (count) order[count - 1]->next = nullptr;
        m_passes = count ? order[0] : nullptr;

        // ponytail: transient array as DFS scratch — can't sort a one-field intrusive list in
        // place (a dep emitted before the driver reaches it clobbers its `next`, dropping
        // disconnected nodes). recursive DFS, no cycle detection: graph is author-acyclic; add an
        // `onstack` flag (+2 lines) only if a cyclic graph ever needs catching.
    }

#if RG_VALIDATE
    // post-cull validation (development aid; compiled out when RG_VALIDATE==0 -- e.g. release/NDEBUG --
    // exactly like assert, so a shipping build pays none of this per-frame walk). Over the FINAL schedule
    // (m_passes is now culled + in execution order), a read of a TRANSIENT resource that no earlier pass
    // has produced is an authoring error -- the reader would sample uninitialized contents (its writer was
    // declared after it, or culled). walking the surviving passes makes this culling-correct and catches
    // every surviving reader, not just the first.
    //   imported resources                        -> exempt (their value comes from outside the graph).
    //   resources with no writer at all (e.g. a host-uploaded uniform) -> exempt (hasWriter stays false).
    // bail before phase 3 so the caller never realize()/execute()s a misordered graph.
    {
        bool* hasWriter = m_allocator->scratch_alloc<bool>(next_id);   // some surviving pass writes id
        bool* produced  = m_allocator->scratch_alloc<bool>(next_id);   // ...has written it so far, in order
        bool* imported  = m_allocator->scratch_alloc<bool>(next_id);
        defer { m_allocator->reset_scratch(); };
        for (ResourceNode* r = m_resouces; r; r = r->next) imported[r->handle.id] = r->imported;
        for (PassNode* p = m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (access_is_write(p->accesses[i].type)) hasWriter[p->accesses[i].handle.id] = true;

        bool hadError = false;
        for (PassNode* p = m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                uint32_t id = p->accesses[i].handle.id;
                if (access_is_write(p->accesses[i].type)) { produced[id] = true; continue; }
                if (id == 0 || imported[id] || produced[id] || !hasWriter[id]) continue;
                ResourceNode* r = node(p->accesses[i].handle);
                WGPUStringView rn = r ? r->name : WGPUStringView{};
                std::printf("[RenderGraph] error: pass \"%.*s\" reads resource \"%.*s\" before any pass "
                            "writes it -- declare a writer of \"%.*s\" first.\n",
                            (int)p->name.length, p->name.data ? p->name.data : "",
                            (int)rn.length, rn.data ? rn.data : "",
                            (int)rn.length, rn.data ? rn.data : "");
                hadError = true;
            }
        if (hadError) return false;
    }
#endif

    // phase 3: frame-independent CPU analysis -> accumulate WGPU usage + resolve concrete sizes.
    // WebGPU requires the usage bit at create time; realize() then only does the device create calls.
    {
        // id->node table, same scratch_alloc-over-next_id trick as phases 1/2.
        ResourceNode** byId = m_allocator->scratch_alloc<ResourceNode*>(next_id);
        defer { m_allocator->reset_scratch(); };
        for (ResourceNode* r = m_resouces; r; r = r->next) byId[r->handle.id] = r;

        for (PassNode* p = m_passes; p; p = p->next)          // m_passes == surviving (post-cull) passes
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                ResourceNode* r = byId[p->accesses[i].handle.id];
                if (!r) continue;
                switch (p->accesses[i].type) {
                  case AccessType::ColorAttachment:
                  case AccessType::DepthStencilAttachment:
                  case AccessType::DepthStencilReadOnly:   r->texUsage |= WGPUTextureUsage_RenderAttachment; break;
                  case AccessType::Sampled:                r->texUsage |= WGPUTextureUsage_TextureBinding;  break;
                  case AccessType::StorageRead:
                  case AccessType::StorageWrite:
                      // texture vs buffer go to different usage fields (distinct types -> if/else, not ?:)
                      if (r->kind == ResourceNode::Kind::Texture) r->texUsage |= WGPUTextureUsage_StorageBinding;
                      else                                        r->bufUsage |= WGPUBufferUsage_Storage;
                      break;
                  // CopyDst here is a deliberate host-upload affordance: the graph can't see a host-side
                  // wgpuQueueWriteBuffer, so uniform buffers get CopyDst by default (matches Renderer.cpp).
                  case AccessType::Uniform: r->bufUsage |= WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst; break;
                  // copy src/dst are kind-aware: a buffer copy needs the buffer-usage bit, not the texture one.
                  case AccessType::CopySrc:
                      if (r->kind == ResourceNode::Kind::Texture) r->texUsage |= WGPUTextureUsage_CopySrc;
                      else                                        r->bufUsage |= WGPUBufferUsage_CopySrc;
                      break;
                  case AccessType::CopyDst:
                      if (r->kind == ResourceNode::Kind::Texture) r->texUsage |= WGPUTextureUsage_CopyDst;
                      else                                        r->bufUsage |= WGPUBufferUsage_CopyDst;
                      break;
                  case AccessType::Vertex:   r->bufUsage |= WGPUBufferUsage_Vertex;   break;
                  case AccessType::Index:    r->bufUsage |= WGPUBufferUsage_Index;    break;
                  case AccessType::Indirect: r->bufUsage |= WGPUBufferUsage_Indirect; break;
                }
            }

        // resolve concrete sizes here (CPU-only -> belongs in compile, not realize) by walking each
        // texture's relativeTo chain. memoized + recursive, so chains and any declaration order work.
        for (ResourceNode* r = m_resouces; r; r = r->next)
            if (r->kind == ResourceNode::Kind::Texture) resolve_size(r, byId);

        // ponytail: usage==0 here == untouched by a live pass -> future realize() skips it = free
        // dead-resource culling. no separate resource liveness list needed.
    }

    return true;
}

// debug-only: position of `target` in the pass list = its Mermaid node id. O(n) per call.
static uint32_t pass_index(PassNode* head, PassNode* target)
{
    uint32_t i = 0;
    for (PassNode* p = head; p; p = p->next, ++i)
        if (p == target) return i;
    return 0;
}

// dump the graph as a Mermaid flowchart on stdout. passes are nodes; an edge dep -->|res| Q means Q
// depends on dep via resource res (data/order flow points dep -> Q). edges come from the SAME
// versioning sweep compile() uses, so the dump matches the real graph -- RAW (unlabelled), plus WAW
// and WAR tagged in the edge label -- rather than an approximation. safe before or after compile():
// the topo sort preserves the relative order of any two passes touching the same resource, so the
// rediscovered edges are unchanged. resources with a single touch (imported inputs, unread sinks)
// produce no pass->pass edge and don't appear.
void debug_print_mermaid(RenderGraph* rg)
{
    std::printf("flowchart LR\n");

    // node decl: stable id Pi -> pass name, indexed by list position.
    uint32_t idx = 0;
    for (PassNode* p = rg->m_passes; p; p = p->next, ++idx)
        std::printf("  P%u[\"%.*s\"]\n", idx, (int)p->name.length, p->name.data ? p->name.data : "");

    // one edge per discovered hazard, labelled with the resource name and (for WAW/WAR) the kind.
    sweep_resource_versions(rg->m_allocator, rg->m_passes, rg->next_id,
        [&](PassNode* dependent, PassNode* dep, uint32_t id, HazardKind kind) {
            ResourceNode* r = rg->node({ id });
            WGPUStringView nm = r ? r->name : WGPUStringView{};
            const char* tag = kind == HazardKind::WAW ? " (WAW)" : kind == HazardKind::WAR ? " (WAR)" : "";
            std::printf("  P%u -->|\"%.*s%s\"| P%u\n",
                        pass_index(rg->m_passes, dep), (int)nm.length, nm.data ? nm.data : "", tag,
                        pass_index(rg->m_passes, dependent));
        });

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

WGPUTextureView PassContext::view(ResourceHandle h) const
{
    return graph->node(h)->view;
}

WGPUTexture PassContext::texture(ResourceHandle h) const
{
    return graph->node(h)->texture;
}

WGPUBuffer PassContext::buffer(ResourceHandle h) const
{
    return graph->node(h)->buffer;
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
                } else if (a.type == AccessType::DepthStencilAttachment || a.type == AccessType::DepthStencilReadOnly) {
                    depth = WGPURenderPassDepthStencilAttachment{
                        .view            = r->view,
                        .depthLoadOp     = a.loadOp,
                        .depthStoreOp    = a.storeOp,
                        .depthClearValue = a.clearDepth,
                        .depthReadOnly   = a.type == AccessType::DepthStencilReadOnly,
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

// records one access on the pass currently being built -- the one primitive every GraphBuilder
// helper below wraps. load/store/clear are only meaningful for the two attachment AccessTypes;
// every other call site leaves them at their (ignored) defaults.
void GraphBuilder::use(ResourceHandle handle, AccessType type,
                       WGPULoadOp load, WGPUStoreOp store, WGPUColor clear, float clearDepth)
{
#if RG_VALIDATE
    // immediate (declaration-time) usage check -- fires at the exact b.sampled()/b.storage_write() call
    // site, not deferred to compile(). A pass is one WebGPU usage scope; a resource may not be aliased
    // read+write (e.g. sampled + storage_write -- the named case) or written more than once (the graph
    // can't order two writes inside an atomic pass). read+read and the StorageRead+StorageWrite RMW pair
    // are fine -- see in_pass_accesses_conflict.
    // ponytail: WebGPU itself permits multiple writable-storage uses in one scope; the graph is stricter
    // because it has no way to synchronize two writes inside a pass -- relax if a shader ever needs it.
    if (m_new_pass && handle.id) {
        const bool w = access_is_write(type);
        for (uint32_t i = 0; i < m_new_pass->accessCount; ++i) {
            if (m_new_pass->accesses[i].handle.id != handle.id) continue;
            if (in_pass_accesses_conflict(type, m_new_pass->accesses[i].type)) {
                std::printf("[RenderGraph] error: pass \"%.*s\" uses resource id %u %s in one pass -- a "
                            "written resource must be its only use in the pass.\n",
                            (int)m_new_pass->name.length, m_new_pass->name.data ? m_new_pass->name.data : "",
                            handle.id, (w && access_is_write(m_new_pass->accesses[i].type))
                                           ? "as more than one write (unsynchronized)"
                                           : "as both written and read");
                assert(false && "RenderGraph: illegal in-pass resource usage (read+write or double write in one pass)");
            }
        }
    }
#endif

    if (m_new_pass && m_new_pass->accessCount < PassNode::kMaxAccess)
        m_new_pass->accesses[m_new_pass->accessCount++] = { handle, type, load, store, clear, clearDepth };
    // ponytail: silently drops past kMaxAccess; add assert/grow when a real pass hits it
}

void GraphBuilder::color(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, WGPUColor clear)
{
    use(handle, AccessType::ColorAttachment, load, store, clear);
}

void GraphBuilder::depth_stencil(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, float clearDepth)
{
    use(handle, AccessType::DepthStencilAttachment, load, store, {}, clearDepth);
}

void GraphBuilder::depth_stencil_read_only(ResourceHandle handle)
{
    use(handle, AccessType::DepthStencilReadOnly);   // load/store/clear default Undefined/{} -- required when read-only
}

void GraphBuilder::sampled(ResourceHandle handle)
{
    use(handle, AccessType::Sampled);
}

void GraphBuilder::storage_read(ResourceHandle handle)
{
    use(handle, AccessType::StorageRead);
}

void GraphBuilder::storage_write(ResourceHandle handle)
{
    use(handle, AccessType::StorageWrite);
}

void GraphBuilder::uniform(ResourceHandle handle)
{
    use(handle, AccessType::Uniform);
}

void GraphBuilder::copy_src(ResourceHandle handle)
{
    use(handle, AccessType::CopySrc);
}

void GraphBuilder::copy_dst(ResourceHandle handle)
{
    use(handle, AccessType::CopyDst);
}

void GraphBuilder::vertex_buffer(ResourceHandle handle)
{
    use(handle, AccessType::Vertex);
}

void GraphBuilder::index_buffer(ResourceHandle handle)
{
    use(handle, AccessType::Index);
}

void GraphBuilder::indirect_buffer(ResourceHandle handle)
{
    use(handle, AccessType::Indirect);
}

} // RG


// standalone smoke-test driver, compiled as part of this TU (sees the internal structs above)
#include "RenderGraph_main.cpp"
