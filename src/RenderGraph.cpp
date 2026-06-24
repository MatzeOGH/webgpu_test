#include "RenderGraph.h"
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <webgpu/webgpu.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "AlpUtils.h"

// Graph validation: the post-cull "reads a resource before any pass writes it" check in compile().
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

// how a pass touches a resource -> read/write (hazards) and WGPU usage flags.
// comment column = WebGPU internal usage (usage-scope class) -> hazard -> usage bit.
enum struct AccessType : uint8_t
{
    ColorAttachment,         // attachment      write   tex RenderAttachment
    DepthStencilAttachment,  // attachment      write   tex RenderAttachment  (depth/stencil test + write)
    DepthStencilReadOnly,    // attachment-read read    tex RenderAttachment  (test only, depthReadOnly; no write hazard)
    ResolveAttachment,       // attachment      write   tex RenderAttachment  (single-sample target an MSAA color resolves into)
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

// one recorded access on a pass (PassNode stores a fixed inline array of these).
struct ResourceAccess
{
    ResourceHandle handle{};
    AccessType     type{};

    // attachment-only (ColorAttachment / DepthStencilAttachment); ignored for other access types.
    WGPULoadOp  loadOp{};
    WGPUStoreOp storeOp{};
    WGPUColor   clearColor{};
    float       clearDepth{};
    // stencil aspect of a DepthStencilAttachment: set for a depth+stencil format (e.g. Depth24PlusStencil8),
    // leave Undefined/0 for a depth-only format (e.g. Depth32Float). ignored for non-attachment accesses.
    WGPULoadOp  stencilLoadOp{};
    WGPUStoreOp stencilStoreOp{};
    uint32_t    stencilClear{};

    // subresource this access touches. for an attachment, the single mip/layer execute() renders into;
    // for a read, the level/layer the body samples (drives only the in-pass conflict check). 0 = mip 0/layer 0.
    uint32_t baseMip{};
    uint32_t baseLayer{};
};

// length of a (possibly WGPU_STRLEN-sentinel) string view, measured like copy_string.
static size_t sv_length(WGPUStringView s)
{
    return (s.length == WGPU_STRLEN) ? (s.data ? std::strlen(s.data) : 0) : s.length;
}

// Owns GPU resources that must outlive the per-frame graph teardown -- today, temporal/history textures
// (temporal accumulation, history feedback). One Entry per logical temporal resource, keyed by name content (the
// graph copies names into the per-frame arena, so the pointers don't survive between frames). Each Entry
// holds two physical textures the graph ping-pongs: layer 0 (current) maps to the opposite slot each
// frame, so last frame's "current" is this frame's "previous" for free.
struct PersistentResourcePool
{
    static constexpr uint32_t kLayers = 2;   // ping-pong: current + previous. N>2 deliberately unsupported.

    struct Entry
    {
        std::string          name;               // identity across frames (arena names don't persist)
        uint64_t             frame  = 0;          // rotation counter, bumped once per touch (declaration)

        WGPUTexture          tex[kLayers]  = {};
        WGPUTextureView      view[kLayers] = {};

        // what the live textures were created with; a mismatch forces a recreate (resize/format/usage).
        bool                 created       = false;
        WGPUExtent3D         size          = {};
        WGPUTextureFormat    format        = WGPUTextureFormat_Undefined;
        WGPUTextureDimension dim           = WGPUTextureDimension_2D;
        uint32_t             mipLevelCount = 1;
        uint32_t             sampleCount   = 1;
        WGPUTextureUsage     usage         = {};  // running union of every layer's usage
        WGPUTextureUsage     usageAtCreate = {};
    };
    std::vector<Entry> entries;   // ponytail: linear scan + memcmp; fine for the handful of temporal resources

    Entry* find(WGPUStringView name)
    {
        const size_t len = sv_length(name);   // names may arrive as WGPU_STRLEN (see screenTex)
        for (Entry& e : entries)
            if (e.name.size() == len &&
                (len == 0 || std::memcmp(e.name.data(), name.data, len) == 0))
                return &e;
        return nullptr;
    }

    // declaration-time: ensure the entry exists and advance its rotation. create_temporal_image is the only
    // caller and declares each resource once per frame, so one touch == one frame == one rotation. Declaring
    // the same name twice in a frame is an authoring error (it would double-rotate the slot mapping).
    Entry* touch(WGPUStringView name)
    {
        if (Entry* e = find(name)) { ++e->frame; return e; }   // next frame: rotate the ping-pong
        entries.emplace_back();
        Entry& e = entries.back();
        e.name.assign(name.data ? name.data : "", sv_length(name));   // WGPU_STRLEN -> measured, not SIZE_MAX
        return &e;                                             // first frame: no rotation yet
    }

    // physical slot backing logical layer `layerIndex` (0 == current, 1 == previous) this frame.
    uint32_t slot(const Entry& e, uint32_t layerIndex) const
    {
        return (uint32_t)((e.frame + layerIndex) & 1);         // ping-pong: flips each frame
    }

    // (re)create the two textures when missing or the descriptor changed (lazy: needs the device + the
    // usage union, both known only at realize()).
    void realize_entry(Entry* e, WGPUDevice device, WGPUExtent3D size, WGPUTextureFormat format,
                       WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount)
    {
        bool same = e->created
            && e->size.width == size.width && e->size.height == size.height
            && e->size.depthOrArrayLayers == size.depthOrArrayLayers
            && e->format == format && e->dim == dim && e->mipLevelCount == mipLevelCount
            && e->sampleCount == sampleCount
            && e->usageAtCreate == e->usage;
        if (same) return;
        destroy(e);
        e->size = size; e->format = format; e->dim = dim; e->mipLevelCount = mipLevelCount; e->sampleCount = sampleCount; e->usageAtCreate = e->usage;
        for (uint32_t i = 0; i < kLayers; ++i) {
            WGPUTextureDescriptor d{
                .label         = WGPUStringView{ e->name.c_str(), e->name.size() },
                .usage         = e->usage,
                .dimension     = dim,
                .size          = size,
                .format        = format,
                .mipLevelCount = mipLevelCount,
                .sampleCount   = sampleCount,
            };
            e->tex[i]  = wgpuDeviceCreateTexture(device, &d);
            e->view[i] = wgpuTextureCreateView(e->tex[i], nullptr);
        }
        e->created = true;
    }

    void destroy(Entry* e)
    {
        for (uint32_t i = 0; i < kLayers; ++i) {
            if (e->view[i]) { wgpuTextureViewRelease(e->view[i]); e->view[i] = nullptr; }
            if (e->tex[i])  { wgpuTextureRelease(e->tex[i]);      e->tex[i]  = nullptr; }
        }
        e->created = false;
    }

    ~PersistentResourcePool() { for (Entry& e : entries) destroy(&e); }
};

// Caches transient textures across the per-frame teardown so realize()/release_resources() stop
// churning the driver: one physical texture per Entry, matched by descriptor (size/format/dim/usage)
// and handed back out next frame instead of recreated + destroyed. Sibling to PersistentResourcePool
// (name-keyed ping-pong for history); this one is descriptor-keyed and evicts textures left idle.
// ponytail: linear scan over a vector -- fine for the ~dozen transients a frame declares. This same
// pool is the substrate for within-frame aliasing later: release a claim at a resource's lastUse
// instead of frame end and a later disjoint-lifetime resource reclaims the texture.
struct TransientResourcePool
{
    static constexpr uint64_t kRetain = 4;   // destroy a texture left unclaimed this many frames

    struct Entry
    {
        WGPUExtent3D         size   = {};
        WGPUTextureFormat    format = WGPUTextureFormat_Undefined;
        WGPUTextureDimension dim    = WGPUTextureDimension_2D;
        uint32_t             mipLevelCount = 1;
        uint32_t             sampleCount   = 1;
        WGPUTextureUsage     usage  = {};

        WGPUTexture          tex    = {};
        WGPUTextureView      view   = {};

        bool                 inUse         = false;  // claimed right now; released in end_frame (or mid-frame once aliasing lands)
        uint64_t             lastUsedFrame = 0;      // recency, eviction only
    };
    std::vector<Entry> entries;
    uint64_t frame            = 0;
    uint32_t createdThisFrame = 0;                   // debug: cache misses this frame (reset each end_frame)

    // debug event log: one record per texture create/evict so a UI can prove steady-state reuse --
    // no Create records after warmup means the cache hands the same textures back, not new ones. Ring
    // buffer, newest wraps over oldest; never read by the pool itself.
    enum class Event : uint8_t { Create, Evict };
    struct LogRec { uint64_t frame; Event kind; WGPUExtent3D size; WGPUTextureFormat format; };
    static constexpr uint32_t kLog = 128;
    LogRec   eventLog[kLog] = {};
    uint64_t eventCount     = 0;                      // total ever logged; UI shows last min(eventCount, kLog)

    void log_event(Event kind, const Entry& e)
    {
        eventLog[eventCount++ % kLog] = { frame, kind, e.size, e.format };
    }

    // hand out a free texture matching the descriptor, else create one. inUse (not the frame stamp)
    // is the claim, so two simultaneously-live same-descriptor resources get two distinct textures.
    void acquire(WGPUDevice device, WGPUExtent3D size, WGPUTextureFormat format,
                 WGPUTextureDimension dim, uint32_t mipLevelCount, uint32_t sampleCount, WGPUTextureUsage usage,
                 WGPUTexture& outTex, WGPUTextureView& outView)
    {
        for (Entry& e : entries) {
            if (!e.inUse && e.format == format && e.dim == dim && e.usage == usage
                && e.mipLevelCount == mipLevelCount && e.sampleCount == sampleCount
                && e.size.width == size.width && e.size.height == size.height
                && e.size.depthOrArrayLayers == size.depthOrArrayLayers) {
                e.inUse = true;
                e.lastUsedFrame = frame;
                outTex = e.tex; outView = e.view;
                return;
            }
        }
        entries.emplace_back();
        Entry& e = entries.back();
        e.size = size; e.format = format; e.dim = dim; e.mipLevelCount = mipLevelCount; e.sampleCount = sampleCount; e.usage = usage;
        WGPUTextureDescriptor d{
            .usage         = usage,
            .dimension     = dim,
            .size          = size,
            .format        = format,
            .mipLevelCount = mipLevelCount,
            .sampleCount   = sampleCount,
        };
        e.tex  = wgpuDeviceCreateTexture(device, &d);
        e.view = wgpuTextureCreateView(e.tex, nullptr);
        e.inUse = true;
        e.lastUsedFrame = frame;
        ++createdThisFrame;
        log_event(Event::Create, e);
        outTex = e.tex; outView = e.view;
    }

    // end of frame: drop every claim, destroy textures idle >= kRetain frames, advance the clock.
    void end_frame()
    {
        for (size_t i = entries.size(); i-- > 0; ) {
            Entry& e = entries[i];
            e.inUse = false;
            if (frame - e.lastUsedFrame >= kRetain) {   // lastUsedFrame <= frame always -> no underflow
                log_event(Event::Evict, e);
                destroy(&e);
                entries[i] = entries.back();
                entries.pop_back();
            }
        }
        createdThisFrame = 0;
        ++frame;
    }

    void destroy(Entry* e)
    {
        if (e->view) { wgpuTextureViewRelease(e->view); e->view = nullptr; }
        if (e->tex)  { wgpuTextureRelease(e->tex);      e->tex  = nullptr; }
    }

    ~TransientResourcePool() { for (Entry& e : entries) destroy(&e); }
};

// arena allocator: bumps from both ends of one buffer. `used` grows up from base[0] for
// permanent per-frame nodes (reset once per frame); `scratchUsed` grows down from base[capacity]
// for compile()-local temporaries, reset per-scope via defer (AlpUtils.h) instead of calloc/free.
// Also owns the two resource pools (below): they share the allocator's lifetime, so when it goes they go.
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
    // High-water mark of scratchUsed within the frame. scratch is rewound to 0 per scope, so by the
    // time the debug UI reads the arena the live value is back to 0 -- this keeps the peak for it.
    size_t scratchHighWater{};

    // resource pools folded in. reset() below only rewinds the bump cursors -- it leaves these be,
    // because they cache GPU textures across frames on purpose (history ping-pong + transient reuse).
    PersistentResourcePool pool;        // name-keyed temporal/history textures
    TransientResourcePool  transient;   // descriptor-keyed per-frame texture cache

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

    // Raw aligned allocation from the TOP of the buffer, growing scratchUsed down. Short-lived,
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
        if (scratchUsed > scratchHighWater) scratchHighWater = scratchUsed;
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
        scratchHighWater = 0;
    }

    void reset_scratch()
    {
        scratchUsed = 0;
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

    // temporal/history resource backed by the PersistentResourcePool: survives the per-frame teardown
    // and is rotated each frame. persistent == one layer of such a resource.
    bool     persistent{};
    uint32_t temporalIndex{};    // 0 == current frame (write target); 1 == previous frame (read)

    // storage not owned by the per-frame graph (imported or pool-backed) -> realize()/release_resources()
    // skip it and lifetime aliasing excludes it.
    bool is_external() const { return imported || persistent; }

    // texture fields
    WGPUTextureDimension dimension = WGPUTextureDimension_Undefined;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;

    SizeKind sizeKind = SizeKind::Absolute;
    float scaleX = 1.0f, scaleY = 1.0f;
    ResourceHandle relativeToHandle{};
    WGPUExtent3D absolute = WGPU_EXTENT_3D_INIT;
    uint32_t mipLevelCount = 1;   // > 1 = mip chain; created once, per-mip views built at bind/attach time
    uint32_t sampleCount = 1;     // > 1 = MSAA (multisampled attachment)

    // buffer fields
    uint64_t bufferSize{};

    // realized / registered GPU handles
    WGPUTexture      texture{};                       // created: the texture object backing `view`
    WGPUTextureView  view{};                         // imported: the registered swapchain view
    WGPUBuffer       buffer{};                        // imported: the registered buffer
    WGPUExtent3D     resolved = WGPU_EXTENT_3D_INIT;  // imported: registered size (base for future Relative resolution)
    WGPUTextureUsage texUsage{};                      // accumulated in compile() from the access list
    WGPUBufferUsage  bufUsage{};                      //   "       (WebGPU needs these at create time)

    // first/last surviving pass (execution-order index) to touch this, filled in compile() phase 3.
    // kNoPass = no live pass touched it (dead transient) or imported -- imported is left out: the graph
    // doesn't own its memory, so an aliasing lifetime would be meaningless.
    static constexpr uint32_t kNoPass = ~0u;
    uint32_t firstUse = kNoPass;
    uint32_t lastUse  = kNoPass;

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
    bool sink{}; // mark a pass as a sink

    PassNode* next{}; // ptr to the next pass node of the render graph
};

struct NodeAdjacency
{
    PassNode* pass{};
    NodeAdjacency* next{};
};

// RenderGraph carries no data members; its state lives here, bump-allocated immediately after the
// RenderGraph object (see create_render_graph) and recovered via a fixed offset from `this`. Keeps
// the header a pure method-only interface.
struct RenderGraphStorage
{
    GraphAllocator*         m_allocator{};   // owns the pools; reach them via m_allocator->pool / ->transient
    ResourceNode*           m_resouces{};
    PassNode*               m_passes{};
    WGPUDevice              m_device{};
    uint32_t                next_id = 1; // 0 = invalid handle
    ResourceNode**          byId{};      // id->node (sized next_id); built in compile() phase 3 -> O(1) find_node
};

static RenderGraphStorage* storage(RenderGraph* rg)
{
    return reinterpret_cast<RenderGraphStorage*>(
        reinterpret_cast<uint8_t*>(rg)
        + GraphAllocator::align_up(sizeof(RenderGraph), alignof(RenderGraphStorage)));
}

// resolve a handle to its node. O(1) through the byId table compile() builds (persisted on storage for
// the frame); before that table exists (declaration time) or for the invalid id 0, fall back to a linear
// walk. an unknown handle resolves to null.
static ResourceNode* find_node(RenderGraph* rg, ResourceHandle h)
{
    RenderGraphStorage& s = *storage(rg);
    if (s.byId && h.id && h.id < s.next_id) return s.byId[h.id];   // fast path: compiled graph
    for (ResourceNode* r = s.m_resouces; r; r = r->next)           // pre-compile or id 0: walk
        if (r->handle.id == h.id) return r;
    return nullptr;
}

GraphAllocator* create_allocator(size_t arenaSize){
    GraphAllocator* allocator = new GraphAllocator;
    size_t capacity = arenaSize;
    allocator->base = (uint8_t*)malloc(capacity);
    allocator->capacity = capacity;
    return allocator;
}

RenderGraph* create_render_graph(GraphAllocator* allocator)
{
    allocator->reset();
    RenderGraph* rg = allocator->make<RenderGraph>();
    RenderGraphStorage* st = allocator->make<RenderGraphStorage>();
    assert(st == storage(rg) && "storage must sit immediately after the RenderGraph");
    st->m_allocator = allocator;
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
// absent: read-only depth is `attachment-read`, a read; adding it here would reintroduce the false
// write hazard this distinction exists to remove.
static bool access_is_write(AccessType t)
{
    return t == AccessType::ColorAttachment
        || t == AccessType::DepthStencilAttachment
        || t == AccessType::ResolveAttachment   // the resolve writes its single-sample target
        || t == AccessType::StorageWrite
        || t == AccessType::CopyDst;
}

#if RG_VALIDATE
// do two accesses to the SAME resource in ONE pass (one usage scope) conflict? read+read never does.
// disjoint subresources never do either: WebGPU usage scopes are per-(mip,layer), so sampling mip i while
// rendering into mip j!=i is legal (a mip-chain downsample/upsample pass). the lone same-subresource read+write
// exception is StorageRead+StorageWrite: that is how the graph spells a read-modify-write storage binding
// (var<storage, read_write>). One writable-storage usage, not an alias (the "multi-writer chain" test +
// the sweep's WAR self-guard depend on it). Any other same-subresource pairing involving a write is
// illegal: a read-only binding aliasing a write (e.g. Sampled+StorageWrite, the named case), or two writes
// the graph can't order within an atomic pass ("multiple unsynchronized writes").
// ponytail: each access is one (mip,layer) point, so a wide sampled range overlapping a written mip slips
// past here -- WebGPU validation is the backstop.
static bool in_pass_accesses_conflict(AccessType a, uint32_t aMip, uint32_t aLayer,
                                      AccessType b, uint32_t bMip, uint32_t bLayer)
{
    if (aMip != bMip || aLayer != bLayer) return false;
    if (!access_is_write(a) && !access_is_write(b)) return false;
    if ((a == AccessType::StorageRead  && b == AccessType::StorageWrite) ||
        (a == AccessType::StorageWrite && b == AccessType::StorageRead)) return false;
    return true;
}
#endif


ResourceHandle RenderGraph::create_image(WGPUStringView name, const TextureDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Texture;
    resouce->dimension = desc.dimension;
    resouce->format = desc.format;
    resouce->sizeKind = desc.sizeKind;
    resouce->scaleX = desc.scaleX;
    resouce->scaleY = desc.scaleY;
    resouce->relativeToHandle = desc.relativeTo;
    resouce->absolute = desc.absolute;
    resouce->mipLevelCount = desc.mipLevelCount ? desc.mipLevelCount : 1;
    resouce->sampleCount = desc.sampleCount ? desc.sampleCount : 1;

    list_append(&s.m_resouces, resouce);

    return resouce->handle;
}


ResourceHandle RenderGraph::create_buffer(WGPUStringView name, const BufferDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Buffer;

    resouce->bufferSize = desc.size;

    list_append(&s.m_resouces, resouce);

    return resouce->handle;
}


// imported resources are managed outside the graph (swapchain, etc). they carry no desc;
// the graph only needs the `imported` flag so passes that write them count as sinks (compile()).
ResourceHandle RenderGraph::importe_image(WGPUStringView name, WGPUTextureView view, WGPUExtent3D size)
{
    RenderGraphStorage& s = *storage(this);
    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Texture;
    resouce->imported = true;
    resouce->view = view;
    resouce->resolved = size;

    list_append(&s.m_resouces, resouce);

    return resouce->handle;
}


ResourceHandle RenderGraph::import_buffer(WGPUStringView name, WGPUBuffer buffer)
{
    RenderGraphStorage& s = *storage(this);
    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();

    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Buffer;
    resouce->imported = true;
    resouce->buffer = buffer;

    list_append(&s.m_resouces, resouce);

    return resouce->handle;
}


// temporal/history resource: two rotating physical textures owned by the PersistentResourcePool. allocates
// two ResourceNodes (curr = layer 0, prev = layer 1); the pool backs them and swaps which physical texture
// each maps to every frame (see realize()), so this frame's curr is next frame's prev.
TemporalImage RenderGraph::create_temporal_image(WGPUStringView name, const TextureDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    s.m_allocator->pool.touch(name);   // ensure the pool entry exists + advance its rotation

    TemporalImage out{};
    for (uint32_t i = 0; i < PersistentResourcePool::kLayers; ++i) {
        ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
        resouce->handle = { s.next_id++ };
        resouce->name = s.m_allocator->copy_string(name);
        resouce->kind = ResourceNode::Kind::Texture;
        resouce->persistent = true;
        resouce->temporalIndex = i;
        resouce->dimension = desc.dimension;
        resouce->format = desc.format;
        resouce->sizeKind = desc.sizeKind;
        resouce->scaleX = desc.scaleX;
        resouce->scaleY = desc.scaleY;
        resouce->relativeToHandle = desc.relativeTo;
        resouce->absolute = desc.absolute;
        resouce->mipLevelCount = desc.mipLevelCount ? desc.mipLevelCount : 1;
        resouce->sampleCount = desc.sampleCount ? desc.sampleCount : 1;
        // ponytail: a multisampled history layer is creatable but prev can only be an attachment/resolve
        // target -- sampling an MSAA texture is illegal (Dawn rejects it) and reading prev is the whole
        // point of history. left to Dawn rather than forbidden here.
        list_append(&s.m_resouces, resouce);
        if (i == 0) out.curr = resouce->handle; else out.prev = resouce->handle;
    }
    return out;
}


GraphBuilder RenderGraph::begin_pass(WGPUStringView name, PassKind kind)
{
    RenderGraphStorage& s = *storage(this);
    PassNode* pass = s.m_allocator->make<PassNode>();
    pass->name = s.m_allocator->copy_string(name);
    pass->kind = kind;

    GraphBuilder builder;
    builder.m_new_pass = pass;
    return builder;
}

void RenderGraph::end_pass(GraphBuilder& builder)
{
    list_append(&storage(this)->m_passes, builder.m_new_pass);
}

void* RenderGraph::alloc_exec(size_t size, size_t align)
{
    return storage(this)->m_allocator->alloc_raw(size, align);
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
// discovered hazard: RAW (read -> producer), WAW (write -> prev writer), WAR (write -> readers of
// the version being clobbered); `dependent` is always later in the walk than `dep`. A read seen before
// any writer of its resource simply binds to "no producer" (no edge). Detecting that authoring error
// is left to compile()'s post-cull pass, which sees the final schedule; the sweep stays edge-only.
template<typename OnEdge>
static void sweep_resource_versions(GraphAllocator* alloc, PassNode* head, uint32_t next_id, OnEdge&& onEdge)
{
    // per resource id (1..next_id-1): the pass holding the current version, and the readers of that
    // version not yet retired by a newer write.
    PassNode** currentProducer = alloc->scratch_alloc<PassNode*>(next_id);
    NodeAdjacency** pendingReaders  = alloc->scratch_alloc<NodeAdjacency*>(next_id);
    defer { alloc->reset_scratch(); };

    for (PassNode* p = head; p; p = p->next)
        for (uint32_t i = 0; i < p->accessCount; ++i) {
            uint32_t id = p->accesses[i].handle.id;
            if (!id) continue;   // invalid/default handle: nothing to version (post-cull check skips id 0 too)
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
                // writer binds to "no producer" (no edge); compile()'s post-cull pass flags it.
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
// emits < N nodes), this does not; we assume an author-acyclic graph and would need an `onstack`
// flag to catch back-edges (also noted in phase 2).
static void topo_visit(PassNode* p, PassNode** order, uint32_t& count)
{
    if (p->placed) return;
    p->placed = true;
    for (NodeAdjacency* a = p->adjacency; a; a = a->next)
        topo_visit(a->pass, order, count);
    order[count++] = p;          // all deps already placed
}

// a sink/output pass writes at least one resource whose value leaves the frame: an imported resource
// (swapchain, etc) or a temporal/persistent one (this frame's history layer, read next frame). these are
// the only roots that keep a pass alive; anything not reachable from one is dead.
static bool is_sink(PassNode* p, const bool* external)
{
    for (uint32_t i = 0; i < p->accessCount; ++i)
        if (access_is_write(p->accesses[i].type) && external[p->accesses[i].handle.id])
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
    // scale only width/height; layer count (cube/array) is the node's own, not the base's.
    uint32_t layers = r->absolute.depthOrArrayLayers ? r->absolute.depthOrArrayLayers : 1;
    return r->resolved = { (uint32_t)(b.width * r->scaleX), (uint32_t)(b.height * r->scaleY), layers };
}

bool RenderGraph::compile()
{
    RenderGraphStorage& s = *storage(this);

    // phase 1: build adjacency (pass dependency DAG, "depends-on" direction). The versioning sweep
    // (see sweep_resource_versions) discovers every RAW/WAW/WAR hazard in declaration order; here all
    // three collapse to add_dependency; its dedup folds multiple hazards between one pass pair into
    // the single ordering edge phase 2 needs, so the resource id and hazard kind are ignored. Reads
    // before any writer get no edge here; the post-cull pass below turns them into errors.
    sweep_resource_versions(s.m_allocator, s.m_passes, s.next_id,
        [&](PassNode* dependent, PassNode* dep, uint32_t /*id*/, HazardKind /*kind*/) {
            add_dependency(s.m_allocator, dependent, dep);
        });

    // phase 2: dead-node removal + topo sort, fused into one DFS seeded from sinks.
    {
        // sinks = passes writing an imported resource. accesses store only handle.id, so flatten
        // the imported flags into an id-indexed table first (same scratch_alloc-over-next_id trick
        // as phase 1's currentProducer).
        // external = imported OR persistent (temporal). both are output sinks when written (their value
        // leaves the frame) and exempt from the read-before-write check (value comes from outside the frame).
        bool* external = s.m_allocator->scratch_alloc<bool>(s.next_id);
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            external[r->handle.id] = r->is_external();

        // topo into a transient array, then relink the intrusive list into execution order. The
        // result lives in m_passes itself; the array is just DFS scratch, reclaimed by the
        // deferred reset_scratch() below.
        uint32_t N = 0;
        for (PassNode* p = s.m_passes; p; p = p->next) ++N;

        PassNode** order = s.m_allocator->scratch_alloc<PassNode*>(N);
        defer { s.m_allocator->reset_scratch(); };
        uint32_t count = 0;
        for (PassNode* p = s.m_passes; p; p = p->next)
            if (is_sink(p, external))
            {
                p->sink = true;
                topo_visit(p, order, count);          // only reaches passes that feed a sink
            }

        // relink next-pointers to follow topo order; m_passes is now == execution order, and any
        // pass not reachable from a sink was never emitted -> dead, dropped here for free.
        for (uint32_t i = 0; i + 1 < count; ++i) order[i]->next = order[i + 1];
        if (count) order[count - 1]->next = nullptr;
        s.m_passes = count ? order[0] : nullptr;

        // ponytail: transient array as DFS scratch; can't sort a one-field intrusive list in
        // place (a dep emitted before the driver reaches it clobbers its `next`, dropping
        // disconnected nodes). recursive DFS, no cycle detection: graph is author-acyclic; add an
        // `onstack` flag (+2 lines) only if a cyclic graph ever needs catching.
    }

#if RG_VALIDATE
    // post-cull validation (development aid; compiled out when RG_VALIDATE==0, e.g. release/NDEBUG,
    // exactly like assert, so a shipping build pays none of this per-frame walk). Over the FINAL schedule
    // (m_passes is now culled + in execution order), a read of a TRANSIENT resource that no earlier pass
    // has produced is an authoring error: the reader would sample uninitialized contents (its writer was
    // declared after it, or culled). walking the surviving passes makes this culling-correct and catches
    // every surviving reader, not just the first.
    //   imported + temporal/persistent resources  -> exempt (their value comes from outside this frame).
    //   resources with no writer at all (e.g. a host-uploaded uniform) -> exempt (hasWriter stays false).
    // bail before phase 3 so the caller never realize()/execute()s a misordered graph.
    {
        bool* hasWriter = s.m_allocator->scratch_alloc<bool>(s.next_id);   // some surviving pass writes id
        bool* produced  = s.m_allocator->scratch_alloc<bool>(s.next_id);   // ...has written it so far, in order
        bool* external  = s.m_allocator->scratch_alloc<bool>(s.next_id);   // imported OR temporal: value from outside the frame
        bool* prevLayer = s.m_allocator->scratch_alloc<bool>(s.next_id);   // temporal layer k>0: read-only this frame
        defer { s.m_allocator->reset_scratch(); };
        for (ResourceNode* r = s.m_resouces; r; r = r->next) {
            external[r->handle.id]  = r->is_external();
            prevLayer[r->handle.id] = r->persistent && r->temporalIndex != 0;
        }
        for (PassNode* p = s.m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i)
                if (access_is_write(p->accesses[i].type)) hasWriter[p->accesses[i].handle.id] = true;

        bool hadError = false;
        for (PassNode* p = s.m_passes; p; p = p->next)
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                uint32_t id = p->accesses[i].handle.id;
                if (access_is_write(p->accesses[i].type)) {
                    produced[id] = true;
                    // older layers of a temporal resource are read-only this frame: writing layer k>0 clobbers
                    // a slot that becomes a future "current". layer 0 (create_temporal_image's handle) is the
                    // only legal write target.
                    if (prevLayer[id]) {
                        ResourceNode* w = find_node(this, p->accesses[i].handle);
                        WGPUStringView wn = w ? w->name : WGPUStringView{};
                        std::printf("[RenderGraph] error: pass \"%.*s\" writes temporal resource \"%.*s\" "
                                    "layer %u -- only layer 0 (create_temporal_image's handle) is writable.\n",
                                    (int)p->name.length, p->name.data ? p->name.data : "",
                                    (int)wn.length, wn.data ? wn.data : "", w ? w->temporalIndex : 0u);
                        hadError = true;
                    }
                    continue;
                }
                if (id == 0 || external[id] || produced[id] || !hasWriter[id]) continue;
                ResourceNode* r = find_node(this, p->accesses[i].handle);
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
        // id->node table: O(1) handle resolution. front-arena (lives until the next create_render_graph
        // reset) and persisted on storage so find_node / PassContext reuse it through realize + execute.
        s.byId = s.m_allocator->alloc<ResourceNode*>(s.next_id);
        ResourceNode** byId = s.byId;   // local alias for the loop + resolve_size below
        for (ResourceNode* r = s.m_resouces; r; r = r->next) byId[r->handle.id] = r;

        uint32_t passIdx = 0;
        for (PassNode* p = s.m_passes; p; p = p->next, ++passIdx)  // m_passes == surviving (post-cull) passes
            for (uint32_t i = 0; i < p->accessCount; ++i) {
                ResourceNode* r = byId[p->accesses[i].handle.id];
                if (!r) continue;
                // lifetime: the walk is already in execution order, so the first touch is firstUse and
                // each later touch overwrites lastUse. imported resources are skipped -- excluded from
                // aliasing, so a span would be meaningless.
                if (!r->is_external()) {   // persistent is pool-owned -> excluded from aliasing, like imported
                    if (r->firstUse == ResourceNode::kNoPass) r->firstUse = passIdx;
                    r->lastUse = passIdx;
                }
                switch (p->accesses[i].type) {
                  case AccessType::ColorAttachment:
                  case AccessType::DepthStencilAttachment:
                  case AccessType::DepthStencilReadOnly:
                  case AccessType::ResolveAttachment:      r->texUsage |= WGPUTextureUsage_RenderAttachment; break;
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
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
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

// debug-only: name of the pass at execution-order position idx (empty past the end). O(n) per call.
static WGPUStringView pass_name_at(PassNode* head, uint32_t idx)
{
    for (PassNode* p = head; p; p = p->next) {
        if (idx == 0) return p->name;
        --idx;
    }
    return WGPUStringView{};
}

// dump the graph as a Mermaid flowchart on stdout. passes are nodes; an edge dep -->|res| Q means Q
// depends on dep via resource res (data/order flow points dep -> Q). edges come from the SAME
// versioning sweep compile() uses, so the dump matches the real graph: RAW (unlabelled), plus WAW
// and WAR tagged in the edge label, rather than an approximation. safe before or after compile():
// the topo sort preserves the relative order of any two passes touching the same resource, so the
// rediscovered edges are unchanged. resources with a single touch (imported inputs, unread sinks)
// produce no pass->pass edge and don't appear.
void debug_print_mermaid(RenderGraph* rg)
{
    RenderGraphStorage& s = *storage(rg);
    std::printf("flowchart LR\n");

    // node decl: stable id Pi -> pass name, indexed by list position.
    uint32_t idx = 0;
    for (PassNode* p = s.m_passes; p; p = p->next, ++idx)
        std::printf("  P%u[\"%.*s\"]\n", idx, (int)p->name.length, p->name.data ? p->name.data : "");

    // one edge per discovered hazard, labelled with the resource name and (for WAW/WAR) the kind.
    sweep_resource_versions(s.m_allocator, s.m_passes, s.next_id,
        [&](PassNode* dependent, PassNode* dep, uint32_t id, HazardKind kind) {
            ResourceNode* r = find_node(rg, { id });
            WGPUStringView nm = r ? r->name : WGPUStringView{};
            const char* tag = kind == HazardKind::WAW ? " (WAW)" : kind == HazardKind::WAR ? " (WAR)" : "";
            std::printf("  P%u -->|\"%.*s%s\"| P%u\n",
                        pass_index(s.m_passes, dep), (int)nm.length, nm.data ? nm.data : "", tag,
                        pass_index(s.m_passes, dependent));
        });

    std::fflush(stdout);
    // ponytail: pass_index is O(n) so the edge loop is O(n*edges); fine for a debug dump of a
    // handful of passes. names assumed pipe/quote-free (they're identifiers) -> no escaping.
}

// dump resource lifetimes as a Mermaid Gantt on stdout. the timeline runs over passes in execution
// order: a "passes" row names each slot (one bar per pass), and below it each transient resource is a
// bar from its first to its last pass. overlapping bars can't share memory; gaps between them are
// aliasing candidates. imported resources are excluded (the graph doesn't own them); a transient no
// surviving pass touched is reported dead and gets no bar. reads firstUse/lastUse, so call after compile().
void debug_print_lifetimes(RenderGraph* rg)
{
    RenderGraphStorage& s = *storage(rg);

    // text summary first: the chart below has no bar for a dead resource, so name them here. spans are
    // named by the passes that bound them, not by index.
    std::printf("resource lifetimes (by pass):\n");
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external()) continue;   // imported + pool-backed: no per-frame lifetime span
        if (r->firstUse == ResourceNode::kNoPass) {
            std::printf("  %.*s -- unused (dead)\n", (int)r->name.length, r->name.data ? r->name.data : "");
            continue;
        }
        WGPUStringView f = pass_name_at(s.m_passes, r->firstUse);
        if (r->firstUse == r->lastUse)
            std::printf("  %.*s -- alive in %.*s\n",
                        (int)r->name.length, r->name.data ? r->name.data : "",
                        (int)f.length, f.data ? f.data : "");
        else {
            WGPUStringView l = pass_name_at(s.m_passes, r->lastUse);
            std::printf("  %.*s -- alive %.*s..%.*s\n",
                        (int)r->name.length, r->name.data ? r->name.data : "",
                        (int)f.length, f.data ? f.data : "",
                        (int)l.length, l.data ? l.data : "");
        }
    }

    // mermaid gantt is time-based, so map pass index i to a one-second slot (dateFormat x reads bar
    // values as unix ms -> i*1000). the "passes" section drops a named one-second bar in each slot so
    // the timeline reads as pass names rather than bare seconds; each resource bar then spans from its
    // first pass to its last (+1 so a single-pass resource still fills its slot).
    std::printf("gantt\n");
    std::printf("  title RenderGraph resource lifetimes (timeline = passes in execution order)\n");
    std::printf("  dateFormat x\n");
    std::printf("  axisFormat %%S\n");

    std::printf("  section passes\n");
    uint32_t idx = 0;
    for (PassNode* p = s.m_passes; p; p = p->next, ++idx)
        std::printf("    %.*s :%u, %u\n",
                    (int)p->name.length, p->name.data ? p->name.data : "", idx * 1000, (idx + 1) * 1000);

    std::printf("  section transient\n");
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->imported || r->firstUse == ResourceNode::kNoPass) continue;
        WGPUStringView f = pass_name_at(s.m_passes, r->firstUse);
        WGPUStringView l = pass_name_at(s.m_passes, r->lastUse);
        std::printf("    %.*s (%.*s..%.*s) :%u, %u\n",
                    (int)r->name.length, r->name.data ? r->name.data : "",
                    (int)f.length, f.data ? f.data : "",
                    (int)l.length, l.data ? l.data : "",
                    r->firstUse * 1000, (r->lastUse + 1) * 1000);
    }

    std::fflush(stdout);
    // ponytail: %S labels the axis 00..59, so it wraps past 60 passes; fine for the handful here. bump
    // the axisFormat if a graph ever exceeds that. names assumed colon/pipe-free -> no escaping.
}

WGPUTextureView PassContext::view(ResourceHandle h) const
{
    return find_node(graph, h)->view;
}

WGPUTexture PassContext::texture(ResourceHandle h) const
{
    return find_node(graph, h)->texture;
}

WGPUBuffer PassContext::buffer(ResourceHandle h) const
{
    return find_node(graph, h)->buffer;
}

// create the GPU resources compile() worked out (size in `resolved`, usage in tex/bufUsage).
// imported resources are caller-owned and skipped; a resource with no accumulated usage was
// untouched by a live pass -> skipped too (the free dead-resource cull compile() phase 3 set up).
void RenderGraph::realize(WGPUDevice device)
{
    RenderGraphStorage& s = *storage(this);
    s.m_device = device;

    // temporal/persistent resources: back each layer with a rotating pool texture instead of a per-frame
    // one. first union the usage over all layers (each physical texture cycles through every layer role
    // across frames), then (re)create on demand, then point each layer node at its rotated slot. size +
    // usage came from compile(); the pool owns lifetime, so release_resources() leaves these alone.
    PersistentResourcePool& pool      = s.m_allocator->pool;
    TransientResourcePool&  transient = s.m_allocator->transient;

    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->persistent || !r->texUsage) continue;
        if (PersistentResourcePool::Entry* e = pool.find(r->name)) e->usage |= r->texUsage;
    }
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->persistent || !r->texUsage) continue;
        PersistentResourcePool::Entry* e = pool.find(r->name);
        if (!e) continue;
        pool.realize_entry(e, device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount);
        uint32_t sl = pool.slot(*e, r->temporalIndex);
        r->texture = e->tex[sl];
        r->view    = e->view[sl];
    }

    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external()) continue;
        if (r->kind == ResourceNode::Kind::Texture) {
            if (!r->texUsage) continue;
            // reuse a pooled texture matching this descriptor instead of recreating one. the pool
            // owns it -> release_resources() leaves it alone and recycles/evicts at end_frame().
            transient.acquire(device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount, r->texUsage,
                              r->texture, r->view);
        } else {
            // buffers stay on the direct path: the lone transient buffer (scene ubo) is tiny and
            // fully host-overwritten each frame, so pooling it saves nothing.
            if (!r->bufUsage) continue;
            WGPUBufferDescriptor d{
                .label = r->name,
                .usage = r->bufUsage,
                .size  = r->bufferSize,
            };
            r->buffer = wgpuDeviceCreateBuffer(device, &d);
        }
    }
}

// record the compiled passes (already in execution order) into a caller-owned encoder: open the
// right pass kind, wire the attachments declared in setup, invoke the stored body against a live
// PassContext. caller owns submit + present.
// ponytail: mirrors RenderPassBuilder/ComputePassBuilder in Renderer.cpp; reimplemented inline
// rather than shared because those live in Renderer.cpp (not a header) and this TU is standalone.
void RenderGraph::execute(WGPUCommandEncoder encoder, WGPUQueue queue)
{
    for (PassNode* p = storage(this)->m_passes; p; p = p->next) {
        PassContext ctx{};
        ctx.encoder = encoder;
        ctx.graph = this;
        ctx.queue = queue;

        if (p->kind == PassKind::Compute && p->exec_fn) {
            WGPUComputePassDescriptor cd{ .label = p->name };
            ctx.compute = wgpuCommandEncoderBeginComputePass(encoder, &cd);
            p->exec_fn(p->exec_obj, ctx);
            wgpuComputePassEncoderEnd(ctx.compute);
            wgpuComputePassEncoderRelease(ctx.compute);
        }
        else if (p->kind == PassKind::Graphics && p->exec_fn) {
            // gather declared attachments from the access list -> WebGPU render pass descriptor
            WGPURenderPassColorAttachment color[8]{};
            uint32_t nc = 0;
            uint32_t lastColorSlot = ~0u;   // slot of the most recent color() -> a following resolve() patches its resolveTarget
            WGPURenderPassDepthStencilAttachment depth{};
            bool hasDepth = false;

            // an attachment view is exactly one subresource. a graph-created texture may be a mip chain or
            // array (its default full view spans many subresources -> illegal here), so build the single
            // (baseMip, baseLayer) slice the access named and release it after the pass. imported textures
            // (r->texture null, e.g. swapchain) keep their caller-owned registered view.
            // ponytail: rebuilt per pass per frame; cache on the node keyed by subresource if it ever shows
            // up in a profile.
            // attach_view runs at most once per access; sized to the per-pass access ceiling so a
            // malformed pass (e.g. >1 depth, or 8 color + extra depth) can't overrun before Dawn validates.
            WGPUTextureView made[PassNode::kMaxAccess]{};
            uint32_t nmade = 0;
            auto attach_view = [&](ResourceNode* r, const ResourceAccess& a) -> WGPUTextureView {
                if (!r->texture) return r->view;
                WGPUTextureViewDescriptor vd{
                    .format          = r->format,
                    .dimension       = WGPUTextureViewDimension_2D,
                    .baseMipLevel    = a.baseMip,   .mipLevelCount   = 1,
                    .baseArrayLayer  = a.baseLayer, .arrayLayerCount = 1,
                };
                return made[nmade++] = wgpuTextureCreateView(r->texture, &vd);
            };

            for (uint32_t i = 0; i < p->accessCount; ++i) {
                const ResourceAccess& a = p->accesses[i];
                ResourceNode* r = find_node(this, a.handle);
                if (!r) continue;
                if (a.type == AccessType::ColorAttachment && nc < 8) {
                    color[nc] = WGPURenderPassColorAttachment{
                        .view          = attach_view(r, a),
                        .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
                        .resolveTarget = nullptr,   // patched below if a resolve() in this pass targets this slot
                        .loadOp        = a.loadOp,
                        .storeOp       = a.storeOp,
                        .clearValue    = a.clearColor,
                    };
                    lastColorSlot = nc++;
                } else if (a.type == AccessType::ResolveAttachment) {
                    // resolve target for the most recent color() (positional pairing). attach_view builds the
                    // right single-sample view -- sample count is a texture property, not in the view
                    // descriptor -- or returns the caller-owned view for an imported target (e.g. swapchain).
                    if (lastColorSlot != ~0u) color[lastColorSlot].resolveTarget = attach_view(r, a);
                } else if (a.type == AccessType::DepthStencilAttachment || a.type == AccessType::DepthStencilReadOnly) {
                    depth = WGPURenderPassDepthStencilAttachment{
                        .view              = attach_view(r, a),
                        .depthLoadOp       = a.loadOp,
                        .depthStoreOp      = a.storeOp,
                        .depthClearValue   = a.clearDepth,
                        .depthReadOnly     = a.type == AccessType::DepthStencilReadOnly,
                        // stencil aspect: non-default only when the caller passed stencil ops (depth+stencil
                        // format). depth-only formats keep these Undefined/0, matching the old behavior.
                        // ponytail: depth + stencil share one read-only flag (DepthStencilReadOnly = both).
                        // depth-write + stencil-gate (portals/mirrors/masked regions) needs no flag -- use a
                        // writable depth_stencil() and set the pipeline's stencil ops to Keep. the shared flag
                        // only blocks sampling the stencil aspect as a texture in the same pass you depth-write
                        // (would need depthReadOnly=false + stencilReadOnly=true); add a per-aspect flag then.
                        .stencilLoadOp     = a.stencilLoadOp,
                        .stencilStoreOp    = a.stencilStoreOp,
                        .stencilClearValue = a.stencilClear,
                        .stencilReadOnly   = a.type == AccessType::DepthStencilReadOnly,
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
            p->exec_fn(p->exec_obj, ctx);
            wgpuRenderPassEncoderEnd(ctx.render);
            wgpuRenderPassEncoderRelease(ctx.render);
            for (uint32_t i = 0; i < nmade; ++i) wgpuTextureViewRelease(made[i]);   // attachment views are pass-scoped
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
    RenderGraphStorage& s = *storage(this);
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external()) continue;   // pool owns temporal textures; freed on resize/pool destroy
        // textures are pool-owned -> drop our borrowed refs, don't release. buffers are never pooled
        // (see realize), so always release them here.
        if (r->kind == ResourceNode::Kind::Texture) { r->texture = nullptr; r->view = nullptr; continue; }
        if (r->buffer) { wgpuBufferRelease(r->buffer); r->buffer = nullptr; }
    }
    s.m_allocator->transient.end_frame();   // release every claim + evict idle textures
}

// records one access on the pass currently being built: the one primitive every GraphBuilder
// helper below wraps. load/store/clear are only meaningful for the two attachment AccessTypes;
// every other call site leaves them at their (ignored) defaults.
void GraphBuilder::use(ResourceHandle handle, AccessType type,
                       WGPULoadOp load, WGPUStoreOp store, WGPUColor clear, float clearDepth,
                       uint32_t baseMip, uint32_t baseLayer,
                       WGPULoadOp stencilLoad, WGPUStoreOp stencilStore, uint32_t stencilClear)
{
    if (!handle.id) return;   // invalid handle (id 0): record nothing -- no dependency, no usage bit, no view lookup later
#if RG_VALIDATE
    // immediate (declaration-time) usage check: fires at the exact b.sampled()/b.storage_write() call
    // site, not deferred to compile(). A pass is one WebGPU usage scope; a resource may not be aliased
    // read+write (e.g. sampled + storage_write, the named case) or written more than once (the graph
    // can't order two writes inside an atomic pass). read+read and the StorageRead+StorageWrite RMW pair
    // are fine; see in_pass_accesses_conflict.
    // ponytail: WebGPU itself permits multiple writable-storage uses in one scope; the graph is stricter
    // because it has no way to synchronize two writes inside a pass; relax if a shader ever needs it.
    if (handle.id) {
        const bool w = access_is_write(type);
        for (uint32_t i = 0; i < m_new_pass->accessCount; ++i) {
            if (m_new_pass->accesses[i].handle.id != handle.id) continue;
            if (in_pass_accesses_conflict(type, baseMip, baseLayer,
                                          m_new_pass->accesses[i].type,
                                          m_new_pass->accesses[i].baseMip, m_new_pass->accesses[i].baseLayer)) {
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

    if (m_new_pass->accessCount < PassNode::kMaxAccess)
        m_new_pass->accesses[m_new_pass->accessCount++] =
            { handle, type, load, store, clear, clearDepth, stencilLoad, stencilStore, stencilClear, baseMip, baseLayer };
    // ponytail: silently drops past kMaxAccess; add assert/grow when a real pass hits it
}

void GraphBuilder::color(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, WGPUColor clear, uint32_t baseMip, uint32_t baseLayer)
{
    use(handle, AccessType::ColorAttachment, load, store, clear, {}, baseMip, baseLayer);
}

void GraphBuilder::resolve(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    // single-sample target the preceding color() resolves into; execute() pairs it with the most recent
    // color() by order. load/store/clear are unused (a resolve has no load op of its own).
#if RG_VALIDATE
    // pairing is positional, so a resolve() before any color(), or a second resolve() on a color that
    // already has one, is silently dropped in execute() -- the access never reaches Dawn, so Dawn can't
    // catch it. assert at the bad call instead. scan back to whichever came last: a color (this resolve
    // pairs with it) or another resolve (that color is already resolved).
    bool pairedColor = false, doubleResolve = false;
    for (uint32_t i = m_new_pass->accessCount; i-- > 0; ) {
        AccessType t = m_new_pass->accesses[i].type;
        if (t == AccessType::ColorAttachment)   { pairedColor   = true; break; }
        if (t == AccessType::ResolveAttachment) { doubleResolve = true; break; }
    }
    if (doubleResolve) {
        std::printf("[RenderGraph] error: pass \"%.*s\" calls resolve() twice for one color() -- one "
                    "resolve target per color attachment.\n",
                    (int)m_new_pass->name.length, m_new_pass->name.data ? m_new_pass->name.data : "");
        assert(false && "RenderGraph: resolve() called twice for one color() in a pass");
    } else if (!pairedColor) {
        std::printf("[RenderGraph] error: pass \"%.*s\" calls resolve() with no preceding color() -- a "
                    "resolve target pairs with the color() declared just before it.\n",
                    (int)m_new_pass->name.length, m_new_pass->name.data ? m_new_pass->name.data : "");
        assert(false && "RenderGraph: resolve() with no preceding color() in a pass");
    }
#endif
    use(handle, AccessType::ResolveAttachment, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

void GraphBuilder::depth_stencil(ResourceHandle handle, WGPULoadOp load, WGPUStoreOp store, float clearDepth, uint32_t baseMip, uint32_t baseLayer, WGPULoadOp stencilLoad, WGPUStoreOp stencilStore, uint32_t stencilClear)
{
    use(handle, AccessType::DepthStencilAttachment, load, store, {}, clearDepth, baseMip, baseLayer, stencilLoad, stencilStore, stencilClear);
}

void GraphBuilder::depth_stencil_read_only(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    // load/store/clear default Undefined/{}; required when read-only
    use(handle, AccessType::DepthStencilReadOnly, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

void GraphBuilder::sampled(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    use(handle, AccessType::Sampled, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

void GraphBuilder::storage_read(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    use(handle, AccessType::StorageRead, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
}

void GraphBuilder::storage_write(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    use(handle, AccessType::StorageWrite, WGPULoadOp_Undefined, WGPUStoreOp_Undefined, {}, {}, baseMip, baseLayer);
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
