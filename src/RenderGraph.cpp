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

// content equality of two string views (length-normalized, NUL-safe).
static bool sv_eq(WGPUStringView a, WGPUStringView b)
{
    size_t na = sv_length(a), nb = sv_length(b);
    return na == nb && (na == 0 || std::memcmp(a.data, b.data, na) == 0);
}

// group label of a pass name: the span before the first '.', empty if none. passes are named dotted
// by convention (shadow.cascade, bloom.down) -- that prefix is the logical group the debug tooling
// (markers, DAG, mermaid) brackets. shares the same backing chars, so it's a view, not a copy.
static WGPUStringView group_prefix(WGPUStringView name)
{
    size_t n = sv_length(name);
    for (size_t i = 0; i < n; ++i)
        if (name.data[i] == '.') return WGPUStringView{ .data = name.data, .length = i };
    return WGPUStringView{};
}

// Owns GPU resources that must outlive the per-frame graph teardown -- temporal/history textures AND
// buffers (accumulation, history feedback, GPU-authored particle state). One Entry per logical temporal
// resource, keyed by name content (the
// graph copies names into the per-frame arena, so the pointers don't survive between frames). Each Entry
// holds two physical textures the graph ping-pongs: layer 0 (current) maps to the opposite slot each
// frame, so last frame's "current" is this frame's "previous" for free.
struct PersistentResourcePool
{
    static constexpr uint32_t kLayers = 2;   // ping-pong: current + previous. N>2 deliberately unsupported.
    static constexpr uint64_t kRetain = 4;   // free an entry no pass has declared (touched) for this many frames

    struct Entry
    {
        std::string          name;               // identity across frames (arena names don't persist)
        uint64_t             frame  = 0;          // rotation counter, bumped once per touch (declaration)
        uint64_t             lastTouched = 0;     // pool evictClock at the last touch; stale entries are freed
        uint32_t             layers = kLayers;    // physical instances: kLayers = ping-pong (temporal), 1 = single in-place
        uint64_t             initHash = 0;        // initialize(): settings hash of the content baked in (recorded at execute)
        bool                 baked = false;       // initialize(): true once an init pass baked content in; cleared on (re)create

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

        // buffer arm: mutually exclusive with the texture arm per named entry (create_temporal_image
        // fills the texture fields, create_temporal_buffer fills these), so both can coexist on Entry --
        // only one is ever populated. `created` + destroy() are shared across both arms.
        // ponytail: both arms on one Entry rather than a union/kind tag -- a few unused bytes per
        // temporal resource (there are a handful) isn't worth the tagged-variant ceremony.
        WGPUBuffer           buf[kLayers]     = {};
        uint64_t             bufferSize       = 0;
        WGPUBufferUsage      bufUsage         = {};  // running union across layers/frames
        WGPUBufferUsage      bufUsageAtCreate = {};
    };
    std::vector<Entry> entries;   // ponytail: linear scan + memcmp; fine for the handful of temporal resources
    uint64_t evictClock = 0;      // advanced once per frame (end_frame); entries idle kRetain frames are freed

    Entry* find(WGPUStringView name)
    {
        const size_t len = sv_length(name);   // names may arrive as WGPU_STRLEN (see screenTex)
        for (Entry& e : entries)
            if (e.name.size() == len &&
                (len == 0 || std::memcmp(e.name.data(), name.data, len) == 0))
                return &e;
        return nullptr;
    }

    // declaration-time: ensure the entry exists and advance its rotation. create_temporal_image/_buffer are
    // the only callers and declare each resource once per frame, so one touch == one frame == one rotation.
    // Declaring the same name twice in a frame (or once as an image and once as a buffer) is an authoring
    // error: it double-rotates the slot mapping, and a cross-kind name clash thrashes the shared `created`
    // flag. Unenforced -- ponytail: add a kind tag + assert if a real collision ever happens.
    Entry* touch(WGPUStringView name, uint32_t layers = kLayers)
    {
        if (Entry* e = find(name)) { ++e->frame; e->lastTouched = evictClock; e->layers = layers; return e; }   // next frame: rotate
        entries.emplace_back();
        Entry& e = entries.back();
        e.name.assign(name.data ? name.data : "", sv_length(name));   // WGPU_STRLEN -> measured, not SIZE_MAX
        e.lastTouched = evictClock;
        e.layers = layers;
        return &e;                                             // first frame: no rotation yet
    }

    // physical slot backing logical layer `layerIndex` (0 == current, 1 == previous) this frame.
    // ping-pong (layers == 2) flips each frame; a single-layer entry (layers == 1) always resolves to 0
    // (no rotation -> stable in-place storage), even though `frame` still ticks.
    uint32_t slot(const Entry& e, uint32_t layerIndex) const
    {
        return (uint32_t)((e.frame + layerIndex) % e.layers);
    }

    // (re)create the entry's `layers` textures when missing or the descriptor changed (lazy: needs the
    // device + the usage union, both known only at realize()).
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
        for (uint32_t i = 0; i < e->layers; ++i) {
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

    // buffer twin of realize_entry: (re)create the entry's `layers` buffers when missing or size/usage
    // changed (1 for a single in-place buffer, kLayers for a ping-pong). usage is the union realize()
    // accumulated into e->bufUsage (each physical buffer cycles through every layer role across frames),
    // mirroring how realize_entry reads e->usage.
    void realize_buffer_entry(Entry* e, WGPUDevice device, uint64_t size)
    {
        bool same = e->created && e->bufferSize == size && e->bufUsageAtCreate == e->bufUsage;
        if (same) return;
        destroy(e);
        e->bufferSize = size; e->bufUsageAtCreate = e->bufUsage;
        for (uint32_t i = 0; i < e->layers; ++i) {
            WGPUBufferDescriptor d{
                .label = WGPUStringView{ e->name.c_str(), e->name.size() },
                .usage = e->bufUsage,
                .size  = size,
            };
            e->buf[i] = wgpuDeviceCreateBuffer(device, &d);
        }
        e->created = true;
    }

    void destroy(Entry* e)
    {
        for (uint32_t i = 0; i < kLayers; ++i) {
            if (e->view[i]) { wgpuTextureViewRelease(e->view[i]); e->view[i] = nullptr; }
            if (e->tex[i])  { wgpuTextureRelease(e->tex[i]);      e->tex[i]  = nullptr; }
            if (e->buf[i])  { wgpuBufferRelease(e->buf[i]);       e->buf[i]  = nullptr; }
        }
        e->created = false;
        e->baked   = false;   // content gone -> an initialize() pass must re-bake (re-armed even for hash 0)
    }

    // per-frame teardown: free entries no pass has declared (touched) for kRetain frames, then advance the
    // clock. a temporal resource stops being touched when its demo/feature goes inactive (demo switch,
    // fog/taa toggled off) -- without this the pool holds its physical textures/buffers for the process
    // lifetime. mirrors TransientResourcePool::end_frame's idle eviction; call once per realized frame.
    void end_frame()
    {
        for (size_t i = entries.size(); i-- > 0; )
            if (evictClock - entries[i].lastTouched >= kRetain) {   // lastTouched <= evictClock -> no underflow
                destroy(&entries[i]);
                entries[i] = entries.back();
                entries.pop_back();
            }
        ++evictClock;
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
            // always-on (release too): assert is stripped under NDEBUG, so without this an OOM returns null and
            // the caller (begin_pass, make<T>, store_exec, ...) deref-crashes silently. announce it loudly.
            std::printf("[RenderGraph] error: arena OOM -- need %zu bytes, %zu/%zu used, %zu scratch. "
                        "raise arenaSize in create_allocator.\n", size, used, capacity, scratchUsed);
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
            std::printf("[RenderGraph] error: scratch OOM -- need %zu bytes, scratch %zu/%zu.\n",
                        size, scratchUsed, capacity);
            assert(false && "GraphAllocator scratch OOM");
            return nullptr;
        }

        size_t rawTop = (capacity - scratchUsed - size) & ~(align - 1);   // round start down to align
        size_t newScratchUsed = capacity - rawTop;

        if (newScratchUsed > capacity - used)   // would cross into the live front/permanent region
        {
            std::printf("[RenderGraph] error: scratch OOM -- need %zu bytes, would cross the front region "
                        "(%zu used).\n", size, used);
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

    // aliasing (compile() phase 4, only when enableAlias): the physical slot this transient shares with
    // other disjoint-lifetime resources. kNoSlot = its own object (ineligible, or aliasing off). hasWriter
    // + firstDefines are the eligibility inputs captured during the phase-3 access walk: a transient may
    // take over another's storage only if some pass writes it (else its bytes are host-owned) AND its first
    // touch fully overwrites (else it would read the previous occupant's leftovers).
    static constexpr uint32_t kNoSlot = ~0u;
    uint32_t aliasSlot    = kNoSlot;
    bool     hasWriter    = false;
    bool     firstDefines = false;

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

    // initialize(): if set, this pass (re)bakes the persistent resource `initTarget`. compile() sets
    // skipInit (drops the pass this frame) once the target's pool entry is populated AND was baked with this
    // same initHash; a fresh/evicted target or a changed hash re-arms it. 0 target = ordinary pass.
    ResourceHandle initTarget{};
    uint64_t       initHash{};   // settings digest this bake produces (see GraphBuilder::initialize)
    bool           skipInit{};

    PassNode* next{}; // ptr to the next pass node of the render graph
};

struct NodeAdjacency
{
    PassNode* pass{};
    NodeAdjacency* next{};
};

// one physical GPU object shared by >=1 transient with disjoint lifetimes + identical signature (aliasing,
// compile() phase 4). texture-only: there is no transient-buffer path (every buffer is imported or
// pool-backed), so buffer aliasing is moot. ponytail: add a buffer arm here if transient buffers ever land.
struct PhysicalResource
{
    WGPUTextureDimension dimension{};   // signature: members must match exactly to share this slot
    WGPUTextureFormat    format{};
    WGPUExtent3D         size{};
    WGPUTextureUsage     texUsage{};    // union over members (WebGPU needs every member's bits at create)
    uint32_t             freeFrom{};    // occupant lastUse; reusable iff the next member's firstUse > this (STRICT)
    WGPUTexture          texture{};     // filled by realize() via transient.acquire()
    WGPUTextureView      view{};
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
    // aliasing: physical slots computed by compile() phase 4 (front-arena, this frame). m_slotCount == 0
    // whenever aliasing is off -> realize()/release_resources() fall back to the per-resource path unchanged.
    PhysicalResource*       m_slots{};
    uint32_t                m_slotCount{};
    // execute() scratch: subresource views built for the current pass (attachments + the body's ctx.view()),
    // released after the body. reset per pass; one view per access, so the per-pass access ceiling bounds it.
    WGPUTextureView         viewScratch[PassNode::kMaxAccess]{};
    uint32_t                viewScratchN{};
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

// does this access fully define the resource's contents, so an aliased slot's previous occupant bytes are
// never observed? a cleared color/depth attachment, a storage write, or a copy-dst overwrite. drives the
// aliasing eligibility check (compile() phase 4). NOT a write-load distinction: a LoadOp_Load attachment
// or storage read keeps prior contents and so cannot safely take over another resource's storage.
static bool access_defines(const ResourceAccess& a)
{
    if (a.type == AccessType::ColorAttachment || a.type == AccessType::DepthStencilAttachment)
        return a.loadOp == WGPULoadOp_Clear;
    return a.type == AccessType::StorageWrite || a.type == AccessType::CopyDst;
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
TemporalResource RenderGraph::create_temporal_image(WGPUStringView name, const TextureDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    s.m_allocator->pool.touch(name);   // ensure the pool entry exists + advance its rotation

    TemporalResource out{};
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


// temporal/history BUFFER: the GPU-buffer twin of create_temporal_image. two rotating physical buffers
// owned by the PersistentResourcePool; allocates two ResourceNodes (curr = layer 0, prev = layer 1) and
// the pool swaps which physical buffer each maps to every frame, so this frame's curr is next frame's prev.
TemporalResource RenderGraph::create_temporal_buffer(WGPUStringView name, const BufferDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    s.m_allocator->pool.touch(name);   // ensure the pool entry exists + advance its rotation

    TemporalResource out{};
    for (uint32_t i = 0; i < PersistentResourcePool::kLayers; ++i) {
        ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
        resouce->handle = { s.next_id++ };
        resouce->name = s.m_allocator->copy_string(name);
        resouce->kind = ResourceNode::Kind::Buffer;
        resouce->persistent = true;
        resouce->temporalIndex = i;
        resouce->bufferSize = desc.size;
        list_append(&s.m_resouces, resouce);
        if (i == 0) out.curr = resouce->handle; else out.prev = resouce->handle;
    }
    return out;
}


// persistent (cross-frame) SINGLE GPU buffer: one pool-backed buffer (no ping-pong), survives the per-frame
// teardown and is auto-evicted when no longer declared. the in-place own-slot / atomic-accumulator twin of
// create_temporal_buffer -- read+write it in one pass (var<storage, read_write>) and the graph models that
// as the StorageRead+StorageWrite RMW pair (no self-ordering). one ResourceNode, one physical buffer.
ResourceHandle RenderGraph::create_persistent_buffer(WGPUStringView name, const BufferDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    s.m_allocator->pool.touch(name, 1);   // single layer: slot() always resolves to 0, no rotation

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Buffer;
    resouce->persistent = true;
    resouce->temporalIndex = 0;       // the only layer
    resouce->bufferSize = desc.size;
    list_append(&s.m_resouces, resouce);
    return resouce->handle;
}


// persistent (cross-frame) SINGLE texture: the image twin of create_persistent_buffer -- one pool-backed
// texture (no ping-pong), survives the per-frame teardown, auto-evicted when no longer declared. for a
// precomputed/baked resource written once and sampled every frame (IBL/env map, BRDF LUT). pair it with an
// initialize() pass to fill it on the first frame / after eviction / when its settings hash changes. one ResourceNode, one texture.
ResourceHandle RenderGraph::create_persistent_image(WGPUStringView name, const TextureDesc& desc)
{
    RenderGraphStorage& s = *storage(this);
    s.m_allocator->pool.touch(name, 1);   // single layer: slot() always resolves to 0, no rotation

    ResourceNode* resouce = s.m_allocator->make<ResourceNode>();
    resouce->handle = { s.next_id++ };
    resouce->name = s.m_allocator->copy_string(name);
    resouce->kind = ResourceNode::Kind::Texture;
    resouce->persistent = true;
    resouce->temporalIndex = 0;       // the only layer
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

    for (PassNode* p = head; p; p = p->next) {
        if (p->skipInit) continue;   // initialize() pass already satisfied -> treat as absent (no versions/edges)
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
    // round, don't truncate: at scale 0.5 an odd base dim (1281 -> 640.5) must land on 641, not 640 --
    // truncation dropped a pixel, the half-res scaling bug. +0.5 then cast = round-half-up (sizes are >= 0).
    return r->resolved = { (uint32_t)(b.width * r->scaleX + 0.5f), (uint32_t)(b.height * r->scaleY + 0.5f), layers };
}

bool RenderGraph::compile(bool enableAlias)
{
    RenderGraphStorage& s = *storage(this);

    // phase 0: resolve initialize() passes. such a pass (re)bakes a persistent target and should run only
    // while that target needs it: skip it this frame iff the pool entry exists, was realized on a prior frame
    // (`created`), AND was baked with the same settings hash (`initHash`). a fresh/evicted target or a changed
    // hash leaves skipInit false -> the pass runs. when it runs, execute() records the new hash into the entry.
    // the baked result lives in the pool, so on skipped frames readers bind to it with no in-graph writer
    // (legal: persistent is external). skipped passes are then dropped: phase 1's sweep and phase 2's sink
    // seeding both ignore skipInit, so the pass produces no version (no reader depends on it) and is not a
    // root -> unreachable -> culled.
    for (PassNode* p = s.m_passes; p; p = p->next) {
        p->skipInit = false;
        if (!p->initTarget.id) continue;
        for (ResourceNode* r = s.m_resouces; r; r = r->next)        // byId isn't built until phase 3 -> linear
            if (r->handle.id == p->initTarget.id) {
                assert(r->persistent && "initialize() target must be a persistent resource "
                                        "(create_persistent_image/_buffer); else the pass never caches");
                PersistentResourcePool::Entry* e = s.m_allocator->pool.find(r->name);
                // skip only when the target is realized AND holds content baked with this exact hash.
                // `baked` (cleared by destroy()) re-arms after any (re)create, so a recreated-blank texture
                // is re-baked even for hash 0 -- closes the resize/descriptor-change stale-bake hole.
                p->skipInit = (e && e->created && e->baked && e->initHash == p->initHash);
                break;
            }
    }

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
            if (!p->skipInit && is_sink(p, external))   // a satisfied initialize() pass is not a root (its target is already baked)
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
                    // a slot that becomes a future "current". layer 0 (the `.curr` handle from
                    // create_temporal_image/_buffer) is the only legal write target.
                    if (prevLayer[id]) {
                        ResourceNode* w = find_node(this, p->accesses[i].handle);
                        WGPUStringView wn = w ? w->name : WGPUStringView{};
                        std::printf("[RenderGraph] error: pass \"%.*s\" writes temporal resource \"%.*s\" "
                                    "layer %u -- only layer 0 (the .curr handle) is writable.\n",
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
                    if (r->firstUse == ResourceNode::kNoPass) {
                        r->firstUse = passIdx;
                        r->firstDefines = access_defines(p->accesses[i]);   // aliasing: does the first touch overwrite?
                    }
                    r->lastUse = passIdx;
                    if (access_is_write(p->accesses[i].type)) r->hasWriter = true;
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

    // phase 4: transient memory aliasing (opt-in). pack disjoint-lifetime, same-signature transients onto a
    // shared physical object so peak VRAM tracks max simultaneous overlap, not the sum of all transients.
    // CPU-only -> it belongs here next to phase 3, not realize(). the schedule is a single linear queue, so
    // each transient's liveness is a closed [firstUse,lastUse] interval and greedy left-edge first-fit is
    // optimal (same trick as linear-scan register allocation). off (enableAlias==false) -> no slots, every
    // aliasSlot stays kNoSlot, and realize() takes the unchanged per-resource path.
    if (enableAlias) {
        // eligible transients (see ResourceNode::aliasSlot): texture-only, single mip + sample so the slot's
        // default view fits every member; first touch must fully define (no reading a previous occupant's
        // bytes) and some pass must write it (don't stomp host-uploaded contents). imported/persistent and
        // dead (kNoPass) excluded. collect into scratch, then sort by firstUse for the left-edge sweep.
        ResourceNode** elig = s.m_allocator->scratch_alloc<ResourceNode*>(s.next_id);
        defer { s.m_allocator->reset_scratch(); };
        uint32_t nElig = 0;
        for (ResourceNode* r = s.m_resouces; r; r = r->next)
            if (r->kind == ResourceNode::Kind::Texture && !r->is_external()
                && r->firstUse != ResourceNode::kNoPass
                && r->mipLevelCount == 1 && r->sampleCount == 1
                && r->hasWriter && r->firstDefines)
                elig[nElig++] = r;

        // insertion sort by firstUse asc (nElig is a handful -> the O(n^2) is free; ponytail).
        for (uint32_t i = 1; i < nElig; ++i) {
            ResourceNode* key = elig[i];
            uint32_t j = i;
            while (j && elig[j - 1]->firstUse > key->firstUse) { elig[j] = elig[j - 1]; --j; }
            elig[j] = key;
        }

        // first-fit: reuse the first signature-matching slot whose occupant is already dead (STRICT
        // freeFrom < firstUse: equality means they share a pass -> binding one object as two resources in
        // one usage scope, illegal), else open a new one. slots live in the front arena (read this frame by
        // realize/release). m_slots == null / m_slotCount == 0 when nothing is eligible.
        if (nElig) s.m_slots = s.m_allocator->alloc<PhysicalResource>(nElig);
        for (uint32_t i = 0; i < nElig; ++i) {
            ResourceNode* r = elig[i];
            uint32_t slot = ResourceNode::kNoSlot;
            for (uint32_t k = 0; k < s.m_slotCount; ++k) {
                PhysicalResource& ph = s.m_slots[k];
                if (ph.dimension == r->dimension && ph.format == r->format
                    && ph.size.width == r->resolved.width && ph.size.height == r->resolved.height
                    && ph.size.depthOrArrayLayers == r->resolved.depthOrArrayLayers
                    && ph.freeFrom < r->firstUse) { slot = k; break; }
            }
            if (slot == ResourceNode::kNoSlot) {
                slot = s.m_slotCount++;
                PhysicalResource& ph = s.m_slots[slot];
                ph.dimension = r->dimension; ph.format = r->format; ph.size = r->resolved;
            }
            PhysicalResource& ph = s.m_slots[slot];
            ph.texUsage |= r->texUsage;   // widen to the union: every member is created on this one object
            ph.freeFrom  = r->lastUse;
            r->aliasSlot = slot;
#if RG_VALIDATE
            assert(ph.dimension == r->dimension && ph.format == r->format
                   && ph.size.width == r->resolved.width && ph.size.height == r->resolved.height
                   && "alias slot signature mismatch");
#endif
        }
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

    // bracket each contiguous run of same-prefix passes (shadow.cascade, bloom.*) in a subgraph -- the
    // mermaid analogue of the DAG's group frames and the encoder debug groups. node ids resolve wherever
    // declared, so nodes -> subgraphs -> edges renders correctly.
    uint32_t gi = 0;
    for (PassNode* a = s.m_passes; a; ) {
        WGPUStringView pre = group_prefix(a->name);
        PassNode* b = a->next; uint32_t gj = gi + 1;
        while (b && sv_length(pre) && sv_eq(group_prefix(b->name), pre)) { b = b->next; ++gj; }
        if (sv_length(pre) && gj - gi >= 2) {
            std::printf("  subgraph \"%.*s\"\n", (int)sv_length(pre), pre.data);
            for (uint32_t k = gi; k < gj; ++k) std::printf("    P%u\n", k);
            std::printf("  end\n");
        }
        a = b; gi = gj;
    }

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

// rough bytes-per-texel for the formats this project creates -- used only by the aliasing memory report
// (peak-allocation accounting), never for allocation. no block-compressed formats here, so a linear
// width*height*layers*texel estimate is exact enough; unknown formats fall back to 4.
static uint32_t texel_bytes(WGPUTextureFormat f)
{
    switch (f) {
      case WGPUTextureFormat_R8Unorm: case WGPUTextureFormat_R8Uint: case WGPUTextureFormat_Stencil8: return 1;
      case WGPUTextureFormat_R16Float: case WGPUTextureFormat_RG8Unorm: case WGPUTextureFormat_Depth16Unorm: return 2;
      case WGPUTextureFormat_RGBA8Unorm: case WGPUTextureFormat_BGRA8Unorm: case WGPUTextureFormat_RG16Float:
      case WGPUTextureFormat_R32Float: case WGPUTextureFormat_Depth32Float: case WGPUTextureFormat_Depth24Plus: return 4;
      case WGPUTextureFormat_RGBA16Float: case WGPUTextureFormat_RG32Float: return 8;
      case WGPUTextureFormat_RGBA32Float: return 16;
      default: return 4;
    }
}

// bytes one texture of this size+format would occupy (base mip only -- aliasing members are single-mip).
static uint64_t texture_bytes(WGPUExtent3D size, WGPUTextureFormat format)
{
    uint32_t layers = size.depthOrArrayLayers ? size.depthOrArrayLayers : 1;
    return (uint64_t)size.width * size.height * layers * texel_bytes(format);
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

    // physical slots (only present when compile() ran phase 4 with aliasing on): each slot is one GPU
    // object shared by the transients listed under it. the totals are the peak-allocation win.
    if (s.m_slotCount) {
        uint32_t  logical = 0;
        uint64_t  logicalBytes = 0, physicalBytes = 0;
        std::printf("physical slots (aliasing):\n");
        for (uint32_t i = 0; i < s.m_slotCount; ++i) {
            PhysicalResource& ph = s.m_slots[i];
            uint64_t slotBytes = texture_bytes(ph.size, ph.format);
            physicalBytes += slotBytes;
            std::printf("  slot %u  %ux%u:", i, ph.size.width, ph.size.height);
            for (ResourceNode* r = s.m_resouces; r; r = r->next)
                if (r->aliasSlot == i) {
                    ++logical; logicalBytes += slotBytes;   // a member shares the slot's signature -> same bytes
                    std::printf(" %.*s", (int)r->name.length, r->name.data ? r->name.data : "");
                }
            std::printf("\n");
        }
        std::printf("  %u logical transients -> %u physical objects; aliased VRAM %llu -> %llu KB (saved %llu KB)\n",
                    logical, s.m_slotCount, (unsigned long long)(logicalBytes / 1024),
                    (unsigned long long)(physicalBytes / 1024),
                    (unsigned long long)((logicalBytes - physicalBytes) / 1024));
    }

    std::fflush(stdout);
    // ponytail: %S labels the axis 00..59, so it wraps past 60 passes; fine for the handful here. bump
    // the axisFormat if a graph ever exceeds that. names assumed colon/pipe-free -> no escaping.
}

// a ctx resolver got a handle no live node matches (id 0, an uncompiled handle, or one from another graph).
// always-on (release too): announce it, then the resolver hands back an empty object so the body binds nothing
// rather than dereferencing null. a valid graph never reaches here.
static void rg_resolve_miss(const char* fn, ResourceHandle h)
{
    std::printf("[RenderGraph] error: ctx.%s(): no live resource for handle id %u.\n", fn, h.id);
}

WGPUTextureView PassContext::view(ResourceHandle h) const
{
    ResourceNode* r = find_node(graph, h);
    if (!r) { rg_resolve_miss("view", h); return {}; }   // unknown / default handle: loud, bind nothing
    if (!r->texture) return r->view;     // imported: the caller-registered view (e.g. swapchain)

    // a single-mip 2D texture's full view IS its only subresource, and a 3D volume is sampled whole, so the
    // view already on the node is the right one -- hand it back (no per-call churn, and a 3D volume must not
    // be sliced to one 2D layer). only a mip chain / 2D array needs a per-subresource view.
    uint32_t layers = (r->dimension == WGPUTextureDimension_2D)
                    ? (r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1) : 1;
    if (!pass || (r->mipLevelCount <= 1 && layers <= 1)) return r->view;

    // mip chain / array: hand back the (baseMip, baseLayer) 2D slice the body's READ access declared, pooled
    // with the attachment views for release after the pass (exactly like attach_view in execute()).
    // ponytail: a 3D mip chain would fall through and get a 2D view -- none exist; revisit if one does.
    const ResourceAccess* rd = nullptr;
    for (uint32_t i = 0; i < pass->accessCount; ++i) {
        const ResourceAccess& a = pass->accesses[i];
        if (a.handle.id == h.id && !access_is_write(a.type)) {
            assert(!rd && "ctx.view: two reads of one handle in a pass -- ambiguous subresource; use ctx.texture");
            rd = &a;
        }
    }
    if (!rd) return r->view;   // viewed but not declared as a read in this pass -> the full view
    RenderGraphStorage& s = *storage(graph);
    assert(rd->baseMip < r->mipLevelCount && "ctx.view: baseMip past the texture's mip count");
    assert(rd->baseLayer < layers && "ctx.view: baseLayer past the texture's layer count");
    WGPUTextureViewDescriptor vd{
        .format          = r->format,
        .dimension       = WGPUTextureViewDimension_2D,
        .baseMipLevel    = rd->baseMip,   .mipLevelCount   = 1,
        .baseArrayLayer  = rd->baseLayer, .arrayLayerCount = 1,
    };
    assert(s.viewScratchN < PassNode::kMaxAccess);
    return s.viewScratch[s.viewScratchN++] = wgpuTextureCreateView(r->texture, &vd);
}

WGPUTexture PassContext::texture(ResourceHandle h) const
{
    ResourceNode* r = find_node(graph, h);
    if (!r) { rg_resolve_miss("texture", h); return {}; }
    return r->texture;   // null for an imported texture (sample via ctx.view) -- the body's concern, not a graph bug
}

WGPUBuffer PassContext::buffer(ResourceHandle h) const
{
    ResourceNode* r = find_node(graph, h);
    if (!r) { rg_resolve_miss("buffer", h); return {}; }
    // realize() skips a buffer with no accumulated usage (no live pass wrote it) -> r->buffer stays null. an
    // imported buffer always carries its registered handle, so a null here means a declared-but-unwritten graph
    // buffer (e.g. a temporal whose .curr no pass writes); binding it is a Dawn error -- assert at the resolve
    // so it points at the missing write, not a downstream bind.
    assert(r->buffer && "ctx.buffer(): resource declared but never realized (no live writer)");
    return r->buffer;
}

WGPUExtent3D PassContext::size(ResourceHandle h) const
{
    ResourceNode* r = find_node(graph, h);
    if (!r) { rg_resolve_miss("size", h); return {}; }
    return r->resolved;   // compile() phase 3 resolved every live resource
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
        if (!r->persistent) continue;
        PersistentResourcePool::Entry* e = pool.find(r->name);
        if (!e) continue;
        if (r->kind == ResourceNode::Kind::Texture) e->usage    |= r->texUsage;
        else                                        e->bufUsage |= r->bufUsage;
    }
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (!r->persistent) continue;
        PersistentResourcePool::Entry* e = pool.find(r->name);
        if (!e) continue;
        uint32_t sl = pool.slot(*e, r->temporalIndex);
        if (r->kind == ResourceNode::Kind::Texture) {
            if (!r->texUsage) continue;
            pool.realize_entry(e, device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount);
            r->texture = e->tex[sl];
            r->view    = e->view[sl];
        } else {
            if (!r->bufUsage) continue;
            pool.realize_buffer_entry(e, device, r->bufferSize);
            r->buffer = e->buf[sl];
        }
    }

    // aliasing (compile() phase 4): acquire one pooled texture per physical slot with the union usage, then
    // point every member resource at it. m_slotCount == 0 when aliasing is off -> this is skipped and the
    // per-resource loop below realizes each transient on its own, exactly as before. one acquire per slot
    // (not per member) keeps the pool's claim count right: distinct slots get distinct textures, members share.
    for (uint32_t i = 0; i < s.m_slotCount; ++i) {
        PhysicalResource& ph = s.m_slots[i];
        transient.acquire(device, ph.size, ph.format, ph.dimension, 1, 1, ph.texUsage, ph.texture, ph.view);
    }
    for (ResourceNode* r = s.m_resouces; r; r = r->next)
        if (r->aliasSlot != ResourceNode::kNoSlot) {
            PhysicalResource& ph = s.m_slots[r->aliasSlot];
            r->texture = ph.texture; r->view = ph.view;
        }

    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        // non-external (graph-owned, per-frame) resources are all transient TEXTURES: every buffer is
        // imported or pool-backed (temporal/persistent), so there is no transient-buffer path. reuse a
        // pooled texture matching this descriptor; release_resources() leaves it alone and the pool
        // recycles/evicts at end_frame(). a member already placed on an alias slot (above) skips this.
        if (r->is_external() || r->kind != ResourceNode::Kind::Texture || !r->texUsage
            || r->aliasSlot != ResourceNode::kNoSlot) continue;
        transient.acquire(device, r->resolved, r->format, r->dimension, r->mipLevelCount, r->sampleCount, r->texUsage,
                          r->texture, r->view);
    }
}

// record the compiled passes (already in execution order) into a caller-owned encoder: open the
// right pass kind, wire the attachments declared in setup, invoke the stored body against a live
// PassContext. caller owns submit + present.
// ponytail: mirrors RenderPassBuilder/ComputePassBuilder in Renderer.cpp; reimplemented inline
// rather than shared because those live in Renderer.cpp (not a header) and this TU is standalone.
void RenderGraph::execute(WGPUCommandEncoder encoder, WGPUQueue queue)
{
    RenderGraphStorage& s = *storage(this);
    // bracket each contiguous run of same-prefix passes (shadow.cascade x3, bloom.*) in an encoder
    // debug group, so a RenderDoc/PIX capture shows collapsible regions over the per-pass labels.
    // push/pop happen between passes (no pass open at that point) -> balanced + nested by construction.
    WGPUStringView openGroup{};
    for (PassNode* p = s.m_passes; p; p = p->next) {
        WGPUStringView grp = group_prefix(p->name);
        if (!sv_eq(grp, openGroup)) {
            if (sv_length(openGroup)) wgpuCommandEncoderPopDebugGroup(encoder);
            if (sv_length(grp))       wgpuCommandEncoderPushDebugGroup(encoder, grp);
            openGroup = grp;
        }

        // initialize() pass that survived the cull == it (re)bakes this frame: mark the target's pool entry
        // baked + stamp the settings hash, so next frame's compile() skips the bake until the hash changes
        // (or the entry is recreated, which clears `baked`). recorded here, not in compile(), so a frame that
        // fails compile never claims a bake it didn't run; gated on exec_fn so a body-less pass can't claim
        // one either. skipped (already-baked) init passes were culled -> not in this loop.
        if (p->initTarget.id && p->exec_fn)
            if (ResourceNode* t = find_node(this, p->initTarget))
                if (PersistentResourcePool::Entry* e = s.m_allocator->pool.find(t->name)) {
                    e->initHash = p->initHash;
                    e->baked    = true;
                }

        PassContext ctx{};
        ctx.encoder = encoder;
        ctx.graph = this;
        ctx.queue = queue;
        ctx.pass = p;
        s.viewScratchN = 0;   // per-pass: attachment + body ctx.view() views accumulate here, freed after the body

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
            // built views land in the shared per-pass scratch (s.viewScratch, drained after the body) so the
            // body's own ctx.view() reads pool with these attachments. one view per access -> the per-pass
            // access ceiling bounds it; the assert guards an overrun that construction already prevents.
            auto attach_view = [&](ResourceNode* r, const ResourceAccess& a) -> WGPUTextureView {
                if (!r->texture) {   // imported: the caller-owned view spans the whole resource; a subresource pick is ignored
                    assert(a.baseMip == 0 && a.baseLayer == 0 &&
                           "subresource (baseMip/baseLayer) selection on an imported attachment is ignored");
                    return r->view;
                }
                uint32_t layers = r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1;
                assert(a.baseMip < r->mipLevelCount && "attachment baseMip past the texture's mip count");
                assert(a.baseLayer < layers && "attachment baseLayer past the texture's layer count");
                WGPUTextureViewDescriptor vd{
                    .format          = r->format,
                    .dimension       = WGPUTextureViewDimension_2D,
                    .baseMipLevel    = a.baseMip,   .mipLevelCount   = 1,
                    .baseArrayLayer  = a.baseLayer, .arrayLayerCount = 1,
                };
                assert(s.viewScratchN < PassNode::kMaxAccess);
                return s.viewScratch[s.viewScratchN++] = wgpuTextureCreateView(r->texture, &vd);
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
        }
        else { // Transfer / None: body records straight onto the encoder
            if (p->exec_fn) p->exec_fn(p->exec_obj, ctx);
        }

        // pass-scoped subresource views (attachments built above + any the body built via ctx.view()) -- free
        // them now the pass has ended. runs for every kind, so ctx.view() auto-releases in compute/transfer too.
        for (uint32_t i = 0; i < s.viewScratchN; ++i) wgpuTextureViewRelease(s.viewScratch[i]);
    }
    if (sv_length(openGroup)) wgpuCommandEncoderPopDebugGroup(encoder);   // close the last open group
}

// release graph-created GPU handles (imported ones are caller-owned -> left alone). pairs with
// realize(); call once the frame's commands have been submitted.
void RenderGraph::release_resources()
{
    RenderGraphStorage& s = *storage(this);
    for (ResourceNode* r = s.m_resouces; r; r = r->next) {
        if (r->is_external()) continue;   // pool owns temporal/persistent, caller owns imported -> not ours
        // every non-external resource is a transient texture (no transient-buffer path) -> it's pool-owned;
        // drop our borrowed refs, don't release. the TransientResourcePool recycles/evicts it at end_frame().
        r->texture = nullptr; r->view = nullptr;
    }
    // aliased members just had their borrowed refs cleared; drop the per-slot refs too. the pool owns each
    // physical texture and frees it at end_frame() -- releasing here would double-free. m_slots is arena
    // garbage next frame regardless; this only keeps a stale handle from being read mid-teardown.
    for (uint32_t i = 0; i < s.m_slotCount; ++i) { s.m_slots[i].texture = nullptr; s.m_slots[i].view = nullptr; }
    s.m_allocator->pool.end_frame();        // free temporal resources gone idle (demo switch / feature toggle)
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

    if (m_new_pass->accessCount < PassNode::kMaxAccess) {
        m_new_pass->accesses[m_new_pass->accessCount++] =
            { handle, type, load, store, clear, clearDepth, stencilLoad, stencilStore, stencilClear, baseMip, baseLayer };
    } else {
        // hard structural cap hit (PassNode::kMaxAccess). dropping the access silently mis-renders (a missing
        // attachment/binding), so be loud right here -- always-on (not RG_VALIDATE), at the offending b.*() call.
        std::printf("[RenderGraph] error: pass \"%.*s\" hit kMaxAccess (%u) -- access on resource id %u dropped; "
                    "raise PassNode::kMaxAccess.\n",
                    (int)m_new_pass->name.length, m_new_pass->name.data ? m_new_pass->name.data : "",
                    (unsigned)PassNode::kMaxAccess, handle.id);
        assert(false && "RenderGraph: pass exceeded kMaxAccess");
    }
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

// in-place read-modify-write: the API mirror of WGSL `var<storage, read_write>`. records the
// StorageRead+StorageWrite pair on one handle -- in_pass_accesses_conflict whitelists exactly this
// pairing, and the sweep's self-guards (no WAR/WAW/RAW edge from a pass to itself) keep it acyclic.
// one logical binding, two recorded accesses.
void GraphBuilder::storage_read_write(ResourceHandle handle, uint32_t baseMip, uint32_t baseLayer)
{
    storage_read(handle, baseMip, baseLayer);
    storage_write(handle, baseMip, baseLayer);
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

// gate this pass on a persistent target -- compile() runs it only while the target needs (re)baking: it is
// unrealized, or `hash` differs from the hash last baked in. not an access (records no hazard/usage), just a
// marker; declare the actual write to `target` separately. hash 0 (default) == bake once.
void GraphBuilder::initialize(ResourceHandle target, uint64_t hash)
{
    m_new_pass->initTarget = target;
    m_new_pass->initHash   = hash;
}

} // RG


// standalone smoke-test driver, compiled as part of this TU (sees the internal structs above)
#include "RenderGraph_main.cpp"
