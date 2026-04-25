#include "Renderer.h"
#include "Framework.h"
#include "ClusteredMesh.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <array>
#include <vector>

// ---- Math ------------------------------------------------------------------

using Mat4 = std::array<float, 16>;  // column-major

struct Vec3 { float x, y, z; };

static Vec3  v3_cross(Vec3 a, Vec3 b)  { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static float v3_dot  (Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3  v3_norm (Vec3 a)          { float inv = 1.f/std::sqrt(v3_dot(a,a)); return {a.x*inv, a.y*inv, a.z*inv}; }
static Vec3  v3_add  (Vec3 a, Vec3 b)  { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3  v3_scale(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }

// right-handed look-at, column-major
static Mat4 mat4_look_at(Vec3 eye, Vec3 fwd, Vec3 world_up) {
    Vec3 r = v3_norm(v3_cross(fwd, world_up));
    Vec3 u = v3_cross(r, fwd);
    Mat4 m{};
    m[0]=r.x; m[1]=u.x; m[2]=-fwd.x;
    m[4]=r.y; m[5]=u.y; m[6]=-fwd.y;
    m[8]=r.z; m[9]=u.z; m[10]=-fwd.z;
    m[12]=-v3_dot(r,eye); m[13]=-v3_dot(u,eye); m[14]=v3_dot(fwd,eye); m[15]=1.f;
    return m;
}

// depth [0..1], right-handed, column-major
static Mat4 mat4_perspective(float fov_y, float aspect, float near_z, float far_z) {
    float f = 1.f / std::tan(fov_y * .5f);
    float d = near_z - far_z;
    Mat4 m{};
    m[0]=f/aspect; m[5]=f;
    m[10]=far_z/d; m[11]=-1.f;
    m[14]=(near_z*far_z)/d;
    return m;
}

// ---- Uniform structs -------------------------------------------------------

// Must match the CameraUniforms WGSL struct exactly (160 bytes total)
struct CameraUniforms {
    float view[16];           // 64 bytes
    float proj[16];           // 64 bytes
    float cameraPos[3];       // 12 bytes
    float _pad0       = 0.f;  //  4 bytes
    float clusterThreshold;   //  4 bytes
    float cameraProj;         //  4 bytes (proj[1][1] = 1/tan(fov/2))
    float znear;              //  4 bytes
    float _pad1       = 0.f;  //  4 bytes
};
static_assert(sizeof(CameraUniforms) == 160);

struct DrawIndexedIndirectParams {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  baseVertex;
    uint32_t firstInstance;
};

// ---- WEBGPU helpers --------------------------------------------------------

// Only valid for string literals and const char[] arrays (sizeof gives array size)
#define WEBGPU_STR(str) WGPUStringView{ .data = (str), .length = sizeof(str) - 1 }

static WGPUShaderModule createShaderModule(WGPUDevice device, WGPUStringView code)
{
    WGPUShaderSourceWGSL wgslDesc{
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code  = code,
    };
    WGPUShaderModuleDescriptor desc{
        .nextInChain = &wgslDesc.chain,
    };
    return wgpuDeviceCreateShaderModule(device, &desc);
}

struct RenderPassBuilder
{
    static constexpr uint32_t kMax = 8;
    WGPURenderPassColorAttachment       colorAttachments[kMax]{};
    uint32_t                            colorCount = 0;
    WGPURenderPassDepthStencilAttachment depth{};
    bool                                hasDepth = false;

    RenderPassBuilder& color(WGPUTextureView view,
                             WGPULoadOp  load  = WGPULoadOp_Clear,
                             WGPUStoreOp store = WGPUStoreOp_Store,
                             WGPUColor   clear = {0,0,0,1})
    {
        assert(colorCount < kMax);
        colorAttachments[colorCount++] = WGPURenderPassColorAttachment{
            .view       = view,
            .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
            .loadOp     = load,
            .storeOp    = store,
            .clearValue = clear,
        };
        return *this;
    }

    RenderPassBuilder& depth_stencil(WGPUTextureView view,
                                     float clearDepth = 1.f)
    {
        depth = WGPURenderPassDepthStencilAttachment{
            .view            = view,
            .depthLoadOp     = WGPULoadOp_Clear,
            .depthStoreOp    = WGPUStoreOp_Store,
            .depthClearValue = clearDepth,
        };
        hasDepth = true;
        return *this;
    }

    WGPURenderPassEncoder begin(WGPUCommandEncoder enc) const
    {
        WGPURenderPassDescriptor desc{
            .colorAttachmentCount   = colorCount,
            .colorAttachments       = colorAttachments,
            .depthStencilAttachment = hasDepth ? &depth : nullptr,
        };
        return wgpuCommandEncoderBeginRenderPass(enc, &desc);
    }
};

struct ComputePassBuilder
{
    const char* label = nullptr;
    WGPUComputePassEncoder begin(WGPUCommandEncoder enc) const
    {
        WGPUComputePassDescriptor desc{};
        if (label) desc.label = WGPUStringView{ .data = label, .length = strlen(label) };
        return wgpuCommandEncoderBeginComputePass(enc, &desc);
    }
};

// ---- Static WebGPU state ---------------------------------------------------

// C++ RAII owners
static wgpu::Device  g_device;
static wgpu::Queue   g_queue;
static wgpu::Surface g_surface;

// Raw C handles (valid as long as wgpu:: objects live)
static WGPUDevice  gDevice{};
static WGPUQueue   gQueue{};
static WGPUSurface gSurface{};

static WGPUTextureFormat gSurfaceFormat = WGPUTextureFormat_BGRA8Unorm;
static uint32_t gWidth  = kWidth;
static uint32_t gHeight = kHeight;

// Fallback pipeline (hardcoded triangle + camera uniforms)
static WGPURenderPipeline gPipeline{};
static WGPUBuffer         viewUniforms{};
static WGPUBindGroup      cameraBindGroup{};

// Cluster culling compute
static WGPUComputePipeline gCullPipeline{};
static WGPUBindGroupLayout gCullBGL{};
static WGPUBindGroup       gCullBindGroup{};
static WGPUBuffer          gVisibleClustersBuffer{};
static uint32_t            gCullClusterCount = 0;

// Cluster index-expansion compute
static WGPUComputePipeline gExpandPipeline{};
static WGPUBindGroupLayout gExpandBGL{};
static WGPUBindGroup       gExpandBindGroup{};
static WGPUBuffer          gOutputIndexBuffer{};
static WGPUBuffer          gDrawArgsBuffer{};
static uint32_t            gExpandClusterCount = 0;

// Clustered render
static WGPURenderPipeline  gClusteredPipeline{};
static WGPUBindGroupLayout gClusteredBGL{};
static WGPUBindGroup       gClusteredBG{};

// Depth buffer
static WGPUTexture     gDepthTexture{};
static WGPUTextureView gDepthView{};

// Camera
static Vec3  gCamPos   = {0.f, 0.f, 3.f};
static float gCamYaw   = 0.f;
static float gCamPitch = 0.f;
static auto  gLastTime = std::chrono::steady_clock::now();

// ---- Surface/depth helpers -------------------------------------------------

static void reconfigure_surface(uint32_t w, uint32_t h)
{
    WGPUSurfaceConfiguration cfg{
        .device      = gDevice,
        .format      = gSurfaceFormat,
        .usage       = WGPUTextureUsage_RenderAttachment,
        .width       = w,
        .height      = h,
        .alphaMode   = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(gSurface, &cfg);
}

static void recreate_depth_texture(uint32_t w, uint32_t h)
{
    if (gDepthView)    { wgpuTextureViewRelease(gDepthView);    gDepthView    = nullptr; }
    if (gDepthTexture) { wgpuTextureRelease(gDepthTexture);     gDepthTexture = nullptr; }
    WGPUTextureDescriptor desc{
        .usage         = WGPUTextureUsage_RenderAttachment,
        .dimension     = WGPUTextureDimension_2D,
        .size          = { w, h, 1 },
        .format        = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount   = 1,
    };
    gDepthTexture = wgpuDeviceCreateTexture(gDevice, &desc);
    gDepthView    = wgpuTextureCreateView(gDepthTexture, nullptr);
}

// ---- Per-mesh init ---------------------------------------------------------

static void initClusterCullPass(const ClusteredMeshGPU& mesh)
{
    if (gVisibleClustersBuffer) { wgpuBufferRelease(gVisibleClustersBuffer); gVisibleClustersBuffer = {}; }
    if (gCullBindGroup)         { wgpuBindGroupRelease(gCullBindGroup);      gCullBindGroup         = {}; }

    gCullClusterCount = mesh.clusterCount;
    const uint64_t bufSize = sizeof(uint32_t) + (uint64_t)mesh.clusterCount * sizeof(uint32_t);

    WGPUBufferDescriptor bufDesc{
        .label = WEBGPU_STR("visible clusters"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
        .size  = bufSize,
    };
    gVisibleClustersBuffer = wgpuDeviceCreateBuffer(gDevice, &bufDesc);

    WGPUBindGroupEntry entries[3] = {
        { .binding = 0, .buffer = viewUniforms,           .offset = 0, .size = sizeof(CameraUniforms) },
        { .binding = 1, .buffer = mesh.clusterBuffer,     .offset = 0, .size = (uint64_t)mesh.clusterCount * sizeof(ClusterN) },
        { .binding = 2, .buffer = gVisibleClustersBuffer, .offset = 0, .size = bufSize },
    };
    WGPUBindGroupDescriptor bgDesc{
        .layout     = gCullBGL,
        .entryCount = 3,
        .entries    = entries,
    };
    gCullBindGroup = wgpuDeviceCreateBindGroup(gDevice, &bgDesc);
}

static void initClusterExpandPass(const ClusteredMeshGPU& mesh)
{
    if (gOutputIndexBuffer) { wgpuBufferRelease(gOutputIndexBuffer); gOutputIndexBuffer = {}; }
    if (gDrawArgsBuffer)    { wgpuBufferRelease(gDrawArgsBuffer);    gDrawArgsBuffer    = {}; }
    if (gExpandBindGroup)   { wgpuBindGroupRelease(gExpandBindGroup); gExpandBindGroup  = {}; }

    gExpandClusterCount = mesh.clusterCount;

    const uint32_t totalTriangles = mesh.meshletTriangleByteCount / 3u;
    const uint64_t indexBufSize   = (uint64_t)totalTriangles * 3u * sizeof(uint32_t);

    {
        WGPUBufferDescriptor desc{
            .label = WEBGPU_STR("output index buffer"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
            .size  = indexBufSize,
        };
        gOutputIndexBuffer = wgpuDeviceCreateBuffer(gDevice, &desc);
    }
    {
        WGPUBufferDescriptor desc{
            .label = WEBGPU_STR("draw args"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst,
            .size  = sizeof(DrawIndexedIndirectParams),
        };
        gDrawArgsBuffer = wgpuDeviceCreateBuffer(gDevice, &desc);
        DrawIndexedIndirectParams init{ .instanceCount = 1 };
        wgpuQueueWriteBuffer(gQueue, gDrawArgsBuffer, 0, &init, sizeof(init));
    }

    const uint64_t trisBufSize = ((uint64_t)mesh.meshletTriangleByteCount + 3u) & ~3ull;
    const uint64_t visSize     = sizeof(uint32_t) + (uint64_t)mesh.clusterCount * sizeof(uint32_t);

    WGPUBindGroupEntry entries[5] = {
        { .binding = 0, .buffer = gVisibleClustersBuffer,     .offset = 0, .size = visSize },
        { .binding = 1, .buffer = mesh.clusterBuffer,         .offset = 0, .size = (uint64_t)mesh.clusterCount * sizeof(ClusterN) },
        { .binding = 2, .buffer = mesh.meshletTriangleBuffer, .offset = 0, .size = trisBufSize },
        { .binding = 3, .buffer = gDrawArgsBuffer,            .offset = 0, .size = sizeof(DrawIndexedIndirectParams) },
        { .binding = 4, .buffer = gOutputIndexBuffer,         .offset = 0, .size = indexBufSize },
    };
    WGPUBindGroupDescriptor bgDesc{
        .layout     = gExpandBGL,
        .entryCount = 5,
        .entries    = entries,
    };
    gExpandBindGroup = wgpuDeviceCreateBindGroup(gDevice, &bgDesc);
}

static void initClusteredRenderPipeline(const ClusteredMeshGPU& mesh)
{
    if (gClusteredBG)  { wgpuBindGroupRelease(gClusteredBG);         gClusteredBG  = {}; }
    if (gClusteredPipeline) { wgpuRenderPipelineRelease(gClusteredPipeline); gClusteredPipeline = {}; }
    if (gClusteredBGL) { wgpuBindGroupLayoutRelease(gClusteredBGL);  gClusteredBGL = {}; }

    const char clusteredShaderCode[] = R"(
enable primitive_index;

struct CameraUniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
}

struct ClusterN {
    refined              : i32,
    groupCenterX         : f32,
    groupCenterY         : f32,
    groupCenterZ         : f32,
    groupRadius          : f32,
    groupError           : f32,
    refinedCenterX       : f32,
    refinedCenterY       : f32,
    refinedCenterZ       : f32,
    refinedRadius        : f32,
    refinedError         : f32,
    meshletVertexOffset  : u32,
    meshletTriangleOffset: u32,
    packedCounts         : u32,
}

// must match MeshVertex in C++ (8 x f32 = 32 bytes)
struct MeshVertex {
    x:  f32, y:  f32, z:  f32,
    nx: f32, ny: f32, nz: f32,
    tu: f32, tv: f32,
}

@group(0) @binding(0) var<uniform>        camera        : CameraUniforms;
@group(0) @binding(1) var<storage, read>  clusters      : array<ClusterN>;
@group(0) @binding(2) var<storage, read>  meshlet_verts : array<u32>;
@group(0) @binding(3) var<storage, read>  vertices      : array<MeshVertex>;

struct VertexOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       color : vec3<f32>,
    @location(1)       normal: vec3<f32>,
}

fn cluster_color(id: u32) -> vec3<f32> {
    let h = id * 2246822519u + 2654435761u;
    let r = f32((h       ) & 0xFFu) / 255.0;
    let g = f32((h >>  8u) & 0xFFu) / 255.0;
    let b = f32((h >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

// taken from unreal engine
fn murmurMix(hash: u32) -> u32 {
    var h = hash;

    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);

    return h;
}

// taken from unreal engine
fn intToColor(index: u32) -> vec3<f32> {
    let hash = murmurMix(index);

    let r = f32((hash >> 0u) & 255u);
    let g = f32((hash >> 8u) & 255u);
    let b = f32((hash >> 16u) & 255u);

    return vec3<f32>(r, g, b) * (1.0 / 255.0);
}

@vertex
fn vs_main(@builtin(vertex_index) packed: u32) -> VertexOut {
    let cluster_id  = packed >> 8u;
    let local_idx   = packed & 0xFFu;
    let cluster     = clusters[cluster_id];
    let global_vtx  = meshlet_verts[cluster.meshletVertexOffset + local_idx];
    let v           = vertices[global_vtx];
    var out: VertexOut;
    out.pos    = camera.proj * camera.view * vec4<f32>(v.x, v.y, v.z, 1.0);
    out.color  = intToColor(cluster_id);
    out.normal = vec3<f32>(v.nx, v.ny, v.nz);
    return out;
}



@fragment
fn fs_main(in: VertexOut, @builtin(primitive_index) prim_id: u32) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let diffuse   = max(dot(normalize(in.normal), light_dir), 0.08);
    let color = intToColor(prim_id);
    return vec4<f32>(in.color, 1.0);
}
    )";

    WGPUShaderModule shader = createShaderModule(gDevice, WEBGPU_STR(clusteredShaderCode));
    // shader released after pipeline creation via defer-like cleanup below

    WGPUBindGroupLayoutEntry bglEntries[4] = {
        { .binding = 0, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms) } },
        { .binding = 1, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 2, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 3, .visibility = WGPUShaderStage_Vertex,
          .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
    };
    WGPUBindGroupLayoutDescriptor bglDesc{ .entryCount = 4, .entries = bglEntries };
    gClusteredBGL = wgpuDeviceCreateBindGroupLayout(gDevice, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{ .bindGroupLayoutCount = 1, .bindGroupLayouts = &gClusteredBGL };
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(gDevice, &plDesc);

    WGPUDepthStencilState depthStencil{
        .format            = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = WGPUOptionalBool_True,
        .depthCompare      = WGPUCompareFunction_Less,
    };
    WGPUColorTargetState colorTarget{
        .format    = gSurfaceFormat,
        .writeMask = WGPUColorWriteMask_All,
    };
    WGPUFragmentState fragment{
        .module      = shader,
        .entryPoint  = WEBGPU_STR("fs_main"),
        .targetCount = 1,
        .targets     = &colorTarget,
    };
    WGPURenderPipelineDescriptor pipeDesc{
        .layout   = layout,
        .vertex   = { .module = shader, .entryPoint = WEBGPU_STR("vs_main") },
        .primitive = {
            .topology         = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Undefined,
            .frontFace        = WGPUFrontFace_CCW,
            .cullMode         = WGPUCullMode_Back,
        },
        .depthStencil = &depthStencil,
        .multisample  = { .count = 1, .mask = ~0u },
        .fragment     = &fragment,
    };
    gClusteredPipeline = wgpuDeviceCreateRenderPipeline(gDevice, &pipeDesc);

    wgpuShaderModuleRelease(shader);
    wgpuPipelineLayoutRelease(layout);

    WGPUBindGroupEntry bgEntries[4] = {
        { .binding = 0, .buffer = viewUniforms,             .offset = 0, .size = sizeof(CameraUniforms) },
        { .binding = 1, .buffer = mesh.clusterBuffer,       .offset = 0, .size = (uint64_t)mesh.clusterCount      * sizeof(ClusterN)   },
        { .binding = 2, .buffer = mesh.meshletVertexBuffer, .offset = 0, .size = (uint64_t)mesh.meshletVertexCount * sizeof(uint32_t)   },
        { .binding = 3, .buffer = mesh.vertexBuffer,        .offset = 0, .size = (uint64_t)mesh.vertexCount        * sizeof(MeshVertex) },
    };
    WGPUBindGroupDescriptor bgDesc{ .layout = gClusteredBGL, .entryCount = 4, .entries = bgEntries };
    gClusteredBG = wgpuDeviceCreateBindGroup(gDevice, &bgDesc);
}

static void loadMeshForRendering(const char* path)
{
    static ClusteredMeshGPU sMesh{};
    if (!loadClusteredMeshFromFile(path, gDevice, gQueue, sMesh))
    {
        fprintf(stderr, "loadMeshForRendering: failed to load %s — falling back to triangle\n", path);
        return;
    }
    initClusterCullPass(sMesh);
    initClusterExpandPass(sMesh);
    initClusteredRenderPipeline(sMesh);
    printf("Mesh loaded and clustered render pipeline ready.\n");
}

// ---- Public API ------------------------------------------------------------

void renderer_set_initial_size(uint32_t w, uint32_t h)
{
    gWidth = w; gHeight = h;
}

void renderer_init(wgpu::Device device, wgpu::Queue queue, wgpu::Surface surface)
{
    g_device  = std::move(device);
    g_queue   = std::move(queue);
    g_surface = std::move(surface);
    gDevice   = g_device.Get();
    gQueue    = g_queue.Get();
    gSurface  = g_surface.Get();

    reconfigure_surface(gWidth, gHeight);
    recreate_depth_texture(gWidth, gHeight);

    // Camera uniform buffer
    {
        WGPUBufferDescriptor desc{
            .label = WEBGPU_STR("camera uniforms"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size  = sizeof(CameraUniforms),
        };
        viewUniforms = wgpuDeviceCreateBuffer(gDevice, &desc);
    }

    // ---- Fallback pipeline (hardcoded triangle, CameraUniforms bind group) ----
    {
        const char fallbackShader[] = R"(
enable primitive_index;

struct Uniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> u : Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = vec2f(0.0, 0.0);
    if (vi == 0u) { p = vec2f(-0.5, -0.5); }
    else if (vi == 1u) { p = vec2f(0.5, -0.5); }
    else { p = vec2f(0.0, 0.5); }
    return u.proj * u.view * vec4f(p, 0.0, 1.0);
}

// taken from unreal engine
fn murmurMix(hash: u32) -> u32 {
    var h = hash;

    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);

    return h;
}

// taken from unreal engine
fn intToColor(index: u32) -> vec3<f32> {
    let hash = murmurMix(index);

    let r = f32((hash >> 0u) & 255u);
    let g = f32((hash >> 8u) & 255u);
    let b = f32((hash >> 16u) & 255u);

    return vec3<f32>(r, g, b) * (1.0 / 255.0);
}

@fragment
fn fs_main( @builtin(primitive_index) prim_id: u32) -> @location(0) vec4f {
    let color = intToColor(prim_id);
    return vec4f(color, 1.0);
}
        )";

        WGPUShaderModule sm = createShaderModule(gDevice, WEBGPU_STR(fallbackShader));

        WGPUBindGroupLayoutEntry bgle{
            .binding    = 0,
            .visibility = WGPUShaderStage_Vertex,
            .buffer     = { .type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms) },
        };
        WGPUBindGroupLayoutDescriptor bgld{ .entryCount = 1, .entries = &bgle };
        WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(gDevice, &bgld);

        WGPUBindGroupEntry bge{ .binding = 0, .buffer = viewUniforms, .size = sizeof(CameraUniforms) };
        WGPUBindGroupDescriptor bgd{ .layout = bgl, .entryCount = 1, .entries = &bge };
        cameraBindGroup = wgpuDeviceCreateBindGroup(gDevice, &bgd);

        WGPUPipelineLayoutDescriptor pld{ .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl };
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(gDevice, &pld);

        WGPUDepthStencilState ds{
            .format            = WGPUTextureFormat_Depth24Plus,
            .depthWriteEnabled = WGPUOptionalBool_True,
            .depthCompare      = WGPUCompareFunction_Less,
        };
        WGPUColorTargetState ct{ .format = gSurfaceFormat, .writeMask = WGPUColorWriteMask_All };
        WGPUFragmentState frag{ .module = sm, .entryPoint = WEBGPU_STR("fs_main"), .targetCount = 1, .targets = &ct };
        WGPURenderPipelineDescriptor pd{
            .layout   = pl,
            .vertex   = { .module = sm, .entryPoint = WEBGPU_STR("vs_main") },
            .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
            .depthStencil = &ds,
            .multisample  = { .count = 1, .mask = ~0u },
            .fragment     = &frag,
        };
        gPipeline = wgpuDeviceCreateRenderPipeline(gDevice, &pd);
        wgpuShaderModuleRelease(sm);
        wgpuPipelineLayoutRelease(pl);
        wgpuBindGroupLayoutRelease(bgl);
    }

    // ---- Cluster culling compute pipeline ----------------------------------
    {
        const char cullShader[] = R"(
struct CameraUniforms {
    view             : mat4x4<f32>,
    proj             : mat4x4<f32>,
    cameraPos        : vec3<f32>,
    _pad0            : f32,
    clusterThreshold : f32,
    cameraProj       : f32,
    znear            : f32,
    _pad1            : f32,
}
struct ClusterN {
    refined              : i32,
    groupCenterX         : f32, groupCenterY  : f32, groupCenterZ  : f32,
    groupRadius          : f32, groupError    : f32,
    refinedCenterX       : f32, refinedCenterY: f32, refinedCenterZ: f32,
    refinedRadius        : f32, refinedError  : f32,
    meshletVertexOffset  : u32, meshletTriangleOffset: u32,
    packedCounts         : u32,
}
struct VisibleClusters { count : atomic<u32>, ids : array<u32>, }

@group(0) @binding(0) var<uniform>             camera   : CameraUniforms;
@group(0) @binding(1) var<storage, read>       clusters : array<ClusterN>;
@group(0) @binding(2) var<storage, read_write> visible  : VisibleClusters;

fn bounds_error(center: vec3<f32>, error: f32, radius: f32) -> f32 {
    let d = max(length(center - camera.cameraPos) - radius, camera.znear);
    return error / d * (camera.cameraProj * 0.5);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&clusters)) { return; }
    let c             = clusters[idx];
    let groupCenter   = vec3<f32>(c.groupCenterX,   c.groupCenterY,   c.groupCenterZ);
    let refinedCenter = vec3<f32>(c.refinedCenterX, c.refinedCenterY, c.refinedCenterZ);
    let groupError    = bounds_error(groupCenter,   c.groupError,   c.groupRadius);
    let refinedError  = bounds_error(refinedCenter, c.refinedError, c.refinedRadius);
    let render = groupError > camera.clusterThreshold &&
                (c.refined < 0 || refinedError <= camera.clusterThreshold);
    if (render) {
        let slot = atomicAdd(&visible.count, 1u);
        visible.ids[slot] = idx;
    }
}
        )";

        WGPUShaderModule sm = createShaderModule(gDevice, WEBGPU_STR(cullShader));

        WGPUBindGroupLayoutEntry entries[3] = {
            { .binding = 0, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(CameraUniforms) } },
            { .binding = 1, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
            { .binding = 2, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Storage } },
        };
        WGPUBindGroupLayoutDescriptor bgld{ .entryCount = 3, .entries = entries };
        gCullBGL = wgpuDeviceCreateBindGroupLayout(gDevice, &bgld);

        WGPUPipelineLayoutDescriptor pld{ .bindGroupLayoutCount = 1, .bindGroupLayouts = &gCullBGL };
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(gDevice, &pld);

        WGPUComputePipelineDescriptor cpd{
            .label   = WEBGPU_STR("cluster cull pipeline"),
            .layout  = pl,
            .compute = { .module = sm, .entryPoint = WEBGPU_STR("cs_main") },
        };
        gCullPipeline = wgpuDeviceCreateComputePipeline(gDevice, &cpd);
        wgpuShaderModuleRelease(sm);
        wgpuPipelineLayoutRelease(pl);
    }

    // ---- Cluster index-expansion compute pipeline --------------------------
    //
    // Each workgroup handles one visible cluster (128 threads, one per triangle).
    // Packs (cluster_id << 8 | local_vtx) into the output index buffer.
    {
        const char expandShader[] = R"(
struct ClusterN {
    refined              : i32,
    groupCenterX         : f32, groupCenterY  : f32, groupCenterZ  : f32,
    groupRadius          : f32, groupError    : f32,
    refinedCenterX       : f32, refinedCenterY: f32, refinedCenterZ: f32,
    refinedRadius        : f32, refinedError  : f32,
    meshletVertexOffset  : u32, meshletTriangleOffset: u32,
    packedCounts         : u32,
}
struct VisibleClusters { count : u32, ids : array<u32>, }
struct DrawArgs {
    index_count    : atomic<u32>,
    instance_count : u32,
    first_index    : u32,
    base_vertex    : i32,
    first_instance : u32,
}

@group(0) @binding(0) var<storage, read>       visible      : VisibleClusters;
@group(0) @binding(1) var<storage, read>       clusters     : array<ClusterN>;
@group(0) @binding(2) var<storage, read>       meshlet_tris : array<u32>;
@group(0) @binding(3) var<storage, read_write> draw_args    : DrawArgs;
@group(0) @binding(4) var<storage, read_write> out_indices  : array<u32>;

var<workgroup> ws_base    : u32;
var<workgroup> ws_cluster : ClusterN;

fn read_u8(buf: ptr<storage, array<u32>, read>, byte_offset: u32) -> u32 {
    let word  = (*buf)[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(128)
fn cs_main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    if (wgid.x >= visible.count) { return; }
    let cluster_id = visible.ids[wgid.x];
    if (lid.x == 0u) {
        ws_cluster = clusters[cluster_id];
        let tri_count = (ws_cluster.packedCounts >> 8u) & 0xFFu;
        ws_base    = atomicAdd(&draw_args.index_count, tri_count * 3u);
    }
    workgroupBarrier();
    let tri_idx = lid.x;
    let tri_count = (ws_cluster.packedCounts >> 8u) & 0xFFu;
    if (tri_idx >= tri_count) { return; }
    let tri_byte = ws_cluster.meshletTriangleOffset + tri_idx * 3u;
    let i0 = read_u8(&meshlet_tris, tri_byte + 0u);
    let i1 = read_u8(&meshlet_tris, tri_byte + 1u);
    let i2 = read_u8(&meshlet_tris, tri_byte + 2u);
    let out_base = ws_base + tri_idx * 3u;
    out_indices[out_base + 0u] = (cluster_id << 8u) | i0;
    out_indices[out_base + 1u] = (cluster_id << 8u) | i1;
    out_indices[out_base + 2u] = (cluster_id << 8u) | i2;
}
        )";

        WGPUShaderModule sm = createShaderModule(gDevice, WEBGPU_STR(expandShader));

        WGPUBindGroupLayoutEntry entries[5] = {
            { .binding = 0, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
            { .binding = 1, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
            { .binding = 2, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
            { .binding = 3, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Storage } },
            { .binding = 4, .visibility = WGPUShaderStage_Compute,
              .buffer = { .type = WGPUBufferBindingType_Storage } },
        };
        WGPUBindGroupLayoutDescriptor bgld{ .entryCount = 5, .entries = entries };
        gExpandBGL = wgpuDeviceCreateBindGroupLayout(gDevice, &bgld);

        WGPUPipelineLayoutDescriptor pld{ .bindGroupLayoutCount = 1, .bindGroupLayouts = &gExpandBGL };
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(gDevice, &pld);

        WGPUComputePipelineDescriptor cpd{
            .label   = WEBGPU_STR("cluster expand pipeline"),
            .layout  = pl,
            .compute = { .module = sm, .entryPoint = WEBGPU_STR("cs_main") },
        };
        gExpandPipeline = wgpuDeviceCreateComputePipeline(gDevice, &cpd);
        wgpuShaderModuleRelease(sm);
        wgpuPipelineLayoutRelease(pl);
    }

    // Load mesh (falls back to triangle if file not found)
    loadMeshForRendering("assets/rsc.nanite");
}

void renderer_resize(uint32_t w, uint32_t h)
{
    gWidth = w; gHeight = h;
    reconfigure_surface(w, h);
    recreate_depth_texture(w, h);
}

void renderer_frame()
{
    // Delta time
    auto  now = std::chrono::steady_clock::now();
    float dt  = std::chrono::duration<float>(now - gLastTime).count();
    gLastTime = now;
    dt = std::min(dt, 0.1f);

    // Camera — RMB to look, WASD to move
    if (input_mouse_down(0)) {
        gCamYaw   += input_mouse_delta_x() * 0.002f;
        gCamPitch += input_mouse_delta_y() * 0.002f;
        gCamPitch  = std::clamp(gCamPitch, -1.5f, 1.5f);
    }
    // Thumbstick look (2 rad/s at full deflection)
    float jx = input_look_joystick_x(), jy = input_look_joystick_y();
    if (jx != 0.f || jy != 0.f) {
        gCamYaw   += jx * 0.5f * dt;
        gCamPitch -= jy * 0.5f * dt;
        gCamPitch  = std::clamp(gCamPitch, -1.5f, 1.5f);
    }

    float cp = std::cos(gCamPitch), sp = std::sin(gCamPitch);
    float cy = std::cos(gCamYaw),   sy = std::sin(gCamYaw);
    Vec3 fwd   = {sy*cp, sp, -cy*cp};
    Vec3 right = v3_norm(v3_cross(fwd, {0,1,0}));

    float spd = 50.f * dt;
    // Analog move joystick (mobile) — knob down = backward, so negate my
    float mx = input_move_joystick_x(), my = input_move_joystick_y();
    gCamPos = v3_add(gCamPos, v3_scale(fwd,   spd * (-my)));
    gCamPos = v3_add(gCamPos, v3_scale(right,  spd * mx));
    // Digital WASD (desktop; joystick values are 0 when no touch)
    if (input_key_down(Key::W)) gCamPos = v3_add(gCamPos, v3_scale(fwd,   spd));
    if (input_key_down(Key::S)) gCamPos = v3_add(gCamPos, v3_scale(fwd,  -spd));
    if (input_key_down(Key::A)) gCamPos = v3_add(gCamPos, v3_scale(right, -spd));
    if (input_key_down(Key::D)) gCamPos = v3_add(gCamPos, v3_scale(right,  spd));

    // Upload camera uniforms
    {
        Mat4 view = mat4_look_at(gCamPos, v3_norm(fwd), {0,1,0});
        Mat4 proj = mat4_perspective(1.047f, (float)gWidth / (float)gHeight, 0.01f, 100000.f);
        CameraUniforms cu{};
        std::copy(view.begin(), view.end(), cu.view);
        std::copy(proj.begin(), proj.end(), cu.proj);
        cu.cameraPos[0]      = gCamPos.x;
        cu.cameraPos[1]      = gCamPos.y;
        cu.cameraPos[2]      = gCamPos.z;
        cu.clusterThreshold  = 0.002f;
        cu.cameraProj        = proj[5]; // 1/tan(fov/2)
        cu.znear             = 0.01f;
        wgpuQueueWriteBuffer(gQueue, viewUniforms, 0, &cu, sizeof(cu));
    }

    // Acquire swapchain texture
    WGPUSurfaceTexture surfaceTex{};
    wgpuSurfaceGetCurrentTexture(gSurface, &surfaceTex);
    switch (surfaceTex.status)
    {
    case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
    case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
        break;
    default:
        return;
    }

    WGPUTextureViewDescriptor viewDesc{
        .format          = gSurfaceFormat,
        .dimension       = WGPUTextureViewDimension_2D,
        .baseMipLevel    = 0, .mipLevelCount  = 1,
        .baseArrayLayer  = 0, .arrayLayerCount = 1,
        .aspect          = WGPUTextureAspect_All,
    };
    WGPUTextureView frameView = wgpuTextureCreateView(surfaceTex.texture, &viewDesc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gDevice, nullptr);

    // Cluster cull pass
    if (gCullBindGroup)
    {
        const uint32_t zero = 0;
        wgpuQueueWriteBuffer(gQueue, gVisibleClustersBuffer, 0, &zero, sizeof(zero));

        WGPUComputePassEncoder cpass = ComputePassBuilder{"cluster cull"}.begin(encoder);
        wgpuComputePassEncoderSetPipeline(cpass, gCullPipeline);
        wgpuComputePassEncoderSetBindGroup(cpass, 0, gCullBindGroup, 0, nullptr);
        const uint32_t cullGroups = (gCullClusterCount + 63u) / 64u;
        wgpuComputePassEncoderDispatchWorkgroups(cpass, cullGroups, 1, 1);
        wgpuComputePassEncoderEnd(cpass);
        wgpuComputePassEncoderRelease(cpass);
    }

    // Cluster expand pass
    if (gExpandBindGroup)
    {
        const uint32_t zero = 0;
        wgpuQueueWriteBuffer(gQueue, gDrawArgsBuffer, 0, &zero, sizeof(zero));

        WGPUComputePassEncoder epass = ComputePassBuilder{"cluster expand"}.begin(encoder);
        wgpuComputePassEncoderSetPipeline(epass, gExpandPipeline);
        wgpuComputePassEncoderSetBindGroup(epass, 0, gExpandBindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(epass, gExpandClusterCount, 1, 1);
        wgpuComputePassEncoderEnd(epass);
        wgpuComputePassEncoderRelease(epass);
    }

    // Render pass
    WGPURenderPassEncoder pass = RenderPassBuilder{}
        .color(frameView, WGPULoadOp_Clear, WGPUStoreOp_Store, {0.95, 0.05, 0.95, 1.0})
        .depth_stencil(gDepthView)
        .begin(encoder);

    if (gClusteredBG && gClusteredPipeline && gOutputIndexBuffer && gDrawArgsBuffer)
    {
        wgpuRenderPassEncoderSetPipeline(pass, gClusteredPipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, gClusteredBG, 0, nullptr);
        wgpuRenderPassEncoderSetIndexBuffer(pass, gOutputIndexBuffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexedIndirect(pass, gDrawArgsBuffer, 0);
    }
    else
    {
        wgpuRenderPassEncoderSetPipeline(pass, gPipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, cameraBindGroup, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gQueue, 1, &cmd);

#ifndef __EMSCRIPTEN__
    wgpuSurfacePresent(gSurface);
#endif

    wgpuTextureViewRelease(frameView);
    wgpuTextureRelease(surfaceTex.texture);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}
