#include "Renderer.h"
#include "Framework.h"
#include <cmath>
#include <array>
#include <chrono>
#include <algorithm>

// ---- Math ------------------------------------------------------------------

using Mat4 = std::array<float, 16>;  // column-major

struct Vec3 { float x, y, z; };

static Vec3 v3_cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static float v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 v3_norm(Vec3 a) {
    float inv = 1.f / std::sqrt(v3_dot(a, a));
    return {a.x*inv, a.y*inv, a.z*inv};
}

// eye + fwd_unit → view matrix (column-major, right-handed, Y-up)
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

// Perspective, depth [0..1], right-handed (column-major)
static Mat4 mat4_perspective(float fov_y, float aspect, float near_z, float far_z) {
    float f = 1.f / std::tan(fov_y * .5f);
    float d = near_z - far_z;
    Mat4 m{};
    m[0]=f/aspect; m[5]=f;
    m[10]=far_z/d; m[11]=-1.f;
    m[14]=(near_z*far_z)/d;
    return m;
}

static Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 c{};
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 4; k++)
                c[col*4+row] += a[k*4+row] * b[col*4+k];
    return c;
}

// ---- Scene geometry --------------------------------------------------------

struct Vertex { float x, y, z, r, g, b; };

// Floor + 4 colored pillars at cardinal points
static const Vertex kScene[] = {
    // Floor (gray)
    {-10,0,-10, .4f,.4f,.4f}, { 10,0,-10, .4f,.4f,.4f}, { 10,0, 10, .4f,.4f,.4f},
    {-10,0,-10, .4f,.4f,.4f}, { 10,0, 10, .4f,.4f,.4f}, {-10,0, 10, .4f,.4f,.4f},
    // North pillar z=-5 (red)
    {-.5f,0,-5, .9f,.2f,.2f}, {.5f,0,-5, .9f,.2f,.2f}, {.5f,3,-5, .9f,.2f,.2f},
    {-.5f,0,-5, .9f,.2f,.2f}, {.5f,3,-5, .9f,.2f,.2f}, {-.5f,3,-5, .9f,.2f,.2f},
    // East pillar x=5 (green)
    {5,0,-.5f, .2f,.9f,.2f}, {5,0,.5f, .2f,.9f,.2f}, {5,3,.5f, .2f,.9f,.2f},
    {5,0,-.5f, .2f,.9f,.2f}, {5,3,.5f, .2f,.9f,.2f}, {5,3,-.5f, .2f,.9f,.2f},
    // West pillar x=-5 (blue)
    {-5,0,-.5f, .2f,.2f,.9f}, {-5,0,.5f, .2f,.2f,.9f}, {-5,3,.5f, .2f,.2f,.9f},
    {-5,0,-.5f, .2f,.2f,.9f}, {-5,3,.5f, .2f,.2f,.9f}, {-5,3,-.5f, .2f,.2f,.9f},
    // South pillar z=5 (yellow)
    {-.5f,0,5, .9f,.9f,.2f}, {.5f,0,5, .9f,.9f,.2f}, {.5f,3,5, .9f,.9f,.2f},
    {-.5f,0,5, .9f,.9f,.2f}, {.5f,3,5, .9f,.9f,.2f}, {-.5f,3,5, .9f,.9f,.2f},
};
static constexpr uint32_t kVertexCount = sizeof(kScene) / sizeof(kScene[0]);

// ---- Shaders ---------------------------------------------------------------

static const char kVertexShader[] = R"(
struct Uniforms { view_proj : mat4x4f }
@group(0) @binding(0) var<uniform> u : Uniforms;
struct VOut {
    @builtin(position) clip : vec4f,
    @location(0) col : vec3f,
}
@vertex fn vs_main(@location(0) pos : vec3f, @location(1) col : vec3f) -> VOut {
    return VOut(u.view_proj * vec4f(pos, 1.0), col);
}
)";

static const char kFragmentShader[] = R"(
@fragment fn fs_main(@location(0) col : vec3f) -> @location(0) vec4f {
    return vec4f(col, 1.0);
}
)";

// ---- GPU objects -----------------------------------------------------------

static wgpu::Device         g_device;
static wgpu::Queue          g_queue;
static wgpu::Surface        g_surface;
static wgpu::RenderPipeline g_pipeline;
static wgpu::Buffer         g_vertex_buf;
static wgpu::Buffer         g_uniform_buf;
static wgpu::BindGroup      g_bind_group;
static wgpu::Texture        g_depth_tex;
static wgpu::TextureView    g_depth_view;
static uint32_t             g_width  = kWidth;
static uint32_t             g_height = kHeight;

// ---- Camera ----------------------------------------------------------------

struct Camera { float x=0, y=1.7f, z=0, yaw=0, pitch=0; };
static Camera g_cam;
static auto   g_last_time = std::chrono::steady_clock::now();

// ---- Helpers ---------------------------------------------------------------

static wgpu::ShaderModule make_shader(const char* code) {
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = code;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl;
    return g_device.CreateShaderModule(&desc);
}

static void configure_surface(uint32_t w, uint32_t h) {
    wgpu::SurfaceConfiguration cfg{};
    cfg.device = g_device;
    cfg.format = wgpu::TextureFormat::BGRA8Unorm;
    cfg.width  = w;
    cfg.height = h;
    g_surface.Configure(&cfg);
}

static void create_depth_texture(uint32_t w, uint32_t h) {
    wgpu::TextureDescriptor td{};
    td.size   = {w, h, 1};
    td.format = wgpu::TextureFormat::Depth24Plus;
    td.usage  = wgpu::TextureUsage::RenderAttachment;
    g_depth_tex  = g_device.CreateTexture(&td);
    g_depth_view = g_depth_tex.CreateView();
}

static void build_pipeline() {
    wgpu::VertexAttribute attrs[2]{};
    attrs[0].format        = wgpu::VertexFormat::Float32x3;
    attrs[0].offset        = 0;
    attrs[0].shaderLocation = 0;
    attrs[1].format        = wgpu::VertexFormat::Float32x3;
    attrs[1].offset        = 3 * sizeof(float);
    attrs[1].shaderLocation = 1;

    wgpu::VertexBufferLayout vbl{};
    vbl.arrayStride    = 6 * sizeof(float);
    vbl.attributeCount = 2;
    vbl.attributes     = attrs;

    wgpu::BindGroupLayoutEntry bgle{};
    bgle.binding            = 0;
    bgle.visibility         = wgpu::ShaderStage::Vertex;
    bgle.buffer.type        = wgpu::BufferBindingType::Uniform;
    bgle.buffer.minBindingSize = 64;

    wgpu::BindGroupLayoutDescriptor bgld{};
    bgld.entryCount = 1;
    bgld.entries    = &bgle;
    wgpu::BindGroupLayout bgl = g_device.CreateBindGroupLayout(&bgld);

    wgpu::PipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &bgl;

    wgpu::DepthStencilState ds{};
    ds.format             = wgpu::TextureFormat::Depth24Plus;
    ds.depthWriteEnabled  = wgpu::OptionalBool::True;
    ds.depthCompare       = wgpu::CompareFunction::Less;

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = wgpu::TextureFormat::BGRA8Unorm;

    wgpu::FragmentState frag{};
    frag.module      = make_shader(kFragmentShader);
    frag.entryPoint  = "fs_main";
    frag.targetCount = 1;
    frag.targets     = &colorTarget;

    wgpu::RenderPipelineDescriptor pipeDesc{};
    pipeDesc.layout             = g_device.CreatePipelineLayout(&pld);
    pipeDesc.vertex.module      = make_shader(kVertexShader);
    pipeDesc.vertex.entryPoint  = "vs_main";
    pipeDesc.vertex.bufferCount = 1;
    pipeDesc.vertex.buffers     = &vbl;
    pipeDesc.fragment           = &frag;
    pipeDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeDesc.primitive.cullMode = wgpu::CullMode::None;
    pipeDesc.depthStencil       = &ds;
    pipeDesc.multisample.count  = 1;

    g_pipeline = g_device.CreateRenderPipeline(&pipeDesc);
}

// ---- Public API ------------------------------------------------------------

void renderer_set_initial_size(uint32_t w, uint32_t h) {
    g_width = w; g_height = h;
}

void renderer_init(wgpu::Device device, wgpu::Queue queue, wgpu::Surface surface) {
    g_device  = std::move(device);
    g_queue   = std::move(queue);
    g_surface = std::move(surface);

    configure_surface(g_width, g_height);
    create_depth_texture(g_width, g_height);
    build_pipeline();

    wgpu::BufferDescriptor vbd{};
    vbd.size  = sizeof(kScene);
    vbd.usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
    g_vertex_buf = g_device.CreateBuffer(&vbd);
    g_queue.WriteBuffer(g_vertex_buf, 0, kScene, sizeof(kScene));

    wgpu::BufferDescriptor ubd{};
    ubd.size  = 64;  // mat4x4f
    ubd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    g_uniform_buf = g_device.CreateBuffer(&ubd);

    wgpu::BindGroupEntry bge{};
    bge.binding = 0;
    bge.buffer  = g_uniform_buf;
    bge.size    = 64;

    wgpu::BindGroupDescriptor bgd{};
    bgd.layout     = g_pipeline.GetBindGroupLayout(0);
    bgd.entryCount = 1;
    bgd.entries    = &bge;
    g_bind_group = g_device.CreateBindGroup(&bgd);
}

void renderer_resize(uint32_t w, uint32_t h) {
    g_width = w; g_height = h;
    configure_surface(w, h);
    create_depth_texture(w, h);
}

void renderer_frame() {
    // Delta time
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - g_last_time).count();
    g_last_time = now;
    dt = std::min(dt, 0.1f);

    // Camera — mouse look
    const float kSens  = 0.002f;
    const float kSpeed = 5.0f;
    g_cam.yaw   += input_mouse_delta_x() * kSens;
    g_cam.pitch -= input_mouse_delta_y() * kSens;
    g_cam.pitch  = std::max(-1.5f, std::min(1.5f, g_cam.pitch));

    // Forward vector from yaw/pitch (unit length by construction)
    float cp = std::cos(g_cam.pitch), sp = std::sin(g_cam.pitch);
    float cy = std::cos(g_cam.yaw),   sy = std::sin(g_cam.yaw);
    Vec3 fwd = {sy*cp, sp, -cy*cp};

    // WASD — move on XZ plane, right = rotate fwd 90° in XZ
    float move = kSpeed * dt;
    if (input_key_down(Key::W)) { g_cam.x += fwd.x*move; g_cam.z += fwd.z*move; }
    if (input_key_down(Key::S)) { g_cam.x -= fwd.x*move; g_cam.z -= fwd.z*move; }
    if (input_key_down(Key::A)) { g_cam.x -= cy*move;    g_cam.z -= sy*move; }
    if (input_key_down(Key::D)) { g_cam.x += cy*move;    g_cam.z += sy*move; }

    // Build view-projection and upload
    Vec3 eye = {g_cam.x, g_cam.y, g_cam.z};
    Mat4 view_mat = mat4_look_at(eye, v3_norm(fwd), {0,1,0});
    Mat4 proj_mat = mat4_perspective(1.2f, (float)g_width/(float)g_height, 0.1f, 100.f);
    Mat4 vp = mat4_mul(proj_mat, view_mat);
    g_queue.WriteBuffer(g_uniform_buf, 0, vp.data(), 64);

    // Acquire frame
    wgpu::SurfaceTexture surfTex;
    g_surface.GetCurrentTexture(&surfTex);
    if (surfTex.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surfTex.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal)
        return;

    wgpu::TextureView frame_view = surfTex.texture.CreateView();

    wgpu::RenderPassColorAttachment color{};
    color.view       = frame_view;
    color.loadOp     = wgpu::LoadOp::Clear;
    color.storeOp    = wgpu::StoreOp::Store;
    color.clearValue = {0.05f, 0.05f, 0.05f, 1.f};

    wgpu::RenderPassDepthStencilAttachment depth{};
    depth.view            = g_depth_view;
    depth.depthLoadOp     = wgpu::LoadOp::Clear;
    depth.depthStoreOp    = wgpu::StoreOp::Store;
    depth.depthClearValue = 1.f;

    wgpu::RenderPassDescriptor pass{};
    pass.colorAttachmentCount   = 1;
    pass.colorAttachments       = &color;
    pass.depthStencilAttachment = &depth;

    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::RenderPassEncoder rp   = encoder.BeginRenderPass(&pass);
    rp.SetPipeline(g_pipeline);
    rp.SetBindGroup(0, g_bind_group);
    rp.SetVertexBuffer(0, g_vertex_buf);
    rp.Draw(kVertexCount);
    rp.End();

    wgpu::CommandBuffer cmd = encoder.Finish();
    g_queue.Submit(1, &cmd);
#ifndef __EMSCRIPTEN__
    g_surface.Present();
#endif
}
