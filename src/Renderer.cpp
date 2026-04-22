#include "Renderer.h"

static wgpu::Device         g_device;
static wgpu::Queue          g_queue;
static wgpu::Surface        g_surface;
static wgpu::RenderPipeline g_pipeline;

static const char kVertexShader[] = R"(
@vertex
fn vs_main(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(
        vec2f( 0.0,  0.5),
        vec2f(-0.5, -0.5),
        vec2f( 0.5, -0.5),
    );
    return vec4f(pos[idx], 0.0, 1.0);
}
)";

static const char kFragmentShader[] = R"(
@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(0.2, 0.6, 1.0, 1.0);
}
)";

static wgpu::ShaderModule make_shader(const char* code) {
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = code;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl;
    return g_device.CreateShaderModule(&desc);
}

static void build_pipeline() {
    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = wgpu::TextureFormat::BGRA8Unorm;

    wgpu::FragmentState frag{};
    frag.module      = make_shader(kFragmentShader);
    frag.entryPoint  = "fs_main";
    frag.targetCount = 1;
    frag.targets     = &colorTarget;

    wgpu::RenderPipelineDescriptor pipeDesc{};
    pipeDesc.vertex.module      = make_shader(kVertexShader);
    pipeDesc.vertex.entryPoint  = "vs_main";
    pipeDesc.fragment           = &frag;
    pipeDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeDesc.multisample.count  = 1;

    g_pipeline = g_device.CreateRenderPipeline(&pipeDesc);
}

void renderer_init(wgpu::Device device, wgpu::Queue queue, wgpu::Surface surface) {
    g_device  = std::move(device);
    g_queue   = std::move(queue);
    g_surface = std::move(surface);
    build_pipeline();
}

void renderer_frame() {
    wgpu::SurfaceTexture surfTex;
    g_surface.GetCurrentTexture(&surfTex);
    if (surfTex.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surfTex.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        return;
    }

    wgpu::TextureView view = surfTex.texture.CreateView();

    wgpu::RenderPassColorAttachment color{};
    color.view       = view;
    color.loadOp     = wgpu::LoadOp::Clear;
    color.storeOp    = wgpu::StoreOp::Store;
    color.clearValue = {0.05f, 0.05f, 0.05f, 1.0f};

    wgpu::RenderPassDescriptor pass{};
    pass.colorAttachmentCount = 1;
    pass.colorAttachments     = &color;

    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();
    wgpu::RenderPassEncoder rp   = encoder.BeginRenderPass(&pass);
    rp.SetPipeline(g_pipeline);
    rp.Draw(3);
    rp.End();

    wgpu::CommandBuffer cmd = encoder.Finish();
    g_queue.Submit(1, &cmd);
#ifndef __EMSCRIPTEN__
    g_surface.Present();
#endif
}
