#pragma once
// Shared scaffolding for the RenderGraph smoke-test demos. A demo is a whole render graph living in
// its own .cpp, #included into the single TU (after RenderGraph.cpp's internals) and listed once in
// the RG_DEMO_LIST registry in RenderGraph_main.cpp. Adding one = new file with the four <id>_*
// hooks + one #include + one registry row. Included after RenderGraph.h + the ImGui headers.
#include "RenderGraph.h"
#include <string>   // assemble shader sources from snippets
#include <cmath>    // camera basis

using namespace RG;   // matches imgui_layer.h; demos + main get the RG names unqualified

// the swapchain format, shared: surface config, ImGui init, and every demo's final present pipeline.
constexpr WGPUTextureFormat kSwapFormat = WGPUTextureFormat_BGRA8Unorm;

// fullscreen-triangle vertex shader, shared by every demo's fullscreen passes. `ndc` is clip-space
// xy in [-1,1] (y up); `casc` carries instance_index (shadow cascades use it, everyone else gets 0).
static const char* kVS = R"(
struct VsOut { @builtin(position) pos : vec4f, @location(0) ndc : vec2f, @location(1) @interpolate(flat) casc : u32 };
@vertex fn vs(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> VsOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o : VsOut;
    o.pos = vec4f(p[vid], 0.0, 1.0);
    o.ndc = p[vid];
    o.casc = iid;                                  // shadow draws one instance per cascade; everyone else draws instance 0
    return o;
}
)";

// free-fly camera: position + yaw/pitch, with the orthonormal basis recomputed by basis(). main owns
// one and drives pos/yaw/pitch from SDL input; demos read the basis to fill their uniforms.
struct Camera {
    float pos[3]   = { 0.0f, 1.9f, 5.5f };
    float yaw      = 0.0f;        // 0 -> looking down -Z
    float pitch    = -0.36f;      // slight downward tilt
    float fwd[3]   = {};
    float right[3] = {};
    float up[3]    = {};
    void basis() {
        float cp = cosf(pitch), sp = sinf(pitch), sy = sinf(yaw), cy = cosf(yaw);
        fwd[0]   =  cp * sy; fwd[1]   = sp;   fwd[2]   = -cp * cy;
        right[0] =  cy;      right[1] = 0.0f; right[2] =  sy;
        up[0]    = -sy * sp; up[1]    = cp;   up[2]    =  cy * sp;
    }
};

// per-frame context handed to a demo's hooks. rebuilt by main every frame.
struct DemoEnv {
    WGPUDevice        device;
    WGPUQueue         queue;
    WGPUTextureFormat swapFormat;
    uint32_t          width, height;
    float             time, dt;
    uint64_t          frame;        // monotonic; a gap means the demo was inactive (re-entry detect)
    const Camera&     camera;
};

// a demo = a whole render graph. all four hooks are required (write an empty body if unused); the
// RG_DEMO_LIST registry wires them by naming convention. build() declares the frame's passes AND
// host-uploads its uniforms (each demo owns its UBO buffer and imports it -- no post-realize step).
struct Demo {
    const char* name;
    void (*init)(const DemoEnv&);                                         // once: pipelines / samplers / UBO buffer
    void (*shutdown)();                                                   // once: release them
    void (*build)(const DemoEnv&, RenderGraph*, ResourceHandle swapchain); // per-frame: declare + upload
    void (*ui)();                                                        // per-frame: ImGui controls
};

// a shader module from a literal or an assembled std::string (snippets concatenated).
static WGPUShaderModule make_shader(WGPUDevice dev, WGPUStringView code)
{
    WGPUShaderSourceWGSL wgsl{ .chain = { .sType = WGPUSType_ShaderSourceWGSL }, .code = code };
    WGPUShaderModuleDescriptor d{ .nextInChain = &wgsl.chain };
    return wgpuDeviceCreateShaderModule(dev, &d);
}
static WGPUShaderModule make_shader(WGPUDevice dev, const std::string& code)
{
    return make_shader(dev, WGPUStringView{ code.c_str(), code.size() });
}

// a single-mip 2D view of one texture -- how a pass body samples a chosen mip (bloom's mip chain).
static WGPUTextureView mip_view_2d(WGPUTexture tex, WGPUTextureFormat fmt, uint32_t mip)
{
    WGPUTextureViewDescriptor vd{
        .format = fmt, .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = mip, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1,
    };
    return wgpuTextureCreateView(tex, &vd);
}
