// Windowed sample for the render graph — immediate mode.
// Not a standalone TU: #included at the end of RenderGraph.cpp so it sees the internal node structs.
// Opens an SDL3 window and rebuilds the WHOLE graph every frame (declare -> compile -> realize ->
// execute -> release), arena-allocated from a single persistent GraphAllocator. Which passes exist
// depends on a runtime toggle (SPACE):
//   glow ON  : scene (graphics) -> sobel (compute) -> blur (compute) -> compose (graphics, additive)
//   glow OFF : scene (graphics) -> present (graphics blit of the plain scene)
// The graph derives pass order from the read/write accesses each frame; toggling glow literally
// changes how many passes exist that frame -- the point of an immediate-mode graph.

#include "RenderGraph.h"
#include <webgpu/webgpu_cpp.h>   // C++ wrappers, only for instance/surface/device bring-up
#include <cmath>                 // sin/cos for the animated edge color

#define SDL_MAIN_HANDLED         // we keep our own main(); just call SDL_SetMainReady()
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>       // SDL_SetMainReady (no main redefinition under SDL_MAIN_HANDLED)

double getTime() {
    static const double freq =
        (double)SDL_GetPerformanceFrequency();

    return (double)SDL_GetPerformanceCounter() / freq;
}

namespace {

constexpr WGPUTextureFormat kSwapFormat    = WGPUTextureFormat_BGRA8Unorm;  // matches the surface
constexpr WGPUTextureFormat kSceneFormat   = WGPUTextureFormat_RGBA8Unorm;  // offscreen scene color (attachment + sampled)
constexpr WGPUTextureFormat kStorageFormat = WGPUTextureFormat_RGBA8Unorm;  // sobel/blur outputs; must match the rgba8unorm storage decls in the shaders

// sobel edge color, animated per frame and uploaded to the sobel pass's uniform buffer.
struct Params { float edgeColor[4]; };

// same shape as Renderer.cpp::createShaderModule
WGPUShaderModule make_shader(WGPUDevice dev, WGPUStringView code)
{
    WGPUShaderSourceWGSL wgsl{ .chain = { .sType = WGPUSType_ShaderSourceWGSL }, .code = code };
    WGPUShaderModuleDescriptor d{ .nextInChain = &wgsl.chain };
    return wgpuDeviceCreateShaderModule(dev, &d);
}

// adapter + device against the given surface, pumped synchronously via the instance (kept alive by
// the caller). mirrors Framework.cpp's request flow, plus compatibleSurface so the adapter can present.
wgpu::Device acquire_device(wgpu::Instance instance, wgpu::Surface surface)
{
    struct AdState { wgpu::Adapter adapter; bool done = false; } as;
    wgpu::RequestAdapterOptions ao{};
    ao.powerPreference   = wgpu::PowerPreference::HighPerformance;
    ao.compatibleSurface = surface;
    ao.backendType       = wgpu::BackendType::D3D12;   // or D3D12 -- force a backend instead of Dawn's default pick
    instance.RequestAdapter(&ao, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView m, AdState* st){
            if (s != wgpu::RequestAdapterStatus::Success)
                std::printf("RequestAdapter failed: %.*s\n", (int)m.length, m.data);
            st->adapter = std::move(a);
            st->done = true;
        }, &as);
    while (!as.done) instance.ProcessEvents();
    if (!as.adapter) return {};

    wgpu::DeviceDescriptor devDesc{};
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m){
            std::printf("GPU error (%d): %.*s\n", (int)t, (int)m.length, m.data);
        });
    // ponytail: no optional features needed -- every storage texture is rgba8unorm (a core storage
    // format) and present is a graphics blit, so the old BGRA8UnormStorage requirement is gone.

    struct DevState { wgpu::Device device; bool done = false; } ds;
    as.adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m, DevState* st){
            if (s != wgpu::RequestDeviceStatus::Success)
                std::printf("RequestDevice failed: %.*s\n", (int)m.length, m.data);
            st->device = std::move(d);
            st->done = true;
        }, &ds);
    while (!ds.done) instance.ProcessEvents();
    return ds.device;
}

} // anon

int main()
{
    using namespace RG;
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: prints/errors show live (and survive a kill)

    // ---- window ----
    SDL_SetMainReady();
    if (!SDL_Init(SDL_INIT_VIDEO)) { std::printf("SDL_Init failed: %s\n", SDL_GetError()); return 1; }
    uint32_t curW = 1280, curH = 720;
    SDL_Window* window = SDL_CreateWindow("RenderGraph sample", (int)curW, (int)curH, SDL_WINDOW_RESIZABLE);
    if (!window) { std::printf("SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }

    // ---- WebGPU instance + surface from the window (HWND glue, same as main_SDL.cpp) ----
    wgpu::InstanceDescriptor instDesc{};
    wgpu::Instance instance = wgpu::CreateInstance(&instDesc);

    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND src{};
    src.hwnd = hwnd; src.hinstance = hinst;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &src;
    wgpu::Surface surface = instance.CreateSurface(&surfDesc);

    // ---- device ----
    wgpu::Device device = acquire_device(instance, surface);
    if (!device) { std::printf("no device, aborting\n"); return 1; }
    wgpu::Queue queue = device.GetQueue();
    WGPUDevice  dev  = device.Get();
    WGPUQueue   q    = queue.Get();
    WGPUSurface surf = surface.Get();

    // ---- configure surface (field order copied verbatim from Renderer.cpp::reconfigure_surface) ----
    WGPUSurfaceConfiguration cfg{
        .device      = dev,
        .format      = kSwapFormat,
        .usage       = WGPUTextureUsage_RenderAttachment,   // compose/present write it as a color attachment
        .width       = curW,
        .height      = curH,
        .alphaMode   = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(surf, &cfg);

    // ---- pipelines (created once: they depend on formats, not on the per-frame sizes) -------------
    // triangle: positions from vertex_index, no vertex buffers. targets the offscreen scene color.
    WGPUShaderModule triSM = make_shader(dev, WEBGPU_STR(R"(
        @vertex fn vs(@builtin(vertex_index) vid : u32) -> @builtin(position) vec4f {
            var p = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
            return vec4f(p[vid], 0.0, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.6, 0.1, 1.0); }
    )"));
    WGPUColorTargetState triCT{ .format = kSceneFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState    triFrag{ .module = triSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &triCT };
    WGPURenderPipelineDescriptor triPD{
        .label       = WEBGPU_STR("triangle pipeline"),
        .layout      = nullptr,
        .vertex      = { .module = triSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &triFrag,
    };
    WGPURenderPipeline triPipe = wgpuDeviceCreateRenderPipeline(dev, &triPD);
    wgpuShaderModuleRelease(triSM);

    // sobel: read scene color via textureLoad, write edge magnitude into a write-only storage texture.
    WGPUShaderModule sobelSM = make_shader(dev, WEBGPU_STR(R"(
        @group(0) @binding(0) var src : texture_2d<f32>;
        @group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;

        struct Params {
            edgeColor : vec4f
        };

        @group(0) @binding(2)
        var<uniform> params : Params;

        fn lum(p : vec2i, dim : vec2i) -> f32 {
            let c = clamp(p, vec2i(0, 0), dim - vec2i(1, 1));
            return dot(textureLoad(src, c, 0).rgb, vec3f(0.299, 0.587, 0.114));
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let dim = vec2i(textureDimensions(src));
            let p   = vec2i(i32(gid.x), i32(gid.y));
            if (p.x >= dim.x || p.y >= dim.y) { return; }

            let a = lum(p + vec2i(-1,-1), dim); let b = lum(p + vec2i(0,-1), dim); let c = lum(p + vec2i(1,-1), dim);
            let d = lum(p + vec2i(-1, 0), dim);                                    let f = lum(p + vec2i(1, 0), dim);
            let g = lum(p + vec2i(-1, 1), dim); let h = lum(p + vec2i(0, 1), dim); let i = lum(p + vec2i(1, 1), dim);

            let gx  = (c + 2.0 * f + i) - (a + 2.0 * d + g);
            let gy  = (g + 2.0 * h + i) - (a + 2.0 * b + c);

            let mag = sqrt(gx * gx + gy * gy);
            let color = params.edgeColor.rgb * mag * 10;
            textureStore(dst, p, vec4f(color, 1.0));
        }
    )"));
    WGPUComputePipelineDescriptor sobelPD{
        .label   = WEBGPU_STR("sobel pipeline"),
        .layout  = nullptr,
        .compute = { .module = sobelSM, .entryPoint = WEBGPU_STR("main") },
    };
    WGPUComputePipeline sobelPipe = wgpuDeviceCreateComputePipeline(dev, &sobelPD);
    wgpuShaderModuleRelease(sobelSM);

    // blur: box-blur the sobel result into a second storage texture (the "glow" source).
    WGPUShaderModule blurSM = make_shader(dev, WEBGPU_STR(R"(
        @group(0) @binding(0) var src : texture_2d<f32>;
        @group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let dim = vec2i(textureDimensions(src));
            let p   = vec2i(i32(gid.x), i32(gid.y));
            if (p.x >= dim.x || p.y >= dim.y) { return; }

            let R = 8;
            var sum   = vec3f(0.0);
            var count = 0.0;
            for (var dy = -R; dy <= R; dy = dy + 1) {
                for (var dx = -R; dx <= R; dx = dx + 1) {
                    let c = clamp(p + vec2i(dx, dy), vec2i(0, 0), dim - vec2i(1, 1));
                    sum   = sum + textureLoad(src, c, 0).rgb;
                    count = count + 1.0;
                }
            }
            textureStore(dst, p, vec4f(sum / count, 1.0));
        }
    )"));
    WGPUComputePipelineDescriptor blurPD{
        .label   = WEBGPU_STR("blur pipeline"),
        .layout  = nullptr,
        .compute = { .module = blurSM, .entryPoint = WEBGPU_STR("main") },
    };
    WGPUComputePipeline blurPipe = wgpuDeviceCreateComputePipeline(dev, &blurPD);
    wgpuShaderModuleRelease(blurSM);

    // present (glow off): fullscreen-triangle blit of a single texture (the plain scene) onto the swapchain.
    WGPUShaderModule blitSM = make_shader(dev, WEBGPU_STR(R"(
        @group(0) @binding(0) var img : texture_2d<f32>;
        @group(0) @binding(1) var smp : sampler;
        struct VsOut { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
        @vertex fn vs(@builtin(vertex_index) vid : u32) -> VsOut {
            var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            var o : VsOut;
            o.pos  = vec4f(p[vid], 0.0, 1.0);
            o.uv   = p[vid] * 0.5 + vec2f(0.5, 0.5);
            o.uv.y = 1.0 - o.uv.y;
            return o;
        }
        @fragment fn fs(in : VsOut) -> @location(0) vec4f { return textureSample(img, smp, in.uv); }
    )"));
    WGPUColorTargetState blitCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState    blitFrag{ .module = blitSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &blitCT };
    WGPURenderPipelineDescriptor blitPD{
        .label       = WEBGPU_STR("blit pipeline"),
        .layout      = nullptr,
        .vertex      = { .module = blitSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &blitFrag,
    };
    WGPURenderPipeline blitPipe = wgpuDeviceCreateRenderPipeline(dev, &blitPD);
    wgpuShaderModuleRelease(blitSM);

    // compose (glow on): fullscreen-triangle blit that adds the blurred edges over the scene.
    WGPUShaderModule composeSM = make_shader(dev, WEBGPU_STR(R"(
        @group(0) @binding(0) var sceneTex : texture_2d<f32>;
        @group(0) @binding(1) var edgeTex  : texture_2d<f32>;
        @group(0) @binding(2) var smp : sampler;
        struct VsOut { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
        @vertex fn vs(@builtin(vertex_index) vid : u32) -> VsOut {
            var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            var o : VsOut;
            o.pos  = vec4f(p[vid], 0.0, 1.0);
            o.uv   = p[vid] * 0.5 + vec2f(0.5, 0.5);
            o.uv.y = 1.0 - o.uv.y;
            return o;
        }
        @fragment fn fs(in : VsOut) -> @location(0) vec4f {
            let scene = textureSample(sceneTex, smp, in.uv).rgb;
            let edges = textureSample(edgeTex,  smp, in.uv).rgb;
            return vec4f(scene + edges, 1.0);   // additive glow
        }
    )"));
    WGPUColorTargetState composeCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState    composeFrag{ .module = composeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &composeCT };
    WGPURenderPipelineDescriptor composePD{
        .label       = WEBGPU_STR("compose pipeline"),
        .layout      = nullptr,
        .vertex      = { .module = composeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &composeFrag,
    };
    WGPURenderPipeline composePipe = wgpuDeviceCreateRenderPipeline(dev, &composePD);
    wgpuShaderModuleRelease(composeSM);

    WGPUSamplerDescriptor sampDesc{
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter    = WGPUFilterMode_Linear,
        .minFilter    = WGPUFilterMode_Linear,
        .maxAnisotropy = 1,
    };
    WGPUSampler sampler = wgpuDeviceCreateSampler(dev, &sampDesc);

    // single persistent arena. create_render_graph() resets it and arena-allocates a fresh
    // RenderGraph (+ all its nodes) from it each frame -> the allocator is the only graph-side
    // object that lives across frames.
    GraphAllocator* allocator = create_allocator();

    // ---- frame loop: declare + compile + realize + execute + release the graph EVERY frame -------
    bool glow      = true;   // SPACE toggles the compute blur/compose glow
    int  shownGlow = -1;     // last glow state we printed the execution order for
    bool running   = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_SPACE) {
                glow = !glow;
                std::printf("glow %s\n", glow ? "ON" : "OFF");
            }
            else if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                cfg.width = (uint32_t)e.window.data1; cfg.height = (uint32_t)e.window.data2;
                wgpuSurfaceConfigure(surf, &cfg);
            }
        }

        WGPUSurfaceTexture st{};
        wgpuSurfaceGetCurrentTexture(surf, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            wgpuSurfaceConfigure(surf, &cfg);     // surface went stale (resize/minimize) -> reconfigure, skip frame
            continue;
        }

        WGPUTextureViewDescriptor vd{
            .format          = kSwapFormat,
            .dimension       = WGPUTextureViewDimension_2D,
            .baseMipLevel    = 0, .mipLevelCount   = 1,
            .baseArrayLayer  = 0, .arrayLayerCount = 1,
            .aspect          = WGPUTextureAspect_All,
        };
        WGPUTextureView view = wgpuTextureCreateView(st.texture, &vd);

        // ---- declare the whole graph for THIS frame (immediate mode) ----
        RenderGraph* rg = create_render_graph(allocator, nullptr);   // resets the arena, fresh RenderGraph

        // import this frame's swapchain view directly; its size also roots the Relative chain below.
        auto swapchain  = rg->importe_image(WEBGPU_STR("swapchain"), view, { cfg.width, cfg.height, 1 });
        auto sceneColor = rg->create_image(WEBGPU_STR("scene.color"), {
            .dimension = WGPUTextureDimension_2D, .format = kSceneFormat,
            .sizeKind  = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
        });

        // multi-writer smoke test (SSA versioning): a depth buffer written by two passes (v1 then
        // v2 -> WAW edge) and read by scene/compose -> RAW; scene's read of v1 makes depth.main's
        // overwrite a WAR. the two writers are graph-shape-only no-op Transfer passes: execute()
        // auto-wires a depth attachment for *graphics* passes (which would then need a depth-enabled
        // pipeline), but skips that for Transfer -- so this drives compile()'s versioning without
        // touching the render pipelines. a real renderer would z-prepass / draw geometry here.
        auto depth = rg->create_image(WEBGPU_STR("depth"), {
            .dimension = WGPUTextureDimension_2D, .format = WGPUTextureFormat_Depth32Float,
            .sizeKind  = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
        });
        rg->add_pass(WEBGPU_STR("depth.prepass"), PassKind::Transfer,
            [&](GraphBuilder& b) { b.depth_stencil(depth, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f); },
            [](PassContext&){});                                  // writes depth v1

        rg->add_pass(WEBGPU_STR("scene"), PassKind::Graphics,
            [&](GraphBuilder& b) {
                b.color(sceneColor, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0.05, 0.05, 0.08, 1.0});
                b.sampled(depth);                                 // reads depth v1 (graph-shape only; not bound in the body)
            },
            [triPipe](PassContext& ctx){
                wgpuRenderPassEncoderSetPipeline(ctx.render, triPipe);
                wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            });

        rg->add_pass(WEBGPU_STR("depth.main"), PassKind::Transfer,
            [&](GraphBuilder& b) { b.depth_stencil(depth, WGPULoadOp_Load, WGPUStoreOp_Store, 1.0f); },
            [](PassContext&){});      // writes depth v2 -> WAW edge to depth.prepass, WAR edge to scene's read of v1

        ResourceHandle ubo{};   // only created on the glow path; needed after realize() to upload Params
        if (glow) {
            auto sobelOut = rg->create_image(WEBGPU_STR("sobel.out"), {
                .dimension = WGPUTextureDimension_2D, .format = kStorageFormat,
                .sizeKind  = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = sceneColor,
            });
            auto blurOut = rg->create_image(WEBGPU_STR("blur.out"), {
                .dimension = WGPUTextureDimension_2D, .format = kStorageFormat,
                .sizeKind  = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = sceneColor,
            });
            ubo = rg->create_buffer(WEBGPU_STR("ubo"), { .size = sizeof(Params) });

            const uint32_t groupsX = (cfg.width + 7) / 8, groupsY = (cfg.height + 7) / 8;

            // bind groups reference this frame's freshly-realized views, so build them in the pass
            // body from the resolved PassContext, then release right after recording -- the command
            // buffer retains them until the GPU is done.
            rg->add_pass(WEBGPU_STR("sobel"), PassKind::Compute,
                [&](GraphBuilder& b) {
                    b.sampled(sceneColor);
                    b.storage_write(sobelOut);
                    b.uniform(ubo);
                },
                [dev, sobelPipe, sceneColor, sobelOut, ubo, groupsX, groupsY](PassContext& ctx){
                    WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(sobelPipe, 0);
                    WGPUBindGroupEntry e[3] = {
                        { .binding = 0, .textureView = ctx.view(sceneColor) },
                        { .binding = 1, .textureView = ctx.view(sobelOut) },
                        { .binding = 2, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(Params) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuComputePassEncoderSetPipeline(ctx.compute, sobelPipe);
                    wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
                    wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, groupsX, groupsY, 1);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });

            rg->add_pass(WEBGPU_STR("blur"), PassKind::Compute,
                [&](GraphBuilder& b) {
                    b.sampled(sobelOut);
                    b.storage_write(blurOut);
                },
                [dev, blurPipe, sobelOut, blurOut, groupsX, groupsY](PassContext& ctx){
                    WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(blurPipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(sobelOut) },
                        { .binding = 1, .textureView = ctx.view(blurOut) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuComputePassEncoderSetPipeline(ctx.compute, blurPipe);
                    wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
                    wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, groupsX, groupsY, 1);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });

            rg->add_pass(WEBGPU_STR("compose"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(sceneColor);
                    b.sampled(blurOut);
                    b.sampled(depth);                             // reads depth v2 -> RAW edge to depth.main
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, composePipe, sampler, sceneColor, blurOut](PassContext& ctx){
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(composePipe, 0);
                    WGPUBindGroupEntry e[3] = {
                        { .binding = 0, .textureView = ctx.view(sceneColor) },
                        { .binding = 1, .textureView = ctx.view(blurOut) },
                        { .binding = 2, .sampler = sampler },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, composePipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        } else {
            rg->add_pass(WEBGPU_STR("present"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(sceneColor);
                    b.sampled(depth);                             // reads depth v2 -> RAW edge to depth.main
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, blitPipe, sampler, sceneColor](PassContext& ctx){
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(blitPipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(sceneColor) },
                        { .binding = 1, .sampler = sampler },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, blitPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        }

        rg->compile();
        rg->realize(dev);     // creates this frame's offscreen textures (+ ubo) from the resolved usages/sizes

        // proof the per-frame graph really changes shape: print the order whenever glow flips.
        if ((int)glow != shownGlow) {
            shownGlow = (int)glow;
            std::printf("execution order:");
            for (PassNode* p = rg->m_passes; p; p = p->next) std::printf(" %s", p->name.data);
            std::printf("\n");
            rg->debug_print_mermaid();
        }

        if (glow) {
            Params p;
            float t = (float)(getTime() * 10);
            p.edgeColor[0] = sin(t) * 0.5f + 0.5f;
            p.edgeColor[1] = cos(t) * 0.5f + 0.5f;
            p.edgeColor[2] = 1.0f;
            p.edgeColor[3] = 1.0f;
            wgpuQueueWriteBuffer(q, rg->node(ubo)->buffer, 0, &p, sizeof(p));
        }

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev, nullptr);
        rg->execute(enc, q);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(q, 1, &cmd);
        wgpuSurfacePresent(surf);

        rg->release_resources();   // destroy this frame's graph textures/buffers (imported swapchain left alone)

        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        instance.ProcessEvents();   // pump device/error callbacks
    }

    // ---- teardown --------------------------------------------------------------------------------
    wgpuSamplerRelease(sampler);
    wgpuRenderPipelineRelease(composePipe);
    wgpuRenderPipelineRelease(blitPipe);
    wgpuComputePipelineRelease(blurPipe);
    wgpuComputePipelineRelease(sobelPipe);
    wgpuRenderPipelineRelease(triPipe);
    SDL_DestroyWindow(window);
    SDL_Quit();
    // ponytail: the GraphAllocator (1 MB block) is leaked at exit -- one-time, process reclaims it.
    return 0;
}
