// Windowed driver for the render-graph smoke test. The whole graph is declared -> compiled -> realized
// -> executed -> torn down EVERY frame; only the GraphAllocator (arena + resource pools) survives.
//
// This file is just the framework: window/device bring-up, a shared free-fly camera, ImGui, and the
// frame loop. Each actual render graph ("demo") lives in its own .cpp, #included below and registered in
// RG_DEMO_LIST. Adding a demo = a new file defining the four <id>_init/_shutdown/_build/_ui hooks + one
// #include + one RG_DEMO_LIST row. F1..Fn (or the "Demos" window) switch between them live.
//
// Camera: WASD to fly, hold left mouse to look around.
//
// Not a standalone TU: #included at the end of RenderGraph.cpp so the demos see the internal node structs.

#include "RenderGraph.h"
#include <cstdio>
#include <string>                // pass-name signature for the reshaping print
#include <webgpu/webgpu_cpp.h>   // C++ wrappers, only for instance/surface/device bring-up
#include <cmath>

#define SDL_MAIN_HANDLED         // we keep our own main(); just call SDL_SetMainReady()
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>       // SDL_SetMainReady (no main redefinition under SDL_MAIN_HANDLED)

#include "imgui_layer.h"         // ImGui SDL3 + WebGPU backends + the DAG widget
#include "RenderGraph_demo.h"    // Camera / DemoEnv / Demo + shared helpers + kSwapFormat + kVS

double getTime() {
    static const double freq = (double)SDL_GetPerformanceFrequency();
    return (double)SDL_GetPerformanceCounter() / freq;
}

namespace {

// adapter + device against the given surface, pumped synchronously via the instance (kept alive by
// the caller). mirrors Framework.cpp's request flow, plus compatibleSurface so the adapter can present.
wgpu::Device acquire_device(wgpu::Instance instance, wgpu::Surface surface)
{
    struct AdState { wgpu::Adapter adapter; bool done = false; } as;
    wgpu::RequestAdapterOptions ao{};
    ao.powerPreference   = wgpu::PowerPreference::HighPerformance;
    ao.compatibleSurface = surface;
    ao.backendType       = wgpu::BackendType::D3D12;   // force a backend instead of Dawn's default pick
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

// ---- demos: each defines <id>_init / <id>_shutdown / <id>_build / <id>_ui --------------------------
#include "RenderGraph_deferred.cpp"
#include "RenderGraph_pathtracer.cpp"

// the registry. one row per demo; RG_DEMO_LIST(X) builds the table from the naming convention, so the
// hooks never need a hand-written Demo literal. add a demo above, then add its row here.
#define RG_DEMO_LIST(X) \
    X("Deferred",    deferred)   \
    X("Path Tracer", pathtracer)

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

    // ---- WebGPU instance + surface from the window (HWND glue) ----
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

    // ---- configure surface ----
    WGPUSurfaceConfiguration cfg{
        .device      = dev,
        .format      = kSwapFormat,
        .usage       = WGPUTextureUsage_RenderAttachment,
        .width       = curW,
        .height      = curH,
        .alphaMode   = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(surf, &cfg);

    // single persistent arena: create_render_graph() resets it and bump-allocates a fresh graph each
    // frame. It also owns the resource pools (temporal/transient), so it outlives the frame loop.
    GraphAllocator* allocator = create_allocator();

    imgui_layer_init(window, dev, kSwapFormat);

    // ---- the demo registry ----
    #define X(label, id) Demo{ label, id##_init, id##_shutdown, id##_build, id##_ui },
    static const Demo demos[] = { RG_DEMO_LIST(X) };
    #undef X
    constexpr int kDemoCount = (int)(sizeof(demos) / sizeof(demos[0]));
    int active = 0;   // default demo (set to 1 to launch the path tracer for non-interactive checks)

    // free-fly camera, shared by every demo. main drives pos/yaw/pitch from SDL; demos read the basis.
    Camera camera;
    bool   dragging = false;
    const float kMouseSens = 0.0025f, kMoveSpeed = 4.0f;

    // init every demo once (pipelines depend on formats, not per-frame size). a startup DemoEnv with the
    // initial size is enough -- init() only needs device/queue/swapFormat.
    DemoEnv initEnv{ dev, q, kSwapFormat, cfg.width, cfg.height, 0.0f, 0.0f, 0, camera };
    for (const Demo& d : demos) d.init(initEnv);

    double      prevTime = getTime();
    uint64_t    frame    = 0;
    std::string lastSig;          // pass-name signature; reprint the execution order when the graph reshapes
    bool        running  = true;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN &&
                     e.key.scancode >= SDL_SCANCODE_F1 && e.key.scancode < SDL_SCANCODE_F1 + kDemoCount) {
                active = (int)(e.key.scancode - SDL_SCANCODE_F1);
                std::printf("demo: %s\n", demos[active].name);
            }
            else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT
                     && !ImGui::GetIO().WantCaptureMouse) {     // don't start a look-drag on a UI click
                dragging = true;
                SDL_SetWindowRelativeMouseMode(window, true);   // capture + hide cursor while looking
            }
            else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP && e.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
                SDL_SetWindowRelativeMouseMode(window, false);
            }
            else if (e.type == SDL_EVENT_MOUSE_MOTION && dragging) {
                camera.yaw   += e.motion.xrel * kMouseSens;
                camera.pitch -= e.motion.yrel * kMouseSens;
                const float lim = 1.5533f;                       // ~89 deg, avoid gimbal flip
                camera.pitch = camera.pitch > lim ? lim : (camera.pitch < -lim ? -lim : camera.pitch);
            }
            else if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                cfg.width = (uint32_t)e.window.data1; cfg.height = (uint32_t)e.window.data2;
                wgpuSurfaceConfigure(surf, &cfg);
            }
        }

        // camera basis from yaw/pitch, then WASD fly-movement (frame-rate independent via dt).
        double now = getTime();
        float  dt  = (float)(now - prevTime);
        prevTime   = now;
        camera.basis();
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            auto ks = SDL_GetKeyboardState(nullptr);   // const bool* (SDL3); index with scancodes
            float mv = kMoveSpeed * dt;
            float f = (ks[SDL_SCANCODE_W] ? mv : 0.0f) - (ks[SDL_SCANCODE_S] ? mv : 0.0f);
            float r = (ks[SDL_SCANCODE_D] ? mv : 0.0f) - (ks[SDL_SCANCODE_A] ? mv : 0.0f);
            for (int i = 0; i < 3; ++i) camera.pos[i] += camera.fwd[i] * f + camera.right[i] * r;
        }

        imgui_layer_begin_frame();   // NewFrame only; the DAG window is built after compile, Render() in end_frame

        ImGui::Begin("Demos");
        for (int i = 0; i < kDemoCount; ++i) {
            if (i) ImGui::SameLine();
            if (ImGui::RadioButton(demos[i].name, active == i)) active = i;   // F1..Fn also switch
        }
        ImGui::Separator();
        demos[active].ui();
        ImGui::End();

        WGPUSurfaceTexture st{};
        wgpuSurfaceGetCurrentTexture(surf, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            wgpuSurfaceConfigure(surf, &cfg);     // surface went stale (resize/minimize) -> reconfigure, skip frame
            imgui_layer_end_frame();              // balance begin_frame's NewFrame on the skipped frame
            continue;
        }

        WGPUTextureViewDescriptor vd{
            .format = kSwapFormat, .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
        };
        WGPUTextureView view = wgpuTextureCreateView(st.texture, &vd);

        // ---- declare the whole graph for THIS frame (immediate mode) ----
        RenderGraph* rg = create_render_graph(allocator);   // resets the arena (pools persist in the allocator)
        auto swapchain = rg->importe_image(WEBGPU_STR("swapchain"), view, { cfg.width, cfg.height, 1 });

        ++frame;   // counts frames that reach build (surface-stale skips above don't); demos detect re-entry from gaps
        DemoEnv env{ dev, q, kSwapFormat, cfg.width, cfg.height, (float)now, dt, frame, camera };
        demos[active].build(env, rg, swapchain);

        // ImGui overlay: last pass. Load keeps the rendered scene; the write to the imported swapchain
        // makes it a sink, and a WAW edge orders it after the demo's present.
        rg->add_pass(WEBGPU_STR("imgui"), PassKind::Graphics,
            [&](GraphBuilder& b) { b.color(swapchain, WGPULoadOp_Load, WGPUStoreOp_Store); },
            [](PassContext& ctx) { ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), ctx.render); });

        if (!rg->compile()) {
            // ordering error (compile() already printed it). skip this frame's GPU work.
            wgpuTextureViewRelease(view);
            wgpuTextureRelease(st.texture);
            instance.ProcessEvents();
            imgui_layer_end_frame();
            continue;
        }
        rg->realize(dev);     // creates this frame's transient textures

        imgui_layer_draw_graph(rg);   // build the DAG window now the graph is compiled + realized
        imgui_layer_end_frame();      // ImGui::Render(); the "imgui" pass consumes the draw data at execute

        // reprint the execution order whenever the graph reshapes (demo switch, toggle, resize-driven mip
        // count). a pass-name signature diff catches it for ANY demo, no per-demo cooperation.
        std::string sig;
        for (PassNode* p = storage(rg)->m_passes; p; p = p->next) { sig.append(p->name.data, p->name.length); sig.push_back('|'); }
        if (sig != lastSig) {
            lastSig = sig;
            std::printf("execution order:");
            for (PassNode* p = storage(rg)->m_passes; p; p = p->next) std::printf(" %.*s", (int)p->name.length, p->name.data);
            std::printf("\ntransient pool: %zu textures, %u created this frame\n",
                        allocator->transient.entries.size(), allocator->transient.createdThisFrame);
        }

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev, nullptr);
        rg->execute(enc, q);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(q, 1, &cmd);
        wgpuSurfacePresent(surf);

        rg->release_resources();   // destroy this frame's graph textures (imported swapchain left alone)

        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        instance.ProcessEvents();   // pump device/error callbacks
    }

    // ---- teardown ----
    for (const Demo& d : demos) d.shutdown();
    imgui_layer_shutdown();
    SDL_DestroyWindow(window);
    SDL_Quit();
    // ponytail: the GraphAllocator (arena + pools + null-default textures) is leaked at exit -- one-time, process reclaims it.
    return 0;
}
