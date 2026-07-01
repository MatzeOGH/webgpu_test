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

#ifdef __EMSCRIPTEN__            // web driver: canvas surface, async device, requestAnimationFrame loop
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <utility>               // std::move across the async adapter/device callbacks
#endif

#include "imgui_layer.h"         // ImGui SDL3 + WebGPU backends + the DAG widget
#include "RenderGraph_demo.h"    // Camera / DemoEnv / Demo + shared helpers + kSwapFormat + kVS

double getTime() {
    static const double freq = (double)SDL_GetPerformanceFrequency();
    return (double)SDL_GetPerformanceCounter() / freq;
}

#ifndef __EMSCRIPTEN__   // native only: the busy-wait below can't work under -sASYNCIFY=0 (callbacks
                         // only fire once we yield to the browser). the web path acquires async instead.
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

    // opt-in per-pass GPU timing needs the TimestampQuery feature. request it only when the adapter has it
    // -- requesting an unsupported feature fails device creation. main re-checks via wgpuDeviceHasFeature.
    static const wgpu::FeatureName kTimestamp[] = { wgpu::FeatureName::TimestampQuery };
    if (as.adapter.HasFeature(wgpu::FeatureName::TimestampQuery)) {
        devDesc.requiredFeatures     = kTimestamp;
        devDesc.requiredFeatureCount = 1;
    }

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
#endif // !__EMSCRIPTEN__

// ---- demos: each defines <id>_init / <id>_shutdown / <id>_build / <id>_ui --------------------------
#include "RenderGraph_deferred.cpp"
#include "RenderGraph_pathtracer.cpp"
#include "RenderGraph_particles.cpp"
#include "RenderGraph_bake.cpp"
#include "RenderGraph_msaa.cpp"

// the registry. one row per demo; RG_DEMO_LIST(X) builds the table from the naming convention, so the
// hooks never need a hand-written Demo literal. add a demo above, then add its row here.
#define RG_DEMO_LIST(X) \
    X("Deferred",    deferred)   \
    X("Path Tracer", pathtracer) \
    X("Particles",   particles)  \
    X("Bake",        bake)        \
    X("MSAA",        msaa)

// All per-frame state. On native this could be main() locals; on web it must outlive main() (which
// returns immediately after handing the frame callback to the browser), so main() keeps one file-static
// instance and threads &app through the loop callback.
struct AppState {
    SDL_Window*    window = nullptr;
    wgpu::Instance instance;                 // keep the C++ objects alive: RAII owns the GPU handles, and
    wgpu::Surface  surface;                  // the async web callbacks store into here -- a raw WGPU* cache
#ifdef __EMSCRIPTEN__                         // alone would let the wrapper release the device on return.
    wgpu::Adapter  adapter;
#endif
    wgpu::Device   deviceObj;
    wgpu::Queue    queueObj;
    WGPUDevice     dev  = nullptr;            // cached .Get() handles used every frame
    WGPUQueue      q    = nullptr;
    WGPUSurface    surf = nullptr;
    WGPUSurfaceConfiguration cfg{};           // width/height mutate on resize
    GraphAllocator* allocator = nullptr;

    int    active   = 0;                       // selected demo (set to 1 for the path tracer non-interactively)
    Camera camera;
    bool   dragging = false;

    double      startTime = 0.0;               // getTime() at first frame; env.time is relative to this so the
    double      prevTime  = 0.0;                // cast to float below doesn't lose precision to a large clock epoch
    uint64_t    frame    = 0;
    std::string lastSig;                       // pass-name signature; reprint execution order on reshape

    bool gpuTimingAvailable = false;
    bool enableAlias        = true;            // phase-4 transient aliasing (folded into the reshape sig)
    bool enableProfiling    = false;           // opt-in per-pass GPU timestamps; hidden if unsupported
    bool running            = true;            // native loop guard; the web RAF loop never ends

    bool ready  = false;                       // device acquired (immediate on native, async on web)
    bool inited = false;                        // one-time init_after_device done
};

// the demo registry. one row per demo; RG_DEMO_LIST(X) builds the table from the naming convention.
#define X(label, id) Demo{ label, id##_init, id##_shutdown, id##_build, id##_ui },
static const Demo g_demos[] = { RG_DEMO_LIST(X) };
#undef X
static constexpr int   kDemoCount = (int)(sizeof(g_demos) / sizeof(g_demos[0]));
static constexpr float kMouseSens = 0.0025f, kMoveSpeed = 4.0f;

#ifdef __EMSCRIPTEN__
// Async adapter->device, mirroring Framework.cpp. Under -sASYNCIFY=0 the callbacks fire once we yield to
// the browser (i.e. during instance.ProcessEvents() from the RAF loop), so we can't block on them; we set
// app.ready and let the frame callback pick it up. Store every wgpu object into AppState to keep it alive.
static void web_on_device(wgpu::RequestDeviceStatus status, wgpu::Device device,
                          wgpu::StringView msg, AppState* app)
{
    if (status != wgpu::RequestDeviceStatus::Success) {
        std::printf("RequestDevice failed: %.*s\n", (int)msg.length, msg.data);
        return;
    }
    app->deviceObj = std::move(device);
    app->queueObj  = app->deviceObj.GetQueue();
    app->dev  = app->deviceObj.Get();
    app->q    = app->queueObj.Get();
    app->surf = app->surface.Get();
    app->gpuTimingAvailable = wgpuDeviceHasFeature(app->dev, WGPUFeatureName_TimestampQuery);
    app->ready = true;
}

static void web_on_adapter(wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                           wgpu::StringView msg, AppState* app)
{
    if (status != wgpu::RequestAdapterStatus::Success) {
        std::printf("RequestAdapter failed: %.*s\n", (int)msg.length, msg.data);
        return;
    }
    app->adapter = std::move(adapter);

    wgpu::DeviceDescriptor devDesc{};
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m){
            std::printf("GPU error (%d): %.*s\n", (int)t, (int)m.length, m.data);
        });
    // per-pass GPU timing needs TimestampQuery; browsers almost never expose it. request it only if the
    // adapter has it (an unsupported required feature fails device creation) -- the UI hides itself otherwise.
    static const wgpu::FeatureName kTimestamp[] = { wgpu::FeatureName::TimestampQuery };
    if (app->adapter.HasFeature(wgpu::FeatureName::TimestampQuery)) {
        devDesc.requiredFeatures     = kTimestamp;
        devDesc.requiredFeatureCount = 1;
    }
    app->adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous, web_on_device, app);
}

static void request_adapter_then_device(AppState& app)
{
    wgpu::RequestAdapterOptions ao{};
    ao.powerPreference   = wgpu::PowerPreference::HighPerformance;
    ao.compatibleSurface = app.surface;   // no backendType force on web -- the browser picks
    app.instance.RequestAdapter(&ao, wgpu::CallbackMode::AllowSpontaneous, web_on_adapter, &app);
}

// browser-window resize -> follow the canvas' backing size + reconfigure the surface (cf. main_web.cpp).
// A named function, not an inline lambda: emscripten_set_resize_callback is a macro and a braced lambda
// body trips its argument counting.
static EM_BOOL web_on_resize(int, const EmscriptenUiEvent*, void* ud)
{
    auto& s = *static_cast<AppState*>(ud);
    int cw = 0, ch = 0;
    if (s.ready && emscripten_get_canvas_element_size("#canvas", &cw, &ch) == EMSCRIPTEN_RESULT_SUCCESS
        && cw > 0 && ch > 0) {
        s.cfg.width = (uint32_t)cw; s.cfg.height = (uint32_t)ch;
        wgpuSurfaceConfigure(s.surf, &s.cfg);
    }
    return EM_FALSE;
}
#endif // __EMSCRIPTEN__

// One-time setup after the device exists: configure the surface, bring up ImGui, init every demo.
// Runs on both paths (native: right after acquire; web: on the first ready frame).
static void init_after_device(AppState& app)
{
    app.q    = app.queueObj.Get();
    app.surf = app.surface.Get();

    app.cfg = WGPUSurfaceConfiguration{
        .device      = app.dev,
        .format      = kSwapFormat,
        .usage       = WGPUTextureUsage_RenderAttachment,
        .width       = app.cfg.width,      // main() seeded these from the window/canvas
        .height      = app.cfg.height,
        .alphaMode   = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(app.surf, &app.cfg);

    imgui_layer_init(app.window, app.dev, kSwapFormat);

    // init every demo once (pipelines depend on formats, not per-frame size).
    DemoEnv initEnv{ app.dev, app.q, kSwapFormat, app.cfg.width, app.cfg.height, 0.0f, 0.0f, 0, app.camera };
    for (const Demo& d : g_demos) d.init(initEnv);

    app.startTime = app.prevTime = getTime();
}

// One frame: pump input, update the camera, declare + compile + execute + present the graph. Identical on
// native (called from a while loop) and web (called from the requestAnimationFrame callback).
static void frame(AppState& app)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL3_ProcessEvent(&e);
        if (e.type == SDL_EVENT_QUIT) app.running = false;
        else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE) app.running = false;
        else if (e.type == SDL_EVENT_KEY_DOWN &&
                 e.key.scancode >= SDL_SCANCODE_F1 && e.key.scancode < SDL_SCANCODE_F1 + kDemoCount) {
            app.active = (int)(e.key.scancode - SDL_SCANCODE_F1);
            std::printf("demo: %s\n", g_demos[app.active].name);
        }
        else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT
                 && !ImGui::GetIO().WantCaptureMouse) {     // don't start a look-drag on a UI click
            app.dragging = true;
            SDL_SetWindowRelativeMouseMode(app.window, true);   // capture + hide cursor while looking
        }
        else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP && e.button.button == SDL_BUTTON_LEFT) {
            app.dragging = false;
            SDL_SetWindowRelativeMouseMode(app.window, false);
        }
        else if (e.type == SDL_EVENT_MOUSE_MOTION && app.dragging) {
            app.camera.yaw   += e.motion.xrel * kMouseSens;
            app.camera.pitch -= e.motion.yrel * kMouseSens;
            const float lim = 1.5533f;                       // ~89 deg, avoid gimbal flip
            app.camera.pitch = app.camera.pitch > lim ? lim : (app.camera.pitch < -lim ? -lim : app.camera.pitch);
        }
        else if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {   // data1/data2 are device pixels; fires on resize AND DPI change
            app.cfg.width = (uint32_t)e.window.data1; app.cfg.height = (uint32_t)e.window.data2;
            wgpuSurfaceConfigure(app.surf, &app.cfg);
        }
    }

    // camera basis from yaw/pitch, then WASD fly-movement (frame-rate independent via dt).
    double now = getTime();
    float  dt  = (float)(now - app.prevTime);
    app.prevTime = now;
    app.camera.basis();
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        auto ks = SDL_GetKeyboardState(nullptr);   // const bool* (SDL3); index with scancodes
        float mv = kMoveSpeed * dt;
        float f = (ks[SDL_SCANCODE_W] ? mv : 0.0f) - (ks[SDL_SCANCODE_S] ? mv : 0.0f);
        float r = (ks[SDL_SCANCODE_D] ? mv : 0.0f) - (ks[SDL_SCANCODE_A] ? mv : 0.0f);
        for (int i = 0; i < 3; ++i) app.camera.pos[i] += app.camera.fwd[i] * f + app.camera.right[i] * r;
    }

    imgui_layer_begin_frame();   // NewFrame only; the DAG window is built after compile, Render() in end_frame

    ImGui::Begin("Demos");
    for (int i = 0; i < kDemoCount; ++i) {
        if (i) ImGui::SameLine();
        if (ImGui::RadioButton(g_demos[i].name, app.active == i)) app.active = i;   // F1..Fn also switch
    }
    ImGui::Separator();
    ImGui::Checkbox("alias transients", &app.enableAlias);
    if (app.gpuTimingAvailable) ImGui::Checkbox("gpu timings", &app.enableProfiling);
    g_demos[app.active].ui();
    ImGui::End();

    WGPUSurfaceTexture st{};
    wgpuSurfaceGetCurrentTexture(app.surf, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        wgpuSurfaceConfigure(app.surf, &app.cfg);   // surface went stale (resize/minimize) -> reconfigure, skip frame
        imgui_layer_end_frame();                    // balance begin_frame's NewFrame on the skipped frame
        return;
    }

    WGPUTextureViewDescriptor vd{
        .format = kSwapFormat, .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    WGPUTextureView view = wgpuTextureCreateView(st.texture, &vd);

    // ---- declare the whole graph for THIS frame (immediate mode) ----
    RenderGraph* rg = create_render_graph(app.allocator);   // resets the arena (pools persist in the allocator)
    auto swapchain = rg->importe_image("swapchain"_rid, view, { app.cfg.width, app.cfg.height, 1 });

    ++app.frame;   // counts frames that reach build (surface-stale skips don't); demos detect re-entry from gaps
    DemoEnv env{ app.dev, app.q, kSwapFormat, app.cfg.width, app.cfg.height, (float)(now - app.startTime), dt, app.frame, app.camera };
    g_demos[app.active].build(env, rg, swapchain);

    // ImGui overlay: last pass. Load keeps the rendered scene; the write to the imported swapchain
    // makes it a sink, and a WAW edge orders it after the demo's present.
    rg->add_pass("imgui"_rid, PassKind::Graphics,
        [&](PassBuilder& b) { b.color(swapchain, WGPULoadOp_Load, WGPUStoreOp_Store); },
        [](PassContext& ctx) { ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), ctx.render); });

    rg->compile(app.enableAlias);

    imgui_layer_draw_graph(rg);   // build the DAG window now the graph is compiled + realized
    imgui_layer_end_frame();      // ImGui::Render(); the "imgui" pass consumes the draw data at execute

    // reprint the execution order whenever the graph reshapes (demo switch, toggle, resize-driven mip
    // count). a pass-name signature diff catches it for ANY demo, no per-demo cooperation.
    std::string sig;
    for (PassNode* p = storage(rg)->m_passes; p; p = p->next) { sig.append(p->id.name.data, p->id.name.length); sig.push_back('|'); }
    sig += app.enableAlias ? "A1" : "A0";   // fold the aliasing toggle in so flipping it reprints the pool stats
    if (sig != app.lastSig) {
        app.lastSig = sig;
        std::printf("execution order:");
        for (PassNode* p = storage(rg)->m_passes; p; p = p->next) std::printf(" %.*s", (int)p->id.name.length, p->id.name.data);
        size_t tpTex = 0, tpBuf = 0;   // one pool, tagged by kind
        for (const auto& en : app.allocator->transient.entries) (en.isBuffer ? tpBuf : tpTex)++;
        std::printf("\ntransient pool: %zu textures, %zu buffers (%u created this frame)\n",
                    tpTex, tpBuf, app.allocator->transient.createdThisFrame);
        if (storage(rg)->m_slotCount) {   // phase-4 aliasing ran (enableAlias): logical transients -> physical slots
            uint32_t logical = 0;
            for (ResourceNode* r = storage(rg)->m_resouces; r; r = r->next)
                if (r->aliasSlot != ResourceNode::kNoSlot) ++logical;
            std::printf("aliasing: %u transients share %u physical slots\n", logical, storage(rg)->m_slotCount);
        }
    }

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(app.dev, nullptr);

    rg->execute(app.dev, enc, app.q, app.enableProfiling && app.gpuTimingAvailable);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(app.q, 1, &cmd);
    rg->collect_gpu_timings();   // kick the async timestamp read-back now the copy is submitted
#ifndef __EMSCRIPTEN__
    wgpuSurfacePresent(app.surf);   // web: emdawnwebgpu drives present from requestAnimationFrame -- calling
#endif                              // wgpuSurfacePresent there aborts the runtime, so skip it (cf. Renderer.cpp).

    wgpuTextureViewRelease(view);
    wgpuTextureRelease(st.texture);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    app.instance.ProcessEvents();   // pump device/error callbacks (and the async adapter/device ones on web)
}

int main()
{
    using namespace RG;
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: prints/errors show live (and survive a kill)

    static AppState app;   // static: on web, main() returns and the RAF loop keeps using &app

    // ---- window (SDL owns the OS window natively; the "#canvas" element on web) ----
    SDL_SetMainReady();
    if (!SDL_Init(SDL_INIT_VIDEO)) { std::printf("SDL_Init failed: %s\n", SDL_GetError()); return 1; }
    // Logical (point) size. On web the shell already sized #canvas via CSS and SDL adopts that, ignoring
    // these; on native it's the starting window size. HIGH_PIXEL_DENSITY is load-bearing: without it SDL
    // reports pixel-size == point-size, so ImGui's DisplayFramebufferScale stays 1 while the WebGPU surface
    // below is sized in device pixels -- under DPI scaling (dpr>1) the overlay then renders shrunk into a
    // corner and the mouse no longer lands on it.
    int initW = 1280, initH = 720;
    app.window = SDL_CreateWindow("RenderGraph sample", initW, initH,
                                  SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!app.window) { std::printf("SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }

    // Surface size is in DEVICE PIXELS, not points. ImGui derives DisplayFramebufferScale from the same
    // points/pixels pair, so overlay and cursor stay aligned at any DPI scale.
    int pxW = initW, pxH = initH;
    SDL_GetWindowSizeInPixels(app.window, &pxW, &pxH);
    app.cfg.width = (uint32_t)pxW; app.cfg.height = (uint32_t)pxH;

    wgpu::InstanceDescriptor instDesc{};
    app.instance = wgpu::CreateInstance(&instDesc);

    // single persistent arena: create_render_graph() resets it and bump-allocates a fresh graph each
    // frame. It also owns the resource pools (temporal/transient), so it outlives the frame loop.
    app.allocator = create_allocator();

#ifndef __EMSCRIPTEN__
    // ---- native: surface from the window (HWND glue) + synchronous device ----
    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(app.window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(app.window), SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND src{};
    src.hwnd = hwnd; src.hinstance = hinst;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &src;
    app.surface = app.instance.CreateSurface(&surfDesc);

    app.deviceObj = acquire_device(app.instance, app.surface);
    if (!app.deviceObj) { std::printf("no device, aborting\n"); return 1; }
    app.queueObj = app.deviceObj.GetQueue();
    app.dev = app.deviceObj.Get();
    app.gpuTimingAvailable = wgpuDeviceHasFeature(app.dev, WGPUFeatureName_TimestampQuery);

    init_after_device(app);
    app.ready = app.inited = true;

    while (app.running) frame(app);

    // ---- teardown ----
    for (const Demo& d : g_demos) d.shutdown();
    imgui_layer_shutdown();
    SDL_DestroyWindow(app.window);
    SDL_Quit();
    // ponytail: the GraphAllocator (arena + pools + null-default textures) is leaked at exit -- one-time, process reclaims it.
    return 0;
#else
    // ---- web: surface from the canvas + async device + requestAnimationFrame loop ----
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasSrc{};
    canvasSrc.selector = "#canvas";
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &canvasSrc;
    app.surface = app.instance.CreateSurface(&surfDesc);

    request_adapter_then_device(app);   // async; sets app.ready when the device lands

    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, &app, false, web_on_resize);

    // start the loop immediately; the callback pumps events, waits for the device, inits once, then draws.
    emscripten_set_main_loop_arg([](void* ud){
        auto& s = *static_cast<AppState*>(ud);
        s.instance.ProcessEvents();       // pump the async adapter/device callbacks (mirrors Framework.cpp)
        if (!s.ready) return;
        if (!s.inited) { init_after_device(s); s.inited = true; }
        frame(s);
    }, &app, 0, false);
    return 0;
#endif
}
