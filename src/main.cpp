#include "Renderer.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cstdio>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------
struct InputState {
    bool  keys[SDL_SCANCODE_COUNT]{};
    float mouseX = 0, mouseY = 0;
    bool  mouseButtons[3]{};  // index 0=left 1=middle 2=right
};

static InputState  g_input;
static SDL_Window* g_window  = nullptr;
static bool        g_ready   = false;
static bool        g_running = true;

static void process_events() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:
            g_running = false;
            break;
        case SDL_EVENT_KEY_DOWN:
            g_input.keys[e.key.scancode] = true;
            break;
        case SDL_EVENT_KEY_UP:
            g_input.keys[e.key.scancode] = false;
            if (e.key.scancode == SDL_SCANCODE_ESCAPE) g_running = false;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            g_input.mouseX = e.motion.x;
            g_input.mouseY = e.motion.y;
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (e.button.button >= 1 && e.button.button <= 3)
                g_input.mouseButtons[e.button.button - 1] =
                    (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Unified surface creation + renderer kick-off
// ---------------------------------------------------------------------------
static void start_rendering(wgpu::Instance instance, wgpu::Device device, wgpu::Queue queue) {
#ifdef __EMSCRIPTEN__
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{};
    canvas.selector = "#canvas";
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &canvas;
#else
    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(g_window),
                      SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(g_window),
                      SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND hwndSrc{};
    hwndSrc.hinstance = hinst;
    hwndSrc.hwnd      = hwnd;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &hwndSrc;
#endif
    wgpu::Surface surface = instance.CreateSurface(&surfDesc);

    wgpu::SurfaceConfiguration cfg{};
    cfg.device = device;
    cfg.format = wgpu::TextureFormat::BGRA8Unorm;
    cfg.width  = kWidth;
    cfg.height = kHeight;
    surface.Configure(&cfg);

    renderer_init(std::move(device), std::move(queue), std::move(surface));

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop([]() {
        process_events();
        if (!g_running) { emscripten_cancel_main_loop(); return; }
        renderer_frame();
    }, 0, false);
#else
    g_ready = true;
#endif
}

// ---------------------------------------------------------------------------
// Unified adapter → device init
// ---------------------------------------------------------------------------
struct InitCtx {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
};

static void do_init(wgpu::Instance instance) {
    auto* ctx = new InitCtx{instance};

    wgpu::RequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;

    instance.RequestAdapter(
        &adapterOpts,
        wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
           wgpu::StringView msg, InitCtx* ctx) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::printf("RequestAdapter failed: %.*s\n",
                            static_cast<int>(msg.length), msg.data);
                delete ctx; return;
            }
            ctx->adapter = adapter;

            wgpu::DeviceDescriptor devDesc{};
            devDesc.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason r, wgpu::StringView m) {
                    std::printf("Device lost (%d): %.*s\n",
                                static_cast<int>(r), static_cast<int>(m.length), m.data);
                });
            devDesc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
                    std::printf("GPU error (%d): %.*s\n",
                                static_cast<int>(t), static_cast<int>(m.length), m.data);
                });

            ctx->adapter.RequestDevice(
                &devDesc,
                wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus status, wgpu::Device device,
                   wgpu::StringView msg, InitCtx* ctx) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        std::printf("RequestDevice failed: %.*s\n",
                                    static_cast<int>(msg.length), msg.data);
                        delete ctx; return;
                    }
                    wgpu::Device   dev  = std::move(device);
                    wgpu::Queue    q    = dev.GetQueue();
                    wgpu::Instance inst = ctx->instance;
                    delete ctx;
                    start_rendering(inst, std::move(dev), std::move(q));
                },
                ctx);
        },
        ctx);
}

// ---------------------------------------------------------------------------
// Entry point (SDL_main.h maps this to WinMain on Windows)
// ---------------------------------------------------------------------------
int main(int /*argc*/, char* /*argv*/[]) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    g_window = SDL_CreateWindow("WebGPU App", kWidth, kHeight, 0);

    wgpu::InstanceDescriptor instDesc{};
    wgpu::Instance instance = wgpu::CreateInstance(&instDesc);
    do_init(instance);

#ifdef __EMSCRIPTEN__
    // Main loop is registered inside start_rendering once the GPU device is ready.
#else
    while (g_running) {
        process_events();
        instance.ProcessEvents();
        if (g_ready) renderer_frame();
    }
    SDL_DestroyWindow(g_window);
    SDL_Quit();
#endif
    return 0;
}
