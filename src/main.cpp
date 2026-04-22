// Web: no SDL3 — emscripten main-loop + WebGPU canvas directly.
// Native: SDL3 for window, events, and main loop.

#include "Framework.h"


#ifdef __EMSCRIPTEN__

#include "Renderer.h"
#include <emscripten/emscripten.h>
#include <webgpu/webgpu_cpp.h>
#include <cstdio>
#include <emscripten.h>

EM_JS(void, log_js, (const char* str), {
    console.log(UTF8ToString(str));
});
#define LOG(...)                                  \
    do {                                          \
        char buffer[256];                         \
        std::snprintf(buffer, sizeof(buffer), __VA_ARGS__); \
        log_js(buffer);                           \
    } while (0)

static wgpu::Instance g_instance;
static bool           g_ready = false;

static void frame_cb(void*) {
    g_instance.ProcessEvents();
    if (g_ready) renderer_frame();
}

struct InitCtx {
    wgpu::Adapter adapter;
};

static void on_device(wgpu::RequestDeviceStatus status, wgpu::Device device,
                      wgpu::StringView msg, InitCtx* ctx) {
    delete ctx;
    if (status != wgpu::RequestDeviceStatus::Success) {
        LOG("RequestDevice failed: %.*s\n", (int)msg.length, msg.data);
        return;
    }

    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{};
    canvas.selector = "#canvas";
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &canvas;
    wgpu::Surface surface = g_instance.CreateSurface(&surfDesc);

    wgpu::SurfaceConfiguration cfg{};
    cfg.device = device;
    cfg.format = wgpu::TextureFormat::BGRA8Unorm;
    cfg.width  = kWidth;
    cfg.height = kHeight;
    surface.Configure(&cfg);

    wgpu::Queue queue = device.GetQueue();
    renderer_init(std::move(device), std::move(queue), std::move(surface));
    g_ready = true;
}

static void on_adapter(wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                       wgpu::StringView msg, InitCtx* ctx) {
    if (status != wgpu::RequestAdapterStatus::Success) {
        LOG("RequestAdapter failed: %.*s\n", (int)msg.length, msg.data);
        delete ctx; return;
    }
    ctx->adapter = adapter;

    wgpu::DeviceDescriptor devDesc{};
    devDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason r, wgpu::StringView m) {
            LOG("Device lost (%d): %.*s\n", (int)r, (int)m.length, m.data);
        });
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
            LOG("GPU error (%d): %.*s\n", (int)t, (int)m.length, m.data);
        });

    ctx->adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous,
                               on_device, ctx);
}

int main() {
    wgpu::InstanceDescriptor instDesc{};
    g_instance = wgpu::CreateInstance(&instDesc);

    auto* ctx = new InitCtx{};
    wgpu::RequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;
    g_instance.RequestAdapter(&adapterOpts, wgpu::CallbackMode::AllowSpontaneous,
                              on_adapter, ctx);
    LOG("finished %s", "WebGPU");
    emscripten_set_main_loop_arg(frame_cb, nullptr, 0, false);
    return 0;
}

// ---------------------------------------------------------------------------
#else // Native Windows — SDL3
// ---------------------------------------------------------------------------

#define SDL_MAIN_USE_CALLBACKS
#include "Renderer.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdio>
#define LOG(...) std::printf(__VA_ARGS__)

struct AppState {
    SDL_Window*    window = nullptr;
    wgpu::Instance instance;
    bool           ready  = false;

    bool  keys[SDL_SCANCODE_COUNT]{};
    float mouseX = 0, mouseY = 0;
    bool  mouseButtons[3]{};  // 0=left 1=middle 2=right
};

struct InitCtx {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    AppState*      app;
};

static void start_rendering(AppState* app, wgpu::Device device, wgpu::Queue queue) {
    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(app->window),
                      SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(app->window),
                      SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND hwndSrc{};
    hwndSrc.hinstance = hinst;
    hwndSrc.hwnd      = hwnd;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &hwndSrc;

    wgpu::Surface surface = app->instance.CreateSurface(&surfDesc);

    wgpu::SurfaceConfiguration cfg{};
    cfg.device = device;
    cfg.format = wgpu::TextureFormat::BGRA8Unorm;
    cfg.width  = kWidth;
    cfg.height = kHeight;
    surface.Configure(&cfg);

    renderer_init(std::move(device), std::move(queue), std::move(surface));
    app->ready = true;
}

static void do_init(AppState* app) {
    auto* ctx = new InitCtx{app->instance, {}, app};

    wgpu::RequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;

    app->instance.RequestAdapter(
        &adapterOpts,
        wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
           wgpu::StringView msg, InitCtx* ctx) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                LOG("RequestAdapter failed: %.*s\n",
                            (int)msg.length, msg.data);
                delete ctx; return;
            }
            ctx->adapter = adapter;

            wgpu::DeviceDescriptor devDesc{};
            devDesc.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason r, wgpu::StringView m) {
                    LOG("Device lost (%d): %.*s\n",
                                (int)r, (int)m.length, m.data);
                });
            devDesc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
                    LOG("GPU error (%d): %.*s\n",
                                (int)t, (int)m.length, m.data);
                });

            ctx->adapter.RequestDevice(
                &devDesc,
                wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus status, wgpu::Device device,
                   wgpu::StringView msg, InitCtx* ctx) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        LOG("RequestDevice failed: %.*s\n",
                                    (int)msg.length, msg.data);
                        delete ctx; return;
                    }
                    wgpu::Device dev = std::move(device);
                    wgpu::Queue  q   = dev.GetQueue();
                    AppState*    app = ctx->app;
                    delete ctx;
                    start_rendering(app, std::move(dev), std::move(q));
                },
                ctx);
        },
        ctx);
}

SDL_AppResult SDL_AppInit(void** appstate, int /*argc*/, char* /*argv*/[]) {
    auto* app = new AppState;
    *appstate = app;

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    app->window = SDL_CreateWindow("WebGPU App", kWidth, kHeight, 0);

    wgpu::InstanceDescriptor instDesc{};
    app->instance = wgpu::CreateInstance(&instDesc);
    do_init(app);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    auto* app = static_cast<AppState*>(appstate);
    app->instance.ProcessEvents();
    if (app->ready) renderer_frame();
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    auto* app = static_cast<AppState*>(appstate);
    switch (event->type) {
    case SDL_EVENT_QUIT:
        return SDL_APP_SUCCESS;
    case SDL_EVENT_KEY_DOWN:
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) return SDL_APP_SUCCESS;
        app->keys[event->key.scancode] = true;
        break;
    case SDL_EVENT_KEY_UP:
        app->keys[event->key.scancode] = false;
        break;
    case SDL_EVENT_MOUSE_MOTION:
        app->mouseX = event->motion.x;
        app->mouseY = event->motion.y;
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event->button.button >= 1 && event->button.button <= 3)
            app->mouseButtons[event->button.button - 1] =
                (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        break;
    }
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult /*result*/) {
    auto* app = static_cast<AppState*>(appstate);
    SDL_DestroyWindow(app->window);
    delete app;
}

#endif // __EMSCRIPTEN__
