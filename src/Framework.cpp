#include "Framework.h"
#include "Renderer.h"
#include <cstdio>

// ---- Input -----------------------------------------------------------------

struct InputState {
    bool  keys[512]{};
    float mx = 0, my = 0;
    float mdx = 0, mdy = 0;
    bool  btns[3]{};
};

static InputState* s = nullptr;

// ---- WebGPU ----------------------------------------------------------------

static wgpu::Instance g_instance;
static bool           g_ready = false;

struct InitCtx {
    wgpu::Adapter adapter;
    wgpu::Surface surface;
};

static void on_device(wgpu::RequestDeviceStatus status, wgpu::Device device,
                      wgpu::StringView msg, InitCtx* ctx) {
    if (status != wgpu::RequestDeviceStatus::Success) {
        std::printf("RequestDevice failed: %.*s\n", (int)msg.length, msg.data);
        delete ctx; return;
    }
    wgpu::Queue queue = device.GetQueue();
    renderer_init(std::move(device), std::move(queue), std::move(ctx->surface));
    delete ctx;
    g_ready = true;
}

static void on_adapter(wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                       wgpu::StringView msg, InitCtx* ctx) {
    if (status != wgpu::RequestAdapterStatus::Success) {
        std::printf("RequestAdapter failed: %.*s\n", (int)msg.length, msg.data);
        delete ctx; return;
    }
    ctx->adapter = adapter;

    wgpu::DeviceDescriptor devDesc{};
    devDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason r, wgpu::StringView m) {
            std::printf("Device lost (%d): %.*s\n", (int)r, (int)m.length, m.data);
        });
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m) {
            std::printf("GPU error (%d): %.*s\n", (int)t, (int)m.length, m.data);
        });

    ctx->adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous,
                               on_device, ctx);
}

// ---- Lifecycle -------------------------------------------------------------

void init(int argc, char* argv[])
{
    s = new InputState();
    wgpu::InstanceDescriptor instDesc{};
    g_instance = wgpu::CreateInstance(&instDesc);
}

void shutdown()
{
    delete s;
    s = nullptr;
    g_instance = nullptr;
    g_ready = false;
}

// ---- WebGPU API ------------------------------------------------------------

wgpu::Instance webgpu_instance() { return g_instance; }

void webgpu_init(wgpu::Surface surface)
{
    auto* ctx = new InitCtx{{}, std::move(surface)};
    wgpu::RequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;
    g_instance.RequestAdapter(&adapterOpts, wgpu::CallbackMode::AllowSpontaneous,
                              on_adapter, ctx);
}

void webgpu_tick()
{
    g_instance.ProcessEvents();
    if (g_ready) renderer_frame();
    input_reset_frame();
}

bool webgpu_ready() { return g_ready; }

// ---- Input -----------------------------------------------------------------

void input_set_key(int sc, bool down)           { if (s && sc >= 0 && sc < 512) s->keys[sc] = down; }
void input_set_mouse_pos(float x, float y)      { if (s) { s->mx = x; s->my = y; } }
void input_set_mouse_delta(float dx, float dy)  { if (s) { s->mdx += dx; s->mdy += dy; } }
void input_set_mouse_button(int b, bool down)   { if (s && b >= 0 && b < 3) s->btns[b] = down; }

bool  input_key_down(int sc)     { return s && sc >= 0 && sc < 512 && s->keys[sc]; }
float input_mouse_x()            { return s ? s->mx  : 0.f; }
float input_mouse_y()            { return s ? s->my  : 0.f; }
float input_mouse_delta_x()      { return s ? s->mdx : 0.f; }
float input_mouse_delta_y()      { return s ? s->mdy : 0.f; }
bool  input_mouse_down(int b)    { return s && b >= 0 && b < 3 && s->btns[b]; }
void  input_reset_frame()        { if (s) { s->mdx = 0.f; s->mdy = 0.f; } }
