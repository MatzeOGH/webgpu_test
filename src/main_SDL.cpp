#define SDL_MAIN_USE_CALLBACKS
#include "Framework.h"
#include "Renderer.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <webgpu/webgpu_cpp.h>

struct AppState {
    SDL_Window* window = nullptr;
};

SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    init(argc, argv);

    auto* app = new AppState;
    *appstate = app;

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    app->window = SDL_CreateWindow("WebGPU App", kWidth, kHeight, 0);

    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(app->window),
                      SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(app->window),
                      SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND hwndSrc{};
    hwndSrc.hinstance = hinst;
    hwndSrc.hwnd      = hwnd;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &hwndSrc;
    webgpu_init(webgpu_instance().CreateSurface(&surfDesc));

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* /*appstate*/) {
    webgpu_tick();
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    auto* app = static_cast<AppState*>(appstate);
    switch (event->type) {
    case SDL_EVENT_QUIT:
        return SDL_APP_SUCCESS;
    case SDL_EVENT_KEY_DOWN:
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) {
            if (SDL_GetWindowRelativeMouseMode(app->window)) {
                SDL_SetWindowRelativeMouseMode(app->window, false);
                break;
            }
            return SDL_APP_SUCCESS;
        }
        if (event->key.scancode == SDL_SCANCODE_RETURN && (event->key.mod & SDL_KMOD_ALT)) {
            bool fs = SDL_GetWindowFlags(app->window) & SDL_WINDOW_FULLSCREEN;
            SDL_SetWindowFullscreen(app->window, !fs);
            break;
        }
        input_set_key(event->key.scancode, true);
        break;
    case SDL_EVENT_KEY_UP:
        input_set_key(event->key.scancode, false);
        break;
    case SDL_EVENT_MOUSE_MOTION:
        input_set_mouse_pos(event->motion.x, event->motion.y);
        if (SDL_GetWindowRelativeMouseMode(app->window))
            input_set_mouse_delta(event->motion.xrel, event->motion.yrel);
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (event->button.button == SDL_BUTTON_LEFT && !SDL_GetWindowRelativeMouseMode(app->window)) {
            SDL_SetWindowRelativeMouseMode(app->window, true);
            input_skip_mouse(1);
        }
        input_set_mouse_button(event->button.button - 1, true);
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event->button.button == SDL_BUTTON_LEFT)
            SDL_SetWindowRelativeMouseMode(app->window, false);
        input_set_mouse_button(event->button.button - 1, false);
        break;
    case SDL_EVENT_MOUSE_WHEEL: {
        float dy = event->wheel.y;
        if (event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED) dy = -dy;
        input_set_scroll(dy);
        break;
    }
    case SDL_EVENT_WINDOW_RESIZED:
        if (webgpu_ready())
            renderer_resize((uint32_t)event->window.data1, (uint32_t)event->window.data2);
        break;
    }
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult /*result*/) {
    auto* app = static_cast<AppState*>(appstate);
    SDL_DestroyWindow(app->window);
    delete app;
    shutdown();
}
