#include "Framework.h"
#include "Renderer.h"
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>
#include <cstring>

// Maps a browser KeyboardEvent.code string to a USB HID scancode (= SDL3 scancode).
static int browser_code_to_scancode(const char* code) {
    if (!code || !code[0]) return -1;
    // "KeyA".."KeyZ"  ->  4..29
    if (code[0]=='K' && code[1]=='e' && code[2]=='y' && code[3] && !code[4]) {
        int c = code[3] - 'A';
        if (c >= 0 && c < 26) return 4 + c;
    }
    // "Digit1".."Digit9" -> 30..38,  "Digit0" -> 39
    if (code[0]=='D' && code[1]=='i' && code[2]=='g' && code[3]=='i' &&
        code[4]=='t' && code[5] && !code[6]) {
        int c = code[5] - '0';
        return c == 0 ? 39 : 29 + c;
    }
    // "F1".."F12" -> 58..69
    if (code[0] == 'F' && code[1] >= '1' && code[1] <= '9') {
        int n = code[2] ? (code[1]-'0')*10 + (code[2]-'0') : code[1]-'0';
        if (n >= 1 && n <= 12 && (code[2] ? !code[3] : true)) return 57 + n;
    }
    if (!std::strcmp(code, "Escape"))       return 41;
    if (!std::strcmp(code, "Enter"))        return 40;
    if (!std::strcmp(code, "Space"))        return 44;
    if (!std::strcmp(code, "Tab"))          return 43;
    if (!std::strcmp(code, "Backspace"))    return 42;
    if (!std::strcmp(code, "Insert"))       return 73;
    if (!std::strcmp(code, "Home"))         return 74;
    if (!std::strcmp(code, "PageUp"))       return 75;
    if (!std::strcmp(code, "Delete"))       return 76;
    if (!std::strcmp(code, "End"))          return 77;
    if (!std::strcmp(code, "PageDown"))     return 78;
    if (!std::strcmp(code, "ArrowRight"))   return 79;
    if (!std::strcmp(code, "ArrowLeft"))    return 80;
    if (!std::strcmp(code, "ArrowDown"))    return 81;
    if (!std::strcmp(code, "ArrowUp"))      return 82;
    if (!std::strcmp(code, "ShiftLeft"))    return 225;
    if (!std::strcmp(code, "ShiftRight"))   return 229;
    if (!std::strcmp(code, "ControlLeft"))  return 224;
    if (!std::strcmp(code, "ControlRight")) return 228;
    if (!std::strcmp(code, "AltLeft"))      return 226;
    if (!std::strcmp(code, "AltRight"))     return 230;
    return -1;
}

static EM_BOOL on_keydown(int, const EmscriptenKeyboardEvent* e, void*) {
    input_set_key(browser_code_to_scancode(e->code), true);
    if (e->altKey && !std::strcmp(e->code, "Enter")) {
        EmscriptenFullscreenChangeEvent fse{};
        emscripten_get_fullscreen_status(&fse);
        if (fse.isFullscreen)
            emscripten_exit_fullscreen();
        else
            emscripten_request_fullscreen("#canvas", true);
    }
    return EM_TRUE;
}

static EM_BOOL on_keyup(int, const EmscriptenKeyboardEvent* e, void*) {
    input_set_key(browser_code_to_scancode(e->code), false);
    return EM_TRUE;
}

static EM_BOOL on_mousemove(int, const EmscriptenMouseEvent* e, void*) {
    input_set_mouse_pos((float)e->targetX, (float)e->targetY);
    input_set_mouse_delta((float)e->movementX, (float)e->movementY);
    return EM_FALSE;
}

static EM_BOOL on_mousedown(int, const EmscriptenMouseEvent* e, void*) {
    input_set_mouse_button(e->button, true);
    emscripten_request_pointerlock("#canvas", false);
    return EM_FALSE;
}

static EM_BOOL on_mouseup(int, const EmscriptenMouseEvent* e, void*) {
    input_set_mouse_button(e->button, false);
    return EM_FALSE;
}

static EM_BOOL on_resize(int, const EmscriptenUiEvent*, void*) {
    int w, h;
    emscripten_get_canvas_element_size("#canvas", &w, &h);
    if (w > 0 && h > 0 && webgpu_ready())
        renderer_resize((uint32_t)w, (uint32_t)h);
    return EM_FALSE;
}

int main() {
    init(0, nullptr);

    int cw, ch;
    emscripten_get_canvas_element_size("#canvas", &cw, &ch);
    if (cw > 0 && ch > 0)
        renderer_set_initial_size((uint32_t)cw, (uint32_t)ch);

    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{};
    canvas.selector = "#canvas";
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &canvas;
    webgpu_init(webgpu_instance().CreateSurface(&surfDesc));

    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, on_keydown);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW,   nullptr, true, on_keyup);
    emscripten_set_mousemove_callback("#canvas", nullptr, true, on_mousemove);
    emscripten_set_mousedown_callback("#canvas", nullptr, true, on_mousedown);
    emscripten_set_mouseup_callback(  "#canvas", nullptr, true, on_mouseup);
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, false, on_resize);

    emscripten_set_main_loop_arg([](void*){ webgpu_tick(); }, nullptr, 0, false);
    return 0;
}
