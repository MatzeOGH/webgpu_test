#include "Framework.h"
#include "Renderer.h"
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>
#include <cstring>

// ---- Touch input -----------------------------------------------------------

static int   g_cam_touch_id = -1;
static float g_touch_last_x = 0.f, g_touch_last_y = 0.f;

static EM_BOOL on_touchstart(int, const EmscriptenTouchEvent* e, void*) {
    if (g_cam_touch_id == -1 && e->numTouches > 0) {
        g_cam_touch_id = e->touches[0].identifier;
        g_touch_last_x = (float)e->touches[0].targetX;
        g_touch_last_y = (float)e->touches[0].targetY;
        input_set_mouse_button(0, true);
    }
    return EM_TRUE;
}

static EM_BOOL on_touchmove(int, const EmscriptenTouchEvent* e, void*) {
    for (int i = 0; i < e->numTouches; i++) {
        if (e->touches[i].identifier == g_cam_touch_id) {
            float dx = (float)e->touches[i].targetX - g_touch_last_x;
            float dy = (float)e->touches[i].targetY - g_touch_last_y;
            g_touch_last_x = (float)e->touches[i].targetX;
            g_touch_last_y = (float)e->touches[i].targetY;
            input_set_mouse_delta(dx, dy);
            break;
        }
    }
    return EM_TRUE;
}

// touchend: e->touches holds *remaining* touches (not the one that ended)
static EM_BOOL on_touchend(int, const EmscriptenTouchEvent* e, void*) {
    if (g_cam_touch_id == -1) return EM_TRUE;
    for (int i = 0; i < e->numTouches; i++)
        if (e->touches[i].identifier == g_cam_touch_id) return EM_TRUE;
    g_cam_touch_id = -1;
    input_set_mouse_button(0, false);
    return EM_TRUE;
}

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
    EmscriptenPointerlockChangeEvent ls{};
    if (emscripten_get_pointerlock_status(&ls) == EMSCRIPTEN_RESULT_SUCCESS && ls.isActive) {
        float dx = std::max(-100.f, std::min(100.f, (float)e->movementX));
        float dy = std::max(-100.f, std::min(100.f, (float)e->movementY));
        input_set_mouse_delta(dx, dy);
    }
    return EM_FALSE;
}

static EM_BOOL on_pointerlockchange(int, const EmscriptenPointerlockChangeEvent* e, void*) {
    if (e->isActive)
        input_skip_mouse(3);
    return EM_FALSE;
}

static EM_BOOL on_mousedown(int, const EmscriptenMouseEvent* e, void*) {
    input_set_mouse_button(e->button, true);
    if (e->button == 0) {
        EmscriptenPointerlockChangeEvent ls{};
        if (emscripten_get_pointerlock_status(&ls) != EMSCRIPTEN_RESULT_SUCCESS || !ls.isActive)
            emscripten_request_pointerlock("#canvas", false);
    }
    return EM_FALSE;
}

static EM_BOOL on_mouseup(int, const EmscriptenMouseEvent* e, void*) {
    input_set_mouse_button(e->button, false);
    if (e->button == 0)
        emscripten_exit_pointerlock();
    return EM_FALSE;
}

static EM_BOOL on_wheel(int, const EmscriptenWheelEvent* e, void*) {
    // deltaY > 0 = scroll down in browser; negate so positive = zoom in
    float delta = -(float)e->deltaY;
    // Normalize: pixel mode (~100px/click) → ~1 per click; line mode (~3/click) → use as-is
    if (e->deltaMode == 0) delta /= 100.f;
    input_set_scroll(delta);
    return EM_TRUE;
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

    emscripten_set_pointerlockchange_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, true, on_pointerlockchange);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, on_keydown);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW,   nullptr, true, on_keyup);
    emscripten_set_mousemove_callback("#canvas", nullptr, true, on_mousemove);
    emscripten_set_mousedown_callback("#canvas", nullptr, true, on_mousedown);
    emscripten_set_mouseup_callback(  "#canvas", nullptr, true, on_mouseup);
    emscripten_set_wheel_callback(    "#canvas", nullptr, true, on_wheel);
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, false, on_resize);
    emscripten_set_touchstart_callback( "#canvas", nullptr, true, on_touchstart);
    emscripten_set_touchmove_callback(  "#canvas", nullptr, true, on_touchmove);
    emscripten_set_touchend_callback(   "#canvas", nullptr, true, on_touchend);
    emscripten_set_touchcancel_callback("#canvas", nullptr, true, on_touchend);

    emscripten_set_main_loop_arg([](void*){ webgpu_tick(); }, nullptr, 0, false);
    return 0;
}
