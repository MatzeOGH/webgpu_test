#pragma once
#include <webgpu/webgpu_cpp.h>

// Lifecycle
void init(int argc, char* argv[]);
void shutdown();

// WebGPU — call webgpu_init() after creating the platform surface
wgpu::Instance webgpu_instance();
void           webgpu_init(wgpu::Surface surface);
void           webgpu_tick();
bool           webgpu_ready();

// Input write — called by the platform layer (main.cpp)
void input_set_key(int scancode, bool down);
void input_set_mouse_pos(float x, float y);
void input_set_mouse_delta(float dx, float dy);
void input_set_mouse_button(int button, bool down);
void input_set_scroll(float delta);

// Input read — called by game / renderer code
bool  input_key_down(int scancode);
float input_mouse_x();
float input_mouse_y();
bool  input_mouse_down(int button);
float input_mouse_delta_x();
float input_mouse_delta_y();
float input_scroll_delta();

// Call once per frame after renderer has consumed input
void input_reset_frame();
// Skip the next n mouse delta events (use after enabling pointer lock / relative mode)
void input_skip_mouse(int n);

// Key constants — USB HID position codes, identical to SDL3 scancodes
// and to the values produced by browser KeyboardEvent.code on web builds.
namespace Key {
    constexpr int A = 4,  B = 5,  C = 6,  D = 7;
    constexpr int E = 8,  F = 9,  G = 10, H = 11;
    constexpr int I = 12, J = 13, K = 14, L = 15;
    constexpr int M = 16, N = 17, O = 18, P = 19;
    constexpr int Q = 20, R = 21, S = 22, T = 23;
    constexpr int U = 24, V = 25, W = 26, X = 27;
    constexpr int Y = 28, Z = 29;
    constexpr int Num1 = 30, Num2 = 31, Num3 = 32;
    constexpr int Num4 = 33, Num5 = 34, Num6 = 35;
    constexpr int Num7 = 36, Num8 = 37, Num9 = 38, Num0 = 39;
    constexpr int Enter = 40, Escape = 41, Backspace = 42, Tab = 43, Space = 44;
    constexpr int Insert = 73, Home = 74, PageUp   = 75;
    constexpr int Delete = 76, End  = 77, PageDown = 78;
    constexpr int Right = 79, Left = 80, Down = 81, Up = 82;
    constexpr int F1 = 58, F2 = 59, F3 = 60, F4  = 61;
    constexpr int F5 = 62, F6 = 63, F7 = 64, F8  = 65;
    constexpr int F9 = 66, F10 = 67, F11 = 68, F12 = 69;
    constexpr int LShift = 225, RShift = 229;
    constexpr int LCtrl  = 224, RCtrl  = 228;
    constexpr int LAlt   = 226, RAlt   = 230;
}
