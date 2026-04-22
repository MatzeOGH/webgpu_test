#pragma once
#include <webgpu/webgpu_cpp.h>
#include <cstdint>

inline constexpr uint32_t kWidth  = 800;
inline constexpr uint32_t kHeight = 600;

void renderer_init(wgpu::Device device, wgpu::Queue queue, wgpu::Surface surface);
void renderer_frame();
