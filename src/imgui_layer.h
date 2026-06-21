#pragma once
// Dear ImGui bring-up for the RenderGraph sample: SDL3 platform + WebGPU(Dawn) renderer backends.
// #included once into the single TU (RenderGraph_main.cpp) -- keeps the imgui plumbing out of main(),
// which only calls these three and adds the "imgui" graph pass. Uses ImGui/SDL/WGPU types only.
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_wgpu.h"

static void imgui_layer_init(SDL_Window* window, WGPUDevice dev, WGPUTextureFormat swapFormat)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;                   // no imgui.ini for a smoke test
    ImGui_ImplSDL3_InitForOther(window);
    ImGui_ImplWGPU_InitInfo init{};
    init.Device             = dev;
    init.NumFramesInFlight  = 3;
    init.RenderTargetFormat = swapFormat;                  // BGRA8Unorm
    init.DepthStencilFormat = WGPUTextureFormat_Undefined; // overlay only, no depth
    ImGui_ImplWGPU_Init(&init);
}

// Build the frame's UI and finalize draw data. Runs every frame (even skipped ones) so the
// NewFrame/Render pair stays balanced; the "imgui" pass consumes ImGui::GetDrawData() at execute.
static void imgui_layer_new_frame()
{
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("RenderGraph");
    ImGui::Text("Dear ImGui %s", IMGUI_VERSION);
    ImGui::Text("%.1f FPS  (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Render();
}

static void imgui_layer_shutdown()
{
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}
