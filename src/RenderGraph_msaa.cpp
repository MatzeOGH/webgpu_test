// ===== MSAA (multisample anti-aliasing) demo ================================================
// A rotating pinwheel of solid wedges -- lots of edges at every angle, the classic place aliasing shows.
// Toggle 4x MSAA from the menu and watch the jaggies vanish. A magnifying loupe follows the mouse and
// point-samples (unfiltered) the final resolved image, so you can park it on an edge and see the per-pixel
// difference MSAA makes. When MSAA is on you pick how the 4 samples collapse to one: the in-pass
// resolveTarget (the usual fast path) or an alternative compute pass that averages the samples by hand.
//
// What the graph does each frame:
//   MSAA off          : scene -> resolved (1 sample)                         -> present(+loupe) -> swapchain
//   MSAA, in-pass     : scene -> msaaColor(4), resolve into resolved         -> present -> swapchain
//   MSAA, compute     : scene -> msaaColor(4) stored; resolve.cs -> resolved -> present -> swapchain
// #included after RenderGraph_demo.h, registered in RG_DEMO_LIST.

namespace msaa_demo {

constexpr WGPUTextureFormat kColorFormat = WGPUTextureFormat_RGBA8Unorm;  // multisample + resolve + storage capable

// one UBO shared by the scene (time/aspect) and present (mouse/loupe) shaders. vec4-aligned for std140.
struct MsaaUBO {
    float    resolution[2];
    float    mouse[2];
    float    time;
    float    aspect;
    float    zoom;
    float    radius;
    uint32_t magnifierOn;
    uint32_t pad[3];
};
static_assert(sizeof(MsaaUBO) == 48, "MsaaUBO must stay vec4-aligned to match the WGSL U layout");

static const char* kUStruct = R"(
struct U {
    resolution  : vec2f,
    mouse       : vec2f,
    time        : f32,
    aspect      : f32,
    zoom        : f32,
    radius      : f32,
    magnifierOn : u32,
};
)";

// scene: build a pinwheel straight from vertex_index (no vertex buffers) -- wedge `tri` spans a gap-leaving
// slice of the circle, spun by time and squished by aspect so it stays round in any window.
static const char* kSceneSrc = R"(
@group(0) @binding(0) var<uniform> u : U;

fn hue(h : f32) -> vec3f {
    let k = fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0));
    return clamp(abs(k * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

struct VsOut { @builtin(position) pos : vec4f, @location(0) col : vec3f };
@vertex fn vs(@builtin(vertex_index) vid : u32) -> VsOut {
    let N    = 12u;
    let tau  = 6.2831853;
    let tri  = vid / 3u;
    let corner = vid % 3u;
    let base  = f32(tri) * tau / f32(N) + u.time * 0.5;
    let wedge = tau / f32(N) * 0.6;                 // < full slice -> gaps between blades = more edges
    let R = 0.9;
    var local = vec2f(0.0, 0.0);                    // corner 0 is the hub
    if (corner == 1u) { local = vec2f(cos(base),         sin(base))         * R; }
    if (corner == 2u) { local = vec2f(cos(base + wedge), sin(base + wedge)) * R; }
    local.x = local.x / u.aspect;
    var o : VsOut;
    o.pos = vec4f(local, 0.0, 1.0);
    o.col = hue(f32(tri) / f32(N));
    return o;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4f { return vec4f(in.col, 1.0); }
)";

// present: blit the resolved image, and inside the loupe radius remap the lookup to a zoomed, point-sampled
// (textureLoad = unfiltered) window around the cursor. a thin white ring marks the loupe edge.
static const char* kPresentFs = R"(
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var img : texture_2d<f32>;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px  = in.pos.xy;                            // framebuffer pixel, top-left origin
    let d   = distance(px, u.mouse);
    var src = px;
    if (u.magnifierOn != 0u && d < u.radius) {
        src = u.mouse + (px - u.mouse) / u.zoom;
    }
    let dim = vec2i(textureDimensions(img));
    let ip  = clamp(vec2i(src), vec2i(0), dim - vec2i(1));
    var col = textureLoad(img, ip, 0).rgb;
    if (u.magnifierOn != 0u) {
        let ring = smoothstep(u.radius - 2.0, u.radius, d) - smoothstep(u.radius, u.radius + 2.0, d);
        col = mix(col, vec3f(1.0), ring);
    }
    return vec4f(col, 1.0);
}
)";

// compute resolve: average the 4 samples by hand into a single-sample storage image -- the long way round
// that the in-pass resolveTarget does for free, here so both can be compared.
static const char* kResolveCs = R"(
@group(0) @binding(0) var ms  : texture_multisampled_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let dim = vec2i(textureDimensions(dst));
    let p   = vec2i(i32(gid.x), i32(gid.y));
    if (p.x >= dim.x || p.y >= dim.y) { return; }
    let n = i32(textureNumSamples(ms));
    var acc = vec4f(0.0);
    for (var i = 0; i < n; i = i + 1) { acc = acc + textureLoad(ms, p, i); }
    textureStore(dst, p, acc / f32(n));
}
)";

// ---- state (created once in init) ----
static WGPURenderPipeline  scenePipe1 = nullptr;   // scene, 1 sample (MSAA off)
static WGPURenderPipeline  scenePipe4 = nullptr;   // scene, 4 samples (MSAA on)
static WGPURenderPipeline  presentPipe = nullptr;  // fullscreen blit + loupe
static WGPUComputePipeline resolvePipe = nullptr;  // alternative MSAA resolve
static WGPUBuffer          uboBuf = nullptr;

// ---- settings (ImGui) ----
static bool  msaaOn        = true;
static bool  computeResolve = false;   // false = in-pass resolveTarget, true = compute pass
static bool  magnifierOn   = true;
static bool  animate       = true;
static float animTime      = 0.0f;     // spin clock, advanced by dt only while animating -> pause freezes in place
static float zoom          = 8.0f;
static float radius        = 140.0f;

// one wedge = 3 verts; the pinwheel is 12 of them. keep in sync with N in kSceneSrc.
constexpr uint32_t kSceneVerts = 12u * 3u;

// scene draw, shared by the 1- and 4-sample paths -- same bind group, different pipeline.
static void draw_scene(PassContext& ctx, WGPUDevice dev, ResourceHandle ubo, WGPURenderPipeline pipe)
{
    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(pipe, 0);
    WGPUBindGroupEntry e[1] = {
        { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(MsaaUBO) },
    };
    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
    wgpuRenderPassEncoderSetPipeline(ctx.render, pipe);
    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderDraw(ctx.render, kSceneVerts, 1, 0, 0);
    wgpuBindGroupRelease(bg);
    wgpuBindGroupLayoutRelease(l);
}

// scene pipeline differing only in sample count; both render to kColorFormat.
static WGPURenderPipeline make_scene_pipe(WGPUDevice dev, WGPUShaderModule sm, uint32_t samples)
{
    WGPUColorTargetState ct{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState frag{ .module = sm, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &ct };
    WGPURenderPipelineDescriptor pd{
        .label       = WEBGPU_STR("msaa.scene pipeline"),
        .vertex      = { .module = sm, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = samples, .mask = ~0u },
        .fragment    = &frag,
    };
    return wgpuDeviceCreateRenderPipeline(dev, &pd);
}

} // namespace msaa_demo

static void msaa_init(const DemoEnv& env)
{
    using namespace msaa_demo;
    WGPUDevice dev = env.device;

    WGPUShaderModule sceneSM = make_shader(dev, std::string(kUStruct) + kSceneSrc);
    scenePipe1 = make_scene_pipe(dev, sceneSM, 1);
    scenePipe4 = make_scene_pipe(dev, sceneSM, 4);
    wgpuShaderModuleRelease(sceneSM);

    WGPUShaderModule presentSM = make_shader(dev, std::string(kUStruct) + kVS + kPresentFs);
    WGPUColorTargetState swapCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState presentFrag{ .module = presentSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &swapCT };
    WGPURenderPipelineDescriptor presentPD{
        .label       = WEBGPU_STR("msaa.present pipeline"),
        .vertex      = { .module = presentSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &presentFrag,
    };
    presentPipe = wgpuDeviceCreateRenderPipeline(dev, &presentPD);
    wgpuShaderModuleRelease(presentSM);

    WGPUShaderModule resolveSM = make_shader(dev, kResolveCs);
    WGPUComputePipelineDescriptor resolvePD{
        .label   = WEBGPU_STR("msaa.resolve pipeline"),
        .compute = { .module = resolveSM, .entryPoint = WEBGPU_STR("main") },
    };
    resolvePipe = wgpuDeviceCreateComputePipeline(dev, &resolvePD);
    wgpuShaderModuleRelease(resolveSM);

    WGPUBufferDescriptor bd{ .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(MsaaUBO) };
    uboBuf = wgpuDeviceCreateBuffer(dev, &bd);
}

static void msaa_shutdown()
{
    using namespace msaa_demo;
    wgpuBufferRelease(uboBuf);
    wgpuComputePipelineRelease(resolvePipe);
    wgpuRenderPipelineRelease(presentPipe);
    wgpuRenderPipelineRelease(scenePipe4);
    wgpuRenderPipelineRelease(scenePipe1);
}

static void msaa_build(const DemoEnv& env, RenderGraph* rg, ResourceHandle swapchain)
{
    using namespace msaa_demo;
    WGPUDevice dev = env.device;

    if (animate) animTime += env.dt;

    ImVec2 m = ImGui::GetMousePos();
    MsaaUBO u{};
    u.resolution[0] = (float)env.width;  u.resolution[1] = (float)env.height;
    u.mouse[0]      = m.x;               u.mouse[1]      = m.y;
    u.time          = animTime;
    u.aspect        = (float)env.width / (float)env.height;
    u.zoom          = zoom;
    u.radius        = radius;
    u.magnifierOn   = magnifierOn ? 1u : 0u;
    wgpuQueueWriteBuffer(env.queue, uboBuf, 0, &u, sizeof u);
    ResourceHandle ubo = rg->import_buffer("msaa.ubo"_rid, uboBuf);

    // the final single-sample image the loupe reads. usage adapts to the path (render/resolve target vs
    // compute storage) from the accesses below.
    auto resolved = rg->create_image("msaa.resolved"_rid, {
        .dimension = WGPUTextureDimension_2D, .format = kColorFormat,
        .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
    });

    const WGPUColor clear{ 0.05, 0.06, 0.09, 1.0 };

    if (msaaOn) {
        auto msaaColor = rg->create_image("msaa.color"_rid, {
            .dimension = WGPUTextureDimension_2D, .format = kColorFormat,
            .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
            .sampleCount = 4,
        });

        if (!computeResolve) {
            // in-pass: draw 4x and let the render pass resolve straight into `resolved`. Discard the MS
            // samples afterwards -- only the resolved single-sample result is read.
            rg->add_pass("msaa.scene"_rid, PassKind::Graphics,
                [&](PassBuilder& b) {
                    b.uniform(ubo);
                    b.color(msaaColor, WGPULoadOp_Clear, WGPUStoreOp_Discard, clear);
                    b.resolve(resolved);
                },
                [dev, ubo](PassContext& ctx) { draw_scene(ctx, dev, ubo, scenePipe4); });
        } else {
            // compute: keep the MS samples (Store), then average them in a compute pass.
            rg->add_pass("msaa.scene"_rid, PassKind::Graphics,
                [&](PassBuilder& b) {
                    b.uniform(ubo);
                    b.color(msaaColor, WGPULoadOp_Clear, WGPUStoreOp_Store, clear);
                },
                [dev, ubo](PassContext& ctx) { draw_scene(ctx, dev, ubo, scenePipe4); });

            rg->add_pass("msaa.resolve"_rid, PassKind::Compute,
                [&](PassBuilder& b) {
                    b.sampled(msaaColor);          // multisampled texture, read per-sample in the shader
                    b.storage_write(resolved);
                },
                [dev, msaaColor, resolved](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(resolvePipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(msaaColor) },
                        { .binding = 1, .textureView = ctx.view(resolved) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuComputePassEncoderSetPipeline(ctx.compute, resolvePipe);
                    wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
                    WGPUExtent3D rs = ctx.texture_size(resolved);
                    wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, (rs.width + 7) / 8, (rs.height + 7) / 8, 1);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        }
    } else {
        // no MSAA: render the pinwheel straight into the single-sample image.
        rg->add_pass("msaa.scene"_rid, PassKind::Graphics,
            [&](PassBuilder& b) {
                b.uniform(ubo);
                b.color(resolved, WGPULoadOp_Clear, WGPUStoreOp_Store, clear);
            },
            [dev, ubo](PassContext& ctx) { draw_scene(ctx, dev, ubo, scenePipe1); });
    }

    // present + loupe: sample the resolved image to the swapchain. ImGui overlays on top (Load) afterwards.
    rg->add_pass("msaa.present"_rid, PassKind::Graphics,
        [&](PassBuilder& b) {
            b.uniform(ubo);
            b.sampled(resolved);
            b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, resolved](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(MsaaUBO) },
                { .binding = 1, .textureView = ctx.view(resolved) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, presentPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
}

static void msaa_ui()
{
    using namespace msaa_demo;
    if (ImGui::Button(animate ? "Pause" : "Play")) animate = !animate;
    ImGui::SameLine();
    ImGui::TextDisabled("spin");
    ImGui::Separator();
    ImGui::Checkbox("4x MSAA", &msaaOn);
    if (!msaaOn) ImGui::BeginDisabled();
    int mode = computeResolve ? 1 : 0;
    const char* modes[] = { "In-pass resolve", "Compute resolve" };
    if (ImGui::Combo("Resolve", &mode, modes, 2)) computeResolve = (mode == 1);
    if (!msaaOn) ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::Checkbox("Magnifier", &magnifierOn);
    ImGui::SliderFloat("Zoom",   &zoom,   2.0f, 20.0f);
    ImGui::SliderFloat("Radius", &radius, 40.0f, 320.0f);
    ImGui::TextDisabled("%s, resolve: %s -- park the loupe on an edge",
                        msaaOn ? "4 samples" : "1 sample",
                        msaaOn ? (computeResolve ? "compute" : "in-pass") : "n/a");
}
