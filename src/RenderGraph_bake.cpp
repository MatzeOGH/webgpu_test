// ===== Bake (precompute-once, re-bake on change) demo =======================================
// Demonstrates create_persistent_image + PassBuilder::initialize(target, hash): a procedural environment
// is baked into a persistent texture and then sampled every frame. The bake pass runs ONLY when the result
// is stale -- the first frame (target unrealized), after the pool evicts it, or when the settings `hash`
// changes (drag the sun, switch quality). A steady setting bakes once; nothing re-bakes per frame.
// Watch the printed execution order: it shows "bake.fill bake.show" on frames that (re)bake, "bake.show"
// alone once settled. This is the IBL / precompute pattern with live invalidation. #included after
// RenderGraph_demo.h, registered F4.

namespace bake_demo {

constexpr WGPUTextureFormat kEnvFormat = WGPUTextureFormat_RGBA8Unorm;   // renderable + filterable in core
constexpr uint32_t kEnvW = 1024, kEnvH = 512;   // lat-long panorama, baked on demand (Absolute -> resize-stable)

// bake fragment: a procedural sky panorama driven by the settings UBO -- sun position + warmth from
// `sunAngle`, cloud detail from `octaves` (quality). stands in for a real (costly) IBL bake; the point is
// that it runs only when those settings change, not every frame.
static const char* kBakeFs = R"(
struct Bake { sunAngle : f32, octaves : f32, pad0 : f32, pad1 : f32 };
@group(0) @binding(0) var<uniform> bk : Bake;

fn h2(p : vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn vnoise(p : vec2f) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h2(i), h2(i + vec2f(1.0, 0.0)), u.x),
               mix(h2(i + vec2f(0.0, 1.0)), h2(i + vec2f(1.0, 1.0)), u.x), u.y);
}
fn fbm(p : vec2f, oct : i32) -> f32 {
    var v = 0.0; var a = 0.5; var q = p;
    for (var i = 0; i < oct; i = i + 1) { v = v + a * vnoise(q); q = q * 2.0; a = a * 0.5; }
    return v;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let uv = in.ndc * 0.5 + 0.5;                                  // [0,1]^2, y up
    let noon = 1.0 - abs(bk.sunAngle - 0.5) * 2.0;               // 1 at midday, 0 toward the edges (dawn/dusk)
    let zenith  = mix(vec3f(0.42, 0.18, 0.12), vec3f(0.13, 0.30, 0.72), noon);
    let horizon = mix(vec3f(0.96, 0.55, 0.28), vec3f(0.66, 0.74, 0.88), noon);
    var col = mix(horizon, zenith, clamp(uv.y, 0.0, 1.0));
    let c = fbm(uv * vec2f(10.0, 5.0) + vec2f(bk.sunAngle * 3.0, 0.0), i32(bk.octaves));  // clouds
    col = mix(col, vec3f(1.0, 0.98, 0.95), smoothstep(0.55, 0.95, c) * 0.7 * smoothstep(0.30, 0.70, uv.y));
    let sunPos = vec2f(bk.sunAngle, 0.34 + noon * 0.46);                                  // sun rides higher at noon
    col += vec3f(1.0, 0.92, 0.75) * smoothstep(0.09, 0.0, distance(uv * vec2f(2.0, 1.0), sunPos * vec2f(2.0, 1.0)));
    return vec4f(col, 1.0);
}
)";

// present fragment: sample the baked env to the swapchain, panning with camera yaw so it reads like looking
// around a precomputed panorama -- every frame samples the SAME baked texture.
static const char* kShowFs = R"(
@group(0) @binding(0) var envTex : texture_2d<f32>;
@group(0) @binding(1) var envSmp : sampler;
@group(0) @binding(2) var<uniform> ctl : vec4f;   // x = horizontal pan, in turns
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    var uv = in.ndc * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;                       // texture v is top-down
    uv.x = fract(uv.x + ctl.x);
    return textureSampleLevel(envTex, envSmp, uv, 0.0);
}
)";

// ---- state (created once in init, lives across frames) ----
static WGPURenderPipeline bakePipe = nullptr;
static WGPURenderPipeline showPipe = nullptr;
static WGPUSampler        sampler  = nullptr;
static WGPUBuffer         paramBuf = nullptr;   // bake settings (sunAngle, octaves), imported each frame
static WGPUBuffer         ctlBuf   = nullptr;   // present pan, imported each frame

// ---- settings (ImGui) -- their hash gates the re-bake ----
static float sunAngle = 0.70f;   // [0,1] horizontal sun position across the panorama
static int   quality  = 1;       // 0 = Low (2 cloud octaves), 1 = High (5)

} // namespace bake_demo

static void bake_init(const DemoEnv& env)
{
    using namespace bake_demo;
    WGPUDevice dev = env.device;

    WGPUShaderModule bakeSM = make_shader(dev, std::string(kVS) + kBakeFs);
    WGPUColorTargetState envCT{ .format = kEnvFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bakeFrag{ .module = bakeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &envCT };
    WGPURenderPipelineDescriptor bakePD{
        .label       = WEBGPU_STR("bake.fill pipeline"),
        .vertex      = { .module = bakeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &bakeFrag,
    };
    bakePipe = wgpuDeviceCreateRenderPipeline(dev, &bakePD);
    wgpuShaderModuleRelease(bakeSM);

    WGPUShaderModule showSM = make_shader(dev, std::string(kVS) + kShowFs);
    WGPUColorTargetState swapCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState showFrag{ .module = showSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &swapCT };
    WGPURenderPipelineDescriptor showPD{
        .label       = WEBGPU_STR("bake.show pipeline"),
        .vertex      = { .module = showSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &showFrag,
    };
    showPipe = wgpuDeviceCreateRenderPipeline(dev, &showPD);
    wgpuShaderModuleRelease(showSM);

    WGPUSamplerDescriptor sd{
        .addressModeU = WGPUAddressMode_Repeat, .addressModeV = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear, .minFilter = WGPUFilterMode_Linear, .maxAnisotropy = 1,
    };
    sampler = wgpuDeviceCreateSampler(dev, &sd);

    WGPUBufferDescriptor bd{ .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = 16 };
    paramBuf = wgpuDeviceCreateBuffer(dev, &bd);
    ctlBuf   = wgpuDeviceCreateBuffer(dev, &bd);
}

static void bake_shutdown()
{
    using namespace bake_demo;
    wgpuBufferRelease(ctlBuf);
    wgpuBufferRelease(paramBuf);
    wgpuSamplerRelease(sampler);
    wgpuRenderPipelineRelease(showPipe);
    wgpuRenderPipelineRelease(bakePipe);
}

static void bake_build(const DemoEnv& env, RenderGraph* rg, ResourceHandle swapchain)
{
    using namespace bake_demo;
    WGPUDevice dev = env.device;

    // settings hash: the bake re-runs whenever this changes. quantize the float so slider jitter past the
    // texel-visible threshold is what re-bakes (and pack quality into the high bits -> collision-free here).
    uint64_t hash = (uint64_t)(uint32_t)(sunAngle * 4096.0f) | ((uint64_t)(uint32_t)quality << 32);

    // bake settings UBO (read by the bake shader). written every frame so it's current whenever a bake runs.
    float params[4] = { sunAngle, (quality == 0 ? 2.0f : 5.0f), 0.0f, 0.0f };
    wgpuQueueWriteBuffer(env.queue, paramBuf, 0, params, sizeof params);
    ResourceHandle bparams = rg->import_buffer("bake.params"_rid, paramBuf);

    // present pan from camera yaw (turns). the env itself is unaffected -- only the lookup pans.
    float ctl[4] = { env.camera.yaw * 0.1591549f, 0.0f, 0.0f, 0.0f };
    wgpuQueueWriteBuffer(env.queue, ctlBuf, 0, ctl, sizeof ctl);
    ResourceHandle ubo = rg->import_buffer("bake.ctl"_rid, ctlBuf);

    // the baked environment: one persistent texture, Absolute-sized so a window resize never recreates it.
    // declared every frame (to read it + keep the pool entry alive), filled only by the initialize() pass.
    auto envTex = rg->create_persistent_image("bake.env"_rid, {
        .dimension = WGPUTextureDimension_2D, .format = kEnvFormat,
        .absolute  = { kEnvW, kEnvH, 1 },
    });

    // bake: fill the env from the current settings. initialize(envTex, hash) gates it -> it runs only while
    // bake.env is unrealized OR `hash` differs from the hash last baked in, then compile() culls it.
    rg->add_pass("bake.fill"_rid, PassKind::Graphics,
        [&](PassBuilder& b) {
            b.uniform(bparams);
            b.color(envTex, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
            b.initialize(envTex, hash);
        },
        [dev, bparams](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bakePipe, 0);
            WGPUBindGroupEntry e[1] = {
                { .binding = 0, .buffer = ctx.buffer(bparams), .offset = 0, .size = 16 },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, bakePipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });

    // show: sample the baked env to the swapchain, every frame. on frames the bake is culled, bake.env has
    // no in-graph writer -- legal, because a persistent resource is external (its value is from a prior frame).
    rg->add_pass("bake.show"_rid, PassKind::Graphics,
        [&](PassBuilder& b) {
            b.uniform(ubo);
            b.sampled(envTex);
            b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, envTex](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(showPipe, 0);
            WGPUBindGroupEntry e[3] = {
                { .binding = 0, .textureView = ctx.view(envTex) },
                { .binding = 1, .sampler = sampler },
                { .binding = 2, .buffer = ctx.buffer(ubo), .offset = 0, .size = 16 },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, showPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
}

static void bake_ui()
{
    using namespace bake_demo;
    ImGui::Text("env %ux%u -- baked via initialize(target, hash)", kEnvW, kEnvH);
    ImGui::SliderFloat("Sun angle", &sunAngle, 0.0f, 1.0f);
    const char* qnames[] = { "Low", "High" };
    ImGui::Combo("Quality", &quality, qnames, 2);
    ImGui::TextDisabled("changing a setting re-bakes (watch bake.fill reappear in the order); drag to pan.");
}
