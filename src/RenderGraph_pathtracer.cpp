// ===== Path-tracer demo =====================================================================
// A self-contained SDF path tracer: its own UBO, its own scene baked into the shader. Each frame
// casts spp cosine-sampled paths per pixel and folds the mean into a ping-ponged HDR texture, so the
// image refines while the camera holds still. #included into the single TU after RenderGraph_demo.h.

namespace pt_demo {

constexpr WGPUTextureFormat kAccumFormat = WGPUTextureFormat_RGBA16Float;  // HDR accumulation (renderable + sampleable in core)

// camera basis + accumulation/quality knobs. scene lives in the shader, so nothing else crosses.
struct PtUBO {
    float    camPos[4];      // xyz
    float    camFwd[4];      // xyz (unit)
    float    camRight[4];    // xyz (unit)
    float    camUp[4];       // xyz (unit)
    float    resolution[2];  // pixel size: aspect + per-pixel AA jitter
    uint32_t accumFrame;     // frames blended so far; doubles as the RNG seed. 0 = reset
    uint32_t maxBounces;     // path depth cap (Russian roulette may cut it short sooner)
    uint32_t spp;            // samples per pixel PER FRAME (more = fewer frames to converge)
    uint32_t _pad[3];        // round up to a 16-byte multiple
};
static_assert(sizeof(PtUBO) == 96, "PtUBO must match the std140 WGSL Pt layout");

// path-tracer guts; concatenated after kVS, shared by trace + (the present fs reads only the result).
static const char* kCommon = R"(
struct Pt {
    camPos     : vec4f,
    camFwd     : vec4f,
    camRight   : vec4f,
    camUp      : vec4f,
    resolution : vec2f,
    accumFrame : u32,
    maxBounces : u32,
    spp        : u32,
    pad0 : u32, pad1 : u32, pad2 : u32,
};
@group(0) @binding(0) var<uniform> pt : Pt;
@group(0) @binding(1) var prevTex : texture_2d<f32>;

const PT_PI    = 3.14159265;
const PT_FAR   = 60.0;
const PT_FOV   = 1.0;              // vertical radians -- same lens as the deferred camera
const PT_RR_MIN = 2u;             // bounces taken before Russian roulette may cull a path
const PT_CLAMP = 16.0;            // per-path radiance cap: kills RR fireflies (slight bias, much cleaner)

// PCG hash RNG: advance state, return a float in [0,1).
fn pt_rand(state : ptr<function, u32>) -> f32 {
    var s = *state;
    s = s * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    *state = s;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}

fn pt_ray(ndc : vec2f, aspect : f32) -> vec3f {
    let t = tan(0.5 * PT_FOV);
    return normalize(pt.camFwd.xyz + pt.camRight.xyz * (ndc.x * aspect * t)
                                   + pt.camUp.xyz    * (ndc.y * t));
}

fn pt_sphere(p : vec3f, c : vec3f, r : f32) -> f32 { return length(p - c) - r; }

// scene -> (distance, material id). ground = 0; diffuse spheres = 1,2; emissive lights = 3,4.
fn pt_map(p : vec3f) -> vec2f {
    var d  = p.y + 1.5;                                                                     // ground plane
    var id = 0.0;
    let s1 = pt_sphere(p, vec3f(-1.3, -0.6,  0.2), 0.9); if (s1 < d) { d = s1; id = 1.0; }  // red
    let s2 = pt_sphere(p, vec3f( 1.4, -0.7, -0.3), 0.8); if (s2 < d) { d = s2; id = 2.0; }  // blue
    let l1 = pt_sphere(p, vec3f( 0.0,  2.6,  1.0), 0.7); if (l1 < d) { d = l1; id = 3.0; }  // warm light
    let l2 = pt_sphere(p, vec3f(-2.6,  0.4, -1.6), 0.4); if (l2 < d) { d = l2; id = 4.0; }  // cool light
    return vec2f(d, id);
}

fn pt_normal(p : vec3f) -> vec3f {
    let e = vec2f(0.0015, 0.0);
    return normalize(vec3f(pt_map(p + e.xyy).x - pt_map(p - e.xyy).x,
                           pt_map(p + e.yxy).x - pt_map(p - e.yxy).x,
                           pt_map(p + e.yyx).x - pt_map(p - e.yyx).x));
}

fn pt_albedo(id : f32) -> vec3f {
    if (id < 0.5) { return vec3f(0.62, 0.62, 0.64); }   // ground
    if (id < 1.5) { return vec3f(0.85, 0.32, 0.22); }   // red sphere
    if (id < 2.5) { return vec3f(0.25, 0.45, 0.85); }   // blue sphere
    return vec3f(0.0);                                   // lights don't reflect
}

fn pt_emission(id : f32) -> vec3f {
    if (id > 3.5) { return vec3f(4.0, 6.0, 12.0); }      // cool light
    if (id > 2.5) { return vec3f(16.0, 12.0, 8.0); }     // warm light
    return vec3f(0.0);
}

fn pt_raymarch(ro : vec3f, rd : vec3f) -> vec2f {
    var t = 0.02;
    for (var i = 0; i < 160; i = i + 1) {
        let h = pt_map(ro + rd * t);
        if (h.x < 0.001) { return vec2f(t, h.y); }
        t = t + h.x;
        if (t > PT_FAR) { break; }
    }
    return vec2f(-1.0, 0.0);                             // miss
}

// branchless orthonormal basis around n (Duff et al. 2017); columns: tangent, bitangent, n.
fn pt_onb(n : vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
                   vec3f(b, s + n.y * n.y * a, -n.y),
                   n);
}

// cosine-weighted hemisphere direction around n.
fn pt_cosine_dir(n : vec3f, seed : ptr<function, u32>) -> vec3f {
    let u1  = pt_rand(seed);
    let u2  = pt_rand(seed);
    let r   = sqrt(u1);
    let phi = 2.0 * PT_PI * u2;
    let local = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(pt_onb(n) * local);
}

fn pt_sky(rd : vec3f) -> vec3f {
    let h = 0.5 * (rd.y + 1.0);
    return mix(vec3f(0.20, 0.24, 0.32), vec3f(0.55, 0.68, 0.92), h) * 0.5;   // dim fill so shadows aren't black
}

// one diffuse path: accumulate emission scaled by the throughput surviving each bounce.
fn pt_trace(ro0 : vec3f, rd0 : vec3f, seed : ptr<function, u32>) -> vec3f {
    var ro = ro0;
    var rd = rd0;
    var throughput = vec3f(1.0);
    var radiance   = vec3f(0.0);
    for (var b = 0u; b < pt.maxBounces; b = b + 1u) {
        let hit = pt_raymarch(ro, rd);
        if (hit.x < 0.0) { radiance += throughput * pt_sky(rd); break; }
        let p = ro + rd * hit.x;
        let n = pt_normal(p);
        radiance += throughput * pt_emission(hit.y);
        throughput *= pt_albedo(hit.y);                 // cosine pdf cancels the lambert term -> just *albedo
        // Russian roulette: past a few bounces, kill dim paths with prob (1 - q) and rescale the
        // survivors by 1/q so the estimator stays unbiased -- spends rays where they still matter.
        if (b >= PT_RR_MIN) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            if (pt_rand(seed) > q) { break; }
            throughput /= q;
        }
        rd = pt_cosine_dir(n, seed);
        ro = p + n * 0.003;
    }
    return min(radiance, vec3f(PT_CLAMP));              // clamp fireflies the RR rescale can spike
}
)";

// trace fs: average spp fresh samples this frame, fold the mean into the history read from prevTex.
static const char* kTraceFs = R"(
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px   = vec2u(u32(in.pos.x), u32(in.pos.y));
    var seed = (px.x * 1973u) + (px.y * 9277u) + (pt.accumFrame * 26699u) + 1u;
    let aspect = pt.resolution.x / pt.resolution.y;
    let spp = max(pt.spp, 1u);
    var sample = vec3f(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        let jitter = (vec2f(pt_rand(&seed), pt_rand(&seed)) - 0.5) * (2.0 / pt.resolution);   // sub-pixel AA
        let rd     = pt_ray(in.ndc + jitter, aspect);
        sample += pt_trace(pt.camPos.xyz, rd, &seed);
    }
    sample /= f32(spp);

    let prev = textureLoad(prevTex, vec2i(px), 0).rgb;
    let n    = f32(pt.accumFrame);
    let avg  = select(sample, (prev * n + sample) / (n + 1.0), pt.accumFrame > 0u);
    return vec4f(avg, 1.0);
}
)";

// present fs: tonemap + gamma-encode the HDR accumulation into the (non-sRGB) swapchain.
static const char* kPresentFs = R"(
@group(0) @binding(0) var srcTex : texture_2d<f32>;
fn pt_aces(x : vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px     = vec2i(i32(in.pos.x), i32(in.pos.y));
    let hdr    = textureLoad(srcTex, px, 0).rgb;
    let mapped = pt_aces(hdr);
    return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
)";

// ---- state (created once in init, lives across frames) ----
static WGPURenderPipeline tracePipe   = nullptr;
static WGPURenderPipeline presentPipe = nullptr;
static WGPUBuffer         uboBuf       = nullptr;   // demo-owned, imported into the graph each frame

// ---- accumulation knobs + the snapshot the reset logic compares against ----
static int      maxBounces  = 5;        // ImGui slider: path depth cap
static int      spp         = 1;        // ImGui slider: samples per pixel per frame
static uint32_t accum       = 0;        // frames folded in so far; 0 = start over
static float    lastCam[5]  = {};       // px,py,pz,yaw,pitch snapshot to spot camera motion
static uint32_t lastW = 0, lastH = 0;   // resolution snapshot (resize rebuilds the accum texture)
static int      lastBounces = 5;        // changing bounce depth changes the estimator -> reset accum
static uint64_t lastFrame   = 0;        // last frame this demo was active (gap => re-entry -> reset)

} // namespace pt_demo

static void pathtracer_init(const DemoEnv& env)
{
    using namespace pt_demo;
    WGPUDevice dev = env.device;

    WGPUShaderModule traceSM = make_shader(dev, std::string(kVS) + kCommon + kTraceFs);
    WGPUColorTargetState accumCT{ .format = kAccumFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState traceFrag{ .module = traceSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &accumCT };
    WGPURenderPipelineDescriptor tracePD{
        .label       = WEBGPU_STR("pt.trace pipeline"),
        .vertex      = { .module = traceSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &traceFrag,
    };
    tracePipe = wgpuDeviceCreateRenderPipeline(dev, &tracePD);
    wgpuShaderModuleRelease(traceSM);

    WGPUShaderModule presentSM = make_shader(dev, std::string(kVS) + kPresentFs);
    WGPUColorTargetState swapCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState presentFrag{ .module = presentSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &swapCT };
    WGPURenderPipelineDescriptor presentPD{
        .label       = WEBGPU_STR("pt.present pipeline"),
        .vertex      = { .module = presentSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &presentFrag,
    };
    presentPipe = wgpuDeviceCreateRenderPipeline(dev, &presentPD);
    wgpuShaderModuleRelease(presentSM);

    WGPUBufferDescriptor bd{ .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(PtUBO) };
    uboBuf = wgpuDeviceCreateBuffer(dev, &bd);
}

static void pathtracer_shutdown()
{
    using namespace pt_demo;
    wgpuBufferRelease(uboBuf);
    wgpuRenderPipelineRelease(presentPipe);
    wgpuRenderPipelineRelease(tracePipe);
}

static void pathtracer_build(const DemoEnv& env, RenderGraph* rg, ResourceHandle swapchain)
{
    using namespace pt_demo;
    WGPUDevice dev = env.device;

    // accumulation counter: reset on re-entry, camera motion, resize, or a bounce-depth change (that
    // shifts the estimator's mean); else climb so the trace shader weights each new sample less. spp
    // changes need no reset -- they don't change what the average converges to. Exact float compare is
    // fine: the camera state is bit-identical frame to frame when nothing moved it.
    bool moved = (env.frame != lastFrame + 1)                              // gap => was inactive
        || env.camera.pos[0] != lastCam[0] || env.camera.pos[1] != lastCam[1] || env.camera.pos[2] != lastCam[2]
        || env.camera.yaw != lastCam[3] || env.camera.pitch != lastCam[4]
        || env.width != lastW || env.height != lastH
        || maxBounces != lastBounces;
    accum = moved ? 0u : accum + 1u;
    lastCam[0] = env.camera.pos[0]; lastCam[1] = env.camera.pos[1]; lastCam[2] = env.camera.pos[2];
    lastCam[3] = env.camera.yaw;    lastCam[4] = env.camera.pitch;
    lastW = env.width; lastH = env.height; lastBounces = maxBounces; lastFrame = env.frame;

    // host-upload the UBO into the demo-owned buffer, then import it (no post-realize step needed).
    PtUBO u{};
    for (int i = 0; i < 3; ++i) { u.camPos[i] = env.camera.pos[i]; u.camFwd[i] = env.camera.fwd[i]; u.camRight[i] = env.camera.right[i]; u.camUp[i] = env.camera.up[i]; }
    u.resolution[0] = (float)env.width; u.resolution[1] = (float)env.height;
    u.accumFrame = accum;
    u.maxBounces = (uint32_t)maxBounces;
    u.spp        = (uint32_t)spp;
    wgpuQueueWriteBuffer(env.queue, uboBuf, 0, &u, sizeof(u));
    ResourceHandle ubo = rg->import_buffer(WEBGPU_STR("pt.ubo"), uboBuf);

    // ping-ponged HDR target: write curr, read prev (last frame's accumulated mean).
    auto a = rg->create_temporal_image(WEBGPU_STR("pt.accum"), {
        .dimension = WGPUTextureDimension_2D, .format = kAccumFormat,
        .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
    });

    // trace one new sample, fold it into the running mean carried in prev.
    rg->add_pass(WEBGPU_STR("pt.trace"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.uniform(ubo);
            b.sampled(a.prev);
            b.color(a.curr, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, a](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(tracePipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(PtUBO) },
                { .binding = 1, .textureView = ctx.view(a.prev) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, tracePipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });

    // tonemap the accumulated HDR result into the swapchain.
    rg->add_pass(WEBGPU_STR("pt.present"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(a.curr);
            b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, a](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
            WGPUBindGroupEntry e[1] = {
                { .binding = 0, .textureView = ctx.view(a.curr) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, presentPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
}

static void pathtracer_ui()
{
    using namespace pt_demo;
    ImGui::Text("samples: %u", accum + 1u);          // frames blended (x spp = total paths/pixel)
    ImGui::SliderInt("Max bounces", &maxBounces, 1, 12);
    ImGui::SliderInt("SPP / frame", &spp, 1, 16);
}
