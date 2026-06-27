// ===== GPU particle-simulation demo =========================================================
// A fully GPU-driven particle fountain whose state lives in a temporal BUFFER (create_temporal_buffer):
// particles.sim reads last frame's particles (prev), integrates forces + turbulence + spawn/death, and
// writes this frame's (curr); particles.draw reads curr straight from the storage buffer in the vertex
// stage and billboards each one. No particle ever touches the CPU -- only the byte size and a small UBO
// of parameters cross. This is the load-bearing exercise of the temporal-buffer feature. #included into
// the single TU after RenderGraph_demo.h, registered F3.

namespace particles_demo {

constexpr uint32_t kParticleCount  = 1u << 16;   // 65,536 particles
constexpr uint64_t kParticleStride = 32;         // pos.xyz + age, vel.xyz + life  (WGSL storage stride)
constexpr uint64_t kParticleBytes  = kParticleStride * kParticleCount;

// camera basis + sim parameters; GPU-only particle state lives in the storage buffer, so this is all
// that crosses from the CPU. laid out as vec4 groups to match the WGSL uniform (std140) alignment.
struct PartUBO {
    float    camPos[4];     // xyz
    float    camFwd[4];     // xyz (unit)
    float    camRight[4];   // xyz (unit)
    float    camUp[4];      // xyz (unit)
    float    emitter[4];    // xyz = spawn point, w = sprite half-size (world units)
    float    gravity[4];    // xyz = constant accel, w = viewport aspect (w/h)
    float    params[4];     // x = dt, y = time, z = turbulence amplitude, w = turbulence frequency
    uint32_t counts[4];     // x = reset flag (seed vs integrate), y = particle count, zw spare
};
static_assert(sizeof(PartUBO) == 128, "PartUBO must match the WGSL std140 PartU layout");

// shared declarations for both shaders: the UBO, the particle layout, the camera lens.
static const char* kCommon = R"(
struct PartU {
    camPos   : vec4f,
    camFwd   : vec4f,
    camRight : vec4f,
    camUp    : vec4f,
    emitter  : vec4f,   // xyz pos, w sprite half-size
    gravity  : vec4f,   // xyz accel, w aspect
    params   : vec4f,   // dt, time, turbAmp, turbFreq
    counts   : vec4u,   // reset, count, _, _
};
@group(0) @binding(0) var<uniform> u : PartU;

struct Particle {
    pos  : vec3f,
    age  : f32,         // < 0 = not yet born (staggered spawn); >= life = recycle
    vel  : vec3f,
    life : f32,
};

const PART_FOV = 1.0;   // vertical radians -- same lens as the deferred / path-tracer cameras
)";

// compute: advance every particle. prev (read) -> curr (write); the temporal pool ping-pongs them.
static const char* kSimBody = R"(
@group(0) @binding(1) var<storage, read>       prevP : array<Particle>;
@group(0) @binding(2) var<storage, read_write> currP : array<Particle>;

const PART_SPAWN_SPREAD = 3.0;   // seconds of staggered birth so the fountain fills in, not all at once

// PCG hash -> u32, then [0,1) / vec3 helpers for decorrelated per-particle randomness.
fn pcg(n : u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}
fn h1(n : u32) -> f32   { return f32(pcg(n)) / 4294967296.0; }
fn h3(n : u32) -> vec3f { return vec3f(h1(n), h1(n ^ 0x9e3779b9u), h1(n + 0x85ebca6bu)); }

// cheap divergence-free (curl-like) flow: each component depends only on the OTHER two axes, so the
// analytic divergence is exactly zero -- swirly, volume-preserving turbulence without a noise texture.
fn turb(p : vec3f, t : f32, freq : f32) -> vec3f {
    return vec3f(
        sin(p.y * freq + t)       - cos(p.z * freq - t * 0.7),
        sin(p.z * freq + t * 1.3) - cos(p.x * freq - t),
        sin(p.x * freq + t * 0.6) - cos(p.y * freq - t * 1.1));
}

// a fresh particle launched from the emitter; salt decorrelates successive respawns of the same slot.
fn spawn(i : u32, salt : u32) -> Particle {
    let h = h3(i * 3u + salt);
    var p : Particle;
    p.pos  = u.emitter.xyz + (h3(i * 7u + salt) - 0.5) * 0.4;             // small nozzle jitter
    let dir = normalize(vec3f((h.x - 0.5) * 0.7, 1.0, (h.z - 0.5) * 0.7)); // upward cone
    p.vel  = dir * mix(1.6, 3.6, h.y) * 0;
    p.life = mix(1.5, 4.0, h1(i ^ (salt * 2246822519u)));
    p.age  = 0.0;
    return p;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= u.counts.y) { return; }
    let dt = u.params.x;

    if (u.counts.x == 1u) {                       // reset: seed the whole pool with staggered births
        var p = spawn(i, 0u);
        p.age = -h1(i * 2654435761u) * PART_SPAWN_SPREAD;   // negative => not yet visible
        currP[i] = p;
        return;
    }

    var p = prevP[i];
    p.age = p.age + dt;
    if (p.age >= p.life) {                         // recycle: salt by a coarse frame index so it differs
        p = spawn(i, u32(u.params.y * 60.0) * 9781u + 1u);
    }
    if (p.age >= 0.0) {                            // born: integrate forces + turbulence
        let force = u.gravity.xyz + turb(p.pos, u.params.y, u.params.w) * u.params.z;
        p.vel = p.vel + force * dt;
        p.pos = p.pos + p.vel * dt;
    }
    currP[i] = p;
}
)";

// graphics: one screen-aligned billboard quad per particle, read straight from the storage buffer in the
// vertex stage (read-only storage in a vertex shader is core-legal). projected with the same lens math as
// the path tracer's pt_ray, inverted. additive blend, no depth buffer.
static const char* kDrawBody = R"(
@group(0) @binding(1) var<storage, read> parts : array<Particle>;

struct VsOut {
    @builtin(position) pos   : vec4f,
    @location(0)       uv    : vec2f,   // [-1,1] across the sprite quad
    @location(1)       color : vec3f,
    @location(2)       alpha : f32,
};

@vertex fn vs(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> VsOut {
    var corners = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f( 1.0, 1.0));
    let corner = corners[vid];
    let p = parts[iid];

    var o : VsOut;
    o.uv = corner;
    let l = p.age / p.life;
    let rel = p.pos - u.camPos.xyz;
    let vz  = dot(rel, u.camFwd.xyz);
    // unborn, expired, or behind the camera -> collapse the quad off-screen (no fragments produced)
    if (p.age < 0.0 || l >= 1.0 || vz <= 0.02) {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        o.color = vec3f(0.0); o.alpha = 0.0;
        return o;
    }
    let vx = dot(rel, u.camRight.xyz);
    let vy = dot(rel, u.camUp.xyz);
    let t  = tan(0.5 * PART_FOV);
    let half = u.emitter.w;
    // offset the corner in view space at the particle's depth -> perspective-correct screen size
    let px = vx + corner.x * half;
    let py = vy + corner.y * half;
    o.pos = vec4f(px / (t * u.gravity.w), py / t, 0.5 * vz, vz);   // depth mid-range; no depth test anyway

    let fadeIn  = smoothstep(0.0, 0.08, l);
    let fadeOut = smoothstep(1.0, 0.65, l);
    o.color = mix(vec3f(1.0, 0.85, 0.45), vec3f(1.0, 0.22, 0.08), l);   // young = warm, old = ember
    o.alpha = fadeIn * fadeOut;
    return o;
}

@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    let soft = smoothstep(1.0, 0.0, r);     // round, soft-edged sprite
    let a = in.alpha * soft;
    return vec4f(in.color * a, a);           // premultiplied -> clean additive accumulation
}
)";

// ---- state (created once in init, lives across frames) ----
static WGPUComputePipeline simPipe  = nullptr;
static WGPURenderPipeline  drawPipe = nullptr;
static WGPUBuffer          uboBuf   = nullptr;   // demo-owned, imported into the graph each frame

// ---- tunables (ImGui) + reset bookkeeping ----
static float    gravityY  = -0.0f;
static float    turbAmp   = 3.0f;
static float    turbFreq  = 12.0f;
static float    spriteSize = 0.01f;
static bool     burst     = false;     // re-seed the whole pool on demand
static uint64_t lastFrame = 0;         // last frame this demo was active (gap => re-entry -> reset)
static bool     seeded    = false;     // false until the first build seeds the pool

} // namespace particles_demo

static void particles_init(const DemoEnv& env)
{
    using namespace particles_demo;
    WGPUDevice dev = env.device;

    WGPUShaderModule simSM = make_shader(dev, std::string(kCommon) + kSimBody);
    WGPUComputePipelineDescriptor simPD{
        .label   = WEBGPU_STR("particles.sim pipeline"),
        .compute = { .module = simSM, .entryPoint = WEBGPU_STR("main") },
    };
    simPipe = wgpuDeviceCreateComputePipeline(dev, &simPD);
    wgpuShaderModuleRelease(simSM);

    WGPUShaderModule drawSM = make_shader(dev, std::string(kCommon) + kDrawBody);
    // additive blend: each sprite adds its premultiplied colour, so overlapping particles glow brighter.
    WGPUBlendState addBlend{
        .color = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
        .alpha = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
    };
    WGPUColorTargetState drawCT{ .format = kSwapFormat, .blend = &addBlend, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState drawFrag{ .module = drawSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &drawCT };
    WGPURenderPipelineDescriptor drawPD{
        .label       = WEBGPU_STR("particles.draw pipeline"),
        .vertex      = { .module = drawSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &drawFrag,
    };
    drawPipe = wgpuDeviceCreateRenderPipeline(dev, &drawPD);
    wgpuShaderModuleRelease(drawSM);

    WGPUBufferDescriptor bd{ .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(PartUBO) };
    uboBuf = wgpuDeviceCreateBuffer(dev, &bd);
}

static void particles_shutdown()
{
    using namespace particles_demo;
    wgpuBufferRelease(uboBuf);
    wgpuRenderPipelineRelease(drawPipe);
    wgpuComputePipelineRelease(simPipe);
}

static void particles_build(const DemoEnv& env, RenderGraph* rg, ResourceHandle swapchain)
{
    using namespace particles_demo;
    WGPUDevice dev = env.device;

    // reset on the first build ever, on re-entry (a gap in the global frame counter), or a UI burst.
    bool reset = !seeded || env.frame != lastFrame + 1 || burst;
    seeded = true; burst = false; lastFrame = env.frame;

    // host-fill the parameter UBO, then import it (demo-owned; no post-realize step, like the path tracer).
    PartUBO u{};
    for (int i = 0; i < 3; ++i) {
        u.camPos[i]   = env.camera.pos[i];   u.camFwd[i] = env.camera.fwd[i];
        u.camRight[i] = env.camera.right[i]; u.camUp[i]  = env.camera.up[i];
    }
    u.emitter[0] = 0.0f; u.emitter[1] = 0.5f; u.emitter[2] = 0.0f; u.emitter[3] = spriteSize;
    u.gravity[0] = 0.0f; u.gravity[1] = gravityY; u.gravity[2] = 0.0f;
    u.gravity[3] = env.height ? (float)env.width / (float)env.height : 1.0f;
    u.params[0] = env.dt; u.params[1] = env.time; u.params[2] = turbAmp; u.params[3] = turbFreq;
    u.counts[0] = reset ? 1u : 0u; u.counts[1] = kParticleCount;
    wgpuQueueWriteBuffer(env.queue, uboBuf, 0, &u, sizeof(u));
    ResourceHandle ubo = rg->import_buffer(WEBGPU_STR("part.ubo"), uboBuf);

    // the cross-frame particle state: write curr, read prev. the pool rotates the two physical buffers.
    auto parts = rg->create_temporal_buffer(WEBGPU_STR("particles"), { .size = kParticleBytes });

    // sim: integrate prev -> curr. one invocation per particle.
    const uint32_t groups = (kParticleCount + 63u) / 64u;
    rg->add_pass(WEBGPU_STR("particles.sim"), PassKind::Compute,
        [&](GraphBuilder& b) {
            b.uniform(ubo);
            b.storage_read(parts.prev);
            b.storage_write(parts.curr);
        },
        [dev, ubo, parts, groups](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(simPipe, 0);
            WGPUBindGroupEntry e[3] = {
                { .binding = 0, .buffer = ctx.buffer(ubo),        .offset = 0, .size = sizeof(PartUBO) },
                { .binding = 1, .buffer = ctx.buffer(parts.prev), .offset = 0, .size = kParticleBytes },
                { .binding = 2, .buffer = ctx.buffer(parts.curr), .offset = 0, .size = kParticleBytes },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuComputePassEncoderSetPipeline(ctx.compute, simPipe);
            wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
            assert(ctx.buffer_size(parts.prev) == kParticleBytes);
            wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, groups, 1, 1);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });

    // draw: billboard every particle, additive over a near-black clear.
    rg->add_pass(WEBGPU_STR("particles.draw"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.uniform(ubo);
            b.storage_read(parts.curr);
            b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0.01, 0.01, 0.02, 1.0});
        },
        [dev, ubo, parts](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(drawPipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .buffer = ctx.buffer(ubo),        .offset = 0, .size = sizeof(PartUBO) },
                { .binding = 1, .buffer = ctx.buffer(parts.curr), .offset = 0, .size = kParticleBytes },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, drawPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 6, kParticleCount, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    // ponytail: fixed-pool recycle (each slot respawns on death; staggered ages give spawn-over-time),
    // not a free-list/atomic emitter. an indirect-draw compaction of only-live particles is the upgrade.
}

static void particles_ui()
{
    using namespace particles_demo;
    ImGui::Text("particles: %u (GPU-resident)", kParticleCount);
    ImGui::SliderFloat("Gravity Y",   &gravityY,   -8.0f, 2.0f);
    ImGui::SliderFloat("Turbulence",  &turbAmp,     0.0f, 4.0f);
    ImGui::SliderFloat("Turb freq",   &turbFreq,    0.1f, 16.0f);
    ImGui::SliderFloat("Sprite size", &spriteSize,  0.01f, 0.15f);
    if (ImGui::Button("Reset / burst")) burst = true;
}
