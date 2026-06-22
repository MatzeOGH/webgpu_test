// Windowed sample for the render graph -- immediate mode, the WHOLE graph rebuilt every frame.
// A small DEFERRED renderer driven entirely by the graph's read/write tracking:
//   gbuffer  (graphics, MRT + depth) : raymarch an SDF (spheres in a UBO) -> albedo / normal / roughness / depth
//   shadow   (graphics, depth-only)  : raymarch the same SDF from a directional light -> depth-only shadow map
//   ssao     (compute)               : screen-space AO from gbuffer depth + normal          [SPACE toggles]
//   lighting (graphics, fullscreen)  : gbuffer + shadow map + directional light -> lit color
//   sky      (graphics, fullscreen)  : re-writes lit color's background pixels (2nd writer -> WAW edge)
//   compose  (graphics, fullscreen)  : lit * ao -> swapchain   (a plain "present" of lit color when SSAO is off)
// Pass order is derived from the accesses each frame; toggling SSAO with SPACE adds/drops the ssao pass
// (and swaps compose<->present) -- the graph reshapes itself, the point of an immediate-mode graph.
//
// Camera: WASD to fly, hold left mouse to look around.
// Keys: SPACE toggles SSAO. T toggles TAA. B toggles bloom (a mip-chain downsample/upsample over one
// texture -- mip subresource access). C toggles a cubemap (6 faces rendered per array layer, sampled as
// a cube -- array-layer subresource access). 1/2/3/4/5 = debug-blit a single buffer (albedo/normal/roughness/depth/ssao)
// straight to the swapchain, 0 = back to the lit image. The debug pass writes the swapchain itself and
// becomes the only sink, so the graph culls every pass not needed for that one image: views 1-4 leave
// only gbuffer; view 5 leaves gbuffer+ssao; shadow/lighting/compose drop out.
//
// Matrix-free on purpose: the camera is a position + basis in the UBO (CPU drives it from WASD +
// mouse-look) fed to a ray function in WGSL, so the gbuffer stores LINEAR depth (t / FAR) and the
// lighting/ssao passes reconstruct the exact world hit as camPos + camera_ray(uv) * depth * FAR --
// no projection/inverse matrices, no CPU math lib.
//
// Not a standalone TU: #included at the end of RenderGraph.cpp so it sees the internal node structs.

#include "RenderGraph.h"
#include <cstdio>
#include <ostream>
#include <string>                // assemble shader sources from shared snippets
#include <webgpu/webgpu_cpp.h>   // C++ wrappers, only for instance/surface/device bring-up
#include <cmath>                 // sin/cos for the animated scene + light

#define SDL_MAIN_HANDLED         // we keep our own main(); just call SDL_SetMainReady()
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>       // SDL_SetMainReady (no main redefinition under SDL_MAIN_HANDLED)

#include "imgui_layer.h"         // ImGui SDL3 + WebGPU backends, driven by main() + the "imgui" pass

double getTime() {
    static const double freq =
        (double)SDL_GetPerformanceFrequency();

    return (double)SDL_GetPerformanceCounter() / freq;
}

namespace {

constexpr WGPUTextureFormat kSwapFormat  = WGPUTextureFormat_BGRA8Unorm;   // matches the surface
constexpr WGPUTextureFormat kColorFormat = WGPUTextureFormat_RGBA8Unorm;   // gbuffer albedo/normal/rough + lit color
constexpr WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth32Float; // gbuffer depth + shadow map
constexpr WGPUTextureFormat kAOFormat    = WGPUTextureFormat_RGBA8Unorm;   // ssao output (rgba8unorm = core storage format)
constexpr uint32_t          kShadowSize  = 1024;                           // shadow map is a fixed square
constexpr uint32_t          kNumCascades = 3;                              // CSM: layers in the shadow array
constexpr uint32_t          kMaxSpheres  = 8;

// One uniform buffer feeds gbuffer + shadow + lighting. Layout matches the WGSL `Scene` struct below
// (std140: vec4 members on 16-byte offsets). Host-uploaded each frame -> never written by a pass.
struct SceneUBO {
    float    spheres[kMaxSpheres][4];   // xyz = center, w = radius          (offset 0)
    float    lightDir[4];               // xyz = light TRAVEL direction      (offset 128)
    float    lightColor[4];             // rgb                               (offset 144)
    float    camPos[4];                 // xyz = camera position             (offset 160)
    float    camFwd[4];                 // xyz = camera forward (unit)       (offset 176)
    float    camRight[4];               // xyz = camera right (unit)         (offset 192)
    float    camUp[4];                  // xyz = camera up (unit)            (offset 208)
    float    resolution[2];             // gbuffer pixel size (for aspect)   (offset 224)
    uint32_t count;                     // live sphere count                 (offset 232)
    float    time;                      //                                   (offset 236)
    uint32_t debugMode;                 // 0 = lit, 1..5 = blit a debug image (offset 240)
    uint32_t _pad[3];                   // round up to a 16-byte multiple    (offset 244)
};
static_assert(sizeof(SceneUBO) == 256, "SceneUBO must match the std140 WGSL Scene layout");

// same shape as Renderer.cpp::createShaderModule
WGPUShaderModule make_shader(WGPUDevice dev, WGPUStringView code)
{
    WGPUShaderSourceWGSL wgsl{ .chain = { .sType = WGPUSType_ShaderSourceWGSL }, .code = code };
    WGPUShaderModuleDescriptor d{ .nextInChain = &wgsl.chain };
    return wgpuDeviceCreateShaderModule(dev, &d);
}

// build a shader module from an assembled std::string (snippets concatenated below)
WGPUShaderModule make_shader(WGPUDevice dev, const std::string& code)
{
    return make_shader(dev, WGPUStringView{ code.c_str(), code.size() });
}

// a single-mip 2D view of one texture -- how a pass body samples a chosen mip (the graph resolves the
// texture via ctx.texture(); the body builds whatever view it needs, no graph-side abstraction).
WGPUTextureView mip_view_2d(WGPUTexture tex, WGPUTextureFormat fmt, uint32_t mip)
{
    WGPUTextureViewDescriptor vd{
        .format = fmt, .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = mip, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1,
    };
    return wgpuTextureCreateView(tex, &vd);
}

// ---- shared WGSL snippets ------------------------------------------------------------------------
// Each pipeline's source = a few of these concatenated. Keeping the SDF + camera in one place is what
// makes the gbuffer, shadow and lighting passes agree on world space without passing any matrices.

// fullscreen triangle; `ndc` is clip-space xy in [-1,1] (y up) -> the per-pixel ray + reconstruct uv.
const char* kVS = R"(
struct VsOut { @builtin(position) pos : vec4f, @location(0) ndc : vec2f, @location(1) @interpolate(flat) casc : u32 };
@vertex fn vs(@builtin(vertex_index) vid : u32, @builtin(instance_index) iid : u32) -> VsOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o : VsOut;
    o.pos = vec4f(p[vid], 0.0, 1.0);
    o.ndc = p[vid];
    o.casc = iid;                                  // shadow draws one instance per cascade; every other pass draws instance 0
    return o;
}
)";

// scene UBO at @group(0) @binding(0) -- gbuffer/shadow/lighting all bind it there.
const char* kSceneDecl = R"(
struct Scene {
    spheres    : array<vec4f, 8>,
    lightDir   : vec4f,
    lightColor : vec4f,
    camPos     : vec4f,
    camFwd     : vec4f,
    camRight   : vec4f,
    camUp      : vec4f,
    resolution : vec2f,
    count      : u32,
    time       : f32,
    debugMode  : u32,
};
@group(0) @binding(0) var<uniform> scene : Scene;
)";

// camera + light, matrix-free. The single source of truth for "what world point is this pixel".
const char* kCameraDefs = R"(
const FAR         : f32 = 24.0;
const FOV         : f32 = 1.0;                  // vertical, radians
const LIGHT_DIST  : f32 = 12.0;                 // light "camera" distance from the shadow center
// CSM cascades follow the viewer: center the ortho frustums on the camera, not a fixed world point,
// so the tight near cascade tracks where you're looking. shadow pass + lighting both call this, so
// they agree on the same per-frame center. ponytail: + scene.camFwd.xyz * k to bias the budget ahead
// of the view; add per-cascade texel snapping if the shadow edges shimmer while moving.
fn shadow_center() -> vec3f { return scene.camPos.xyz; }
const NUM_CASCADES: u32 = 3u;
// per-cascade ortho half-extent, near -> far. local var (not module const) so a dynamic index is legal, as in kVS.
fn csm_ortho(c : u32) -> f32 {
    var e = array<f32, 3>(3.0, 6.0, 12.0);
    return e[c];
}

// per-pixel primary ray from the UBO camera basis (CPU drives pos/fwd/right/up from WASD + mouse).
fn camera_ray(ndc : vec2f, aspect : f32) -> vec3f {
    let t = tan(0.5 * FOV);
    return normalize(scene.camFwd.xyz + scene.camRight.xyz * (ndc.x * aspect * t)
                                      + scene.camUp.xyz    * (ndc.y * t));
}
// exact world hit from the LINEAR depth (t / FAR) the gbuffer stored -- the deferred passes' anchor.
fn world_from_depth(ndc : vec2f, aspect : f32, lin : f32) -> vec3f {
    return scene.camPos.xyz + camera_ray(ndc, aspect) * (lin * FAR);
}
// orthonormal basis for the directional light's ortho "camera": columns right, up, forward(=travel).
fn light_basis(dir : vec3f) -> mat3x3f {
    let f = normalize(dir);
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(f.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let r = normalize(cross(up, f));
    let u = cross(f, r);
    return mat3x3f(r, u, f);
}
// world -> light space for a cascade of half-extent `ortho`: xy in [-1,1] across the frustum,
// z in [0,1] matching what the shadow pass writes (z normalization is shared by all cascades).
fn light_space(wp : vec3f, dir : vec3f, ortho : f32) -> vec3f {
    let B   = light_basis(dir);
    let rel = wp - shadow_center();
    return vec3f(dot(rel, B[0]) / ortho,
                 dot(rel, B[1]) / ortho,
                 (dot(rel, B[2]) + LIGHT_DIST) / (2.0 * LIGHT_DIST));
}
)";

// the SDF scene (references the `scene` UBO) -- shared by the gbuffer and shadow raymarches.
const char* kSdfDefs = R"(
fn sd_sphere(p : vec3f, c : vec3f, r : f32) -> f32 { return length(p - c) - r; }
fn smin(a : f32, b : f32, k : f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
fn map(p : vec3f) -> f32 {
    var d = 1e9;
    for (var i = 0u; i < scene.count; i = i + 1u) {
        let s = scene.spheres[i];
        d = smin(d, sd_sphere(p, s.xyz, s.w), 0.6);   // blob the spheres together
    }
    return min(d, p.y + 1.5);                          // crisp ground plane at y = -1.5
}
fn estimate_normal(p : vec3f) -> vec3f {
    let e = vec2f(0.0015, 0.0);
    return normalize(vec3f(map(p + e.xyy) - map(p - e.xyy),
                           map(p + e.yxy) - map(p - e.yxy),
                           map(p + e.yyx) - map(p - e.yyx)));
}
fn raymarch(ro : vec3f, rd : vec3f, maxd : f32) -> f32 {
    var t = 0.05;
    for (var i = 0; i < 96; i = i + 1) {
        let d = map(ro + rd * t);
        if (d < 0.001) { return t; }
        t = t + d;
        if (t > maxd) { break; }
    }
    return -1.0;                                       // miss
}
)";

// gbuffer fragment: raymarch from the camera, write the three G-targets + linear depth.
const char* kGbufferFs = R"(
struct GOut {
    @location(0) albedo : vec4f,
    @location(1) normal : vec4f,
    @location(2) rough  : vec4f,
    @builtin(frag_depth) depth : f32,
};
@fragment fn fs(in : VsOut) -> GOut {
    let aspect = scene.resolution.x / scene.resolution.y;
    let ro = scene.camPos.xyz;
    let rd = camera_ray(in.ndc, aspect);
    let t  = raymarch(ro, rd, FAR);
    if (t < 0.0) { discard; }                          // background: leave cleared, write no depth
    let p = ro + rd * t;
    let n = estimate_normal(p);
    var albedo = vec3f(0.85, 0.30, 0.20);
    var rough  = 0.35;
    // ponytail: material chosen by height (ground vs blob), no per-primitive id; return a material
    // index out of map() if the scene grows more than two surfaces.
    if (p.y < -1.4) {                                  // ground: checkerboard so AO/shadow read clearly
        let chk = floor(p.x) + floor(p.z);
        albedo  = mix(vec3f(0.18), vec3f(0.45), fract(chk * 0.5) * 2.0);
        rough   = 0.9;
    }
    var o : GOut;
    o.albedo = vec4f(albedo, 1.0);
    o.normal = vec4f(n * 0.5 + 0.5, 1.0);              // encode [-1,1] -> [0,1]
    o.rough  = vec4f(rough, 0.0, 0.0, 1.0);
    o.depth  = clamp(t / FAR, 0.0, 1.0);               // LINEAR depth -> reconstructable later
    return o;
}
)";

// shadow fragment: raymarch the same SDF from the light's ortho camera, depth only (no color targets).
const char* kShadowFs = R"(
@fragment fn fs(in : VsOut) -> @builtin(frag_depth) f32 {
    let ortho = csm_ortho(in.casc);                    // this cascade's frustum half-extent (firstInstance -> in.casc)
    let B = light_basis(scene.lightDir.xyz);
    let origin = shadow_center() - B[2] * LIGHT_DIST
               + B[0] * (in.ndc.x * ortho)
               + B[1] * (in.ndc.y * ortho);
    let t = raymarch(origin, B[2], 2.0 * LIGHT_DIST);  // parallel rays along the light direction
    if (t < 0.0) { discard; }
    return t / (2.0 * LIGHT_DIST);
}
)";

// lighting fragment: read gbuffer, reconstruct world pos, directional light + shadow lookup + cheap spec.
const char* kLightingFs = R"(
@group(0) @binding(1) var gAlbedo   : texture_2d<f32>;
@group(0) @binding(2) var gNormal   : texture_2d<f32>;
@group(0) @binding(3) var gRough    : texture_2d<f32>;
@group(0) @binding(4) var gDepth    : texture_depth_2d;
@group(0) @binding(5) var shadowMap : texture_depth_2d_array;

struct Shadow { factor : f32, cascade : i32 };   // factor: 1 lit / 0 shadowed; cascade: chosen layer, -1 = none
// ponytail: single tap -> hard edges; add a 3x3 PCF loop here if the shadow edges look too crunchy.
// pick the tightest cascade (smallest ortho extent) that contains wp -> highest available shadow resolution.
fn sample_shadow(wp : vec3f) -> Shadow {
    let dir = scene.lightDir.xyz;
    for (var c = 0u; c < NUM_CASCADES; c = c + 1u) {
        let ls = light_space(wp, dir, csm_ortho(c));
        if (abs(ls.x) > 1.0 || abs(ls.y) > 1.0 || ls.z > 1.0 || ls.z < 0.0) { continue; }   // not in this cascade
        let dim = vec2f(textureDimensions(shadowMap));
        let tx  = vec2i(i32((ls.x * 0.5 + 0.5) * dim.x), i32((0.5 - ls.y * 0.5) * dim.y));   // flip y for texel space
        let stored = textureLoad(shadowMap, tx, i32(c), 0);
        return Shadow(select(1.0, 0.0, ls.z - 0.004 > stored), i32(c));                      // bias against acne
    }
    return Shadow(1.0, -1);                                                                  // outside all cascades -> lit
}
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px  = vec2i(in.pos.xy);
    let lin = textureLoad(gDepth, px, 0);
    if (lin >= 1.0) { return vec4f(0.015, 0.02, 0.03, 1.0); }                               // background
    let dim    = vec2f(textureDimensions(gDepth));
    let wp     = world_from_depth(in.ndc, dim.x / dim.y, lin);
    let n      = textureLoad(gNormal, px, 0).xyz * 2.0 - 1.0;
    let albedo = textureLoad(gAlbedo, px, 0).rgb;
    let rough  = textureLoad(gRough,  px, 0).r;
    let L = normalize(-scene.lightDir.xyz);
    let V = normalize(scene.camPos.xyz - wp);
    let H = normalize(L + V);
    let ndl  = max(dot(n, L), 0.0);
    let sh   = sample_shadow(wp);
    let spec = pow(max(dot(n, H), 0.0), mix(4.0, 64.0, 1.0 - rough)) * (1.0 - rough);
    let direct  = (ndl + spec) * sh.factor * scene.lightColor.rgb;
    let ambient = 0.12;
    var col = albedo * (ambient + direct);
    if (scene.debugMode == 6u) {                                                            // CSM debug: tint by cascade
        var tint = vec3f(0.35);                                                             // outside all cascades = grey
        if      (sh.cascade == 0) { tint = vec3f(1.0, 0.4, 0.4); }
        else if (sh.cascade == 1) { tint = vec3f(0.4, 1.0, 0.4); }
        else if (sh.cascade == 2) { tint = vec3f(0.4, 0.6, 1.0); }
        col = col * tint;
    }
    return vec4f(col, 1.0);
}
)";

// ssao compute: hemisphere AO from reconstructed positions (matrix-free, screen-space spiral taps).
const char* kSsaoBody = R"(
@group(0) @binding(1) var gDepth  : texture_depth_2d;
@group(0) @binding(2) var gNormal : texture_2d<f32>;
@group(0) @binding(3) var aoOut   : texture_storage_2d<rgba8unorm, write>;

fn hash12(p : vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn ndc_of(px : vec2i, fdim : vec2f) -> vec2f {
    return vec2f((f32(px.x) + 0.5) / fdim.x * 2.0 - 1.0,
                 1.0 - (f32(px.y) + 0.5) / fdim.y * 2.0);   // clip y up
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let dim = vec2i(textureDimensions(gDepth));
    let px  = vec2i(i32(gid.x), i32(gid.y));
    if (px.x >= dim.x || px.y >= dim.y) { return; }
    let lin = textureLoad(gDepth, px, 0);
    if (lin >= 1.0) { textureStore(aoOut, px, vec4f(1.0)); return; }     // background = fully lit

    let fdim   = vec2f(dim);
    let aspect = fdim.x / fdim.y;
    let P = world_from_depth(ndc_of(px, fdim), aspect, lin);
    let N = textureLoad(gNormal, px, 0).xyz * 2.0 - 1.0;

    let focal = fdim.y / (2.0 * tan(0.5 * FOV));
    let rpix  = clamp(0.85 * focal / (lin * FAR), 4.0, 96.0);            // ~constant world radius in pixels
    let rot   = hash12(vec2f(px)) * 6.2831;
    let SAMPLES = 16;
    var ao = 0.0;
    for (var i = 0; i < SAMPLES; i = i + 1) {
        let r = sqrt((f32(i) + 0.5) / f32(SAMPLES));
        let a = f32(i) * 2.39996 + rot;                                  // golden-angle spiral
        let off = vec2f(cos(a), sin(a)) * r * rpix;
        let npx = px + vec2i(i32(round(off.x)), i32(round(off.y)));
        if (npx.x < 0 || npx.y < 0 || npx.x >= dim.x || npx.y >= dim.y) { continue; }
        let nd = textureLoad(gDepth, npx, 0);
        if (nd >= 1.0) { continue; }
        let Pn   = world_from_depth(ndc_of(npx, fdim), aspect, nd);
        let v    = Pn - P;
        let dist = length(v);
        let ndv  = dot(N, v) / (dist + 1e-4);                            // is the sample inside the hemisphere?
        let rng  = 1.0 / (1.0 + dist * dist * 3.0);                      // distance falloff
        ao = ao + max(ndv - 0.02, 0.0) * rng;
    }
    ao = clamp(1.0 - ao / f32(SAMPLES) * 2.5, 0.0, 1.0);
    textureStore(aoOut, px, vec4f(ao, ao, ao, 1.0));
}
)";

// compose fragment: lit color modulated by AO. present = lit color only (SSAO off).
// ponytail: AO multiplies the whole lit result, not just the ambient term; if it over-darkens lit
// surfaces, output ambient/direct separately from the lighting pass and apply AO to ambient only.
const char* kComposeFs = R"(
@group(0) @binding(0) var litColor : texture_2d<f32>;
@group(0) @binding(1) var aoTex    : texture_2d<f32>;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    return vec4f(textureLoad(litColor, px, 0).rgb * textureLoad(aoTex, px, 0).r, 1.0);
}
)";
const char* kPresentFs = R"(
@group(0) @binding(0) var litColor : texture_2d<f32>;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    return vec4f(textureLoad(litColor, px, 0).rgb, 1.0);
}
)";

// temporal accumulation with neighborhood clamping (demo of create_temporal_image). binding 0 = current
// lit, binding 1 = history.prev (last frame, rotated in by the pool). The scene animates every frame
// (spinning spheres, bobbing capstone, rotating light), so a plain history blend smears that motion
// forever and never settles. The standard TAA fix, no motion vectors required: clamp the history sample
// to the colour AABB of the current 3x3 neighborhood before blending. A static pixel's history sits
// inside the box -> the blend accumulates and converges; a pixel whose content moved has its stale
// history fall outside the box and snap to current -> no ghost trails. No motion vectors (out of scope),
// so "reprojection" is the same pixel -- exactly right while the camera is still.
const char* kTaaFs = R"(
@group(0) @binding(0) var curr : texture_2d<f32>;
@group(0) @binding(1) var hist : texture_2d<f32>;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px  = vec2i(in.pos.xy);
    let dim = vec2i(textureDimensions(curr));
    let c   = textureLoad(curr, px, 0).rgb;

    // colour bounding box of the 3x3 current-frame neighborhood = the range history may stay in.
    var lo = c;
    var hi = c;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let q = clamp(px + vec2i(dx, dy), vec2i(0), dim - vec2i(1));
            let n = textureLoad(curr, q, 0).rgb;
            lo = min(lo, n);
            hi = max(hi, n);
        }
    }

    let h = clamp(textureLoad(hist, px, 0).rgb, lo, hi);   // reject stale history where content moved
    return vec4f(mix(h, c, 0.1), 1.0);                     // 10% current/frame: static converges, moving can't ghost
}
)";

// debug blit: dump one gbuffer image straight to the swapchain, picked by scene.debugMode (1..4).
const char* kDebugFs = R"(
@group(0) @binding(1) var gAlbedo : texture_2d<f32>;
@group(0) @binding(2) var gNormal : texture_2d<f32>;
@group(0) @binding(3) var gRough  : texture_2d<f32>;
@group(0) @binding(4) var gDepth  : texture_depth_2d;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    var col = vec3f(1.0, 0.0, 1.0);                          // magenta = unhandled mode
    switch (scene.debugMode) {
        case 1u: { col = textureLoad(gAlbedo, px, 0).rgb; }
        case 2u: { col = textureLoad(gNormal, px, 0).rgb; } // stored as n*0.5+0.5
        case 3u: { col = vec3f(textureLoad(gRough, px, 0).r); }
        case 4u: { col = vec3f(textureLoad(gDepth, px, 0)); } // linear depth as grey
        default: {}
    }
    return vec4f(col, 1.0);
}
)";

// sky fill: re-write litColor on the BACKGROUND pixels only (a vertical gradient). Runs after lighting
// with LoadOp_Load -> a second writer of litColor == a WAW edge (lighting -> sky) in the graph.
const char* kSkyFs = R"(
@group(0) @binding(0) var gDepth : texture_depth_2d;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    if (textureLoad(gDepth, px, 0) < 1.0) { discard; }       // geometry -> keep the lit result
    let h = clamp(in.ndc.y * 0.5 + 0.5, 0.0, 1.0);           // 0 horizon .. 1 zenith
    return vec4f(mix(vec3f(0.45, 0.55, 0.70), vec3f(0.10, 0.20, 0.45), h), 1.0);
}
)";

// ndc (y up, [-1,1]) -> uv (y down, [0,1]) for the sampled-texture passes below.
const char* kUvHelper = R"(
fn uv_of(ndc : vec2f) -> vec2f { return vec2f(ndc.x, -ndc.y) * 0.5 + 0.5; }
)";

// bloom extract: bright-pass litColor into mip 0. soft knee keeps only the over-threshold part.
const char* kBloomExtractFs = R"(
@group(0) @binding(0) var src  : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let c = textureSampleLevel(src, samp, uv_of(in.ndc), 0.0).rgb;
    let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    return vec4f(c * max(l - 0.6, 0.0) / max(l, 1e-4), 1.0);
}
)";

// bloom blit: one bilinear tap from the bound source mip. shared by downsample (write the smaller mip,
// no blend = box filter) and upsample (write the larger mip, additive blend in the pipeline = tent-ish
// accumulate). the bound view is a single mip, so sampling level 0 always hits it.
const char* kBloomBlitFs = R"(
@group(0) @binding(0) var src  : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    return textureSampleLevel(src, samp, uv_of(in.ndc), 0.0);
}
)";

// bloom composite: lit color + the accumulated bloom (mip 0) -> swapchain.
const char* kBloomCompositeFs = R"(
@group(0) @binding(0) var litTex   : texture_2d<f32>;
@group(0) @binding(1) var bloomTex : texture_2d<f32>;
@group(0) @binding(2) var samp     : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let uv = uv_of(in.ndc);
    let base  = textureSampleLevel(litTex,   samp, uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb;
    return vec4f(base + bloom * 0.8, 1.0);
}
)";

// cubemap sample: the 6 faces (rendered as array layers) viewed as a cube, looked up by camera ray ->
// a rotatable skybox. proves per-layer attachment writes + a Cube-dimension view of the same texture.
const char* kCubeFs = R"(
@group(0) @binding(1) var cubeTex : texture_cube<f32>;
@group(0) @binding(2) var samp    : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let rd = camera_ray(in.ndc, scene.resolution.x / scene.resolution.y);
    return textureSample(cubeTex, samp, rd);
}
)";

// adapter + device against the given surface, pumped synchronously via the instance (kept alive by
// the caller). mirrors Framework.cpp's request flow, plus compatibleSurface so the adapter can present.
wgpu::Device acquire_device(wgpu::Instance instance, wgpu::Surface surface)
{
    struct AdState { wgpu::Adapter adapter; bool done = false; } as;
    wgpu::RequestAdapterOptions ao{};
    ao.powerPreference   = wgpu::PowerPreference::HighPerformance;
    ao.compatibleSurface = surface;
    ao.backendType       = wgpu::BackendType::D3D12;   // or D3D12 -- force a backend instead of Dawn's default pick
    instance.RequestAdapter(&ao, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView m, AdState* st){
            if (s != wgpu::RequestAdapterStatus::Success)
                std::printf("RequestAdapter failed: %.*s\n", (int)m.length, m.data);
            st->adapter = std::move(a);
            st->done = true;
        }, &as);
    while (!as.done) instance.ProcessEvents();
    if (!as.adapter) return {};

    wgpu::DeviceDescriptor devDesc{};
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType t, wgpu::StringView m){
            std::printf("GPU error (%d): %.*s\n", (int)t, (int)m.length, m.data);
        });
    // ponytail: no optional features needed -- every storage texture is rgba8unorm (a core storage
    // format) and present is a graphics blit, so the old BGRA8UnormStorage requirement is gone.

    struct DevState { wgpu::Device device; bool done = false; } ds;
    as.adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m, DevState* st){
            if (s != wgpu::RequestDeviceStatus::Success)
                std::printf("RequestDevice failed: %.*s\n", (int)m.length, m.data);
            st->device = std::move(d);
            st->done = true;
        }, &ds);
    while (!ds.done) instance.ProcessEvents();
    return ds.device;
}

} // anon

// ---------------------------------------------------------------------------------------------------
// compile()-only edge-case tests (NO GPU work): build tiny graphs and run them through compile(),
// checking only the bool return. Covers the consumer-before-producer error and the exemptions that
// must NOT false-positive. Bodies never run (compile() doesn't invoke them); realize()/execute() are
// never called. Runs once at startup, before the render loop.
namespace {

int g_rgTestFails = 0;

void rg_expect(bool got, bool want, const char* name)
{
    const bool ok = (got == want);
    if (!ok) ++g_rgTestFails;
    std::printf("[rg-test] %-42s %s  (compile()=%s)\n",
                name, ok ? "PASS" : "FAIL", got ? "true" : "false");
}

// plain pass/fail check for assertions that aren't about compile()'s return (e.g. computed lifetimes).
void rg_check(bool ok, const char* name)
{
    if (!ok) ++g_rgTestFails;
    std::printf("[rg-test] %-42s %s\n", name, ok ? "PASS" : "FAIL");
}

void run_rg_compile_tests(RG::GraphAllocator* alloc)
{
    using namespace RG;
    auto noop = [](PassContext&){};                 // bodies never run -- compile() doesn't invoke them
    // ordering errors are reported only when validation is compiled in (RG_VALIDATE); a release build
    // (NDEBUG) strips the check, so compile() can't fail and the error cases below return true instead.
    const bool errCaught = (RG_VALIDATE != 0);
    std::printf("[rg-test] compile()-only edge cases (validation %s):\n", errCaught ? "ON" : "OFF");

    // OK: producer declared before its consumer.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto x   = g->create_buffer(WEBGPU_STR("x"), { .size = 16 });
        g->add_pass(WEBGPU_STR("producer"), PassKind::Compute,  [&](GraphBuilder& b){ b.storage_write(x); }, noop);
        g->add_pass(WEBGPU_STR("consumer"), PassKind::Graphics, [&](GraphBuilder& b){ b.storage_read(x); b.color(out); }, noop);
        rg_expect(g->compile(), true, "producer before consumer");
    }

    // ERROR: consumer declared before producer of a transient (both survive -- both write the sink).
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto x   = g->create_buffer(WEBGPU_STR("x"), { .size = 16 });
        g->add_pass(WEBGPU_STR("consumer"), PassKind::Graphics, [&](GraphBuilder& b){ b.storage_read(x);  b.color(out); }, noop);
        g->add_pass(WEBGPU_STR("producer"), PassKind::Graphics, [&](GraphBuilder& b){ b.storage_write(x); b.color(out); }, noop);
        rg_expect(g->compile(), !errCaught, "consumer before producer (transient)");
    }

    // OK: a transient read but never written by any pass (host-uploaded uniform) -- hasWriter exemption.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto ubo = g->create_buffer(WEBGPU_STR("ubo"), { .size = 16 });
        g->add_pass(WEBGPU_STR("use"), PassKind::Graphics, [&](GraphBuilder& b){ b.uniform(ubo); b.color(out); }, noop);
        rg_expect(g->compile(), true, "read-never-written transient (ubo)");
    }

    // OK: an imported resource read before an in-graph write (legal "read then overwrite" WAR).
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto imp = g->import_buffer(WEBGPU_STR("imp"), nullptr);
        g->add_pass(WEBGPU_STR("read.imported"),  PassKind::Graphics, [&](GraphBuilder& b){ b.storage_read(imp);  b.color(out); }, noop);
        g->add_pass(WEBGPU_STR("write.imported"), PassKind::Compute,  [&](GraphBuilder& b){ b.storage_write(imp); }, noop);
        rg_expect(g->compile(), true, "imported read-before-write");
    }

    // OK: multi-writer chain (v1 written, read, v2 written, read) -- no false early-read error.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto x   = g->create_buffer(WEBGPU_STR("x"), { .size = 16 });
        g->add_pass(WEBGPU_STR("prepass"), PassKind::Compute,  [&](GraphBuilder& b){ b.storage_write(x); }, noop);                   // x v1
        g->add_pass(WEBGPU_STR("read.v1"), PassKind::Compute,  [&](GraphBuilder& b){ b.storage_read(x); }, noop);
        g->add_pass(WEBGPU_STR("main"),    PassKind::Compute,  [&](GraphBuilder& b){ b.storage_read(x); b.storage_write(x); }, noop); // v1 -> v2
        g->add_pass(WEBGPU_STR("read.v2"), PassKind::Graphics, [&](GraphBuilder& b){ b.storage_read(x); b.color(out); }, noop);
        rg_expect(g->compile(), true, "multi-writer chain");
    }

    // ERROR: a pass reads then writes the same transient that nothing else produced -> reads uninitialized.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto x   = g->create_buffer(WEBGPU_STR("x"), { .size = 16 });
        g->add_pass(WEBGPU_STR("rmw"), PassKind::Graphics, [&](GraphBuilder& b){ b.storage_read(x); b.storage_write(x); b.color(out); }, noop);
        rg_expect(g->compile(), !errCaught, "within-pass read-then-write (unproduced)");
    }

    // OK: same resource sampled twice in one pass (read+read) -- the build-time use() check must NOT trip
    // (no write involved). The illegal cases it DOES catch (sampled+storage_write, double write) assert at
    // declaration time, so they can't be exercised here without aborting the run -- verified manually.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto tex = g->create_image(WEBGPU_STR("tex"), { .dimension = WGPUTextureDimension_2D, .absolute = { 1, 1, 1 } });
        g->add_pass(WEBGPU_STR("read.twice"), PassKind::Graphics, [&](GraphBuilder& b){ b.sampled(tex); b.sampled(tex); b.color(out); }, noop);
        rg_expect(g->compile(), true, "sampled twice (read+read)");
    }

    // OK: read-only depth + sampled of the same texture in one pass (both reads) -- must NOT trip either.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out   = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto depth = g->create_image(WEBGPU_STR("depth"), { .dimension = WGPUTextureDimension_2D, .absolute = { 1, 1, 1 } });
        g->add_pass(WEBGPU_STR("depth.ro+sampled"), PassKind::Graphics, [&](GraphBuilder& b){ b.depth_stencil_read_only(depth); b.sampled(depth); b.color(out); }, noop);
        rg_expect(g->compile(), true, "read-only depth + sampled (read+read)");
    }

    // lifetimes: producer writes transient t (pass 0); consumer samples t and writes the sink (pass 1).
    // phase 3 records firstUse/lastUse over the post-cull execution order; the imported sink is excluded.
    {
        RenderGraph* g = create_render_graph(alloc);
        auto out = g->importe_image(WEBGPU_STR("out"), nullptr, { 1, 1, 1 });
        auto t   = g->create_image(WEBGPU_STR("t"), { .dimension = WGPUTextureDimension_2D, .absolute = { 1, 1, 1 } });
        g->add_pass(WEBGPU_STR("producer"), PassKind::Graphics, [&](GraphBuilder& b){ b.color(t); }, noop);
        g->add_pass(WEBGPU_STR("consumer"), PassKind::Graphics, [&](GraphBuilder& b){ b.sampled(t); b.color(out); }, noop);
        const bool compiled = g->compile();
        ResourceNode* rt = g->node(t);
        ResourceNode* ro = g->node(out);
        const bool span   = rt && rt->firstUse == 0 && rt->lastUse == 1;     // transient spans both passes
        const bool impOut = ro && ro->firstUse == ResourceNode::kNoPass;     // imported left out
        rg_check(compiled && span && impOut, "lifetime first/last use (imported excluded)");
    }

    std::printf("[rg-test] %s (%d failure%s)\n\n",
                g_rgTestFails ? "FAILURES" : "all passed", g_rgTestFails, g_rgTestFails == 1 ? "" : "s");
}

} // anon

int main()
{
    using namespace RG;
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: prints/errors show live (and survive a kill)

    // ---- window ----
    SDL_SetMainReady();
    if (!SDL_Init(SDL_INIT_VIDEO)) { std::printf("SDL_Init failed: %s\n", SDL_GetError()); return 1; }
    uint32_t curW = 1280, curH = 720;
    SDL_Window* window = SDL_CreateWindow("RenderGraph sample", (int)curW, (int)curH, SDL_WINDOW_RESIZABLE);
    if (!window) { std::printf("SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }

    // ---- WebGPU instance + surface from the window (HWND glue, same as main_SDL.cpp) ----
    wgpu::InstanceDescriptor instDesc{};
    wgpu::Instance instance = wgpu::CreateInstance(&instDesc);

    void* hwnd  = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    void* hinst = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
    wgpu::SurfaceSourceWindowsHWND src{};
    src.hwnd = hwnd; src.hinstance = hinst;
    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &src;
    wgpu::Surface surface = instance.CreateSurface(&surfDesc);

    // ---- device ----
    wgpu::Device device = acquire_device(instance, surface);
    if (!device) { std::printf("no device, aborting\n"); return 1; }
    wgpu::Queue queue = device.GetQueue();
    WGPUDevice  dev  = device.Get();
    WGPUQueue   q    = queue.Get();
    WGPUSurface surf = surface.Get();

    // ---- configure surface (field order copied verbatim from Renderer.cpp::reconfigure_surface) ----
    WGPUSurfaceConfiguration cfg{
        .device      = dev,
        .format      = kSwapFormat,
        .usage       = WGPUTextureUsage_RenderAttachment,   // compose/present write it as a color attachment
        .width       = curW,
        .height      = curH,
        .alphaMode   = WGPUCompositeAlphaMode_Auto,
        .presentMode = WGPUPresentMode_Fifo,
    };
    wgpuSurfaceConfigure(surf, &cfg);

    // ---- pipelines (created once: they depend on formats, not on the per-frame sizes) -------------
    // depth-stencil shared by the gbuffer + shadow passes: write depth, Less, stencil disabled
    // (depth-only format -> stencil faces must be the no-op Always/Keep default).
    WGPUStencilFaceState stencilNop{
        .compare     = WGPUCompareFunction_Always,
        .failOp      = WGPUStencilOperation_Keep,
        .depthFailOp = WGPUStencilOperation_Keep,
        .passOp      = WGPUStencilOperation_Keep,
    };
    WGPUDepthStencilState depthWrite{
        .format            = kDepthFormat,
        .depthWriteEnabled = WGPUOptionalBool_True,
        .depthCompare      = WGPUCompareFunction_Less,
        .stencilFront      = stencilNop,
        .stencilBack       = stencilNop,
        .stencilReadMask   = 0xFFFFFFFF,
        .stencilWriteMask  = 0xFFFFFFFF,
    };

    // gbuffer: fullscreen raymarch -> 3 color targets (albedo/normal/rough) + depth.
    WGPUShaderModule gbufferSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kSdfDefs + kGbufferFs);
    WGPUColorTargetState gbCT[3] = {
        { .format = kColorFormat, .writeMask = WGPUColorWriteMask_All },
        { .format = kColorFormat, .writeMask = WGPUColorWriteMask_All },
        { .format = kColorFormat, .writeMask = WGPUColorWriteMask_All },
    };
    WGPUFragmentState gbFrag{ .module = gbufferSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 3, .targets = gbCT };
    WGPURenderPipelineDescriptor gbPD{
        .label        = WEBGPU_STR("gbuffer pipeline"),
        .vertex       = { .module = gbufferSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive    = { .topology = WGPUPrimitiveTopology_TriangleList },
        .depthStencil = &depthWrite,
        .multisample  = { .count = 1, .mask = ~0u },
        .fragment     = &gbFrag,
    };
    WGPURenderPipeline gbufferPipe = wgpuDeviceCreateRenderPipeline(dev, &gbPD);
    wgpuShaderModuleRelease(gbufferSM);

    // shadow: fullscreen raymarch from the light -> depth only (no color targets).
    WGPUShaderModule shadowSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kSdfDefs + kShadowFs);
    WGPUFragmentState shadowFrag{ .module = shadowSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 0, .targets = nullptr };
    WGPURenderPipelineDescriptor shadowPD{
        .label        = WEBGPU_STR("shadow pipeline"),
        .vertex       = { .module = shadowSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive    = { .topology = WGPUPrimitiveTopology_TriangleList },
        .depthStencil = &depthWrite,
        .multisample  = { .count = 1, .mask = ~0u },
        .fragment     = &shadowFrag,
    };
    WGPURenderPipeline shadowPipe = wgpuDeviceCreateRenderPipeline(dev, &shadowPD);
    wgpuShaderModuleRelease(shadowSM);

    // lighting: fullscreen -> lit color from gbuffer + shadow map.
    WGPUShaderModule lightSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kLightingFs);
    WGPUColorTargetState lightCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState lightFrag{ .module = lightSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &lightCT };
    WGPURenderPipelineDescriptor lightPD{
        .label       = WEBGPU_STR("lighting pipeline"),
        .vertex      = { .module = lightSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &lightFrag,
    };
    WGPURenderPipeline lightingPipe = wgpuDeviceCreateRenderPipeline(dev, &lightPD);
    wgpuShaderModuleRelease(lightSM);

    // ssao: compute, reconstructs world positions from gbuffer depth + normal (needs the UBO camera).
    WGPUShaderModule ssaoSM = make_shader(dev, std::string(kSceneDecl) + kCameraDefs + kSsaoBody);
    WGPUComputePipelineDescriptor ssaoPD{
        .label   = WEBGPU_STR("ssao pipeline"),
        .compute = { .module = ssaoSM, .entryPoint = WEBGPU_STR("main") },
    };
    WGPUComputePipeline ssaoPipe = wgpuDeviceCreateComputePipeline(dev, &ssaoPD);
    wgpuShaderModuleRelease(ssaoSM);

    // compose (SSAO on): lit * ao -> swapchain.
    WGPUShaderModule composeSM = make_shader(dev, std::string(kVS) + kComposeFs);
    WGPUColorTargetState composeCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState composeFrag{ .module = composeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &composeCT };
    WGPURenderPipelineDescriptor composePD{
        .label       = WEBGPU_STR("compose pipeline"),
        .vertex      = { .module = composeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &composeFrag,
    };
    WGPURenderPipeline composePipe = wgpuDeviceCreateRenderPipeline(dev, &composePD);
    wgpuShaderModuleRelease(composeSM);

    // present (SSAO off): lit color straight to the swapchain.
    WGPUShaderModule presentSM = make_shader(dev, std::string(kVS) + kPresentFs);
    WGPUColorTargetState presentCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState presentFrag{ .module = presentSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &presentCT };
    WGPURenderPipelineDescriptor presentPD{
        .label       = WEBGPU_STR("present pipeline"),
        .vertex      = { .module = presentSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &presentFrag,
    };
    WGPURenderPipeline presentPipe = wgpuDeviceCreateRenderPipeline(dev, &presentPD);
    wgpuShaderModuleRelease(presentSM);

    // taa accumulation: writes the history texture (kSwapFormat, NOT the swapchain) by blending the
    // current shaded colour with the previous history layer. history is kSwapFormat so the compose/
    // present pipelines can shade the TAA input into the same format that feeds it.
    WGPUShaderModule taaSM = make_shader(dev, std::string(kVS) + kTaaFs);
    WGPUColorTargetState taaCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState taaFrag{ .module = taaSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &taaCT };
    WGPURenderPipelineDescriptor taaPD{
        .label       = WEBGPU_STR("taa pipeline"),
        .vertex      = { .module = taaSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &taaFrag,
    };
    WGPURenderPipeline taaPipe = wgpuDeviceCreateRenderPipeline(dev, &taaPD);
    wgpuShaderModuleRelease(taaSM);

    // sky: fills litColor's background pixels (discards on geometry). 2nd writer of litColor -> WAW.
    WGPUShaderModule skySM = make_shader(dev, std::string(kVS) + kSkyFs);
    WGPUColorTargetState skyCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState skyFrag{ .module = skySM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &skyCT };
    WGPURenderPipelineDescriptor skyPD{
        .label       = WEBGPU_STR("sky pipeline"),
        .vertex      = { .module = skySM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &skyFrag,
    };
    WGPURenderPipeline skyPipe = wgpuDeviceCreateRenderPipeline(dev, &skyPD);
    wgpuShaderModuleRelease(skySM);

    // debug (keys 1..4): blit one gbuffer image straight to the swapchain.
    WGPUShaderModule debugSM = make_shader(dev, std::string(kVS) + kSceneDecl + kDebugFs);
    WGPUColorTargetState debugCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState debugFrag{ .module = debugSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &debugCT };
    WGPURenderPipelineDescriptor debugPD{
        .label       = WEBGPU_STR("debug pipeline"),
        .vertex      = { .module = debugSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &debugFrag,
    };
    WGPURenderPipeline debugPipe = wgpuDeviceCreateRenderPipeline(dev, &debugPD);
    wgpuShaderModuleRelease(debugSM);

    // linear/clamp sampler shared by the bloom + cubemap passes (the deferred passes use textureLoad and
    // need none). filterable because every sampled format here is *Unorm.
    WGPUSamplerDescriptor linSD{
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter    = WGPUFilterMode_Linear,
        .minFilter    = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Nearest,
        .maxAnisotropy = 1,
    };
    WGPUSampler linSampler = wgpuDeviceCreateSampler(dev, &linSD);

    // bloom (key B): bright-pass extract -> mip-chain downsample -> additive upsample -> composite.
    // extract + blit target kColorFormat (the bloom texture); composite targets kSwapFormat.
    WGPUShaderModule bloomExtractSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomExtractFs);
    WGPUColorTargetState bloomCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomExtractFrag{ .module = bloomExtractSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCT };
    WGPURenderPipelineDescriptor bloomExtractPD{
        .label       = WEBGPU_STR("bloom.extract pipeline"),
        .vertex      = { .module = bloomExtractSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &bloomExtractFrag,
    };
    WGPURenderPipeline bloomExtractPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomExtractPD);
    wgpuShaderModuleRelease(bloomExtractSM);

    // down + up share one fragment shader (a single bilinear tap); only the blend differs.
    WGPUShaderModule bloomBlitSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomBlitFs);
    WGPUFragmentState bloomDownFrag{ .module = bloomBlitSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCT };
    WGPURenderPipelineDescriptor bloomDownPD{
        .label       = WEBGPU_STR("bloom.down pipeline"),
        .vertex      = { .module = bloomBlitSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &bloomDownFrag,
    };
    WGPURenderPipeline bloomDownPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomDownPD);

    // upsample: additive (src*1 + dst*1) so each coarser level accumulates onto the finer one (LoadOp_Load).
    WGPUBlendState addBlend{
        .color = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
        .alpha = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
    };
    WGPUColorTargetState bloomUpCT{ .format = kColorFormat, .blend = &addBlend, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomUpFrag{ .module = bloomBlitSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomUpCT };
    WGPURenderPipelineDescriptor bloomUpPD{
        .label       = WEBGPU_STR("bloom.up pipeline"),
        .vertex      = { .module = bloomBlitSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &bloomUpFrag,
    };
    WGPURenderPipeline bloomUpPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomUpPD);
    wgpuShaderModuleRelease(bloomBlitSM);

    WGPUShaderModule bloomCompSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomCompositeFs);
    WGPUColorTargetState bloomCompCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomCompFrag{ .module = bloomCompSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCompCT };
    WGPURenderPipelineDescriptor bloomCompPD{
        .label       = WEBGPU_STR("bloom.composite pipeline"),
        .vertex      = { .module = bloomCompSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &bloomCompFrag,
    };
    WGPURenderPipeline bloomCompositePipe = wgpuDeviceCreateRenderPipeline(dev, &bloomCompPD);
    wgpuShaderModuleRelease(bloomCompSM);

    // cubemap (key C): sample the 6-layer cube by camera ray. needs the scene UBO for camera_ray.
    WGPUShaderModule cubeSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kCubeFs);
    WGPUColorTargetState cubeCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState cubeFrag{ .module = cubeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &cubeCT };
    WGPURenderPipelineDescriptor cubePD{
        .label       = WEBGPU_STR("cube.sample pipeline"),
        .vertex      = { .module = cubeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive   = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment    = &cubeFrag,
    };
    WGPURenderPipeline cubeSamplePipe = wgpuDeviceCreateRenderPipeline(dev, &cubePD);
    wgpuShaderModuleRelease(cubeSM);

    // single persistent arena. create_render_graph() resets it and arena-allocates a fresh
    // RenderGraph (+ all its nodes) from it each frame -> the allocator is the only graph-side
    // object that lives across frames. It also owns the two resource pools (folded in): the
    // persistent pool (temporal/history textures, create_temporal_image) and the transient pool
    // (per-frame gbuffer/lit/shadow textures reused across teardown). Both outlive the frame loop.
    GraphAllocator* allocator = create_allocator();

    run_rg_compile_tests(allocator);   // compile()-only edge-case checks (no GPU); next create_render_graph() resets the arena

    // ---- frame loop: declare + compile + realize + execute + release the graph EVERY frame -------
    bool ssaoOn     = true;  // SPACE toggles the SSAO pass (and compose<->present)
    bool taaOn      = true;  // T toggles temporal accumulation (demo of create_temporal_image)
    bool bloomOn    = false; // B toggles the bloom mip-chain (demo of mip subresource access)
    bool cubeOn     = false; // C toggles the cubemap render+sample (demo of array-layer subresource access)
    int  debugMode  = 0;     // 0 = lit, 1..4 = blit gbuffer albedo/normal/roughness/depth
    int  shownSsao  = -1;    // last (ssao, debug) state we printed the execution order for
    int  shownDebug = -1;
    bool running    = true;
    const char* kDebugName[] = { "off", "albedo", "normal", "roughness", "depth", "ssao", "csm cascades" };

    // free-fly camera: WASD moves, left-drag rotates (yaw/pitch). Inits to the old fixed view.
    float  camPos[3] = { 0.0f, 1.9f, 5.5f };
    float  camYaw    = 0.0f;     // 0 -> looking down -Z
    float  camPitch  = -0.36f;   // slight downward tilt
    bool   dragging  = false;
    double prevTime  = getTime();
    const float kMouseSens = 0.0025f, kMoveSpeed = 4.0f;

    imgui_layer_init(window, dev, kSwapFormat);

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE) running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_SPACE) {
                ssaoOn = !ssaoOn;
                std::printf("ssao %s\n", ssaoOn ? "ON" : "OFF");
            }
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_T) {
                taaOn = !taaOn;
                std::printf("taa %s\n", taaOn ? "ON" : "OFF");
            }
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_B) {
                bloomOn = !bloomOn;
                std::printf("bloom %s\n", bloomOn ? "ON" : "OFF");
            }
            else if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_C) {
                cubeOn = !cubeOn;
                std::printf("cubemap %s\n", cubeOn ? "ON" : "OFF");
            }
            else if (e.type == SDL_EVENT_KEY_DOWN &&
                     (e.key.scancode == SDL_SCANCODE_0 ||
                      (e.key.scancode >= SDL_SCANCODE_1 && e.key.scancode <= SDL_SCANCODE_6))) {
                debugMode = (e.key.scancode == SDL_SCANCODE_0) ? 0 : (int)(e.key.scancode - SDL_SCANCODE_1) + 1;
                std::printf("debug: %s\n", kDebugName[debugMode]);
            }
            else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT
                     && !ImGui::GetIO().WantCaptureMouse) {     // don't start a look-drag on a UI click
                dragging = true;
                SDL_SetWindowRelativeMouseMode(window, true);   // capture + hide cursor while looking
            }
            else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP && e.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
                SDL_SetWindowRelativeMouseMode(window, false);
            }
            else if (e.type == SDL_EVENT_MOUSE_MOTION && dragging) {
                camYaw   += e.motion.xrel * kMouseSens;
                camPitch -= e.motion.yrel * kMouseSens;
                const float lim = 1.5533f;                       // ~89 deg, avoid gimbal flip
                camPitch = camPitch >  lim ?  lim : (camPitch < -lim ? -lim : camPitch);
            }
            else if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                cfg.width = (uint32_t)e.window.data1; cfg.height = (uint32_t)e.window.data2;
                wgpuSurfaceConfigure(surf, &cfg);
            }
        }

        // camera basis from yaw/pitch, then WASD fly-movement (frame-rate independent via dt).
        double now = getTime();
        float  dt  = (float)(now - prevTime);
        prevTime   = now;
        float cp = cosf(camPitch), sp = sinf(camPitch), sy = sinf(camYaw), cy = cosf(camYaw);
        float fwd[3]   = { cp * sy, sp, -cp * cy };
        float right[3] = { cy, 0.0f, sy };
        float up[3]    = { -sy * sp, cp, cy * sp };
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            auto ks = SDL_GetKeyboardState(nullptr);   // const bool* (SDL3); index with scancodes
            float mv = kMoveSpeed * dt;
            float f = (ks[SDL_SCANCODE_W] ? mv : 0.0f) - (ks[SDL_SCANCODE_S] ? mv : 0.0f);
            float r = (ks[SDL_SCANCODE_D] ? mv : 0.0f) - (ks[SDL_SCANCODE_A] ? mv : 0.0f);
            for (int i = 0; i < 3; ++i) camPos[i] += fwd[i] * f + right[i] * r;
        }

        imgui_layer_begin_frame();   // NewFrame only; the DAG window is built after compile, Render() in end_frame

        ImGui::Begin("Features");
        ImGui::Checkbox("SSAO", &ssaoOn);
        ImGui::Checkbox("TAA", &taaOn);
        ImGui::Checkbox("Bloom (B)", &bloomOn);
        ImGui::Checkbox("Cubemap (C)", &cubeOn);
        ImGui::End();

        WGPUSurfaceTexture st{};
        wgpuSurfaceGetCurrentTexture(surf, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            wgpuSurfaceConfigure(surf, &cfg);     // surface went stale (resize/minimize) -> reconfigure, skip frame
            imgui_layer_end_frame();              // balance begin_frame's NewFrame on the skipped frame
            continue;
        }

        WGPUTextureViewDescriptor vd{
            .format          = kSwapFormat,
            .dimension       = WGPUTextureViewDimension_2D,
            .baseMipLevel    = 0, .mipLevelCount   = 1,
            .baseArrayLayer  = 0, .arrayLayerCount = 1,
            .aspect          = WGPUTextureAspect_All,
        };
        WGPUTextureView view = wgpuTextureCreateView(st.texture, &vd);

        // ---- declare the whole graph for THIS frame (immediate mode) ----
        RenderGraph* rg = create_render_graph(allocator);   // resets the arena, fresh RenderGraph (pools persist in the allocator)

        // import this frame's swapchain view; its size roots every Relative texture below.
        auto swapchain = rg->importe_image(WEBGPU_STR("swapchain"), view, { cfg.width, cfg.height, 1 });

        // helper for a swapchain-sized offscreen texture (WGPU_STRLEN -> copy_string measures it).
        auto screenTex = [&](const char* name, WGPUTextureFormat fmt) {
            return rg->create_image(WGPUStringView{ name, WGPU_STRLEN }, {
                .dimension = WGPUTextureDimension_2D, .format = fmt,
                .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
            });
        };
        auto gAlbedo = screenTex("gbuffer.albedo", kColorFormat);
        auto gNormal = screenTex("gbuffer.normal", kColorFormat);
        auto gRough  = screenTex("gbuffer.rough",  kColorFormat);
        auto gDepth  = screenTex("gbuffer.depth",  kDepthFormat);
        auto litColor = screenTex("lighting.color", kColorFormat);
        auto csm = rg->create_image(WEBGPU_STR("shadow.csm"), {
            .dimension = WGPUTextureDimension_2D, .format = kDepthFormat,
            .sizeKind = SizeKind::Absolute, .absolute = { kShadowSize, kShadowSize, kNumCascades },
        });
        auto sceneUbo = rg->create_buffer(WEBGPU_STR("scene.ubo"), { .size = sizeof(SceneUBO) });
        ResourceHandle aoTex{};   // only on the SSAO path

        // 1) gbuffer: raymarch the SDF -> albedo / normal / roughness (MRT) + linear depth.
        rg->add_pass(WEBGPU_STR("gbuffer"), PassKind::Graphics,
            [&](GraphBuilder& b) {
                b.color(gAlbedo, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});         // @location(0)
                b.color(gNormal, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0.5, 0.5, 1.0, 1});   // @location(1)
                b.color(gRough,  WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});         // @location(2)
                b.depth_stencil(gDepth, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f);
                b.uniform(sceneUbo);
            },
            [dev, gbufferPipe, sceneUbo](PassContext& ctx) {
                WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(gbufferPipe, 0);
                WGPUBindGroupEntry e[1] = {
                    { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                };
                WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                wgpuRenderPassEncoderSetPipeline(ctx.render, gbufferPipe);
                wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                wgpuBindGroupRelease(bg);
                wgpuBindGroupLayoutRelease(l);
            });

        // 2) shadow cascades: raymarch the same SDF from the directional light at kNumCascades concentric
        // ortho extents, each into one layer of a 2D-array depth texture (per-layer attachment via baseLayer).
        // firstInstance = c carries the cascade index to the shader (instance_index -> csm_ortho(c)). The 3
        // writes share the handle, so the graph serializes them (WAW) and lighting RAW-depends on all -- same
        // shape as the 6 cube.face passes feeding cube.sample.
        for (uint32_t c = 0; c < kNumCascades; ++c) {
            rg->add_pass(WEBGPU_STR("shadow.cascade"), PassKind::Graphics,
                [&, c](GraphBuilder& b) {
                    b.depth_stencil(csm, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f, 0, c);   // baseMip 0, layer c
                    b.uniform(sceneUbo);
                },
                [dev, shadowPipe, sceneUbo, c](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(shadowPipe, 0);
                    WGPUBindGroupEntry e[1] = {
                        { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, shadowPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, c);   // firstInstance = c -> @builtin(instance_index)
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        }

        // 3) ssao (optional): screen-space AO from gbuffer depth + normal -> aoTex. Also forced on when
        // its debug view (key 5) is selected, so there's an AO buffer to blit.
        if (ssaoOn || debugMode == 5) {
            aoTex = screenTex("ssao.ao", kAOFormat);
            const uint32_t gx = (cfg.width + 7) / 8, gy = (cfg.height + 7) / 8;
            rg->add_pass(WEBGPU_STR("ssao"), PassKind::Compute,
                [&](GraphBuilder& b) {
                    b.uniform(sceneUbo);                 // camera basis for world reconstruction
                    b.sampled(gDepth);
                    b.sampled(gNormal);
                    b.storage_write(aoTex);
                },
                [dev, ssaoPipe, sceneUbo, gDepth, gNormal, aoTex, gx, gy](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(ssaoPipe, 0);
                    WGPUBindGroupEntry e[4] = {
                        { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                        { .binding = 1, .textureView = ctx.view(gDepth) },
                        { .binding = 2, .textureView = ctx.view(gNormal) },
                        { .binding = 3, .textureView = ctx.view(aoTex) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 4, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuComputePassEncoderSetPipeline(ctx.compute, ssaoPipe);
                    wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
                    wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, gx, gy, 1);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        }

        // 4) lighting: gbuffer + shadow map + directional light -> lit color.
        rg->add_pass(WEBGPU_STR("lighting"), PassKind::Graphics,
            [&](GraphBuilder& b) {
                b.sampled(gAlbedo);
                b.sampled(gNormal);
                b.sampled(gRough);
                b.sampled(gDepth);
                b.sampled(csm);
                b.uniform(sceneUbo);
                b.color(litColor, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
            },
            [dev, lightingPipe, sceneUbo, gAlbedo, gNormal, gRough, gDepth, csm](PassContext& ctx) {
                // the body builds its own array view (graph resolves the texture; no graph-side abstraction),
                // mirroring cube.sample below -- binds all kNumCascades layers as a single texture_depth_2d_array.
                WGPUTextureViewDescriptor avd{
                    .format = kDepthFormat, .dimension = WGPUTextureViewDimension_2DArray,
                    .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = kNumCascades,
                };
                WGPUTextureView csmView = wgpuTextureCreateView(ctx.texture(csm), &avd);
                WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(lightingPipe, 0);
                WGPUBindGroupEntry e[6] = {
                    { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                    { .binding = 1, .textureView = ctx.view(gAlbedo) },
                    { .binding = 2, .textureView = ctx.view(gNormal) },
                    { .binding = 3, .textureView = ctx.view(gRough) },
                    { .binding = 4, .textureView = ctx.view(gDepth) },
                    { .binding = 5, .textureView = csmView },
                };
                WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 6, .entries = e };
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                wgpuRenderPassEncoderSetPipeline(ctx.render, lightingPipe);
                wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                wgpuBindGroupRelease(bg);
                wgpuBindGroupLayoutRelease(l);
                wgpuTextureViewRelease(csmView);
            });

        // 4b) sky: 2nd writer of litColor (LoadOp_Load, background pixels only) -> WAW edge after lighting.
        rg->add_pass(WEBGPU_STR("sky"), PassKind::Graphics,
            [&](GraphBuilder& b) {
                b.sampled(gDepth);                                                              // which pixels are background
                b.color(litColor, WGPULoadOp_Load, WGPUStoreOp_Store);                          // keep lit geometry, add sky
            },
            [dev, skyPipe, gDepth](PassContext& ctx) {
                WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(skyPipe, 0);
                WGPUBindGroupEntry e[1] = {
                    { .binding = 0, .textureView = ctx.view(gDepth) },
                };
                WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                wgpuRenderPassEncoderSetPipeline(ctx.render, skyPipe);
                wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                wgpuBindGroupRelease(bg);
                wgpuBindGroupLayoutRelease(l);
            });

        // 5) final blit -> swapchain. debug active: dump one image and let the graph cull whatever isn't
        // needed to produce it (1-4 -> gbuffer, only gbuffer survives; 5 -> ao, gbuffer+ssao survive).
        // else SSAO on: lit * ao; else lit color (present).
        if (debugMode >= 1 && debugMode <= 4) {
            rg->add_pass(WEBGPU_STR("debug.blit"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.uniform(sceneUbo);                     // scene.debugMode picks the image
                    b.sampled(gAlbedo);
                    b.sampled(gNormal);
                    b.sampled(gRough);
                    b.sampled(gDepth);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, debugPipe, sceneUbo, gAlbedo, gNormal, gRough, gDepth](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(debugPipe, 0);
                    WGPUBindGroupEntry e[5] = {
                        { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                        { .binding = 1, .textureView = ctx.view(gAlbedo) },
                        { .binding = 2, .textureView = ctx.view(gNormal) },
                        { .binding = 3, .textureView = ctx.view(gRough) },
                        { .binding = 4, .textureView = ctx.view(gDepth) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 5, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, debugPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        } else if (debugMode == 5) {
            // AO is stored grey (ao,ao,ao) -> the present pipeline (blits .rgb) shows it as-is. Reading
            // aoTex keeps gbuffer+ssao alive; shadow/lighting cull away.
            rg->add_pass(WEBGPU_STR("debug.ssao"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(aoTex);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, presentPipe, aoTex](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
                    WGPUBindGroupEntry e[1] = {
                        { .binding = 0, .textureView = ctx.view(aoTex) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, presentPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        } else if (cubeOn) {
            // CUBEMAP (array-layer subresource): render 6 faces, one array layer each, into a single cube
            // texture, then sample it as a Cube by camera ray -> a rotatable skybox. Nothing reads litColor
            // here, so gbuffer/lighting/sky cull away -- only the 6 faces + sample + imgui survive.
            auto cube = rg->create_image(WEBGPU_STR("cube"), {
                .dimension = WGPUTextureDimension_2D, .format = kColorFormat,
                .sizeKind = SizeKind::Absolute, .absolute = { 256, 256, 6 }, .mipLevelCount = 1,
            });
            // WebGPU cube layer order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z. Clear-only passes (no draw): the
            // attachment clear IS the face; execute() builds the single-layer attachment view from baseLayer.
            const WGPUColor kFace[6] = {
                {0.80, 0.16, 0.16, 1}, {0.13, 0.55, 0.55, 1}, {0.16, 0.70, 0.20, 1},
                {0.66, 0.16, 0.66, 1}, {0.16, 0.22, 0.80, 1}, {0.72, 0.66, 0.16, 1},
            };
            for (uint32_t f = 0; f < 6; ++f) {
                rg->add_pass(WEBGPU_STR("cube.face"), PassKind::Graphics,
                    [&, f](GraphBuilder& b) {
                        b.color(cube, WGPULoadOp_Clear, WGPUStoreOp_Store, kFace[f], 0, f);   // baseMip 0, layer f
                    },
                    [](PassContext&) {});   // clear-only: nothing to record
            }
            rg->add_pass(WEBGPU_STR("cube.sample"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.uniform(sceneUbo);     // camera basis for the ray
                    b.sampled(cube);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, cubeSamplePipe, sceneUbo, cube, linSampler](PassContext& ctx) {
                    WGPUTextureViewDescriptor cvd{
                        .format = kColorFormat, .dimension = WGPUTextureViewDimension_Cube,
                        .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 6,
                    };
                    WGPUTextureView cubeView = wgpuTextureCreateView(ctx.texture(cube), &cvd);
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(cubeSamplePipe, 0);
                    WGPUBindGroupEntry e[3] = {
                        { .binding = 0, .buffer = ctx.buffer(sceneUbo), .offset = 0, .size = sizeof(SceneUBO) },
                        { .binding = 1, .textureView = cubeView },
                        { .binding = 2, .sampler = linSampler },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, cubeSamplePipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                    wgpuTextureViewRelease(cubeView);
                });
        } else if (bloomOn) {
            // BLOOM (mip subresource): bright-pass litColor into a mip chain, progressively downsample
            // (sample mip i, write mip i+1 of the SAME texture), additively upsample back, composite over
            // lit. Each down/up pass reads one mip + writes an adjacent one in one pass -- the subresource-
            // aware in-pass check allows it; the shared-handle RAW/WAW chain serializes the whole pyramid.
            uint32_t minDim = cfg.width < cfg.height ? cfg.width : cfg.height;
            uint32_t bloomMips = 1;
            while (bloomMips < 6 && (1u << bloomMips) <= minDim) ++bloomMips;   // smallest mip stays >= 1px, cap 6
            auto bloom = rg->create_image(WEBGPU_STR("bloom"), {
                .dimension = WGPUTextureDimension_2D, .format = kColorFormat,
                .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
                .mipLevelCount = bloomMips,
            });

            // extract: bright parts of lit color -> mip 0.
            rg->add_pass(WEBGPU_STR("bloom.extract"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(litColor);
                    b.color(bloom, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1}, 0);
                },
                [dev, bloomExtractPipe, litColor, linSampler](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomExtractPipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(litColor) },
                        { .binding = 1, .sampler = linSampler },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, bloomExtractPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });

            // downsample: sample mip i, render into mip i+1 (Clear).
            for (uint32_t i = 0; i + 1 < bloomMips; ++i) {
                rg->add_pass(WEBGPU_STR("bloom.down"), PassKind::Graphics,
                    [&, i](GraphBuilder& b) {
                        b.sampled(bloom, i);
                        b.color(bloom, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1}, i + 1);
                    },
                    [dev, bloomDownPipe, bloom, linSampler, i](PassContext& ctx) {
                        WGPUTextureView srcView = mip_view_2d(ctx.texture(bloom), kColorFormat, i);
                        WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomDownPipe, 0);
                        WGPUBindGroupEntry e[2] = {
                            { .binding = 0, .textureView = srcView },
                            { .binding = 1, .sampler = linSampler },
                        };
                        WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                        wgpuRenderPassEncoderSetPipeline(ctx.render, bloomDownPipe);
                        wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                        wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                        wgpuBindGroupRelease(bg);
                        wgpuBindGroupLayoutRelease(l);
                        wgpuTextureViewRelease(srcView);
                    });
            }

            // upsample: sample mip i (coarser), additively blend into mip i-1 (LoadOp_Load keeps the
            // finer level the downsample wrote this frame).
            for (uint32_t i = bloomMips; i-- > 1; ) {
                rg->add_pass(WEBGPU_STR("bloom.up"), PassKind::Graphics,
                    [&, i](GraphBuilder& b) {
                        b.sampled(bloom, i);
                        b.color(bloom, WGPULoadOp_Load, WGPUStoreOp_Store, WGPUColor{}, i - 1);
                    },
                    [dev, bloomUpPipe, bloom, linSampler, i](PassContext& ctx) {
                        WGPUTextureView srcView = mip_view_2d(ctx.texture(bloom), kColorFormat, i);
                        WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomUpPipe, 0);
                        WGPUBindGroupEntry e[2] = {
                            { .binding = 0, .textureView = srcView },
                            { .binding = 1, .sampler = linSampler },
                        };
                        WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                        wgpuRenderPassEncoderSetPipeline(ctx.render, bloomUpPipe);
                        wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                        wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                        wgpuBindGroupRelease(bg);
                        wgpuBindGroupLayoutRelease(l);
                        wgpuTextureViewRelease(srcView);
                    });
            }

            // composite: lit color + accumulated bloom (mip 0) -> swapchain.
            rg->add_pass(WEBGPU_STR("bloom.composite"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(litColor);
                    b.sampled(bloom, 0);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, bloomCompositePipe, litColor, bloom, linSampler](PassContext& ctx) {
                    WGPUTextureView mip0 = mip_view_2d(ctx.texture(bloom), kColorFormat, 0);
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomCompositePipe, 0);
                    WGPUBindGroupEntry e[3] = {
                        { .binding = 0, .textureView = ctx.view(litColor) },
                        { .binding = 1, .textureView = mip0 },
                        { .binding = 2, .sampler = linSampler },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, bloomCompositePipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                    wgpuTextureViewRelease(mip0);
                });
        } else if (taaOn) {
            // Run TAA on the FINAL shaded colour, not raw lit -- otherwise it bypasses the lit*ao compose
            // and swallows SSAO. With SSAO on, compose lit*ao into an intermediate first and accumulate
            // THAT; with SSAO off, lit IS the final colour, so TAA samples it directly (textureLoad doesn't
            // care that litColor is kColorFormat while history is kSwapFormat).
            ResourceHandle taaInput = litColor;
            if (ssaoOn) {
                auto sceneColor = screenTex("taa.scene", kSwapFormat);
                rg->add_pass(WEBGPU_STR("taa.compose"), PassKind::Graphics,
                    [&](GraphBuilder& b) {
                        b.sampled(litColor);
                        b.sampled(aoTex);
                        b.color(sceneColor, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                    },
                    [dev, composePipe, litColor, aoTex](PassContext& ctx) {
                        WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(composePipe, 0);
                        WGPUBindGroupEntry e[2] = {
                            { .binding = 0, .textureView = ctx.view(litColor) },
                            { .binding = 1, .textureView = ctx.view(aoTex) },
                        };
                        WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                        wgpuRenderPassEncoderSetPipeline(ctx.render, composePipe);
                        wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                        wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                        wgpuBindGroupRelease(bg);
                        wgpuBindGroupLayoutRelease(l);
                    });
                taaInput = sceneColor;
            }

            // history: ping-pong temporal resource. .curr = this frame's write target, .prev = last
            // frame's result (read-only). the pool swaps the two physical textures every frame, so this
            // frame's result becomes next frame's history
            auto [historyCurr, historyPrev] = rg->create_temporal_image(WEBGPU_STR("taa.history"), {
                .dimension = WGPUTextureDimension_2D, .format = kSwapFormat,
                .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swapchain,
            });

            rg->add_pass(WEBGPU_STR("taa"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(taaInput);                 // this frame's shaded colour (lit*ao or lit)
                    b.sampled(historyPrev);             // previous frame (rotated in by the pool)
                    b.color(historyCurr, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, taaPipe, taaInput, historyCurr, historyPrev](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(taaPipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(taaInput) },
                        { .binding = 1, .textureView = ctx.view(historyPrev) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, taaPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });

            rg->add_pass(WEBGPU_STR("taa.present"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(historyCurr);             // this frame's accumulated result
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, presentPipe, historyCurr](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
                    WGPUBindGroupEntry e[1] = {
                        { .binding = 0, .textureView = ctx.view(historyCurr) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, presentPipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        } else if (ssaoOn) {
            rg->add_pass(WEBGPU_STR("compose"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(litColor);
                    b.sampled(aoTex);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, composePipe, litColor, aoTex](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(composePipe, 0);
                    WGPUBindGroupEntry e[2] = {
                        { .binding = 0, .textureView = ctx.view(litColor) },
                        { .binding = 1, .textureView = ctx.view(aoTex) },
                    };
                    WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
                    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
                    wgpuRenderPassEncoderSetPipeline(ctx.render, composePipe);
                    wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
                    wgpuBindGroupRelease(bg);
                    wgpuBindGroupLayoutRelease(l);
                });
        } else {
            rg->add_pass(WEBGPU_STR("present"), PassKind::Graphics,
                [&](GraphBuilder& b) {
                    b.sampled(litColor);
                    b.color(swapchain, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
                },
                [dev, presentPipe, litColor](PassContext& ctx) {
                    WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
                    WGPUBindGroupEntry e[1] = {
                        { .binding = 0, .textureView = ctx.view(litColor) },
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

        // ImGui overlay: last pass. Load keeps the rendered scene; the write to the imported
        // swapchain makes it a sink, and a WAW edge orders it after present/debug. ctx.render is
        // the open render-pass encoder execute() hands us.
        rg->add_pass(WEBGPU_STR("imgui"), PassKind::Graphics,
            [&](GraphBuilder& b) {
                b.color(swapchain, WGPULoadOp_Load, WGPUStoreOp_Store);
            },
            [](PassContext& ctx) {
                ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), ctx.render);
            });

        if (!rg->compile()) {
            // ordering error (a transient resource read before its writer); compile() already printed
            // the offending pass/resource. skip this frame's GPU work instead of rendering garbage.
            wgpuTextureViewRelease(view);
            wgpuTextureRelease(st.texture);
            instance.ProcessEvents();
            imgui_layer_end_frame();   // balance begin_frame's NewFrame on the skipped frame
            continue;
        }
        rg->realize(dev);     // creates this frame's gbuffer/shadow/lit/ao textures (+ scene ubo)

        imgui_layer_draw_graph(rg);   // build the DAG window now the graph is compiled + realized
        imgui_layer_end_frame();      // ImGui::Render(); the "imgui" pass consumes the draw data at execute

        // proof the per-frame graph really changes shape: print the order whenever SSAO or debug flips.
        if ((int)ssaoOn != shownSsao || debugMode != shownDebug) {
            shownSsao = (int)ssaoOn; shownDebug = debugMode;
            std::printf("execution order:");
            for (PassNode* p = storage(rg)->m_passes; p; p = p->next) std::printf(" %s", p->name.data);
            std::printf("\n");
            debug_print_mermaid(rg);
            debug_print_lifetimes(rg);
            // transient cache proof: toggle a feature off then back on within kRetain frames and the
            // return shows 0 created -> textures were reused from the pool, not reallocated.
            std::printf("transient pool: %zu textures, %u created this frame\n",
                        allocator->transient.entries.size(), allocator->transient.createdThisFrame);
        }

        // host-upload the scene + a slowly rotating directional light. The spheres are packed into a
        // pile resting ON the ground (centers within a radius of each other) so they overlap into tight
        // concave necks and sit in ground contact -- that's where SSAO actually darkens. Spread them
        // out and the AO has nothing to occlude.
        SceneUBO sb{};
        float t = (float)getTime();
        const float gy = -1.5f;                                  // ground plane height (matches map())
        sb.count = 7;
        sb.spheres[0][0] = 0.0f; sb.spheres[0][1] = gy + 0.9f; sb.spheres[0][2] = 0.0f; sb.spheres[0][3] = 0.9f;  // center, on ground
        for (uint32_t i = 0; i < 5; ++i) {                       // ring hugging the center + each other + ground
            float a = (float)i * 1.25664f + t * 0.3f;            // 72deg apart, slow spin
            sb.spheres[1 + i][0] = cosf(a) * 1.1f;
            sb.spheres[1 + i][1] = gy + 0.6f;
            sb.spheres[1 + i][2] = sinf(a) * 1.1f;
            sb.spheres[1 + i][3] = 0.6f;
        }
        sb.spheres[6][0] = 0.0f;                                 // capstone nestled on top, gentle bob
        sb.spheres[6][1] = gy + 1.9f + sinf(t * 1.5f) * 0.12f;
        sb.spheres[6][2] = 0.0f; sb.spheres[6][3] = 0.5f;
        float la = 0.8f;                                         // frozen: static light dir -> shadow map stops sweeping
        float lx = cosf(la) * 0.5f, ly = -1.0f, lz = sinf(la) * 0.5f;
        float ln = sqrtf(lx * lx + ly * ly + lz * lz);
        sb.lightDir[0] = lx / ln; sb.lightDir[1] = ly / ln; sb.lightDir[2] = lz / ln;
        sb.lightColor[0] = 1.0f; sb.lightColor[1] = 0.96f; sb.lightColor[2] = 0.9f;
        for (int i = 0; i < 3; ++i) { sb.camPos[i] = camPos[i]; sb.camFwd[i] = fwd[i]; sb.camRight[i] = right[i]; sb.camUp[i] = up[i]; }
        sb.resolution[0] = (float)cfg.width; sb.resolution[1] = (float)cfg.height;
        sb.time = t;
        sb.debugMode = (uint32_t)debugMode;
        wgpuQueueWriteBuffer(q, rg->node(sceneUbo)->buffer, 0, &sb, sizeof(sb));

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev, nullptr);
        rg->execute(enc, q);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(q, 1, &cmd);
        wgpuSurfacePresent(surf);

        rg->release_resources();   // destroy this frame's graph textures/buffers (imported swapchain left alone)

        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        instance.ProcessEvents();   // pump device/error callbacks
    }

    // ---- teardown --------------------------------------------------------------------------------
    wgpuSamplerRelease(linSampler);
    wgpuRenderPipelineRelease(cubeSamplePipe);
    wgpuRenderPipelineRelease(bloomCompositePipe);
    wgpuRenderPipelineRelease(bloomUpPipe);
    wgpuRenderPipelineRelease(bloomDownPipe);
    wgpuRenderPipelineRelease(bloomExtractPipe);
    wgpuRenderPipelineRelease(debugPipe);
    wgpuRenderPipelineRelease(skyPipe);
    wgpuRenderPipelineRelease(taaPipe);
    wgpuRenderPipelineRelease(presentPipe);
    wgpuRenderPipelineRelease(composePipe);
    wgpuComputePipelineRelease(ssaoPipe);
    wgpuRenderPipelineRelease(lightingPipe);
    wgpuRenderPipelineRelease(shadowPipe);
    wgpuRenderPipelineRelease(gbufferPipe);
    imgui_layer_shutdown();
    SDL_DestroyWindow(window);
    SDL_Quit();
    // ponytail: the GraphAllocator (1 MB block + both resource pools) is leaked at exit -- one-time, process reclaims it.
    return 0;
}
