// ===== Deferred-renderer demo ==============================================================
// A small deferred renderer driven entirely by the graph's read/write tracking, structured as a linear
// feature chain rather than a branchy DAG:
//
//   scene = cubeOn ? cube_skybox()
//                  : compose( lighting( gbuffer, shadows ), ssao )   // compose skipped when SSAO off
//   scene = bloom(scene)   // inserts a mip pyramid, or passes through
//   scene = taa(scene)     // accumulates with history, or passes through
//   present(scene)         // final blit to the swapchain
//
// Each stage returns the colour handle the next reads, so a toggle adds or drops passes instead of
// forking the graph. SSAO off: compose's output would be lit * 1 == lit, so the pass is skipped and lit
// passes through (cull-from-sinks drops the rest), same as bloom/taa. Matrix-free: the camera is a
// position + basis in the UBO, fed to a WGSL ray function, so the gbuffer stores LINEAR depth and the
// deferred passes reconstruct the world hit -- no projection/inverse matrices. #included into the single
// TU after RenderGraph_demo.h.

namespace def_demo {

constexpr WGPUTextureFormat kColorFormat = WGPUTextureFormat_RGBA8Unorm;   // gbuffer albedo/normal/rough + lit color
constexpr WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth32Float; // gbuffer depth + shadow map
constexpr WGPUTextureFormat kAOFormat    = WGPUTextureFormat_RGBA8Unorm;   // ssao output (rgba8unorm = core storage format)
constexpr uint32_t          kShadowSize  = 1024;                           // shadow map is a fixed square
constexpr uint32_t          kNumCascades = 3;                              // CSM: layers in the shadow array
constexpr uint32_t          kMaxSpheres  = 8;
constexpr WGPUTextureFormat kFogFormat   = WGPUTextureFormat_RGBA16Float;  // froxel scatter (HDR, filterable, core storage)
constexpr uint32_t          kFroxelX     = 160;                            // view-frustum fog grid: width
constexpr uint32_t          kFroxelY     = 90;                             //                        height
constexpr uint32_t          kFroxelZ     = 64;                             //                        depth slices (exponential)

// One uniform buffer feeds gbuffer + shadow + lighting + ssao + cube. Layout matches the WGSL `Scene`
// struct below (std140: vec4 members on 16-byte offsets). Host-uploaded each frame -> never written by a pass.
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
    float    fog[4];                    // x=density y=g z=heightFalloff w=ambient (offset 240)
    float    fogColor[4];               // rgb = scatter tint, w = noise amount (offset 256)
    float    jitter[4];                 // xyz = froxel jitter, w = history blend alpha (offset 272)
    float    prevCamPos[4];             // last frame's camera (temporal reprojection) (offset 288)
    float    prevCamFwd[4];             //                                   (offset 304)
    float    prevCamRight[4];           //                                   (offset 320)
    float    prevCamUp[4];              //                                   (offset 336)
};
static_assert(sizeof(SceneUBO) == 352, "SceneUBO must match the std140 WGSL Scene layout");

// ---- shared WGSL snippets ------------------------------------------------------------------------
// Each pipeline's source = a few of these concatenated (after the shared kVS). Keeping the SDF + camera
// in one place is what makes gbuffer, shadow and lighting agree on world space without passing matrices.

// scene UBO at @group(0) @binding(0) -- gbuffer/shadow/lighting/ssao all bind it there.
static const char* kSceneDecl = R"(
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
    fog          : vec4f,
    fogColor     : vec4f,
    jitter       : vec4f,
    prevCamPos   : vec4f,
    prevCamFwd   : vec4f,
    prevCamRight : vec4f,
    prevCamUp    : vec4f,
};
@group(0) @binding(0) var<uniform> scene : Scene;
)";

// camera + light, matrix-free. The single source of truth for "what world point is this pixel".
static const char* kCameraDefs = R"(
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
static const char* kSdfDefs = R"(
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
static const char* kGbufferFs = R"(
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
static const char* kShadowFs = R"(
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
static const char* kLightingFs = R"(
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
    let col = albedo * (ambient + direct);
    return vec4f(col, 1.0);
}
)";

// ssao compute: hemisphere AO from reconstructed positions (matrix-free, screen-space spiral taps).
static const char* kSsaoBody = R"(
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

// ndc (y up, [-1,1]) -> uv (y down, [0,1]) for the sampled-texture passes below.
static const char* kUvHelper = R"(
fn uv_of(ndc : vec2f) -> vec2f { return vec2f(ndc.x, -ndc.y) * 0.5 + 0.5; }
)";

// ---- froxel volumetric fog ----------------------------------------------------------------------
// A 160x90x64 view-frustum-aligned grid. inject scatters light per froxel, integrate marches the
// columns front-to-back, apply composites the result over the lit scene. The grid is built straight
// in y-up clip space (camera_ray off the UBO basis), so it stays matrix-free like the rest of the demo
// and the froxel's xy match a screen pixel's ndc directly (no y flip; uv_of is for framebuffer textures).

// froxel<->depth mapping + phase + temporal reprojection. needs kSceneDecl + kCameraDefs (camera_ray,
// FAR, FOV). grid dims are WGSL literals here, same as kCameraDefs hardcodes FAR/FOV.
static const char* kFogDefs = R"(
const FOG_NEAR : f32 = 0.5;
fn froxel_slice_to_dist(zf : f32) -> f32 {                 // slice center -> axial distance, exponential
    return FOG_NEAR * pow(FAR / FOG_NEAR, (zf + 0.5) / 64.0);
}
fn froxel_dist_to_slice(dist : f32) -> f32 {               // inverse, normalized [0,1] for the w texcoord
    let d = clamp(dist, FOG_NEAR, FAR);
    return log(d / FOG_NEAR) / log(FAR / FOG_NEAR);
}
// world center of froxel `coord`, jittered by scene.jitter (xy in-cell, z along the slice) for temporal AA.
fn froxel_world_pos(coord : vec3u, aspect : f32) -> vec3f {
    let ndc  = vec2f((f32(coord.x) + 0.5 + scene.jitter.x) / 160.0 * 2.0 - 1.0,
                     (f32(coord.y) + 0.5 + scene.jitter.y) / 90.0  * 2.0 - 1.0);
    let dist = froxel_slice_to_dist(f32(coord.z) + scene.jitter.z);
    let rd   = camera_ray(ndc, aspect);
    return scene.camPos.xyz + rd * (dist / max(dot(rd, scene.camFwd.xyz), 1e-3));   // axial dist -> ray length
}
fn phase_hg(cosT : f32, g : f32) -> f32 {                  // Henyey-Greenstein: g>0 forward, g<0 back scatter
    let g2 = g * g;
    let d  = 1.0 + g2 - 2.0 * g * cosT;
    return (1.0 - g2) / pow(max(d, 1e-4), 1.5);            // 4*pi folded into the scatter term so fog fades to a visible colour, not black
}
// project a world point into LAST frame's grid (uvw in [0,1]); inverse of froxel_world_pos vs prevCam*.
fn prev_froxel_uvw(wp : vec3f, aspect : f32) -> vec3f {
    let rel   = wp - scene.prevCamPos.xyz;
    let axial = dot(rel, scene.prevCamFwd.xyz);
    if (axial <= FOG_NEAR) { return vec3f(-1.0); }         // behind / nearer than the grid -> reject
    let t  = tan(0.5 * FOV);
    let nx = dot(rel, scene.prevCamRight.xyz) / (axial * aspect * t);
    let ny = dot(rel, scene.prevCamUp.xyz)    / (axial * t);
    return vec3f(nx * 0.5 + 0.5, ny * 0.5 + 0.5, froxel_dist_to_slice(axial));   // same y-up convention as inject
}
// cheap 3D value-noise fbm to break the fog into wisps. STATIC world-space field on purpose: an animated
// one would drift between frames and the camera-only temporal reprojection would smear it (needs motion
// vectors). the per-frame froxel jitter + temporal EMA supersample this field, so the wisps stay smooth.
fn hash13(p : vec3f) -> f32 {
    var q = fract(p * 0.1031);
    q = q + dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}
fn value_noise(p : vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash13(i + vec3f(0.0, 0.0, 0.0)), hash13(i + vec3f(1.0, 0.0, 0.0)), u.x);
    let x10 = mix(hash13(i + vec3f(0.0, 1.0, 0.0)), hash13(i + vec3f(1.0, 1.0, 0.0)), u.x);
    let x01 = mix(hash13(i + vec3f(0.0, 0.0, 1.0)), hash13(i + vec3f(1.0, 0.0, 1.0)), u.x);
    let x11 = mix(hash13(i + vec3f(0.0, 1.0, 1.0)), hash13(i + vec3f(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn fbm3(p : vec3f) -> f32 {
    var v = 0.0; var a = 0.5; var q = p;
    for (var i = 0; i < 4; i = i + 1) { v = v + a * value_noise(q); q = q * 2.02; a = a * 0.5; }
    return v;
}
)";

// CSM lookup for the inject pass. shadowMap at @binding(1); Shadow + sample_shadow are copied verbatim
// from kLightingFs (deliberately duplicated, not abstracted -- the WebGPU API isn't wrapped here).
static const char* kFogShadowDecl = R"(
@group(0) @binding(1) var shadowMap : texture_depth_2d_array;
struct Shadow { factor : f32, cascade : i32 };
fn sample_shadow(wp : vec3f) -> Shadow {
    let dir = scene.lightDir.xyz;
    for (var c = 0u; c < NUM_CASCADES; c = c + 1u) {
        let ls = light_space(wp, dir, csm_ortho(c));
        if (abs(ls.x) > 1.0 || abs(ls.y) > 1.0 || ls.z > 1.0 || ls.z < 0.0) { continue; }
        let dim = vec2f(textureDimensions(shadowMap));
        let tx  = vec2i(i32((ls.x * 0.5 + 0.5) * dim.x), i32((0.5 - ls.y * 0.5) * dim.y));
        let stored = textureLoad(shadowMap, tx, i32(c), 0);
        return Shadow(select(1.0, 0.0, ls.z - 0.004 > stored), i32(c));
    }
    return Shadow(1.0, -1);
}
)";

// inject + scatter + temporal blend: density (height fog) + shadowed in-scatter per froxel, blended with
// the reprojected history slice, written to volume A. one thread per froxel.
static const char* kFogInjectBody = R"(
@group(0) @binding(2) var triSampler : sampler;
@group(0) @binding(3) var histPrev   : texture_3d<f32>;
@group(0) @binding(4) var scatterOut : texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid >= vec3u(160u, 90u, 64u))) { return; }
    let aspect = scene.resolution.x / scene.resolution.y;
    let wp = froxel_world_pos(gid, aspect);
    let h  = max(wp.y - (-1.5), 0.0);                            // height above the ground plane (matches map())
    let noise = fbm3(wp * 0.45);                                // static world-space wisps -> breaks up the haze
    let dens  = mix(1.0, noise * 2.0, scene.fogColor.w);        // fogColor.w = noise amount (0 = uniform haze)
    let sigma_t = scene.fog.x * exp(-h * scene.fog.z) * dens;   // density: height falloff modulated by noise
    let L  = normalize(-scene.lightDir.xyz);                    // toward the sun
    let V  = normalize(scene.camPos.xyz - wp);                  // toward the camera
    let sh = sample_shadow(wp);
    let phase = phase_hg(dot(-V, L), scene.fog.y);             // view ray into the scene vs sun: g>0 glows toward the sun
    let inscat = (scene.lightColor.rgb * sh.factor * phase + vec3f(scene.fog.w)) * scene.fogColor.rgb * sigma_t;
    var cur = vec4f(inscat, sigma_t);
    let puv = prev_froxel_uvw(wp, aspect);                      // reproject into last frame's grid
    if (all(puv >= vec3f(0.0)) && all(puv <= vec3f(1.0))) {     // history valid -> exponential moving average
        let hist = textureSampleLevel(histPrev, triSampler, puv, 0.0);
        cur = mix(hist, cur, scene.jitter.w);
    }
    textureStore(scatterOut, vec3i(gid), cur);
}
)";

// integrate: march each (x,y) column front-to-back, accumulating in-scatter weighted by transmittance.
// reads volume A as a sampled texture (textureLoad, integer coords) -> no read-capable storage needed.
static const char* kFogIntegrateBody = R"(
@group(0) @binding(0) var scatterIn : texture_3d<f32>;
@group(0) @binding(1) var integOut  : texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= 160u || gid.y >= 90u) { return; }
    var accum = vec3f(0.0);
    var transmittance = 1.0;
    for (var z = 0u; z < 64u; z = z + 1u) {
        let s  = textureLoad(scatterIn, vec3i(i32(gid.x), i32(gid.y), i32(z)), 0);
        let dz = froxel_slice_to_dist(f32(z) + 1.0) - froxel_slice_to_dist(f32(z));   // slab thickness
        let sliceT = exp(-s.a * dz);                                                  // Beer-Lambert over the slab
        accum = accum + transmittance * s.rgb * (1.0 - sliceT) / max(s.a, 1e-5);      // energy-conserving slab integral
        transmittance = transmittance * sliceT;
        textureStore(integOut, vec3i(i32(gid.x), i32(gid.y), i32(z)), vec4f(accum, transmittance));
    }
}
)";

// apply: per pixel map the gbuffer's linear depth to a froxel slice, trilinear-sample the integrated
// volume, composite as transmittance*scene + in-scatter. fullscreen, UBO at binding 0.
static const char* kFogApplyFs = R"(
@group(0) @binding(1) var litColor : texture_2d<f32>;
@group(0) @binding(2) var gDepth   : texture_depth_2d;
@group(0) @binding(3) var fogVol   : texture_3d<f32>;
@group(0) @binding(4) var fogSamp  : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px  = vec2i(in.pos.xy);
    let lit = textureLoad(litColor, px, 0).rgb;
    let lin = textureLoad(gDepth, px, 0);
    let aspect = scene.resolution.x / scene.resolution.y;
    let rd = camera_ray(in.ndc, aspect);
    let dist  = select(lin * FAR, FAR, lin >= 1.0);            // background fogs to the far plane
    let axial = dist * dot(rd, scene.camFwd.xyz);              // gbuffer stores ray length -> axial distance
    let uvw = vec3f(in.ndc.x * 0.5 + 0.5, in.ndc.y * 0.5 + 0.5, froxel_dist_to_slice(axial));
    let fog = textureSampleLevel(fogVol, fogSamp, uvw, 0.0);
    return vec4f(lit * fog.a + fog.rgb, 1.0);
}
)";

// compose fragment: lit colour modulated by AO. Only declared when SSAO is on (off, compose is skipped
// and lit is presented directly), so AO here is always a real texture.
// ponytail: AO multiplies the whole lit result, not just the ambient term; if it over-darkens lit
// surfaces, output ambient/direct separately from lighting and apply AO to ambient only.
static const char* kComposeFs = R"(
@group(0) @binding(0) var litColor : texture_2d<f32>;
@group(0) @binding(1) var aoTex    : texture_2d<f32>;
@group(0) @binding(2) var aoSamp   : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    let ao = textureSampleLevel(aoTex, aoSamp, uv_of(in.ndc), 0.0).r;
    return vec4f(textureLoad(litColor, px, 0).rgb * ao, 1.0);
}
)";

// present fragment: blit a colour texture straight to the swapchain (final sink of the feature chain).
static const char* kPresentFs = R"(
@group(0) @binding(0) var srcColor : texture_2d<f32>;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    return vec4f(textureLoad(srcColor, px, 0).rgb, 1.0);
}
)";

// temporal accumulation with neighborhood clamping. binding 0 = current scene colour, binding 1 =
// history.prev (last frame, rotated in by the pool). The scene animates every frame, so a plain history
// blend smears motion forever. The standard fix, no motion vectors: clamp the history sample to the
// colour AABB of the current 3x3 neighborhood before blending -- a static pixel's history sits inside
// the box and converges; a moved pixel's stale history falls outside and snaps to current (no ghosts).
static const char* kTaaFs = R"(
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

// sky fill: re-write the scene colour on the BACKGROUND pixels only (a vertical gradient). Runs after
// lighting with LoadOp_Load -> a second writer of litColor == a WAW edge (lighting -> sky) in the graph.
static const char* kSkyFs = R"(
@group(0) @binding(0) var gDepth : texture_depth_2d;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let px = vec2i(in.pos.xy);
    if (textureLoad(gDepth, px, 0) < 1.0) { discard; }       // geometry -> keep the lit result
    let h = clamp(in.ndc.y * 0.5 + 0.5, 0.0, 1.0);           // 0 horizon .. 1 zenith
    return vec4f(mix(vec3f(0.45, 0.55, 0.70), vec3f(0.10, 0.20, 0.45), h), 1.0);
}
)";

// bloom extract: bright-pass the scene colour into mip 0. soft knee keeps only the over-threshold part.
static const char* kBloomExtractFs = R"(
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
static const char* kBloomBlitFs = R"(
@group(0) @binding(0) var src  : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    return textureSampleLevel(src, samp, uv_of(in.ndc), 0.0);
}
)";

// bloom composite: scene colour + the accumulated bloom (mip 0) -> result.
static const char* kBloomCompositeFs = R"(
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
static const char* kCubeFs = R"(
@group(0) @binding(1) var cubeTex : texture_cube<f32>;
@group(0) @binding(2) var samp    : sampler;
@fragment fn fs(in : VsOut) -> @location(0) vec4f {
    let rd = camera_ray(in.ndc, scene.resolution.x / scene.resolution.y);
    return textureSample(cubeTex, samp, rd);
}
)";

// ---- state (created once in init, lives across frames) ----
static WGPURenderPipeline  gbufferPipe = nullptr, shadowPipe = nullptr, lightingPipe = nullptr;
static WGPURenderPipeline  composePipe = nullptr, presentPipe = nullptr, taaPipe = nullptr, skyPipe = nullptr;
static WGPURenderPipeline  bloomExtractPipe = nullptr, bloomDownPipe = nullptr, bloomUpPipe = nullptr, bloomCompositePipe = nullptr;
static WGPURenderPipeline  cubeSamplePipe = nullptr;
static WGPUComputePipeline ssaoPipe = nullptr;
static WGPUComputePipeline fogInjectPipe = nullptr, fogIntegratePipe = nullptr;   // froxel volumetrics
static WGPURenderPipeline  fogApplyPipe = nullptr;
static WGPUSampler         linSampler = nullptr;   // linear/clamp on all 3 axes; shared by compose/bloom/cube/fog
static WGPUBuffer          uboBuf = nullptr;       // demo-owned scene UBO, imported into the graph each frame

// feature toggles (ImGui checkboxes in deferred_ui; persist across demo switches).
static bool ssaoOn = true, taaOn = true, bloomOn = false, cubeOn = false;
static bool  fogOn = true;   // froxel volumetric fog; default on so the new passes run out of the box
static float fogDensity = 0.08f, fogAnisotropy = 0.6f, fogHeightFalloff = 0.25f, fogAmbient = 0.02f, fogNoise = 0.8f;

// the gbuffer's four outputs, threaded through the chain as one value.
struct GBuffer { ResourceHandle albedo, normal, rough, depth; };

// a swapchain-sized offscreen texture (WGPU_STRLEN -> copy_string measures the literal).
static ResourceHandle screen_tex(RenderGraph* rg, const char* name, WGPUTextureFormat fmt, ResourceHandle swap)
{
    return rg->create_image(WGPUStringView{ name, WGPU_STRLEN }, {
        .dimension = WGPUTextureDimension_2D, .format = fmt,
        .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swap,
    });
}

// gbuffer: raymarch the SDF -> albedo / normal / roughness (MRT) + linear depth.
static GBuffer build_gbuffer(RenderGraph* rg, WGPUDevice dev, ResourceHandle ubo, ResourceHandle swap)
{
    GBuffer g{
        screen_tex(rg, "gbuffer.albedo", kColorFormat, swap),
        screen_tex(rg, "gbuffer.normal", kColorFormat, swap),
        screen_tex(rg, "gbuffer.rough",  kColorFormat, swap),
        screen_tex(rg, "gbuffer.depth",  kDepthFormat, swap),
    };
    rg->add_pass(WEBGPU_STR("gbuffer"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.color(g.albedo, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});         // @location(0)
            b.color(g.normal, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0.5, 0.5, 1.0, 1});   // @location(1)
            b.color(g.rough,  WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});         // @location(2)
            b.depth_stencil(g.depth, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f);
            b.uniform(ubo);
        },
        [dev, ubo](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(gbufferPipe, 0);
            WGPUBindGroupEntry e[1] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 1, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, gbufferPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    return g;
}

// shadow cascades: raymarch the same SDF from the light at kNumCascades concentric ortho extents, each
// into one layer of a 2D-array depth texture (per-layer attachment via baseLayer). firstInstance = c
// carries the cascade index to the shader. The writes share the handle, so the graph serializes them
// (WAW) and lighting RAW-depends on all.
static ResourceHandle build_shadows(RenderGraph* rg, WGPUDevice dev, ResourceHandle ubo)
{
    auto csm = rg->create_image(WEBGPU_STR("shadow.csm"), {
        .dimension = WGPUTextureDimension_2D, .format = kDepthFormat,
        .sizeKind = SizeKind::Absolute, .absolute = { kShadowSize, kShadowSize, kNumCascades },
    });
    for (uint32_t c = 0; c < kNumCascades; ++c) {
        rg->add_pass(WEBGPU_STR("shadow.cascade"), PassKind::Graphics,
            [&, c](GraphBuilder& b) {
                b.depth_stencil(csm, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f, 0, c);   // baseMip 0, layer c
                b.uniform(ubo);
            },
            [dev, ubo, c](PassContext& ctx) {
                WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(shadowPipe, 0);
                WGPUBindGroupEntry e[1] = {
                    { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
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
    return csm;
}

// ssao: screen-space AO from gbuffer depth + normal -> aoTex. Only called when SSAO is enabled; off, the
// caller skips both this and compose.
static ResourceHandle build_ssao(RenderGraph* rg, const DemoEnv& env, ResourceHandle ubo,
                                 ResourceHandle gDepth, ResourceHandle gNormal, ResourceHandle swap)
{
    WGPUDevice dev = env.device;
    auto ao = screen_tex(rg, "ssao.ao", kAOFormat, swap);
    const uint32_t gx = (env.width + 7) / 8, gy = (env.height + 7) / 8;
    rg->add_pass(WEBGPU_STR("ssao"), PassKind::Compute,
        [&](GraphBuilder& b) {
            b.uniform(ubo);                 // camera basis for world reconstruction
            b.sampled(gDepth);
            b.sampled(gNormal);
            b.storage_write(ao);
        },
        [dev, ubo, gDepth, gNormal, ao, gx, gy](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(ssaoPipe, 0);
            WGPUBindGroupEntry e[4] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
                { .binding = 1, .textureView = ctx.view(gDepth) },
                { .binding = 2, .textureView = ctx.view(gNormal) },
                { .binding = 3, .textureView = ctx.view(ao) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 4, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuComputePassEncoderSetPipeline(ctx.compute, ssaoPipe);
            wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, gx, gy, 1);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    return ao;
}

// lighting: gbuffer + shadow map + directional light -> lit colour.
static ResourceHandle build_lighting(RenderGraph* rg, WGPUDevice dev, ResourceHandle ubo, GBuffer g, ResourceHandle csm, ResourceHandle swap)
{
    auto lit = screen_tex(rg, "lighting.color", kColorFormat, swap);
    rg->add_pass(WEBGPU_STR("lighting"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(g.albedo);
            b.sampled(g.normal);
            b.sampled(g.rough);
            b.sampled(g.depth);
            b.sampled(csm);
            b.uniform(ubo);
            b.color(lit, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, g, csm](PassContext& ctx) {
            // the body builds its own array view (graph resolves the texture; no graph-side abstraction):
            // binds all kNumCascades layers as a single texture_depth_2d_array, like cube.sample below.
            WGPUTextureViewDescriptor avd{
                .format = kDepthFormat, .dimension = WGPUTextureViewDimension_2DArray,
                .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = kNumCascades,
            };
            WGPUTextureView csmView = wgpuTextureCreateView(ctx.texture(csm), &avd);
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(lightingPipe, 0);
            WGPUBindGroupEntry e[6] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
                { .binding = 1, .textureView = ctx.view(g.albedo) },
                { .binding = 2, .textureView = ctx.view(g.normal) },
                { .binding = 3, .textureView = ctx.view(g.rough) },
                { .binding = 4, .textureView = ctx.view(g.depth) },
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
    return lit;
}

// sky: 2nd writer of the lit colour (LoadOp_Load, background pixels only) -> WAW edge after lighting.
static void build_sky(RenderGraph* rg, WGPUDevice dev, ResourceHandle gDepth, ResourceHandle lit)
{
    rg->add_pass(WEBGPU_STR("sky"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(gDepth);                                                              // which pixels are background
            b.color(lit, WGPULoadOp_Load, WGPUStoreOp_Store);                               // keep lit geometry, add sky
        },
        [dev, gDepth](PassContext& ctx) {
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
}

// compose: lit * ao -> scene colour. Only declared when SSAO is on; with it off the result would be
// lit * 1 == lit, so we skip the pass and pass `lit` straight through (cull-from-sinks drops the rest).
// Same shape as build_bloom/build_taa: insert a pass or return the colour unchanged.
static ResourceHandle build_compose(RenderGraph* rg, WGPUDevice dev, ResourceHandle lit, ResourceHandle ao, ResourceHandle swap, bool ssaoOn)
{
    if (!ssaoOn) return lit;

    auto scene = screen_tex(rg, "compose.scene", kSwapFormat, swap);
    rg->add_pass(WEBGPU_STR("compose"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(lit);
            b.sampled(ao);
            b.color(scene, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, lit, ao](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(composePipe, 0);
            WGPUBindGroupEntry e[3] = {
                { .binding = 0, .textureView = ctx.view(lit) },
                { .binding = 1, .textureView = ctx.view(ao) },
                { .binding = 2, .sampler = linSampler },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 3, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, composePipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    return scene;
}

// froxel volumetric fog: inject + scatter (compute) -> integrate (compute) -> apply (graphics), composited
// over the lit scene. the scatter volume is temporal (ping-pong): inject blends this frame's froxels with
// last frame's reprojected slice, so the grid converges instead of shimmering. returns the input unchanged
// when off -> the volumes then accumulate zero usage and realize() skips them.
// ponytail: one isotropic height-fog medium + per-froxel jitter only. the medium is static, so reprojection
// is camera-only (no per-froxel motion vectors); add those + blue-noise jitter for moving/animated media.
static ResourceHandle build_volumetrics(RenderGraph* rg, const DemoEnv& env, ResourceHandle ubo,
                                        ResourceHandle scene, ResourceHandle gDepth, ResourceHandle csm,
                                        ResourceHandle swap, bool enabled)
{
    if (!enabled) return scene;
    WGPUDevice dev = env.device;
    const TextureDesc desc3d{
        .dimension = WGPUTextureDimension_3D, .format = kFogFormat,
        .sizeKind = SizeKind::Absolute, .absolute = { kFroxelX, kFroxelY, kFroxelZ }, .mipLevelCount = 1,
    };
    auto scatter  = rg->create_temporal_image(WEBGPU_STR("fog.scatter"), desc3d);   // .curr write, .prev history
    auto volInteg = rg->create_image(WEBGPU_STR("fog.integrated"), desc3d);
    const uint32_t ix = (kFroxelX + 3) / 4, iy = (kFroxelY + 3) / 4, iz = (kFroxelZ + 3) / 4;   // inject 4x4x4
    const uint32_t cx = (kFroxelX + 7) / 8, cy = (kFroxelY + 7) / 8;                            // integrate 8x8x1

    // inject: scene UBO + CSM + last frame's volume -> scatter.curr.
    rg->add_pass(WEBGPU_STR("fog.inject"), PassKind::Compute,
        [&](GraphBuilder& b) {
            b.uniform(ubo);
            b.sampled(csm);
            b.sampled(scatter.prev);
            b.storage_write(scatter.curr);
        },
        [dev, ubo, csm, scatter, ix, iy, iz](PassContext& ctx) {
            // CSM bound as one depth-2d-array view spanning all cascades, exactly like the lighting body.
            WGPUTextureViewDescriptor avd{
                .format = kDepthFormat, .dimension = WGPUTextureViewDimension_2DArray,
                .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = kNumCascades,
            };
            WGPUTextureView csmView = wgpuTextureCreateView(ctx.texture(csm), &avd);
            WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(fogInjectPipe, 0);
            WGPUBindGroupEntry e[5] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
                { .binding = 1, .textureView = csmView },
                { .binding = 2, .sampler = linSampler },
                { .binding = 3, .textureView = ctx.view(scatter.prev) },   // default full-volume 3D view
                { .binding = 4, .textureView = ctx.view(scatter.curr) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 5, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuComputePassEncoderSetPipeline(ctx.compute, fogInjectPipe);
            wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, ix, iy, iz);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
            wgpuTextureViewRelease(csmView);
        });

    // integrate: scatter.curr -> integrated volume, accumulating in-scatter + transmittance per column.
    rg->add_pass(WEBGPU_STR("fog.integrate"), PassKind::Compute,
        [&](GraphBuilder& b) {
            b.sampled(scatter.curr);
            b.storage_write(volInteg);
        },
        [dev, scatter, volInteg, cx, cy](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuComputePipelineGetBindGroupLayout(fogIntegratePipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .textureView = ctx.view(scatter.curr) },
                { .binding = 1, .textureView = ctx.view(volInteg) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuComputePassEncoderSetPipeline(ctx.compute, fogIntegratePipe);
            wgpuComputePassEncoderSetBindGroup(ctx.compute, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(ctx.compute, cx, cy, 1);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });

    // apply: composite the integrated fog over the lit scene -> new scene colour.
    auto out = screen_tex(rg, "fog.scene", kSwapFormat, swap);
    rg->add_pass(WEBGPU_STR("fog.apply"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.uniform(ubo);
            b.sampled(scene);
            b.sampled(gDepth);
            b.sampled(volInteg);
            b.color(out, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, scene, gDepth, volInteg](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(fogApplyPipe, 0);
            WGPUBindGroupEntry e[5] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
                { .binding = 1, .textureView = ctx.view(scene) },
                { .binding = 2, .textureView = ctx.view(gDepth) },
                { .binding = 3, .textureView = ctx.view(volInteg) },
                { .binding = 4, .sampler = linSampler },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 5, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, fogApplyPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    return out;
}

// cubemap skybox (array-layer subresource): clear 6 faces into one cube texture, sample it by camera ray
// -> a rotatable scene colour. Stands in for the deferred source when cubeOn.
static ResourceHandle build_cube(RenderGraph* rg, WGPUDevice dev, ResourceHandle ubo, ResourceHandle swap)
{
    auto scene = screen_tex(rg, "cube.scene", kSwapFormat, swap);
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
            b.uniform(ubo);     // camera basis for the ray
            b.sampled(cube);
            b.color(scene, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, ubo, cube](PassContext& ctx) {
            WGPUTextureViewDescriptor cvd{
                .format = kColorFormat, .dimension = WGPUTextureViewDimension_Cube,
                .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 6,
            };
            WGPUTextureView cubeView = wgpuTextureCreateView(ctx.texture(cube), &cvd);
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(cubeSamplePipe, 0);
            WGPUBindGroupEntry e[3] = {
                { .binding = 0, .buffer = ctx.buffer(ubo), .offset = 0, .size = sizeof(SceneUBO) },
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
    return scene;
}

// bloom (mip subresource): bright-pass the scene into a mip chain, downsample, additively upsample,
// composite over the scene. Returns the composited colour, or the input unchanged when off.
static ResourceHandle build_bloom(RenderGraph* rg, const DemoEnv& env, ResourceHandle scene, ResourceHandle swap, bool enabled)
{
    if (!enabled) return scene;
    WGPUDevice dev = env.device;
    uint32_t minDim = env.width < env.height ? env.width : env.height;
    uint32_t bloomMips = 1;
    while (bloomMips < 6 && (1u << bloomMips) <= minDim) ++bloomMips;   // smallest mip stays >= 1px, cap 6
    auto bloom = rg->create_image(WEBGPU_STR("bloom"), {
        .dimension = WGPUTextureDimension_2D, .format = kColorFormat,
        .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swap,
        .mipLevelCount = bloomMips,
    });

    // extract: bright parts of the scene -> mip 0.
    rg->add_pass(WEBGPU_STR("bloom.extract"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(scene);
            b.color(bloom, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1}, 0);
        },
        [dev, scene](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomExtractPipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .textureView = ctx.view(scene) },
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
            [dev, bloom, i](PassContext& ctx) {
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

    // upsample: sample mip i (coarser), additively blend into mip i-1 (LoadOp_Load keeps the finer
    // level the downsample wrote this frame).
    for (uint32_t i = bloomMips; i-- > 1; ) {
        rg->add_pass(WEBGPU_STR("bloom.up"), PassKind::Graphics,
            [&, i](GraphBuilder& b) {
                b.sampled(bloom, i);
                b.color(bloom, WGPULoadOp_Load, WGPUStoreOp_Store, WGPUColor{}, i - 1);
            },
            [dev, bloom, i](PassContext& ctx) {
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

    // composite: scene + accumulated bloom (mip 0) -> result.
    auto result = screen_tex(rg, "bloom.scene", kSwapFormat, swap);
    rg->add_pass(WEBGPU_STR("bloom.composite"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(scene);
            b.sampled(bloom, 0);
            b.color(result, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, scene, bloom](PassContext& ctx) {
            WGPUTextureView mip0 = mip_view_2d(ctx.texture(bloom), kColorFormat, 0);
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(bloomCompositePipe, 0);
            WGPUBindGroupEntry e[3] = {
                { .binding = 0, .textureView = ctx.view(scene) },
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
    return result;
}

// taa: blend the scene with neighborhood-clamped history (ping-pong temporal). Returns the accumulated
// colour, or the input unchanged when off.
static ResourceHandle build_taa(RenderGraph* rg, const DemoEnv& env, ResourceHandle scene, ResourceHandle swap, bool enabled)
{
    if (!enabled) return scene;
    WGPUDevice dev = env.device;
    // history: ping-pong temporal resource. .curr = this frame's write target, .prev = last frame's
    // result; the pool swaps the two physical textures each frame.
    auto hist = rg->create_temporal_image(WEBGPU_STR("taa.history"), {
        .dimension = WGPUTextureDimension_2D, .format = kSwapFormat,
        .sizeKind = SizeKind::Relative, .scaleX = 1.0f, .scaleY = 1.0f, .relativeTo = swap,
    });
    rg->add_pass(WEBGPU_STR("taa"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(scene);                    // this frame's shaded colour
            b.sampled(hist.prev);               // previous frame (rotated in by the pool)
            b.color(hist.curr, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, scene, hist](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(taaPipe, 0);
            WGPUBindGroupEntry e[2] = {
                { .binding = 0, .textureView = ctx.view(scene) },
                { .binding = 1, .textureView = ctx.view(hist.prev) },
            };
            WGPUBindGroupDescriptor d{ .layout = l, .entryCount = 2, .entries = e };
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev, &d);
            wgpuRenderPassEncoderSetPipeline(ctx.render, taaPipe);
            wgpuRenderPassEncoderSetBindGroup(ctx.render, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);
            wgpuBindGroupLayoutRelease(l);
        });
    return hist.curr;
}

// present: blit the final scene colour to the swapchain (the chain's sink).
static void build_present(RenderGraph* rg, WGPUDevice dev, ResourceHandle scene, ResourceHandle swap)
{
    rg->add_pass(WEBGPU_STR("present"), PassKind::Graphics,
        [&](GraphBuilder& b) {
            b.sampled(scene);
            b.color(swap, WGPULoadOp_Clear, WGPUStoreOp_Store, WGPUColor{0, 0, 0, 1});
        },
        [dev, scene](PassContext& ctx) {
            WGPUBindGroupLayout l = wgpuRenderPipelineGetBindGroupLayout(presentPipe, 0);
            WGPUBindGroupEntry e[1] = {
                { .binding = 0, .textureView = ctx.view(scene) },
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

} // namespace def_demo

static void deferred_init(const DemoEnv& env)
{
    using namespace def_demo;
    WGPUDevice dev = env.device;

    // depth-stencil shared by gbuffer + shadow: write depth, Less, stencil disabled (depth-only format
    // -> stencil faces must be the no-op Always/Keep default).
    WGPUStencilFaceState stencilNop{
        .compare = WGPUCompareFunction_Always, .failOp = WGPUStencilOperation_Keep,
        .depthFailOp = WGPUStencilOperation_Keep, .passOp = WGPUStencilOperation_Keep,
    };
    WGPUDepthStencilState depthWrite{
        .format = kDepthFormat, .depthWriteEnabled = WGPUOptionalBool_True, .depthCompare = WGPUCompareFunction_Less,
        .stencilFront = stencilNop, .stencilBack = stencilNop, .stencilReadMask = 0xFFFFFFFF, .stencilWriteMask = 0xFFFFFFFF,
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
        .label = WEBGPU_STR("gbuffer pipeline"),
        .vertex = { .module = gbufferSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .depthStencil = &depthWrite,
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &gbFrag,
    };
    gbufferPipe = wgpuDeviceCreateRenderPipeline(dev, &gbPD);
    wgpuShaderModuleRelease(gbufferSM);

    // shadow: fullscreen raymarch from the light -> depth only (no color targets).
    WGPUShaderModule shadowSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kSdfDefs + kShadowFs);
    WGPUFragmentState shadowFrag{ .module = shadowSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 0, .targets = nullptr };
    WGPURenderPipelineDescriptor shadowPD{
        .label = WEBGPU_STR("shadow pipeline"),
        .vertex = { .module = shadowSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .depthStencil = &depthWrite,
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &shadowFrag,
    };
    shadowPipe = wgpuDeviceCreateRenderPipeline(dev, &shadowPD);
    wgpuShaderModuleRelease(shadowSM);

    // lighting: fullscreen -> lit color from gbuffer + shadow map.
    WGPUShaderModule lightSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kLightingFs);
    WGPUColorTargetState lightCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState lightFrag{ .module = lightSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &lightCT };
    WGPURenderPipelineDescriptor lightPD{
        .label = WEBGPU_STR("lighting pipeline"),
        .vertex = { .module = lightSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &lightFrag,
    };
    lightingPipe = wgpuDeviceCreateRenderPipeline(dev, &lightPD);
    wgpuShaderModuleRelease(lightSM);

    // ssao: compute, reconstructs world positions from gbuffer depth + normal (needs the UBO camera).
    WGPUShaderModule ssaoSM = make_shader(dev, std::string(kSceneDecl) + kCameraDefs + kSsaoBody);
    WGPUComputePipelineDescriptor ssaoPD{
        .label = WEBGPU_STR("ssao pipeline"),
        .compute = { .module = ssaoSM, .entryPoint = WEBGPU_STR("main") },
    };
    ssaoPipe = wgpuDeviceCreateComputePipeline(dev, &ssaoPD);
    wgpuShaderModuleRelease(ssaoSM);

    // fog inject: density + shadowed in-scatter per froxel, temporal-blended into the scatter volume.
    WGPUShaderModule fogInjSM = make_shader(dev, std::string(kSceneDecl) + kCameraDefs + kFogShadowDecl + kFogDefs + kFogInjectBody);
    WGPUComputePipelineDescriptor fogInjPD{
        .label = WEBGPU_STR("fog.inject pipeline"),
        .compute = { .module = fogInjSM, .entryPoint = WEBGPU_STR("main") },
    };
    fogInjectPipe = wgpuDeviceCreateComputePipeline(dev, &fogInjPD);
    wgpuShaderModuleRelease(fogInjSM);

    // fog integrate: march each froxel column front-to-back into the integrated volume.
    WGPUShaderModule fogIntSM = make_shader(dev, std::string(kSceneDecl) + kCameraDefs + kFogDefs + kFogIntegrateBody);
    WGPUComputePipelineDescriptor fogIntPD{
        .label = WEBGPU_STR("fog.integrate pipeline"),
        .compute = { .module = fogIntSM, .entryPoint = WEBGPU_STR("main") },
    };
    fogIntegratePipe = wgpuDeviceCreateComputePipeline(dev, &fogIntPD);
    wgpuShaderModuleRelease(fogIntSM);

    // fog apply: composite the integrated volume over the lit scene (fullscreen).
    WGPUShaderModule fogAppSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kFogDefs + kFogApplyFs);
    WGPUColorTargetState fogAppCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState fogAppFrag{ .module = fogAppSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &fogAppCT };
    WGPURenderPipelineDescriptor fogAppPD{
        .label = WEBGPU_STR("fog.apply pipeline"),
        .vertex = { .module = fogAppSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &fogAppFrag,
    };
    fogApplyPipe = wgpuDeviceCreateRenderPipeline(dev, &fogAppPD);
    wgpuShaderModuleRelease(fogAppSM);

    // compose: lit * ao -> scene color (samples ao, so a 1x1 identity works when SSAO off).
    WGPUShaderModule composeSM = make_shader(dev, std::string(kVS) + kUvHelper + kComposeFs);
    WGPUColorTargetState composeCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState composeFrag{ .module = composeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &composeCT };
    WGPURenderPipelineDescriptor composePD{
        .label = WEBGPU_STR("compose pipeline"),
        .vertex = { .module = composeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &composeFrag,
    };
    composePipe = wgpuDeviceCreateRenderPipeline(dev, &composePD);
    wgpuShaderModuleRelease(composeSM);

    // present: blit a color texture straight to the swapchain.
    WGPUShaderModule presentSM = make_shader(dev, std::string(kVS) + kPresentFs);
    WGPUColorTargetState presentCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState presentFrag{ .module = presentSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &presentCT };
    WGPURenderPipelineDescriptor presentPD{
        .label = WEBGPU_STR("present pipeline"),
        .vertex = { .module = presentSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &presentFrag,
    };
    presentPipe = wgpuDeviceCreateRenderPipeline(dev, &presentPD);
    wgpuShaderModuleRelease(presentSM);

    // taa: blend current scene colour with the previous history layer.
    WGPUShaderModule taaSM = make_shader(dev, std::string(kVS) + kTaaFs);
    WGPUColorTargetState taaCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState taaFrag{ .module = taaSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &taaCT };
    WGPURenderPipelineDescriptor taaPD{
        .label = WEBGPU_STR("taa pipeline"),
        .vertex = { .module = taaSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &taaFrag,
    };
    taaPipe = wgpuDeviceCreateRenderPipeline(dev, &taaPD);
    wgpuShaderModuleRelease(taaSM);

    // sky: fills the scene colour's background pixels (discards on geometry). 2nd writer -> WAW.
    WGPUShaderModule skySM = make_shader(dev, std::string(kVS) + kSkyFs);
    WGPUColorTargetState skyCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState skyFrag{ .module = skySM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &skyCT };
    WGPURenderPipelineDescriptor skyPD{
        .label = WEBGPU_STR("sky pipeline"),
        .vertex = { .module = skySM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &skyFrag,
    };
    skyPipe = wgpuDeviceCreateRenderPipeline(dev, &skyPD);
    wgpuShaderModuleRelease(skySM);

    // linear/clamp sampler shared by compose + bloom + cube (the deferred passes use textureLoad).
    WGPUSamplerDescriptor linSD{
        .addressModeU = WGPUAddressMode_ClampToEdge, .addressModeV = WGPUAddressMode_ClampToEdge, .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear, .minFilter = WGPUFilterMode_Linear, .mipmapFilter = WGPUMipmapFilterMode_Nearest,
        .maxAnisotropy = 1,
    };
    linSampler = wgpuDeviceCreateSampler(dev, &linSD);

    // bloom: extract + blit target kColorFormat (the bloom texture); composite targets kSwapFormat.
    WGPUShaderModule bloomExtractSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomExtractFs);
    WGPUColorTargetState bloomCT{ .format = kColorFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomExtractFrag{ .module = bloomExtractSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCT };
    WGPURenderPipelineDescriptor bloomExtractPD{
        .label = WEBGPU_STR("bloom.extract pipeline"),
        .vertex = { .module = bloomExtractSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &bloomExtractFrag,
    };
    bloomExtractPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomExtractPD);
    wgpuShaderModuleRelease(bloomExtractSM);

    // down + up share one fragment shader (a single bilinear tap); only the blend differs.
    WGPUShaderModule bloomBlitSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomBlitFs);
    WGPUFragmentState bloomDownFrag{ .module = bloomBlitSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCT };
    WGPURenderPipelineDescriptor bloomDownPD{
        .label = WEBGPU_STR("bloom.down pipeline"),
        .vertex = { .module = bloomBlitSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &bloomDownFrag,
    };
    bloomDownPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomDownPD);

    // upsample: additive (src*1 + dst*1) so each coarser level accumulates onto the finer one (LoadOp_Load).
    WGPUBlendState addBlend{
        .color = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
        .alpha = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_One },
    };
    WGPUColorTargetState bloomUpCT{ .format = kColorFormat, .blend = &addBlend, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomUpFrag{ .module = bloomBlitSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomUpCT };
    WGPURenderPipelineDescriptor bloomUpPD{
        .label = WEBGPU_STR("bloom.up pipeline"),
        .vertex = { .module = bloomBlitSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &bloomUpFrag,
    };
    bloomUpPipe = wgpuDeviceCreateRenderPipeline(dev, &bloomUpPD);
    wgpuShaderModuleRelease(bloomBlitSM);

    WGPUShaderModule bloomCompSM = make_shader(dev, std::string(kVS) + kUvHelper + kBloomCompositeFs);
    WGPUColorTargetState bloomCompCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState bloomCompFrag{ .module = bloomCompSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &bloomCompCT };
    WGPURenderPipelineDescriptor bloomCompPD{
        .label = WEBGPU_STR("bloom.composite pipeline"),
        .vertex = { .module = bloomCompSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &bloomCompFrag,
    };
    bloomCompositePipe = wgpuDeviceCreateRenderPipeline(dev, &bloomCompPD);
    wgpuShaderModuleRelease(bloomCompSM);

    // cubemap: sample the 6-layer cube by camera ray. needs the scene UBO for camera_ray.
    WGPUShaderModule cubeSM = make_shader(dev, std::string(kVS) + kSceneDecl + kCameraDefs + kCubeFs);
    WGPUColorTargetState cubeCT{ .format = kSwapFormat, .writeMask = WGPUColorWriteMask_All };
    WGPUFragmentState cubeFrag{ .module = cubeSM, .entryPoint = WEBGPU_STR("fs"), .targetCount = 1, .targets = &cubeCT };
    WGPURenderPipelineDescriptor cubePD{
        .label = WEBGPU_STR("cube.sample pipeline"),
        .vertex = { .module = cubeSM, .entryPoint = WEBGPU_STR("vs") },
        .primitive = { .topology = WGPUPrimitiveTopology_TriangleList },
        .multisample = { .count = 1, .mask = ~0u },
        .fragment = &cubeFrag,
    };
    cubeSamplePipe = wgpuDeviceCreateRenderPipeline(dev, &cubePD);
    wgpuShaderModuleRelease(cubeSM);

    WGPUBufferDescriptor bd{ .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(SceneUBO) };
    uboBuf = wgpuDeviceCreateBuffer(dev, &bd);
}

static void deferred_shutdown()
{
    using namespace def_demo;
    wgpuBufferRelease(uboBuf);
    wgpuSamplerRelease(linSampler);
    wgpuRenderPipelineRelease(cubeSamplePipe);
    wgpuRenderPipelineRelease(bloomCompositePipe);
    wgpuRenderPipelineRelease(bloomUpPipe);
    wgpuRenderPipelineRelease(bloomDownPipe);
    wgpuRenderPipelineRelease(bloomExtractPipe);
    wgpuRenderPipelineRelease(skyPipe);
    wgpuRenderPipelineRelease(taaPipe);
    wgpuRenderPipelineRelease(presentPipe);
    wgpuRenderPipelineRelease(composePipe);
    wgpuRenderPipelineRelease(fogApplyPipe);
    wgpuComputePipelineRelease(fogIntegratePipe);
    wgpuComputePipelineRelease(fogInjectPipe);
    wgpuComputePipelineRelease(ssaoPipe);
    wgpuRenderPipelineRelease(lightingPipe);
    wgpuRenderPipelineRelease(shadowPipe);
    wgpuRenderPipelineRelease(gbufferPipe);
}

static void deferred_build(const DemoEnv& env, RenderGraph* rg, ResourceHandle swap)
{
    using namespace def_demo;
    WGPUDevice dev = env.device;

    // host-upload the scene + a static directional light into the demo-owned UBO, then import it. The
    // spheres are packed into a pile resting ON the ground so they overlap into tight concave necks --
    // that's where SSAO actually darkens.
    SceneUBO sb{};
    const float t  = env.time;
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
    float la = 0.8f;                                         // static light dir -> shadow map stops sweeping
    float lx = cosf(la) * 0.5f, ly = -1.0f, lz = sinf(la) * 0.5f;
    float ln = sqrtf(lx * lx + ly * ly + lz * lz);
    sb.lightDir[0] = lx / ln; sb.lightDir[1] = ly / ln; sb.lightDir[2] = lz / ln;
    sb.lightColor[0] = 1.0f; sb.lightColor[1] = 0.96f; sb.lightColor[2] = 0.9f;
    for (int i = 0; i < 3; ++i) { sb.camPos[i] = env.camera.pos[i]; sb.camFwd[i] = env.camera.fwd[i]; sb.camRight[i] = env.camera.right[i]; sb.camUp[i] = env.camera.up[i]; }
    sb.resolution[0] = (float)env.width; sb.resolution[1] = (float)env.height;
    sb.time = t;

    // froxel fog params + temporal state. prevCam* drive the reprojection; jitter (Halton, keyed to the
    // frame index) shifts the grid each frame so the temporal blend converges. carried across frames in
    // statics; on the first build prev == current, so prev_froxel_uvw maps a froxel onto itself (no-op).
    static float sPrevPos[3] = {0,0,0}, sPrevFwd[3] = {0,0,1}, sPrevRight[3] = {1,0,0}, sPrevUp[3] = {0,1,0};
    static bool  sHavePrev = false;
    sb.fog[0] = fogDensity; sb.fog[1] = fogAnisotropy; sb.fog[2] = fogHeightFalloff; sb.fog[3] = fogAmbient;
    sb.fogColor[0] = 0.9f; sb.fogColor[1] = 0.95f; sb.fogColor[2] = 1.0f; sb.fogColor[3] = fogNoise;   // w = noise amount
    for (int i = 0; i < 3; ++i) {
        sb.prevCamPos[i]   = sHavePrev ? sPrevPos[i]   : env.camera.pos[i];
        sb.prevCamFwd[i]   = sHavePrev ? sPrevFwd[i]   : env.camera.fwd[i];
        sb.prevCamRight[i] = sHavePrev ? sPrevRight[i] : env.camera.right[i];
        sb.prevCamUp[i]    = sHavePrev ? sPrevUp[i]    : env.camera.up[i];
    }
    auto halton = [](uint32_t i, uint32_t b) { float f = 1.0f, r = 0.0f; while (i > 0) { f /= (float)b; r += f * (float)(i % b); i /= b; } return r; };
    const uint32_t fi = (uint32_t)(env.frame & 1023u) + 1u;       // +1 so index 0 isn't the degenerate origin
    sb.jitter[0] = halton(fi, 2u) - 0.5f;
    sb.jitter[1] = halton(fi, 3u) - 0.5f;
    sb.jitter[2] = halton(fi, 5u) - 0.5f;
    sb.jitter[3] = sHavePrev ? 0.05f : 1.0f;                      // history blend alpha; first frame ignores history
    for (int i = 0; i < 3; ++i) { sPrevPos[i] = env.camera.pos[i]; sPrevFwd[i] = env.camera.fwd[i]; sPrevRight[i] = env.camera.right[i]; sPrevUp[i] = env.camera.up[i]; }
    sHavePrev = true;

    wgpuQueueWriteBuffer(env.queue, uboBuf, 0, &sb, sizeof(sb));
    ResourceHandle ubo = rg->import_buffer(WEBGPU_STR("scene.ubo"), uboBuf);

    // ---- scene source: deferred lighting (default) or the cubemap skybox ----
    ResourceHandle scene;
    if (cubeOn) {
        scene = build_cube(rg, dev, ubo, swap);
    } else {
        GBuffer g          = build_gbuffer(rg, dev, ubo, swap);
        ResourceHandle csm = build_shadows(rg, dev, ubo);
        ResourceHandle ao  = ssaoOn ? build_ssao(rg, env, ubo, g.depth, g.normal, swap) : ResourceHandle{};
        ResourceHandle lit = build_lighting(rg, dev, ubo, g, csm, swap);
        build_sky(rg, dev, g.depth, lit);
        scene = build_compose(rg, dev, lit, ao, swap, ssaoOn);   // SSAO off -> compose skipped, lit passes through
        scene = build_volumetrics(rg, env, ubo, scene, g.depth, csm, swap, fogOn);   // froxel fog over lit + sky
    }

    // ---- post chain: each effect inserts passes or returns the colour unchanged ----
    scene = build_bloom(rg, env, scene, swap, bloomOn);
    scene = build_taa(rg, env, scene, swap, taaOn);
    build_present(rg, dev, scene, swap);
}

static void deferred_ui()
{
    using namespace def_demo;
    ImGui::Checkbox("SSAO", &ssaoOn);
    ImGui::Checkbox("TAA", &taaOn);
    ImGui::Checkbox("Bloom", &bloomOn);
    ImGui::Checkbox("Cubemap", &cubeOn);
    ImGui::Checkbox("Volumetric Fog", &fogOn);
    if (fogOn) {
        ImGui::SliderFloat("Fog density",    &fogDensity,       0.0f, 0.5f);
        ImGui::SliderFloat("Fog anisotropy", &fogAnisotropy,   -0.9f, 0.9f);
        ImGui::SliderFloat("Fog height",     &fogHeightFalloff, 0.0f, 1.0f);
        ImGui::SliderFloat("Fog ambient",    &fogAmbient,       0.0f, 0.2f);
        ImGui::SliderFloat("Fog noise",      &fogNoise,         0.0f, 1.0f);
    }
}
