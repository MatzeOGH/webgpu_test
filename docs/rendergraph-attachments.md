# RenderGraph: MSAA, resolve & stencil attachments

How to use the multisample, resolve, and stencil attachment features. All three are *library*
capabilities — the smoke test (`RenderGraph_main.cpp`) doesn't exercise them yet, so this is the
reference. Rule still holds: the graph schedules + wires attachments, the **pass body owns the pipeline**
(`multisample.count`, depth/stencil state). It does not validate formats — Dawn does.

## MSAA & resolve

Render into a multisampled texture, then resolve it down to a single-sample texture you can sample or
present. Two new pieces:

- `TextureDesc::sampleCount` (default `1`) — set `> 1` (typically `4`) to make a texture multisampled.
  Threaded through both resource pools, so MSAA and non-MSAA textures never alias each other.
- `GraphBuilder::resolve(handle)` — declares a single-sample **resolve target** for the `color()`
  *immediately before it*. Pairing is positional: a `resolve()` binds to the most recent `color()` in the
  same pass. Wired into `WGPURenderPassColorAttachment.resolveTarget` in `execute()`.

```cpp
const WGPUExtent3D size{ w, h, 1 };

// 4x MSAA color + matching 4x MSAA depth.
auto msaaColor = rg->create_image(WEBGPU_STR("msaa.color"),
    { .dimension = WGPUTextureDimension_2D, .format = kColorFormat, .absolute = size, .sampleCount = 4 });
auto msaaDepth = rg->create_image(WEBGPU_STR("msaa.depth"),
    { .dimension = WGPUTextureDimension_2D, .format = WGPUTextureFormat_Depth32Float,
      .absolute = size, .sampleCount = 4 });

// resolve target: single-sample (sampleCount defaults to 1), SAME format + size as msaaColor.
auto resolved = rg->create_image(WEBGPU_STR("resolved"),
    { .dimension = WGPUTextureDimension_2D, .format = kColorFormat, .absolute = size });

rg->add_pass(WEBGPU_STR("forward.msaa"), PassKind::Graphics,
  [&](GraphBuilder& b){
      b.color(msaaColor);          // multisample color (loadOp Clear / storeOp Store by default)
      b.resolve(resolved);         // resolves msaaColor -> resolved; MUST follow the color() it resolves
      b.depth_stencil(msaaDepth);  // multisample depth (depth is NOT resolved -- see below)
  },
  [=](PassContext& ctx){
      // pipeline MUST be built with .multisample = { .count = 4, .mask = ~0u } to match the attachments.
      wgpuRenderPassEncoderSetPipeline(ctx.render, msaaPipe);
      wgpuRenderPassEncoderDraw(ctx.render, 3, 1, 0, 0);
  });

// a later pass samples `resolved` like any single-sample texture; the graph orders it after the resolve.
rg->add_pass(WEBGPU_STR("present"), PassKind::Graphics,
  [&](GraphBuilder& b){ b.sampled(resolved); b.color(swapchain); },
  [=](PassContext& ctx){ /* blit resolved -> swapchain */ });
```

**Resolve straight into the swapchain** (skip the present blit) — the resolve target may be imported:

```cpp
auto swap = rg->importe_image(WEBGPU_STR("swapchain"), view, size);
// ... b.color(msaaColor); b.resolve(swap); ...
```

### Rules (enforced by Dawn, not the graph)

- Every attachment in one pass — all colors *and* depth — must share the same `sampleCount`, and it must
  equal the pipeline's `multisample.count`.
- A resolve target must be single-sample (`sampleCount = 1`) with the same format and size as the color
  it resolves.
- **Depth/stencil cannot be resolved** through a render pass — WebGPU has no resolve slot for it. Make the
  depth attachment multisampled to match, and just don't store/resolve it (or resolve depth yourself in a
  later compute/shader pass).
- MSAA implies a 2D, `mipLevelCount = 1`, non-storage texture.

## Stencil

The depth-stencil attachment now carries the **stencil aspect** too, for stencil-mask effects (outlines,
portals, decal regions, "draw only where marked"). Use a depth+**stencil** format (e.g.
`Depth24PlusStencil8`, `Depth32FloatStencil8`) and pass the stencil load/store/clear to `depth_stencil()`:

```cpp
void depth_stencil(handle, depthLoad = Clear, depthStore = Store, clearDepth = 1.0f,
                   baseMip = 0, baseLayer = 0,
                   stencilLoad = Undefined, stencilStore = Undefined, stencilClear = 0);
```

The stencil params default to `Undefined/0`, so depth-only formats (the gbuffer/shadow passes on
`Depth32Float`) are unaffected — leave them out. The graph only carries the attachment's load/store/clear;
the actual stencil **compare/write** lives in the pipeline's `WGPUDepthStencilState` (don't-abstract rule).

```cpp
// depth+stencil target so the stencil aspect exists.
auto ds = rg->create_image(WEBGPU_STR("mask.ds"),
    { .dimension = WGPUTextureDimension_2D, .format = WGPUTextureFormat_Depth24PlusStencil8,
      .absolute = { w, h, 1 } });

// pass 1 -- write the mask. clear+store BOTH aspects (depth 1.0, stencil 0).
rg->add_pass(WEBGPU_STR("stencil.mask"), PassKind::Graphics,
  [&](GraphBuilder& b){
      b.depth_stencil(ds, WGPULoadOp_Clear, WGPUStoreOp_Store, 1.0f, 0, 0,
                          WGPULoadOp_Clear, WGPUStoreOp_Store, /*stencilClear*/ 0);
  },
  [=](PassContext& ctx){ /* maskPipe: stencil passOp = Replace, ref = 1; draw mask geometry */ });

// pass 2 -- fullscreen effect only where stencil == 1 (compare = Equal, ref = 1, no depth/stencil write).
rg->add_pass(WEBGPU_STR("stencil.effect"), PassKind::Graphics,
  [&](GraphBuilder& b){
      b.depth_stencil_read_only(ds);                      // test only -> reads ds, orders after the mask pass
      b.color(outColor, WGPULoadOp_Load, WGPUStoreOp_Store);
  },
  [=](PassContext& ctx){ /* effectPipe; draw fullscreen triangle */ });
```

`depth_stencil_read_only()` marks **both** depth and stencil read-only (sets `depthReadOnly` +
`stencilReadOnly`), which is what a *pure* test-only pass needs and is the cheapest way to avoid a false
write hazard.

**Depth-write + stencil-as-a-gate** (portals, mirrors, "render only inside a marked region") needs no
special flag: declare a *writable* `depth_stencil()` and let the pipeline test stencil with all stencil ops
set to `Keep`. The stencil test is fixed-function pipeline state; the graph just carries the load/store/clear
and orders the pass after whatever wrote the mask.

```cpp
// pass 2 variant: depth test+write, stencil test == 1, no stencil write.
b.depth_stencil(ds, WGPULoadOp_Load, WGPUStoreOp_Store, /*clearDepth ignored on Load*/ 0.0f, 0, 0,
                    WGPULoadOp_Load, WGPUStoreOp_Store, 0);     // preserve both aspects (depth+stencil format)
// pipeline: depthWriteEnabled = true; stencil compare = Equal, ref = 1, failOp/passOp/depthFailOp = Keep
```

ponytail: depth and stencil share one read-only flag, so the *only* thing not expressible is marking
stencil read-only while depth stays writable — which WebGPU requires only to *sample* the stencil aspect as
a texture in the same pass you depth-write (exotic). Add a per-aspect flag if that ever comes up.

## Scheduling (why ordering just works)

`resolve()`, a writing `depth_stencil()`, and the stencil mask write all start a new SSA version of their
resource, so any later pass that `sampled()` / `depth_stencil_read_only()` reads it is automatically
ordered after (a normal RAW edge — same as color writes). Declare the producing pass before its readers
(the usual def-before-use rule; see `docs/rendergraph-ssa-versioning.md`).

MSAA color/depth and resolve targets are ordinary transients: pooled by descriptor (now including
`sampleCount`), realized in `realize()`, and dropped at frame end. Imported targets (e.g. the swapchain
used as a resolve target) stay caller-owned.
