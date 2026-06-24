#pragma once
// Dear ImGui bring-up for the RenderGraph sample: SDL3 platform + WebGPU(Dawn) renderer backends,
// plus a debug widget that draws the compiled graph. #included once into the single TU
// (RenderGraph_main.cpp), after RenderGraph.h, so imgui_layer_draw_graph can read the RG:: internals.
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_wgpu.h"
#include <cstdio>   // snprintf for node labels
#include <cmath>    // fmod for the canvas grid

static void imgui_layer_init(SDL_Window* window, WGPUDevice dev, WGPUTextureFormat swapFormat)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::GetIO().IniFilename = nullptr;                   // no imgui.ini for a smoke test
	ImGui_ImplSDL3_InitForOther(window);
	ImGui_ImplWGPU_InitInfo init{};
	init.Device = dev;
	init.NumFramesInFlight = 3;
	init.RenderTargetFormat = swapFormat;                  // BGRA8Unorm
	init.DepthStencilFormat = WGPUTextureFormat_Undefined; // overlay only, no depth
	ImGui_ImplWGPU_Init(&init);
}

// NewFrame only. The DAG window is built later (after compile + realize) by imgui_layer_draw_graph,
// then imgui_layer_end_frame() finalizes; the "imgui" pass consumes ImGui::GetDrawData() at execute.
static void imgui_layer_begin_frame()
{
	ImGui_ImplWGPU_NewFrame();
	ImGui_ImplSDL3_NewFrame();
	ImGui::NewFrame();
}

// ImGui::Render(). Call once per frame to match begin_frame -- including skipped frames, so the
// NewFrame/Render pair stays balanced.
static void imgui_layer_end_frame()
{
	ImGui::Render();
}

// Everything below reads the graph's node structs + helpers, which live in namespace RG. The
// directive resolves because RenderGraph.h is included ahead of this header, and matches the
// function-scope `using namespace RG;` the rest of RenderGraph_main.cpp already uses.
using namespace RG;

// short labels/tints for the node boxes.
static const char* rg_kind_name(PassKind k)
{
	switch (k) {
	case PassKind::Graphics: return "gfx";
	case PassKind::Compute:  return "compute";
	case PassKind::Transfer: return "transfer";
	default:                 return "none";
	}
}
static ImU32 rg_kind_color(PassKind k)
{
	switch (k) {
	case PassKind::Graphics: return IM_COL32(54, 96, 156, 255);
	case PassKind::Compute:  return IM_COL32(156, 96, 40, 255);
	case PassKind::Transfer: return IM_COL32(56, 120, 70, 255);
	default:                 return IM_COL32(90, 90, 90, 255);
	}
}
static const char* rg_access_name(AccessType t)
{
	switch (t) {
	case AccessType::ColorAttachment:        return "color";
	case AccessType::DepthStencilAttachment: return "depth";
	case AccessType::DepthStencilReadOnly:   return "depth(ro)";
	case AccessType::Sampled:                return "sampled";
	case AccessType::StorageRead:            return "storage(r)";
	case AccessType::StorageWrite:           return "storage(w)";
	case AccessType::Uniform:                return "uniform";
	case AccessType::CopySrc:                return "copy(src)";
	case AccessType::CopyDst:                return "copy(dst)";
	case AccessType::Vertex:                 return "vertex";
	case AccessType::Index:                  return "index";
	case AccessType::Indirect:               return "indirect";
	}
	return "?";
}
// colour a resource lifetime bar by kind: textures cool, buffers warm.
static ImU32 rg_resource_color(ResourceNode::Kind k)
{
	return k == ResourceNode::Kind::Texture ? IM_COL32(70, 120, 170, 255)
	                                        : IM_COL32(170, 120, 60, 255);
}

// short format label for the transient-pool view. Covers the formats this sample creates; anything
// else falls back to the raw enum so an unexpected recreate still shows something legible.
static const char* rg_format_name(WGPUTextureFormat f)
{
	switch (f) {
	case WGPUTextureFormat_BGRA8Unorm:   return "BGRA8";
	case WGPUTextureFormat_RGBA8Unorm:   return "RGBA8";
	case WGPUTextureFormat_RGBA16Float:  return "RGBA16F";
	case WGPUTextureFormat_RG16Float:    return "RG16F";
	case WGPUTextureFormat_R16Float:     return "R16F";
	case WGPUTextureFormat_R8Unorm:      return "R8";
	case WGPUTextureFormat_R32Float:     return "R32F";
	case WGPUTextureFormat_Depth32Float: return "D32F";
	default: break;
	}
	static char buf[16];
	std::snprintf(buf, sizeof buf, "fmt#%d", (int)f);
	return buf;
}

// usage bits the pool keys on, as a short flag string (legend drawn in the pane header).
static void rg_usage_str(WGPUTextureUsage u, char* out, size_t n)
{
	std::snprintf(out, n, "%s%s%s%s%s",
		(u & WGPUTextureUsage_RenderAttachment) ? "A" : "",
		(u & WGPUTextureUsage_TextureBinding)   ? "T" : "",
		(u & WGPUTextureUsage_StorageBinding)   ? "S" : "",
		(u & WGPUTextureUsage_CopySrc)          ? "r" : "",
		(u & WGPUTextureUsage_CopyDst)          ? "w" : "");
}

// same idea for buffer usage (the temporal pool is textures only, so buffers key on a different set).
static void rg_buf_usage_str(WGPUBufferUsage u, char* out, size_t n)
{
	std::snprintf(out, n, "%s%s%s%s%s%s%s",
		(u & WGPUBufferUsage_Uniform)  ? "U" : "",
		(u & WGPUBufferUsage_Storage)  ? "S" : "",
		(u & WGPUBufferUsage_Vertex)   ? "V" : "",
		(u & WGPUBufferUsage_Index)    ? "I" : "",
		(u & WGPUBufferUsage_Indirect) ? "X" : "",
		(u & WGPUBufferUsage_CopySrc)  ? "r" : "",
		(u & WGPUBufferUsage_CopyDst)  ? "w" : "");
}

// bytes per texel for the formats this sample makes; unknown -> 0 so the UI shows "?" instead of
// inventing a number. tracks the set rg_format_name covers.
static uint32_t rg_format_bytes(WGPUTextureFormat f)
{
	switch (f) {
	case WGPUTextureFormat_R8Unorm:      return 1;
	case WGPUTextureFormat_R16Float:     return 2;
	case WGPUTextureFormat_RG16Float:    return 4;
	case WGPUTextureFormat_R32Float:     return 4;
	case WGPUTextureFormat_RGBA8Unorm:   return 4;
	case WGPUTextureFormat_BGRA8Unorm:   return 4;
	case WGPUTextureFormat_Depth32Float: return 4;
	case WGPUTextureFormat_RGBA16Float:  return 8;
	default: return 0;
	}
}

// one texture's footprint: every mip level, times array layers. ponytail: layers stays constant
// across mips -- right for 2D / 2D-array (all this sample makes), over-counts a true 3D texture whose
// depth also halves each level. shared by the transient pool and the temporal (history) pool.
static uint64_t rg_texture_bytes(WGPUExtent3D size, WGPUTextureFormat format, uint32_t mipLevelCount)
{
	const uint64_t bpp = rg_format_bytes(format);
	if (!bpp) return 0;
	const uint32_t layers = size.depthOrArrayLayers ? size.depthOrArrayLayers : 1;
	uint64_t total = 0;
	for (uint32_t m = 0; m < mipLevelCount; ++m) {
		const uint32_t w = (size.width  >> m) ? (size.width  >> m) : 1u;   // max(1, ..) without <algorithm>
		const uint32_t h = (size.height >> m) ? (size.height >> m) : 1u;
		total += (uint64_t)w * h * layers * bpp;
	}
	return total;
}

static uint64_t rg_entry_bytes(const TransientResourcePool::Entry& e)
{
	return rg_texture_bytes(e.size, e.format, e.mipLevelCount);
}

// byte count -> short human string. ponytail: no GB tier, a transient pool is a few MB.
static void rg_bytes_str(uint64_t bytes, char* out, size_t n)
{
	if      (bytes >= (1u << 20)) std::snprintf(out, n, "%.1f MB", bytes / (1024.0 * 1024.0));
	else if (bytes >= (1u << 10)) std::snprintf(out, n, "%.1f KB", bytes / 1024.0);
	else                          std::snprintf(out, n, "%llu B",  (unsigned long long)bytes);
}

// argb-alpha tweak + warm/cool access tints, shared by the DAG pins below and the lifetime bars.
static ImU32 rg_with_alpha(ImU32 c, ImU32 a) { return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT); }
static constexpr ImU32 kRGWrite = IM_COL32(232, 145, 64, 255);   // write / output pin
static constexpr ImU32 kRGRead = IM_COL32(74, 158, 206, 255);   // read / input pin
static constexpr ImU32 kRGExt = IM_COL32(118, 196, 132, 255);   // external-input source node (imported read)
static constexpr ImU32 kRGPresent = IM_COL32(196, 122, 214, 255);   // present / display sink node (imported output)

// DAG view -----------------------------------------------------------------------------------------
// node-graph layout: one box per pass in dependency columns, one pin per resource access (reads left,
// writes right), edges run producer-output -> consumer-input (true RAW data flow). Hovering a pin lights
// the upstream producer cone -- every pass that must run to make that resource. Reads the .cpp-internal
// node structs directly; assumes a compiled, realized graph like the other dumps.

static constexpr int kRgDagMax = 128;

// one laid-out pass box. its index in box[] == execution-order index (boxes built by walking m_passes),
// so it doubles as the adjacency / cone index.
struct RgDagBox { PassNode* p; int layer; ImVec2 tl; float w, h; int nIn, nOut; };

// does pass p write resource id? one access == one pin, writes land on the right.
static bool rg_pass_writes(PassNode* p, uint32_t id)
{
	for (uint32_t i = 0; i < p->accessCount; ++i)
		if (p->accesses[i].handle.id == id && access_is_write(p->accesses[i].type)) return true;
	return false;
}

// does this access consume the resource's prior contents -> an input pin? plain reads do; so does a
// color/depth attachment with LoadOp_Load (reads what's there, then stores == read-modify-write, e.g. the
// imgui overlay loading the swapchain). a Load attachment therefore gets BOTH an input and an output pin.
static bool rg_access_reads(const ResourceAccess& a)
{
	if (!access_is_write(a.type)) return true;
	return (a.type == AccessType::ColorAttachment || a.type == AccessType::DepthStencilAttachment)
		&& a.loadOp == WGPULoadOp_Load;
}

// do a and b both write a common resource? 0 = no, 1 = yes at DIFFERENT subresources (parallel layers/
// mips, e.g. CSM cascades -- not a true conflict), 2 = yes at the SAME subresource (a real WAW).
static int rg_shared_write(PassNode* a, PassNode* b)
{
	int r = 0;
	for (uint32_t i = 0; i < a->accessCount; ++i) {
		if (!access_is_write(a->accesses[i].type)) continue;
		for (uint32_t j = 0; j < b->accessCount; ++j) {
			if (!access_is_write(b->accesses[j].type) || a->accesses[i].handle.id != b->accesses[j].handle.id) continue;
			if (a->accesses[i].baseLayer == b->accesses[j].baseLayer && a->accesses[i].baseMip == b->accesses[j].baseMip) return 2;
			r = 1;
		}
	}
	return r;
}

// index of pass in box[] (== execution order), or -1.
static int rg_box_index(const RgDagBox* box, int n, PassNode* p)
{
	for (int i = 0; i < n; ++i) if (box[i].p == p) return i;
	return -1;
}

// producer of resource `id` as read by pass `p`: the highest-execution-index writer of `id` among p's
// predecessors. compile() always links the current (RAW) producer into adjacency, and a later writer
// would have started a version p can't see -- so the latest writer-predecessor IS that producer. -1 =
// no in-graph producer (imported value, or read-before-write -> an external input pin).
static int rg_producer_of(const RgDagBox* box, int n, PassNode* p, uint32_t id)
{
	int best = -1;
	for (NodeAdjacency* a = p->adjacency; a; a = a->next) {
		if (!rg_pass_writes(a->pass, id)) continue;
		int idx = rg_box_index(box, n, a->pass);
		if (idx > best) best = idx;
	}
	return best;
}

// output-pin slot index of resource id on pass p (writes only), or -1.
static int rg_out_slot(PassNode* p, uint32_t id)
{
	int slot = 0;
	for (uint32_t k = 0; k < p->accessCount; ++k)
		if (access_is_write(p->accesses[k].type)) { if (p->accesses[k].handle.id == id) return slot; ++slot; }
	return -1;
}

// input-pin slot index of resource id on pass p (reads only), mirroring rg_out_slot. matches the
// encounter-order slot the pin loop assigns, so it lands on the right pin centre.
static int rg_in_slot(PassNode* p, uint32_t id)
{
	int slot = 0;
	for (uint32_t k = 0; k < p->accessCount; ++k)
		if (rg_access_reads(p->accesses[k])) { if (p->accesses[k].handle.id == id) return slot; ++slot; }
	return -1;
}

// mark seed + everything it transitively depends on (the upstream producer cone): DFS over predecessor
// edges. iterative -- a long pass chain would overflow a recursive stack.
static void rg_mark_cone(const RgDagBox* box, int n, int seed, bool* inCone)
{
	if (seed < 0) return;
	int stack[kRgDagMax], sp = 0;
	stack[sp++] = seed; inCone[seed] = true;
	while (sp) {
		PassNode* p = box[stack[--sp]].p;
		for (NodeAdjacency* a = p->adjacency; a; a = a->next) {
			int idx = rg_box_index(box, n, a->pass);
			if (idx >= 0 && !inCone[idx]) { inCone[idx] = true; stack[sp++] = idx; }
		}
	}
}

// same pass name? used to frame + group runs of repeated passes (shadow.cascade, bloom.down/up).
static bool rg_same_name(WGPUStringView a, WGPUStringView b)
{
	if (a.length != b.length) return false;
	if (a.length == 0) return true;
	return a.data && b.data && std::memcmp(a.data, b.data, a.length) == 0;
}

// case-insensitive: does pass name contain needle? (name filter)
static bool rg_name_has(WGPUStringView name, const char* needle)
{
	if (!needle || !needle[0]) return true;
	if (!name.data) return false;
	size_t q = 0; while (needle[q]) ++q;
	auto lc = [](char c) { return (c >= 'A' && c <= 'Z') ? char(c + 32) : c; };
	for (size_t i = 0; i + q <= name.length; ++i) {
		size_t j = 0; while (j < q && lc(name.data[i + j]) == lc(needle[j])) ++j;
		if (j == q) return true;
	}
	return false;
}

// squared distance from point p to segment ab (edge hit-testing).
static float rg_seg_d2(ImVec2 p, ImVec2 a, ImVec2 b)
{
	float vx = b.x - a.x, vy = b.y - a.y, wx = p.x - a.x, wy = p.y - a.y;
	float L = vx * vx + vy * vy, t = L > 0 ? (wx * vx + wy * vy) / L : 0;
	t = t < 0 ? 0 : t > 1 ? 1 : t;
	float dx = a.x + t * vx - p.x, dy = a.y + t * vy - p.y;
	return dx * dx + dy * dy;
}

// true graph output for the halo: writes an *imported* resource (swapchain) whose value leaves the
// frame to be presented. compile()'s p->sink also fires for passes that only write a temporal/history
// layer (persistent, read next frame) -- those keep the pass alive but aren't graph outputs, so the
// halo skips them.
static bool rg_pass_is_sink(RenderGraph* rg, PassNode* p)
{
	for (uint32_t i = 0; i < p->accessCount; ++i) {
		if (!access_is_write(p->accesses[i].type)) continue;
		ResourceNode* r = find_node(rg, p->accesses[i].handle);
		if (r && r->imported) return true;
	}
	return false;
}

// one pin glyph: round for textures, square for buffers. filled = normal, hollow = external input (no
// in-graph producer). square half-extent == circle radius, so the hover/lock ring (kPinR + 3) still
// frames either shape.
static void rg_draw_pin(ImDrawList* dl, ImVec2 c, float r, ImU32 col, bool filled, bool buffer)
{
	if (buffer) {
		ImVec2 a(c.x - r, c.y - r), b(c.x + r, c.y + r);
		if (filled) dl->AddRectFilled(a, b, col, 1.5f);
		else        dl->AddRect(a, b, col, 1.5f, 0, 2.0f);
	}
	else {
		if (filled) dl->AddCircleFilled(c, r, col, 12);
		else        dl->AddCircle(c, r, col, 12, 2.0f);
	}
}

// dashed cubic bezier -- ImDrawList has no dashed stroke, so sample the curve and draw every other span.
// used by the temporal feedback links so they read as cross-frame, distinct from the solid data edges.
static void rg_dashed_cubic(ImDrawList* dl, ImVec2 p0, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImU32 col, float th)
{
	constexpr int kSeg = 24;
	ImVec2 prev = p0;
	for (int i = 1; i <= kSeg; ++i) {
		float t = (float)i / kSeg, u = 1 - t, w0 = u*u*u, w1 = 3*u*u*t, w2 = 3*u*t*t, w3 = t*t*t;
		ImVec2 q(w0*p0.x + w1*p1.x + w2*p2.x + w3*p3.x, w0*p0.y + w1*p1.y + w2*p2.y + w3*p3.y);
		if (i & 1) dl->AddLine(prev, q, col, th);   // every other span = a dash
		prev = q;
	}
}

// little filled arrowhead at `tip`, pointing along (tip - from). marks the read end of a feedback link.
static void rg_arrowhead(ImDrawList* dl, ImVec2 from, ImVec2 tip, ImU32 col, float sz)
{
	float dx = tip.x - from.x, dy = tip.y - from.y, L = std::sqrt(dx*dx + dy*dy);
	if (L < 1e-3f) return;
	dx /= L; dy /= L;
	ImVec2 a(tip.x - dx*sz - dy*sz*0.5f, tip.y - dy*sz + dx*sz*0.5f);
	ImVec2 b(tip.x - dx*sz + dy*sz*0.5f, tip.y - dy*sz - dx*sz*0.5f);
	dl->AddTriangleFilled(tip, a, b, col);
}

static void rg_draw_dag(RenderGraph* rg, RenderGraphStorage& s)
{
	constexpr float kBoxW = 190.0f, kColGap = 64.0f, kRowGap = 20.0f;
	constexpr float kHeaderH = 22.0f, kFooterH = 14.0f, kPinRowH = 18.0f, kMinBodyH = 12.0f;
	constexpr float kPinR = 5.0f, kPinHit = 8.0f;

	// virtual nodes (frame-boundary endpoints: temporal read/write so far) -- toggled from the toolbar.
	// read before the layout pass so disabling them also drops their layout influence (no reserved columns).
	static bool showVirtual = true;
	// imported buffers (uniforms read by many passes): on = one source node fanning faint edges to every
	// reader; off = a node at each use site, like an imported texture. read here so it shapes the layout too.
	static bool fanBuffers = true;

	// ---- layout pass (sink-anchored, Sugiyama-style). columns place each pass left of what depends on
	// it: column = longest distance TO a sink over the FULL dependency graph, so sinks land rightmost and
	// a WAW-ordered pass (a shadow cascade, no pin to its successor) still sits in order instead of
	// floating off as a fake sink. crossing reduction, in contrast, counts only the DRAWN edges -- the
	// RAW producer->consumer pins -- because WAW has no pin to cross. one barycenter sweep from the sinks
	// back (as asked), then a settle. no dummy nodes for multi-column edges; this is a debug view.
	RgDagBox box[kRgDagMax];
	int n = 0;
	for (PassNode* p = s.m_passes; p && n < kRgDagMax; p = p->next, ++n) {
		int nIn = 0, nOut = 0;   // a Load attachment counts as both (read-modify-write)
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (rg_access_reads(p->accesses[k]))      ++nIn;
			if (access_is_write(p->accesses[k].type)) ++nOut;
		}
		int rows = nIn > nOut ? nIn : nOut;
		float h = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
		box[n] = { p, 0, ImVec2(0, 0), kBoxW, h, nIn, nOut };
	}

	// dist = longest path to a sink over all deps; reverse exec order visits sinks first and relaxes
	// each producer. column = maxDist - dist.
	int dist[kRgDagMax] = {}, maxDist = 0;
	for (int i = n - 1; i >= 0; --i) {
		for (NodeAdjacency* a = box[i].p->adjacency; a; a = a->next) {
			int q = rg_box_index(box, n, a->pass);
			if (q >= 0 && dist[i] + 1 > dist[q]) dist[q] = dist[i] + 1;
		}
		if (dist[i] > maxDist) maxDist = dist[i];
	}

	// parallel-writer groups: passes writing one resource at DIFFERENT subresources (CSM cascades: same
	// csm handle, layer 0/1/2) with no data dependency between them. the whole-resource WAW would string
	// them across columns; union them onto one DAG level instead. anc[v][u] = u is a transitive data-
	// ancestor of v (built in exec order, producers precede consumers), so a real chain like the bloom
	// mip pyramid -- each level samples the previous -- is NOT mistaken for parallel.
	static bool anc[kRgDagMax][kRgDagMax];
	for (int v = 0; v < n; ++v) for (int u = 0; u < n; ++u) anc[v][u] = false;
	for (int v = 0; v < n; ++v) {
		PassNode* p = box[v].p;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (!rg_access_reads(p->accesses[k])) continue;
			int u = rg_producer_of(box, n, p, p->accesses[k].handle.id);
			if (u < 0) continue;
			anc[v][u] = true;
			for (int w = 0; w < n; ++w) if (anc[u][w]) anc[v][w] = true;
		}
	}
	int groupRep[kRgDagMax];
	for (int i = 0; i < n; ++i) groupRep[i] = i;
	auto find = [&](int x) { while (groupRep[x] != x) { groupRep[x] = groupRep[groupRep[x]]; x = groupRep[x]; } return x; };
	for (int i = 0; i < n; ++i)
		for (int j = i + 1; j < n; ++j)
			if (rg_shared_write(box[i].p, box[j].p) == 1 && !anc[i][j] && !anc[j][i]) groupRep[find(i)] = find(j);
	for (int i = 0; i < n; ++i) groupRep[i] = find(i);   // flatten to roots

	// column = maxDist - dist, then snap each group to its rightmost member (nearest its consumer) so the
	// siblings share one level, and compact away the columns that leaves empty.
	int colOf[kRgDagMax], groupCol[kRgDagMax] = {};
	for (int i = 0; i < n; ++i) colOf[i] = maxDist - dist[i];
	for (int i = 0; i < n; ++i) if (colOf[i] > groupCol[groupRep[i]]) groupCol[groupRep[i]] = colOf[i];
	for (int i = 0; i < n; ++i) colOf[i] = groupCol[groupRep[i]];
	int remap[kRgDagMax]; for (int c = 0; c <= maxDist; ++c) remap[c] = -1;
	for (int i = 0; i < n; ++i) remap[colOf[i]] = 1;
	int nextCol = 0; for (int c = 0; c <= maxDist; ++c) if (remap[c] == 1) remap[c] = nextCol++;
	for (int i = 0; i < n; ++i) colOf[i] = remap[colOf[i]];
	maxDist = nextCol ? nextCol - 1 : 0;

	// ---- virtual nodes: frame-boundary endpoints drawn as real DAG nodes so the column/barycenter/y code
	// below places them like any pass (no bespoke overlay). each attaches to ONE pass pin -- a read node one
	// column BEFORE its reader, a write node one column AFTER its writer -- so a widely-read texture gets a
	// small node at each use site (like repeated power symbols) instead of one node fanning edges everywhere.
	// three kinds, all gated by the toolbar toggle:
	//   * temporal: create_temporal_image makes two resource nodes sharing a name -- curr (temporalIndex 0,
	//     written THIS frame for next) and prev (temporalIndex 1, read this frame = LAST frame's curr; the
	//     pool rotates two physical textures). cross-frame, so no in-frame edge joins the pair.
	//   * external input: an IMPORTED resource read with no in-graph writer (importe_image'd / import_buffer'd,
	//     value from outside the frame). a texture gets a node per reader pin; a buffer (a uniform read almost
	//     everywhere) is ONE source node fanning faint edges to every reader, so it doesn't swamp the view.
	//   * present: an imported resource that IS written -- the swapchain, whose final value leaves to display.
	struct TNode { bool isRead; int passBox, pin; ResourceNode* res; int col; float w, h; const char* cap; ImU32 tint; int li; };
	static std::vector<TNode> tnodes; tnodes.clear();
	auto push_tnode = [&](bool isRead, int passBox, int pin, ResourceNode* res, const char* cap, ImU32 tint) {
		char b[48]; std::snprintf(b, sizeof b, "%.*s", (int)res->name.length, res->name.data ? res->name.data : "?");
		ImVec2 ns = ImGui::CalcTextSize(b), cs = ImGui::CalcTextSize(cap);
		float w = (ns.x > cs.x ? ns.x : cs.x) + 16, h = ns.y + cs.y + 10;
		tnodes.push_back({ isRead, passBox, pin, res, isRead ? colOf[passBox] - 1 : colOf[passBox] + 1, w, h, cap, tint, -1 });
	};
	if (showVirtual)
		for (ResourceNode* r = s.m_resouces; r; r = r->next) {
			if (r->persistent) {   // temporal: write node at the curr writer, read node at each prev reader
				bool curr = r->temporalIndex == 0;
				for (int i = 0; i < n; ++i) {
					int sl = curr ? rg_out_slot(box[i].p, r->handle.id) : rg_in_slot(box[i].p, r->handle.id);
					if (sl >= 0) push_tnode(!curr, i, sl, r, curr ? "next frame" : "last frame", curr ? kRGWrite : kRGRead);
				}
				continue;
			}
			int lwb = -1, lws = -1;   // last writer, if any
			for (int i = 0; i < n; ++i) { int sl = rg_out_slot(box[i].p, r->handle.id); if (sl >= 0) { lwb = i; lws = sl; } }
			if (lwb < 0) {             // no in-graph writer: external input read from outside the frame
				if (r->imported && r->kind == ResourceNode::Kind::Buffer && fanBuffers) {
					// collapsed: a uniform is read almost everywhere -- emit ONE source node anchored at its
					// earliest reader (far left); the draw loop fans faint edges to every reader, so all
					// consumers show without a node per use site swamping the view.
					int best = -1, bestSl = -1;
					for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, r->handle.id); if (sl >= 0 && (best < 0 || colOf[i] < colOf[best])) { best = i; bestSl = sl; } }
					if (best >= 0) push_tnode(true, best, bestSl, r, "imported", kRGExt);
				}
				else if (r->imported)  // imported texture, or a buffer with fan-out off -> a node at each reader pin
					for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, r->handle.id); if (sl >= 0) push_tnode(true, i, sl, r, "imported", kRGExt); }
			}
			else if (r->imported)      // present: imported + written -> sink node at the last writer
				push_tnode(false, lwb, lws, r, "present", kRGPresent);
		}
	// keep columns in [0, maxDist]: shift everything right if a read node landed left of 0, extend for writes.
	int tmin = 0; for (TNode& t : tnodes) if (t.col < tmin) tmin = t.col;
	if (tmin < 0) { for (int i = 0; i < n; ++i) colOf[i] -= tmin; maxDist -= tmin; for (TNode& t : tnodes) t.col -= tmin; }
	for (TNode& t : tnodes) if (t.col > maxDist) maxDist = t.col;

	// ===== layered routing: real boxes + dummy waypoints. an edge that skips columns gets a dummy in each
	// crossed column, reserving a lane there so it never hides behind a box (Sugiyama virtual nodes).
	struct LNode { int col, box, tn; float x, y; };       // box == -1 => dummy/temporal; tn >= 0 => temporal node
	struct REdge { int src, dst, sOut, dIn; uint32_t id; int chainN, chain[kRgDagMax]; };
	static std::vector<LNode> lnode; lnode.clear();
	static std::vector<REdge> edge;  edge.clear();
	for (int i = 0; i < n; ++i) lnode.push_back({ colOf[i], i, -1, 0, 0 });   // reals: lnode index == box index

	// drawn edges (a read pin fed by its producer + parallel-writer siblings), each with a dummy chain
	// through the columns it skips.
	for (int i = 0; i < n; ++i) {
		PassNode* p = box[i].p; int inS = 0;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (!rg_access_reads(p->accesses[k])) continue;
			uint32_t id = p->accesses[k].handle.id; int dIn = inS++;
			int prod = rg_producer_of(box, n, p, id);
			if (prod < 0) continue;
			for (int w = 0; w < n; ++w) {
				if (groupRep[w] != groupRep[prod] || colOf[w] >= colOf[i]) continue;
				int sOut = rg_out_slot(box[w].p, id);
				if (sOut < 0) continue;
				REdge e{ w, i, sOut, dIn, id, 0, {} };
				for (int c = colOf[w] + 1; c < colOf[i]; ++c) { e.chain[e.chainN++] = (int)lnode.size(); lnode.push_back({ c, -1, -1, 0, 0 }); }
				edge.push_back(e);
			}
		}
	}

	// temporal nodes join the layered list as their own (box == -1, tn >= 0) layered nodes.
	for (int ti = 0; ti < (int)tnodes.size(); ++ti) { tnodes[ti].li = (int)lnode.size(); lnode.push_back({ tnodes[ti].col, -1, ti, 0, 0 }); }

	// per-column members (reals + dummies + temporal) + left/right neighbour lists for the barycenter.
	const int LN = (int)lnode.size();
	static std::vector<int> lcol[kRgDagMax];
	for (int c = 0; c <= maxDist; ++c) lcol[c].clear();
	for (int li = 0; li < LN; ++li) lcol[lnode[li].col].push_back(li);
	static std::vector<std::vector<int>> lleft, lright; lleft.assign(LN, {}); lright.assign(LN, {});
	for (REdge& e : edge) {
		int prev = e.src;
		for (int t = 0; t < e.chainN; ++t) { lright[prev].push_back(e.chain[t]); lleft[e.chain[t]].push_back(prev); prev = e.chain[t]; }
		lright[prev].push_back(e.dst); lleft[e.dst].push_back(prev);
	}
	// temporal adjacency: read node sits left of its reader, write node right of its writer. one link each,
	// enough for the barycenter to order them and y-relax to align them to the pass's row.
	for (TNode& t : tnodes)
		if (t.isRead) { lright[t.li].push_back(t.passBox); lleft[t.passBox].push_back(t.li); }
		else          { lright[t.passBox].push_back(t.li); lleft[t.li].push_back(t.passBox); }

	// barycenter crossing reduction over the layered nodes; from the sinks back, then settle.
	static std::vector<int> lrank; lrank.assign(LN, 0);
	auto reix = [&](int c) { for (int r = 0; r < (int)lcol[c].size(); ++r) lrank[lcol[c][r]] = r; };
	for (int c = 0; c <= maxDist; ++c) reix(c);
	auto bnb = [&](int li, bool right) { std::vector<int>& nb = right ? lright[li] : lleft[li]; float s = 0; int c = 0; for (int x : nb) { s += lrank[x]; ++c; } return c ? s / c : -1.0f; };
	auto lsweep = [&](bool backward) {
		int from = backward ? maxDist - 1 : 1, to = backward ? -1 : maxDist + 1, step = backward ? -1 : 1;
		for (int c = from; c != to; c += step) {
			std::vector<int>& m = lcol[c];
			std::vector<float> key(m.size());
			for (int j = 0; j < (int)m.size(); ++j) { float b = bnb(m[j], backward); key[j] = b < 0 ? (float)lrank[m[j]] : b; }
			for (int a = 1; a < (int)m.size(); ++a) { int mv = m[a]; float kv = key[a]; int b = a - 1; while (b >= 0 && key[b] > kv) { m[b + 1] = m[b]; key[b + 1] = key[b]; --b; } m[b + 1] = mv; key[b + 1] = kv; }
			reix(c);
		}
		};
	lsweep(true); lsweep(false); lsweep(true);

	// y-coordinate assignment: stack each column, then relax every node toward the mean of its neighbours
	// (within column order + min separation) so edges -- the dummy chains especially -- run straight rather
	// than zig-zag. x is just the column.
	const float kLane = 16.0f;
	auto slotH = [&](int li) { int b = lnode[li].box; if (b >= 0) return box[b].h; int t = lnode[li].tn; return t >= 0 ? tnodes[t].h : kLane; };
	static std::vector<float> cy; cy.assign(LN, 0.0f);
	for (int c = 0; c <= maxDist; ++c) { float y = 0; for (int li : lcol[c]) { cy[li] = y + slotH(li) * 0.5f; y += slotH(li) + kRowGap; } }
	for (int it = 0; it < 8; ++it)
		for (int c = 0; c <= maxDist; ++c) {
			std::vector<int>& m = lcol[c];
			for (int r = 0; r < (int)m.size(); ++r) {
				int li = m[r]; float s = 0; int cnt = 0;
				for (int x : lleft[li])  { s += cy[x]; ++cnt; }
				for (int x : lright[li]) { s += cy[x]; ++cnt; }
				if (!cnt) continue;
				float d = s / cnt;
				float lo = r > 0                ? cy[m[r - 1]] + (slotH(m[r - 1]) + slotH(li)) * 0.5f + kRowGap : -1e30f;
				float hi = r + 1 < (int)m.size() ? cy[m[r + 1]] - (slotH(li) + slotH(m[r + 1])) * 0.5f - kRowGap :  1e30f;
				cy[li] = d < lo ? lo : d > hi ? hi : d;
			}
		}

	// write positions + the graph's bounding box (canvas-local), used to centre the view.
	float gxMin = 1e30f, gyMin = 1e30f, gxMax = -1e30f, gyMax = -1e30f;
	for (int c = 0; c <= maxDist; ++c) {
		float cx = c * (kBoxW + kColGap);
		for (int li : lcol[c]) {
			if (lnode[li].box >= 0) {
				int b = lnode[li].box; box[b].tl = ImVec2(cx, cy[li] - box[b].h * 0.5f); box[b].layer = c;
				if (cx < gxMin) gxMin = cx; if (cx + box[b].w > gxMax) gxMax = cx + box[b].w;
				if (box[b].tl.y < gyMin) gyMin = box[b].tl.y; if (box[b].tl.y + box[b].h > gyMax) gyMax = box[b].tl.y + box[b].h;
			}
			else {
				lnode[li].x = cx; lnode[li].y = cy[li];   // dummy: column left edge + lane centre
				int t = lnode[li].tn;
				if (t >= 0) {   // temporal node: centre in the column slot, include in the view bbox
					float ctr = cx + kBoxW * 0.5f, hw = tnodes[t].w * 0.5f, hh = tnodes[t].h * 0.5f;
					lnode[li].x = ctr;
					if (ctr - hw < gxMin) gxMin = ctr - hw; if (ctr + hw > gxMax) gxMax = ctr + hw;
					if (cy[li] - hh < gyMin) gyMin = cy[li] - hh; if (cy[li] + hh > gyMax) gyMax = cy[li] + hh;
				}
			}
		}
	}
	if (gxMax < gxMin) { gxMin = gyMin = gxMax = gyMax = 0; }   // empty graph

	// toolbar: reset/centre the view + a name filter that dims non-matching passes.
	static char filter[64] = "";
	bool doReset = ImGui::Button("Reset view");
	ImGui::SameLine(); ImGui::SetNextItemWidth(180);
	ImGui::InputTextWithHint("##rgfilter", "filter passes...", filter, sizeof filter);
	ImGui::SameLine(); ImGui::Checkbox("virtual nodes", &showVirtual);
	ImGui::SameLine(); ImGui::Checkbox("fan-out buffers", &fanBuffers);   // off -> one virtual node per buffer use site
	const bool filterActive = filter[0] != 0;

	// pannable canvas: a static scroll offset (drag left/middle to pan) + a grid, after the imgui node-
	// graph example. no scrollbar -- navigation is panning, so a big graph isn't boxed in.
	static ImVec2 scrolling(0, 0);
	static bool userMoved = false;   // true once the user pans; until then keep the graph centred
	ImGui::BeginChild("rg_canvas", ImVec2(0, 0), true,
		ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
	ImGuiIO& io = ImGui::GetIO();
	const ImVec2 winPos = ImGui::GetCursorScreenPos();
	const ImVec2 winSize = ImGui::GetContentRegionAvail();
	ImGui::InvisibleButton("canvas", ImVec2(winSize.x > 0 ? winSize.x : 1, winSize.y > 0 ? winSize.y : 1));
	const bool canvasHovered = ImGui::IsItemHovered();
	const bool canvasActive = ImGui::IsItemActive();
	bool panned = false;
	if (canvasActive && (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f))) {
		scrolling.x += io.MouseDelta.x; scrolling.y += io.MouseDelta.y; userMoved = true; panned = true;
	}
	// keep the graph centred in the viewport until the user pans. this rides through the first frames where
	// the child's content region is still settling (a one-shot centre there latches against a bogus size).
	// the Reset button and a double-click re-arm it.
	if (doReset || (canvasHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) userMoved = false;
	if (!userMoved && !panned && winSize.x > 1 && winSize.y > 1) {
		scrolling.x = winSize.x * 0.5f - (gxMin + gxMax) * 0.5f;
		scrolling.y = winSize.y * 0.5f - (gyMin + gyMax) * 0.5f;
	}
	ImDrawList* dl = ImGui::GetWindowDrawList();
	const ImVec2 origin(winPos.x + scrolling.x, winPos.y + scrolling.y);   // panned top-left; node coords add this

	// faint grid, scrolling with the canvas.
	const float kGrid = 48.0f;
	for (float gx = std::fmod(scrolling.x, kGrid); gx < winSize.x; gx += kGrid)
		dl->AddLine(ImVec2(winPos.x + gx, winPos.y), ImVec2(winPos.x + gx, winPos.y + winSize.y), IM_COL32(255, 255, 255, 14));
	for (float gy = std::fmod(scrolling.y, kGrid); gy < winSize.y; gy += kGrid)
		dl->AddLine(ImVec2(winPos.x, winPos.y + gy), ImVec2(winPos.x + winSize.x, winPos.y + gy), IM_COL32(255, 255, 255, 14));

	// ---- group frames: a labelled cluster behind each run of consecutive same-named passes
	// (shadow.cascade x3, bloom.down xN, bloom.up xN). purely visual -- the boxes stay individual.
	for (int gi = 0; gi < n;) {
		int gj = gi + 1;
		while (gj < n && rg_same_name(box[gj].p->name, box[gi].p->name)) ++gj;
		if (gj - gi >= 2) {
			float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
			for (int k = gi; k < gj; ++k) {
				float ax = origin.x + box[k].tl.x, ay = origin.y + box[k].tl.y;
				if (ax < x0) x0 = ax;
				if (ay < y0) y0 = ay;
				if (ax + box[k].w > x1) x1 = ax + box[k].w;
				if (ay + box[k].h > y1) y1 = ay + box[k].h;
			}
			const float pad = 9.0f, lblH = 15.0f;
			ImVec2 a(x0 - pad, y0 - pad - lblH), b(x1 + pad, y1 + pad);
			dl->AddRectFilled(a, b, IM_COL32(255, 255, 255, 8), 6.0f);
			dl->AddRect(a, b, IM_COL32(180, 180, 205, 90), 6.0f, 0, 1.5f);
			WGPUStringView nm = box[gi].p->name;
			char lbl[64]; std::snprintf(lbl, sizeof lbl, "%.*s  x%d", (int)nm.length, nm.data ? nm.data : "", gj - gi);
			dl->AddText(ImVec2(a.x + 6, a.y + 1), IM_COL32(205, 205, 225, 205), lbl);
		}
		gi = gj;
	}

	// pin centres, screen space. reads fill input slots (left) in encounter order; writes fill output
	// slots (right) the same way.
	auto inPin = [&](int b, int slot) { return ImVec2(origin.x + box[b].tl.x,
		origin.y + box[b].tl.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
	auto outPin = [&](int b, int slot) { return ImVec2(origin.x + box[b].tl.x + box[b].w,
		origin.y + box[b].tl.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
	// screen-space polyline of an edge (src out-pin, two points per dummy, dst in-pin); shared by hit-test + draw.
	auto edgePoints = [&](const REdge& e, ImVec2* pts) -> int {
		int np = 0;
		pts[np++] = outPin(e.src, e.sOut);
		for (int t = 0; t < e.chainN; ++t) { float lx = origin.x + lnode[e.chain[t]].x, ly = origin.y + lnode[e.chain[t]].y; pts[np++] = ImVec2(lx, ly); pts[np++] = ImVec2(lx + kBoxW, ly); }
		pts[np++] = inPin(e.dst, e.dIn);
		return np;
	};
	bool matchBox[kRgDagMax];
	for (int i = 0; i < n; ++i) matchBox[i] = rg_name_has(box[i].p->name, filter);

	// ---- find the single hovered pin (manual rect test: pins are small + overlap the box button).
	int hovB = -1, hovSlot = -1; bool hovWrite = false; uint32_t hovId = 0, hovMip = 0, hovLayer = 0; AccessType hovType{};
	if (canvasHovered && !canvasActive) {
		for (int i = 0; i < n && hovB < 0; ++i) {
			int inS = 0, outS = 0; PassNode* p = box[i].p;
			for (uint32_t k = 0; k < p->accessCount && hovB < 0; ++k) {
				const ResourceAccess& acc = p->accesses[k];
				if (rg_access_reads(acc)) {   // input pin (left)
					ImVec2 c = inPin(i, inS);
					if (ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit)))
						{ hovB = i; hovSlot = inS; hovWrite = false; hovId = acc.handle.id; hovType = acc.type; hovMip = acc.baseMip; hovLayer = acc.baseLayer; }
					++inS;
				}
				if (hovB < 0 && access_is_write(acc.type)) {   // output pin (right)
					ImVec2 c = outPin(i, outS);
					if (ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit)))
						{ hovB = i; hovSlot = outS; hovWrite = true; hovId = acc.handle.id; hovType = acc.type; hovMip = acc.baseMip; hovLayer = acc.baseLayer; }
					++outS;
				}
			}
		}
	}
	// hovered box: only used for the fallback reads/writes tooltip when no pin caught the mouse.
	int hovBox = -1;
	if (hovB < 0 && canvasHovered && !canvasActive)
		for (int i = 0; i < n; ++i) {
			ImVec2 tl(origin.x + box[i].tl.x, origin.y + box[i].tl.y);
			if (ImGui::IsMouseHoveringRect(tl, ImVec2(tl.x + box[i].w, tl.y + box[i].h))) { hovBox = i; break; }
		}

	// ---- click a pin to LOCK its cone (stays without holding the mouse); click empty canvas to clear.
	static int lockB = -1, lockSlot = -1; static bool lockWrite = false; static uint32_t lockId = 0;
	static bool pressed = false; static ImVec2 pressAt;
	if (canvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { pressed = true; pressAt = io.MousePos; }
	if (pressed && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
		pressed = false;
		float mdx = io.MousePos.x - pressAt.x, mdy = io.MousePos.y - pressAt.y;
		if (mdx * mdx + mdy * mdy < 16) {   // a click, not a pan
			if (hovB >= 0) {
				if (lockB == hovB && lockWrite == hovWrite && lockSlot == hovSlot) lockB = -1;   // toggle off
				else { lockB = hovB; lockWrite = hovWrite; lockSlot = hovSlot; lockId = hovId; }
			}
			else lockB = -1;
		}
	}
	if (lockB >= n) lockB = -1;   // stale lock after the graph shrank

	// ---- hovered edge (only when not over a pin/box): nearest edge polyline within a few px.
	int hovEdge = -1;
	if (hovB < 0 && hovBox < 0 && canvasHovered && !canvasActive) {
		float best = 6.0f * 6.0f;
		for (int ei = 0; ei < (int)edge.size(); ++ei) {
			ImVec2 pts[2 * kRgDagMax + 2]; int np = edgePoints(edge[ei], pts);
			for (int t = 0; t + 1 < np; ++t) {
				ImVec2 a2 = pts[t], b2 = pts[t + 1]; float dx = (b2.x - a2.x) * 0.5f;
				ImVec2 c1(a2.x + dx, a2.y), c2(b2.x - dx, b2.y), prev = a2;
				for (int s2 = 1; s2 <= 10; ++s2) {   // sample the cubic
					float tt = s2 / 10.0f, u = 1 - tt, w0 = u * u * u, w1 = 3 * u * u * tt, w2 = 3 * u * tt * tt, w3 = tt * tt * tt;
					ImVec2 q(w0 * a2.x + w1 * c1.x + w2 * c2.x + w3 * b2.x, w0 * a2.y + w1 * c1.y + w2 * c2.y + w3 * b2.y);
					float d2 = rg_seg_d2(io.MousePos, prev, q); if (d2 < best) { best = d2; hovEdge = ei; }
					prev = q;
				}
			}
		}
	}

	// ---- cones. seed from the hovered pin, else the locked pin. upstream = producers needed to make the
	// resource; for an output pin also mark downstream = who consumes it.
	int coneB = hovB >= 0 ? hovB : lockB;
	bool coneWrite = hovB >= 0 ? hovWrite : lockWrite;
	uint32_t coneId = hovB >= 0 ? hovId : lockId;
	bool inCone[kRgDagMax] = {}, downCone[kRgDagMax] = {}; bool coneActive = false;
	if (coneB >= 0) {
		int seed = coneWrite ? coneB : rg_producer_of(box, n, box[coneB].p, coneId);
		if (seed >= 0) { rg_mark_cone(box, n, seed, inCone); inCone[coneB] = true; coneActive = true; }   // include the hovered node so its immediate edge lights
		if (coneActive && coneWrite) {   // downstream descendants over the data edges
			int st[kRgDagMax], sp = 0; st[sp++] = coneB;
			while (sp) { int u = st[--sp]; for (REdge& e : edge) if (e.src == u && !downCone[e.dst]) { downCone[e.dst] = true; st[sp++] = e.dst; } }
		}
	}
	auto fout  = [&](int i) { return filterActive && !matchBox[i]; };
	auto isDim = [&](int i) { return fout(i) || (coneActive && !inCone[i] && !downCone[i] && !(coneWrite && i == coneB)); };
	auto inUp  = [&](int i) { return coneActive && inCone[i] && !fout(i); };
	auto inDn  = [&](int i) { return coneActive && (downCone[i] || (coneWrite && i == coneB)) && !fout(i); };

	// ---- data edges, routed through their dummy waypoints so none hides behind a box. gold = upstream
	// cone, teal = downstream consumers, white = hovered, dim otherwise.
	for (int ei = 0; ei < (int)edge.size(); ++ei) {
		REdge& e = edge[ei];
		bool eup = inUp(e.src) && inUp(e.dst), edn = inDn(e.src) && inDn(e.dst);
		ImU32 col; float th;
		if (ei == hovEdge)                     { col = IM_COL32(255, 255, 255, 255); th = 3.0f; }
		else if (eup)                          { col = IM_COL32(245, 222, 120, 235); th = 2.5f; }
		else if (edn)                          { col = IM_COL32(120, 222, 180, 235); th = 2.5f; }
		else if (isDim(e.src) || isDim(e.dst)) { col = IM_COL32(150, 150, 150, 34);  th = 2.0f; }
		else                                   { col = IM_COL32(170, 170, 170, 200); th = 2.0f; }
		ImVec2 pts[2 * kRgDagMax + 2]; int np = edgePoints(e, pts);
		for (int t = 0; t + 1 < np; ++t) {
			ImVec2 a2 = pts[t], b2 = pts[t + 1]; float dx = (b2.x - a2.x) * 0.5f;
			dl->AddBezierCubic(a2, ImVec2(a2.x + dx, a2.y), ImVec2(b2.x - dx, b2.y), b2, col, th);
		}
	}

	// ---- boxes.
	for (int i = 0; i < n; ++i) {
		ImVec2 tl(origin.x + box[i].tl.x, origin.y + box[i].tl.y), br(tl.x + box[i].w, tl.y + box[i].h);
		bool dim = isDim(i), up = inUp(i), dn = inDn(i);

		ImU32 fill = rg_kind_color(box[i].p->kind);
		dl->AddRectFilled(tl, br, dim ? rg_with_alpha(fill, 55) : fill, 5.0f);
		ImU32 edgec = up ? IM_COL32(255, 255, 255, 255) : dn ? IM_COL32(120, 222, 180, 255) : dim ? IM_COL32(40, 40, 40, 120) : IM_COL32(20, 20, 20, 255);
		dl->AddRect(tl, br, edgec, 5.0f, 0, (up || dn) ? 2.5f : 1.0f);
		dl->AddLine(ImVec2(tl.x, tl.y + kHeaderH), ImVec2(br.x, tl.y + kHeaderH), IM_COL32(255, 255, 255, dim ? 18 : 40));

		// sinks: red halo. imported writers only -- temporal/history writers are sinks to the culler
		// (compile sets p->sink) but aren't graph outputs, so they don't get the halo.
		if (rg_pass_is_sink(rg, box[i].p))
			for (int e = 3; e >= 1; --e)
				dl->AddRect(ImVec2(tl.x - e, tl.y - e), ImVec2(br.x + e, br.y + e), IM_COL32(255, 100, 0, dim ? 80 : 200), 6.0f, 0, 1.0f);

		WGPUStringView nm = box[i].p->name;
		char head[96];
		std::snprintf(head, sizeof head, "P%d  %.*s", i, (int)nm.length, nm.data ? nm.data : "");
		dl->PushClipRect(tl, ImVec2(br.x - 4, br.y), true);
		dl->AddText(ImVec2(tl.x + 7, tl.y + 4), dim ? IM_COL32(190, 190, 190, 120) : IM_COL32(255, 255, 255, 255), head);
		dl->PopClipRect();

		const char* kn = rg_kind_name(box[i].p->kind);
		ImVec2 ks = ImGui::CalcTextSize(kn);
		dl->AddText(ImVec2(tl.x + (box[i].w - ks.x) * 0.5f, br.y - kFooterH + 1),
			dim ? IM_COL32(200, 200, 200, 110) : IM_COL32(225, 225, 225, 220), kn);
	}

	// ---- pins + resource labels.
	for (int i = 0; i < n; ++i) {
		PassNode* p = box[i].p; bool dim = isDim(i);
		const float mid = origin.x + box[i].tl.x + box[i].w * 0.5f;
		int inS = 0, outS = 0;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			const ResourceAccess& acc = p->accesses[k];
			ResourceNode* r = find_node(rg, acc.handle);
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
			ImVec2 ls = ImGui::CalcTextSize(lbl);
			const ImU32 lc = dim ? IM_COL32(190, 190, 190, 110) : IM_COL32(230, 230, 230, 255);
			const bool isBuf = r && r->kind == ResourceNode::Kind::Buffer;   // square pin vs round (texture)

			if (rg_access_reads(acc)) {   // input pin (left); hollow if no in-graph producer (external input)
				int slot = inS++; ImVec2 c = inPin(i, slot);
				ImU32 base = dim ? rg_with_alpha(kRGRead, 70) : kRGRead;
				rg_draw_pin(dl, c, kPinR, base, rg_producer_of(box, n, p, acc.handle.id) >= 0, isBuf);
				if (lockB == i && !lockWrite && slot == lockSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
				if (i == hovB && !hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
				dl->PushClipRect(ImVec2(c.x + kPinR + 3, c.y - kPinRowH * 0.5f), ImVec2(mid, c.y + kPinRowH * 0.5f), true);
				dl->AddText(ImVec2(c.x + kPinR + 3, c.y - ls.y * 0.5f), lc, lbl);
				dl->PopClipRect();
			}
			if (access_is_write(acc.type)) {   // output pin (right)
				int slot = outS++; ImVec2 c = outPin(i, slot);
				ImU32 base = dim ? rg_with_alpha(kRGWrite, 70) : kRGWrite;
				rg_draw_pin(dl, c, kPinR, base, true, isBuf);
				if (lockB == i && lockWrite && slot == lockSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
				if (i == hovB && hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
				dl->PushClipRect(ImVec2(mid, c.y - kPinRowH * 0.5f), ImVec2(c.x - kPinR - 3, c.y + kPinRowH * 0.5f), true);
				dl->AddText(ImVec2(c.x - kPinR - 3 - ls.x, c.y - ls.y * 0.5f), lc, lbl);
				dl->PopClipRect();
			}
		}
	}

	// ---- virtual endpoint nodes + dashed links, drawn from the layout positions computed above. most nodes
	// attach to one pass pin in the adjacent column (a read/source node feeds an input pin from the left, a
	// write/sink node is fed by an output pin from the left); an imported buffer fans to all readers (below).
	// tint + caption come from the node's kind.
	for (TNode& t : tnodes) {
		// with fan-out on, an imported buffer (a uniform read by many passes) is ONE source node fanning faint
		// dashed edges to EVERY reader, so all consumers show without a node per use site; it's never wholesale-
		// hidden by the pass filter (the per-reader gate just drops the edges to filtered passes). off -> it's a
		// node per use site like a texture, taking the single-edge branch below.
		const bool fan = fanBuffers && t.isRead && t.res->imported && t.res->kind == ResourceNode::Kind::Buffer;
		if (!fan && fout(t.passBox)) continue;
		ImVec2 c(origin.x + lnode[t.li].x, origin.y + lnode[t.li].y);
		ImVec2 a(c.x - t.w * 0.5f, c.y - t.h * 0.5f), b(c.x + t.w * 0.5f, c.y + t.h * 0.5f);
		if (fan) {
			ImU32 fc = rg_with_alpha(t.tint, 90); ImVec2 p(b.x, c.y);
			for (int i = 0; i < n; ++i) {
				if (fout(i)) continue;
				int sl = rg_in_slot(box[i].p, t.res->handle.id);
				if (sl < 0) continue;
				ImVec2 q = inPin(i, sl); float dx = (q.x - p.x) * 0.5f;
				rg_dashed_cubic(dl, p, ImVec2(p.x + dx, p.y), ImVec2(q.x - dx, q.y), q, fc, 1.5f);
				rg_arrowhead(dl, ImVec2(q.x - 8, q.y), q, fc, 6.0f);
			}
		}
		else {
			ImVec2 p, q;
			if (t.isRead) { q = inPin(t.passBox, t.pin); p = ImVec2(b.x, c.y); }    // node -> reader input pin
			else          { p = outPin(t.passBox, t.pin); q = ImVec2(a.x, c.y); }   // writer output pin -> node
			float dx = (q.x - p.x) * 0.5f;
			rg_dashed_cubic(dl, p, ImVec2(p.x + dx, p.y), ImVec2(q.x - dx, q.y), q, t.tint, 2.0f);
			rg_arrowhead(dl, ImVec2(q.x - 8, q.y), q, t.tint, 7.0f);
		}

		char nm[48]; std::snprintf(nm, sizeof nm, "%.*s", (int)t.res->name.length, t.res->name.data ? t.res->name.data : "?");
		dl->AddRectFilled(a, b, IM_COL32(32, 30, 40, 240), 4.0f);
		dl->AddRect(a, b, t.tint, 4.0f, 0, 1.5f);
		ImVec2 ns = ImGui::CalcTextSize(nm), cs = ImGui::CalcTextSize(t.cap);
		dl->AddText(ImVec2(c.x - ns.x * 0.5f, a.y + 3), IM_COL32(238, 236, 242, 255), nm);
		dl->AddText(ImVec2(c.x - cs.x * 0.5f, b.y - cs.y - 3), rg_with_alpha(t.tint, 220), t.cap);
	}

	// ---- tooltip: hovered pin wins; else fall back to the per-pass reads/writes list.
	if (hovB >= 0) {
		PassNode* p = box[hovB].p; ResourceNode* r = find_node(rg, { hovId });
		WGPUStringView rn = r ? r->name : WGPUStringView{};
		ImGui::BeginTooltip();
		ImGui::Text("%.*s", (int)rn.length, rn.data ? rn.data : "?");
		if (r) {
			if (r->kind == ResourceNode::Kind::Texture) ImGui::Text("texture  %u x %u", r->resolved.width, r->resolved.height);
			else                                         ImGui::Text("buffer  %llu bytes", (unsigned long long)r->bufferSize);
		}
		// subresource this pin touches: layer for an array (CSM cascades), mip for a chain (bloom).
		if (r && r->kind == ResourceNode::Kind::Texture) {
			uint32_t layers = r->resolved.depthOrArrayLayers ? r->resolved.depthOrArrayLayers : 1;
			uint32_t mips   = r->mipLevelCount ? r->mipLevelCount : 1;
			if (layers > 1 || mips > 1 || hovLayer > 0 || hovMip > 0) {
				char sub[64]; int o = 0;
				if (layers > 1 || hovLayer > 0) o += std::snprintf(sub + o, sizeof sub - o, "layer %u / %u", hovLayer, layers);
				if (mips > 1 || hovMip > 0)     o += std::snprintf(sub + o, sizeof sub - o, "%smip %u / %u", o ? "   " : "", hovMip, mips);
				ImGui::TextDisabled("%s", sub);
			}
		}
		const char* verb = hovWrite ? "write"
			: (hovType == AccessType::ColorAttachment || hovType == AccessType::DepthStencilAttachment) ? "load"
			: "read";
		ImGui::TextDisabled("%s on P%d  (%s)", verb, hovB, rg_access_name(hovType));
		ImGui::Separator();
		if (hovWrite) ImGui::Text("produced here");
		else {
			int prod = rg_producer_of(box, n, p, hovId);
			if (prod >= 0) {
				WGPUStringView pn = box[prod].p->name;
				ImGui::Text("produced by P%d %.*s", prod, (int)pn.length, pn.data ? pn.data : "");
			}
			else ImGui::TextDisabled(r && r->imported ? "imported (external input)" : "external input (no producer)");
		}
		ImGui::EndTooltip();
	}
	else if (hovBox >= 0) {
		PassNode* p = box[hovBox].p; WGPUStringView nm = p->name;
		const char* kn = rg_kind_name(p->kind);
		ImGui::BeginTooltip();
		ImGui::Text("%.*s  [%s]", (int)nm.length, nm.data ? nm.data : "", kn);
		ImGui::Separator();
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			const ResourceAccess& acc = p->accesses[k];
			ResourceNode* r = find_node(rg, acc.handle);
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			ImGui::Text("[%s] %.*s  (%s)%s", access_is_write(acc.type) ? "W" : "R",
				(int)rn.length, rn.data ? rn.data : "", rg_access_name(acc.type),
				r ? (r->imported ? "  [imported]" : r->persistent ? "  [temporal]" : "") : "");
		}
		ImGui::EndTooltip();
	}
	else if (hovEdge >= 0) {
		REdge& e = edge[hovEdge]; ResourceNode* r = find_node(rg, { e.id });
		WGPUStringView rn = r ? r->name : WGPUStringView{};
		WGPUStringView sn = box[e.src].p->name, dn = box[e.dst].p->name;
		ImGui::BeginTooltip();
		ImGui::Text("%.*s", (int)rn.length, rn.data ? rn.data : "?");
		ImGui::TextDisabled("P%d %.*s  ->  P%d %.*s", e.src, (int)sn.length, sn.data ? sn.data : "",
			e.dst, (int)dn.length, dn.data ? dn.data : "");
		ImGui::EndTooltip();
	}

	ImGui::EndChild();
}

// imported/persistent resources are left out of compile()'s firstUse/lastUse (the graph doesn't own
// their memory), so the lifetimes view recovers their span by walking the pass list -- only to draw a
// bar, never for aliasing. false if no surviving pass touches the resource.
static bool rg_external_span(RenderGraphStorage& s, ResourceNode* r, uint32_t& first, uint32_t& last)
{
	first = ResourceNode::kNoPass; last = 0;
	uint32_t idx = 0;
	for (PassNode* p = s.m_passes; p; p = p->next, ++idx)
		for (uint32_t i = 0; i < p->accessCount; ++i)
			if (p->accesses[i].handle.id == r->handle.id) {
				if (first == ResourceNode::kNoPass) first = idx;
				last = idx;
			}
	return first != ResourceNode::kNoPass;
}

// what `p` does to resource `id`: bit 1 = reads, bit 2 = writes (0 = doesn't touch it this pass).
static int rg_pass_access(PassNode* p, uint32_t id)
{
	int a = 0;
	for (uint32_t i = 0; i < p->accessCount; ++i)
		if (p->accesses[i].handle.id == id)
			a |= access_is_write(p->accesses[i].type) ? 2 : 1;
	return a;
}

// Resource-lifetime grid. The top axis is the passes in execution order (named); each resource is a
// bar across the passes it stays live for -- first touch to last. The bar is split per pass: a column
// where a pass writes the resource is warm, a column it only reads is cool, read+write splits the
// cell, and a faint kind-tinted band shows passes that merely keep it alive. Transient resources read
// the firstUse/lastUse the aliasing analysis fills in compile() phase 3; imported + temporal are
// caller-owned (toggle adds them, muted). Bars that never share a column are aliasing candidates.
static void rg_draw_lifetimes(RenderGraph* rg, RenderGraphStorage& s)
{
	constexpr int   kMax = 128;
	constexpr float kLabelW = 150.0f, kColW = 88.0f, kRowH = 24.0f, kHeaderH = 24.0f;

	static bool showExternal = true;
	ImGui::Checkbox("show imported / temporal", &showExternal);
	ImGui::SameLine();
	ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGWrite), "write");
	ImGui::SameLine(0, 4);
	ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGRead), "read");
	ImGui::SameLine();
	ImGui::TextDisabled("(band = held alive)");

	PassNode* passAt[kMax];
	int nPass = 0;
	for (PassNode* p = s.m_passes; p && nPass < kMax; p = p->next) passAt[nPass++] = p;

	// rows top-to-bottom: live transients, then (if shown) imported/temporal with a span, then the
	// span-less rows (dead transients, untouched externals) pooled at the bottom.
	struct Row { ResourceNode* r; uint32_t first, last; bool bar; };
	Row row[kMax];
	int nRow = 0;
	for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next)
		if (!r->is_external() && r->firstUse != ResourceNode::kNoPass)
			row[nRow++] = { r, r->firstUse, r->lastUse, true };
	if (showExternal)
		for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next) {
			uint32_t f, l;
			if (r->is_external() && rg_external_span(s, r, f, l)) row[nRow++] = { r, f, l, true };
		}
	for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next)
		if (!r->is_external() && r->firstUse == ResourceNode::kNoPass)
			row[nRow++] = { r, 0, 0, false };
	if (showExternal)
		for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next) {
			uint32_t f, l;
			if (r->is_external() && !rg_external_span(s, r, f, l)) row[nRow++] = { r, 0, 0, false };
		}

	ImGui::BeginChild("rg_life", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::Dummy(ImVec2(kLabelW + nPass * kColW, kHeaderH + nRow * kRowH));   // reserve scroll region
	ImDrawList* dl = ImGui::GetWindowDrawList();

	const float gridR = origin.x + kLabelW + nPass * kColW;
	const float gridT = origin.y + kHeaderH;
	const float gridB = gridT + nRow * kRowH;

	// top axis: one clipped pass name per column + a faint column line; hover a header for the full name.
	for (int c = 0; c < nPass; ++c) {
		float x = origin.x + kLabelW + c * kColW;
		dl->AddLine(ImVec2(x, origin.y), ImVec2(x, gridB), IM_COL32(255, 255, 255, 22));
		WGPUStringView nm = passAt[c]->name;
		if (nm.data) {
			dl->PushClipRect(ImVec2(x + 4, origin.y), ImVec2(x + kColW, gridT), true);
			dl->AddText(ImVec2(x + 6, origin.y + 5), IM_COL32(225, 225, 225, 255), nm.data, nm.data + nm.length);
			dl->PopClipRect();
		}
		ImGui::SetCursorScreenPos(ImVec2(x, origin.y));
		ImGui::PushID(c);
		ImGui::InvisibleButton("h", ImVec2(kColW, kHeaderH));
		if (ImGui::IsItemHovered()) {
			ImGui::BeginTooltip();
			ImGui::Text("P%d  %.*s  [%s]", c, (int)nm.length, nm.data ? nm.data : "", rg_kind_name(passAt[c]->kind));
			ImGui::EndTooltip();
		}
		ImGui::PopID();
	}
	dl->AddLine(ImVec2(origin.x, gridT), ImVec2(gridR, gridT), IM_COL32(255, 255, 255, 40));   // axis underline

	// one row per resource: name in the left gutter, a bar spanning [first, last] columns.
	for (int i = 0; i < nRow; ++i) {
		ResourceNode* r = row[i].r;
		const bool ext = r->is_external();
		const float y = gridT + i * kRowH;
		if (i & 1) dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(gridR, y + kRowH), IM_COL32(255, 255, 255, 10));

		// name colour doubles as the legend: imported = amber, temporal = violet, transient = white,
		// span-less = grey. temporal layers share a name, so tag them with their frame index.
		WGPUStringView nm = r->name;
		ImU32 nameCol = !row[i].bar   ? IM_COL32(135, 135, 135, 255)
		              : r->imported   ? IM_COL32(240, 170, 90, 255)
		              : r->persistent ? IM_COL32(180, 165, 245, 255)
		              :                 IM_COL32(230, 230, 230, 255);
		char label[96];
		if (r->persistent)
			std::snprintf(label, sizeof label, "%.*s #%u", (int)nm.length, nm.data ? nm.data : "", r->temporalIndex);
		else
			std::snprintf(label, sizeof label, "%.*s", (int)nm.length, nm.data ? nm.data : "");
		dl->PushClipRect(ImVec2(origin.x + 4, y), ImVec2(origin.x + kLabelW - 4, y + kRowH), true);
		dl->AddText(ImVec2(origin.x + 6, y + 4), nameCol, label);
		dl->PopClipRect();

		if (!row[i].bar) {   // no surviving pass touched it -> no bar.
			dl->AddText(ImVec2(origin.x + kLabelW + 6, y + 4), nameCol, ext ? "(unused)" : "(dead)");
			continue;
		}

		const float x0 = origin.x + kLabelW + row[i].first * kColW + 3.0f;
		const float x1 = origin.x + kLabelW + (row[i].last + 1) * kColW - 3.0f;
		const ImVec2 tl(x0, y + 3.0f), br(x1, y + kRowH - 3.0f);

		ImGui::SetCursorScreenPos(tl);
		ImGui::PushID(kMax + i);   // keep ids clear of the header buttons
		ImGui::InvisibleButton("b", ImVec2(x1 - x0, kRowH - 6.0f));
		const bool hov = ImGui::IsItemHovered();
		ImGui::PopID();

		// base band = the lifetime itself, faintly kind-tinted so texture/buffer stays legible; held
		// columns show only this. imported/temporal sit fainter still.
		dl->AddRectFilled(tl, br, rg_with_alpha(rg_resource_color(r->kind), ext ? 32 : 60), 0.0f);

		// per-pass cells on top: warm = write, cool = read, split = both. externals dimmed a touch.
		const ImU32 wcol = rg_with_alpha(kRGWrite, ext ? 175 : 255);
		const ImU32 rcol = rg_with_alpha(kRGRead,  ext ? 175 : 255);
		for (uint32_t c = row[i].first; c <= row[i].last; ++c) {
			int acc = rg_pass_access(passAt[c], r->handle.id);
			if (!acc) continue;   // held this pass -> band shows through
			float sx0 = (c == row[i].first) ? x0 : origin.x + kLabelW + c * kColW;
			float sx1 = (c == row[i].last)  ? x1 : origin.x + kLabelW + (c + 1) * kColW;
			if (acc == 3) {   // read + write -> split top (write) / bottom (read)
				float mid = (tl.y + br.y) * 0.5f;
				dl->AddRectFilled(ImVec2(sx0, tl.y), ImVec2(sx1, mid), wcol, 0.0f);
				dl->AddRectFilled(ImVec2(sx0, mid), ImVec2(sx1, br.y), rcol, 0.0f);
			} else
				dl->AddRectFilled(ImVec2(sx0, tl.y), ImVec2(sx1, br.y), acc == 2 ? wcol : rcol, 0.0f);
		}

		// outline marks transient vs imported/temporal (accent edge); white on hover.
		ImU32 edge = hov ? IM_COL32(255, 255, 255, 255) : ext ? nameCol : IM_COL32(20, 20, 20, 160);
		dl->AddRect(tl, br, edge, 0.0f, 0, hov ? 2.0f : 1.0f);

		if (hov) {
			WGPUStringView f = pass_name_at(s.m_passes, row[i].first);
			WGPUStringView l = pass_name_at(s.m_passes, row[i].last);
			ImGui::BeginTooltip();
			ImGui::Text("%s  [%s]", label, r->kind == ResourceNode::Kind::Texture ? "texture" : "buffer");
			if (r->kind == ResourceNode::Kind::Texture)
				ImGui::Text("%u x %u", r->resolved.width, r->resolved.height);
			else
				ImGui::Text("%llu bytes", (unsigned long long)r->bufferSize);
			if (r->imported)        ImGui::TextDisabled("imported; caller-owned, not aliased");
			else if (r->persistent) ImGui::TextDisabled("temporal layer %u; pool-owned, not aliased", r->temporalIndex);
			ImGui::Separator();
			if (row[i].first == row[i].last)
				ImGui::Text("alive in %.*s", (int)f.length, f.data ? f.data : "");
			else
				ImGui::Text("alive %.*s .. %.*s",
					(int)f.length, f.data ? f.data : "", (int)l.length, l.data ? l.data : "");
			for (uint32_t c = row[i].first; c <= row[i].last; ++c) {   // per-pass read/write breakdown
				int a = rg_pass_access(passAt[c], r->handle.id);
				if (!a) continue;
				WGPUStringView pn = passAt[c]->name;
				ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(a == 1 ? kRGRead : kRGWrite),
					"  %s %.*s", a == 3 ? "rw" : a == 2 ? " w" : " r", (int)pn.length, pn.data ? pn.data : "");
			}
			ImGui::EndTooltip();
		}
	}

	ImGui::EndChild();
}

// GPU-memory view across every pool the graph owns: transient textures (descriptor-keyed cache),
// transient buffers (per-frame, on the resource nodes) and temporal/history textures (PersistentResourcePool
// ping-pong). Grand total at the top answers "how much VRAM does the graph cost"; the transient pool also
// keeps its create/evict log so steady-state reuse is still verifiable (0 created after warmup). Drawn after
// realize() and before release_resources()/end_frame(), so every count is this frame's live allocation.
static void rg_draw_transient_pool(RenderGraphStorage& s)
{
	TransientResourcePool&  tp   = s.m_allocator->transient;
	PersistentResourcePool& pool = s.m_allocator->pool;

	// transient textures (pooled, descriptor-keyed).
	int held = (int)tp.entries.size(), inUse = 0;
	uint64_t texBytes = 0, texInUseBytes = 0;
	for (const TransientResourcePool::Entry& e : tp.entries) {
		const uint64_t b = rg_entry_bytes(e);
		texBytes += b;
		if (e.inUse) { ++inUse; texInUseBytes += b; }
	}

	// transient buffers: not pooled, created per-frame in realize() and freed in release_resources() -- we
	// draw between the two, so r->buffer/bufferSize are this frame's real allocations. imported = caller-owned.
	int bufCount = 0;
	uint64_t bufBytes = 0;
	for (ResourceNode* r = s.m_resouces; r; r = r->next)
		if (r->kind == ResourceNode::Kind::Buffer && !r->is_external() && r->buffer) { ++bufCount; bufBytes += r->bufferSize; }

	// temporal/history textures: kLayers physical textures per entry (current + previous), ping-ponged.
	int tmpCount = 0;
	uint64_t tmpBytes = 0;
	for (const PersistentResourcePool::Entry& e : pool.entries)
		if (e.created) { ++tmpCount; tmpBytes += rg_texture_bytes(e.size, e.format, e.mipLevelCount) * PersistentResourcePool::kLayers; }

	const uint64_t grand = texBytes + bufBytes + tmpBytes;

	ImGui::Text("frame %llu  --  transient pool: %d held, %d in use", (unsigned long long)tp.frame, held, inUse);
	ImGui::SameLine();
	if (tp.createdThisFrame == 0)
		ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1), "  --  0 created (reused from pool)");
	else
		ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "  --  %u created this frame", tp.createdThisFrame);

	char gb[24]; rg_bytes_str(grand, gb, sizeof gb);
	ImGui::Text("VRAM %s total", gb);
	{
		char a[24], ib[24], idb[24], bb[24], cb[24];
		rg_bytes_str(texBytes, a, sizeof a);
		rg_bytes_str(texInUseBytes, ib, sizeof ib);
		rg_bytes_str(texBytes - texInUseBytes, idb, sizeof idb);   // in use is a subset sum -> no underflow
		rg_bytes_str(bufBytes, bb, sizeof bb);
		rg_bytes_str(tmpBytes, cb, sizeof cb);
		ImGui::BulletText("transient tex  %-9s  %d held (%s in use, %s idle)", a, held, ib, idb);
		ImGui::BulletText("temporal       %-9s  %d entries x%u layers", cb, tmpCount, PersistentResourcePool::kLayers);
		ImGui::BulletText("buffers        %-9s  %d", bb, bufCount);
	}

	ImGui::TextDisabled("tex usage A=attach T=sampled S=storage r=copy-src w=copy-dst   |   buf usage U=uniform S=storage V=vertex I=index X=indirect   |   evict after %llu idle frames",
		(unsigned long long)TransientResourcePool::kRetain);
	ImGui::Separator();

	const ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

	// currently held physical textures.
	ImGui::TextDisabled("transient textures");
	if (ImGui::BeginTable("tp_live", 7, tf)) {
		ImGui::TableSetupColumn("size");
		ImGui::TableSetupColumn("mips");
		ImGui::TableSetupColumn("layers");
		ImGui::TableSetupColumn("format");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("mem");
		ImGui::TableSetupColumn("state");
		ImGui::TableHeadersRow();
		int idx = 0;
		for (const TransientResourcePool::Entry& e : tp.entries) {
			char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
			const uint64_t eb = rg_entry_bytes(e);
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
			ImGui::TableNextColumn(); ImGui::Text("%u", e.mipLevelCount);
			ImGui::TableNextColumn(); ImGui::Text("%u", e.size.depthOrArrayLayers);
			ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn();
			if (eb) { char mb[24]; rg_bytes_str(eb, mb, sizeof mb); ImGui::Text("%s", mb); }
			else    ImGui::TextDisabled("?");
			ImGui::TableNextColumn();
			if (e.inUse) ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1), "in use");
			else         ImGui::TextDisabled("idle %lluf", (unsigned long long)(tp.frame - e.lastUsedFrame));
		}
		ImGui::EndTable();
	}

	// temporal/history textures (PersistentResourcePool): one row per name, kLayers physical textures each.
	ImGui::Spacing();
	ImGui::TextDisabled("temporal (history) textures");
	if (ImGui::BeginTable("tp_tmp", 7, tf)) {
		ImGui::TableSetupColumn("name");
		ImGui::TableSetupColumn("size");
		ImGui::TableSetupColumn("mips");
		ImGui::TableSetupColumn("layers");
		ImGui::TableSetupColumn("format");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("mem");
		ImGui::TableHeadersRow();
		bool any = false;
		for (const PersistentResourcePool::Entry& e : pool.entries) {
			if (!e.created) continue;
			any = true;
			char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
			const uint64_t eb = rg_texture_bytes(e.size, e.format, e.mipLevelCount) * PersistentResourcePool::kLayers;
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%s", e.name.c_str());
			ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
			ImGui::TableNextColumn(); ImGui::Text("%u", e.mipLevelCount);
			ImGui::TableNextColumn(); ImGui::Text("%u x%u", e.size.depthOrArrayLayers, PersistentResourcePool::kLayers);
			ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn();
			if (eb) { char mb[24]; rg_bytes_str(eb, mb, sizeof mb); ImGui::Text("%s", mb); }
			else    ImGui::TextDisabled("?");
		}
		ImGui::EndTable();
		if (!any) ImGui::TextDisabled("(none)");
	}

	// transient buffers (per-frame, not pooled). walk the resource nodes for graph-owned buffers.
	ImGui::Spacing();
	ImGui::TextDisabled("transient buffers");
	if (ImGui::BeginTable("tp_buf", 3, tf)) {
		ImGui::TableSetupColumn("name");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("size");
		ImGui::TableHeadersRow();
		bool any = false;
		for (ResourceNode* r = s.m_resouces; r; r = r->next) {
			if (r->kind != ResourceNode::Kind::Buffer || r->is_external() || !r->buffer) continue;
			any = true;
			char ub[12]; rg_buf_usage_str(r->bufUsage, ub, sizeof ub);
			char mb[24]; rg_bytes_str(r->bufferSize, mb, sizeof mb);
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%.*s", (int)r->name.length, r->name.data ? r->name.data : "");
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn(); ImGui::Text("%s", mb);
		}
		ImGui::EndTable();
		if (!any) ImGui::TextDisabled("(none this frame)");
	}

	ImGui::Spacing();
	ImGui::Text("transient pool events (newest first)");
	ImGui::BeginChild("tp_log", ImVec2(0, 0), true);
	const uint64_t total = tp.eventCount;
	const uint64_t shown = total < TransientResourcePool::kLog ? total : TransientResourcePool::kLog;
	for (uint64_t k = 0; k < shown; ++k) {
		const TransientResourcePool::LogRec& r = tp.eventLog[(total - 1 - k) % TransientResourcePool::kLog];
		const bool create = r.kind == TransientResourcePool::Event::Create;
		ImGui::TextColored(create ? ImVec4(0.95f, 0.70f, 0.30f, 1) : ImVec4(0.60f, 0.60f, 0.60f, 1),
			"f%-6llu  %-6s  %ux%u  %s", (unsigned long long)r.frame, create ? "CREATE" : "evict",
			r.size.width, r.size.height, rg_format_name(r.format));
	}
	if (shown == 0) ImGui::TextDisabled("(no events yet)");
	ImGui::EndChild();
}

// The per-frame bump arena (GraphAllocator) as a dual-ended usage bar. One fixed buffer backs the
// RenderGraph object, every resource/pass node, the type-erased execute closures and the strings
// copied in: `used` grows up from the base (cool, drawn from the left) and is what the frame holds.
// compile()'s scratch grows down from the top (warm, drawn from the right); it's rewound to 0 before
// we draw, so the bar uses its tracked per-frame peak. The ends can't overlap -- used + scratch peak
// is the worst-case occupancy that decides whether 1 MB is enough (in the hover).
static void rg_draw_arena(GraphAllocator& a)
{
	const double kib         = 1024.0;
	const double cap         = a.capacity ? (double)a.capacity : 1.0;
	const float  usedFrac    = (float)((double)a.used / cap);
	const float  scratchFrac = (float)((double)(a.scratchHighWater) / cap);

	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted("arena");
	ImGui::SameLine();

	const ImVec2 p0 = ImGui::GetCursorScreenPos();
	float        w  = ImGui::GetContentRegionAvail().x;
	if (w < 1.0f) w = 1.0f;                                   // collapsed window -> keep InvisibleButton happy
	const float  h  = ImGui::GetFrameHeight();
	const ImVec2 p1(p0.x + w, p0.y + h);

	ImGui::InvisibleButton("arena_bar", ImVec2(w, h));
	const bool  hov = ImGui::IsItemHovered();
	ImDrawList* dl  = ImGui::GetWindowDrawList();

	dl->AddRectFilled(p0, p1, IM_COL32(28, 28, 28, 255), 3.0f);                       // track
	dl->AddRectFilled(p0, ImVec2(p0.x + w * usedFrac, p1.y), kRGRead, 0.0f);          // used, from the left
	if (scratchFrac > 0.0f)                                                            // scratch peak, from the right
		dl->AddRectFilled(ImVec2(p1.x - w * scratchFrac, p0.y), p1, kRGWrite, 0.0f);
	dl->AddRect(p0, p1, hov ? IM_COL32(255, 255, 255, 255) : IM_COL32(20, 20, 20, 180), 3.0f);

	char ov[64];
	std::snprintf(ov, sizeof ov, "%.1f / %.0f KB  (%.2f%%)", a.used / kib, a.capacity / kib, usedFrac * 100.0);
	const ImVec2 ts = ImGui::CalcTextSize(ov);
	dl->AddText(ImVec2(p0.x + (w - ts.x) * 0.5f, p0.y + (h - ts.y) * 0.5f), IM_COL32(235, 235, 235, 255), ov);

	if (hov) {
		ImGui::BeginTooltip();
		ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGRead),  "front used   %llu B  (%.2f KB)",
			(unsigned long long)a.used, a.used / kib);
		ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGWrite), "scratch peak %llu B  (%.2f KB)",
			(unsigned long long)a.scratchHighWater, a.scratchHighWater / kib);
		ImGui::Text("free         %llu B  (%.2f KB)",
			(unsigned long long)(a.capacity - a.used), (a.capacity - a.used) / kib);
		ImGui::Separator();
		ImGui::Text("capacity     %.0f KB", a.capacity / kib);
		ImGui::Text("worst case   %.2f%%  (used + scratch peak)", (usedFrac + scratchFrac) * 100.0);
		ImGui::EndTooltip();
	}
}

// The RenderGraph debug window: an FPS + arena-usage header, then a tab bar over the dependency DAG,
// the resource-lifetime grid and the transient pool. Built after compile()+realize(), so it all reads
// a finished graph.
static void imgui_layer_draw_graph(RenderGraph* rg)
{
	RenderGraphStorage& s = *storage(rg);

	ImGui::Begin("RenderGraph");
	ImGui::Text(" %.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
	rg_draw_arena(*s.m_allocator);
	ImGui::Separator();

	if (ImGui::BeginTabBar("rg_tabs")) {
		if (ImGui::BeginTabItem("Graph"))     { rg_draw_dag(rg, s);            ImGui::EndTabItem(); }
		if (ImGui::BeginTabItem("Lifetimes")) { rg_draw_lifetimes(rg, s);      ImGui::EndTabItem(); }
		if (ImGui::BeginTabItem("Memory"))    { rg_draw_transient_pool(s);     ImGui::EndTabItem(); }
		ImGui::EndTabBar();
	}
	ImGui::End();
}

static void imgui_layer_shutdown()
{
	ImGui_ImplWGPU_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
}
