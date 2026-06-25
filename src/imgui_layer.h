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

// same idea for buffer usage (a different flag set than textures).
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

// stable distinct colour per group, hashed from the prefix (FNV-1a -> small palette).
static ImU32 group_color(WGPUStringView prefix)
{
	uint32_t h = 2166136261u;
	for (size_t i = 0; i < sv_length(prefix); ++i) { h ^= (uint8_t)prefix.data[i]; h *= 16777619u; }
	static const ImU32 pal[] = {
		IM_COL32(120, 180, 230, 230), IM_COL32(230, 170, 90, 230), IM_COL32(150, 210, 140, 230),
		IM_COL32(210, 140, 200, 230), IM_COL32(220, 205, 110, 230), IM_COL32(140, 205, 210, 230),
	};
	return pal[h % (sizeof pal / sizeof pal[0])];
}

// stable, ID-stack-independent key for a group's collapse state. ImGui::GetID is seeded by the current
// window, so the pre-layout read (outer window) and the in-canvas click write (rg_canvas child) would hash
// the same string to DIFFERENT ids -- a hash of the prefix (salted) is the same wherever it's computed.
static ImGuiID rg_grp_key(WGPUStringView prefix)
{
	ImU32 h = 2166136261u;
	for (const char* s = "rg.grp."; *s; ++s) { h ^= (uint8_t)*s; h *= 16777619u; }
	for (size_t i = 0; i < sv_length(prefix); ++i) { h ^= (uint8_t)prefix.data[i]; h *= 16777619u; }
	return (ImGuiID)h;
}

// DAG view -----------------------------------------------------------------------------------------
// node-graph layout: one box per pass in dependency columns, one pin per resource access (reads left,
// writes right), edges run producer-output -> consumer-input (true RAW data flow). Hovering a pin lights
// the upstream producer cone -- every pass that must run to make that resource. Reads the .cpp-internal
// node structs directly; assumes a compiled, realized graph like the other dumps.

static constexpr int kRgDagMax = 128;
static constexpr int kRgGPinMax = 32;   // max interface pins drawn on a collapsed group node (silently capped)

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

// ---- DAG intermediate representation (see docs/rendergraph-nested-layout.md). built once per frame from
// the passes + frame-boundary endpoints, then drawn. a node is a pass or a virtual endpoint (temporal/
// imported/present), linked by RgEdges. the pass-graph (box[]) still drives layout; the draw consumes this
// IR for the nodes + virtual links. groups live in the GView side-table, not here.
struct RgNode {
	enum class Kind : uint8_t { Pass, Group, Virtual };
	Kind kind{};
	PassNode*      pass{};      // Kind::Pass
	ResourceNode*  res{};       // Kind::Virtual (the endpoint's resource)
	WGPUStringView label{};     // Kind::Virtual caption
	ImVec2 pos{}; float w = 0, h = 0; int col = 0;              // layout result -> consumed by draw
	ImU32 tint = 0;             // Kind::Virtual base colour (read/write/imported/present)
};
struct RgEdge {
	int srcNode = -1, dstNode = -1; uint16_t srcPin = 0, dstPin = 0; uint32_t resId = 0;
	enum class Kind : uint8_t { Raw, Temporal, Fanout } kind = Kind::Raw;
};

// barycenter crossing-reduction + y-relax over a layered node set (Sugiyama). lcol[c] = node indices in
// column c (reordered in place); lleft/lright[i] = a node's neighbours; h[i] = node height. writes cy[i].
// shared by the top-level layout and each expanded group's interior so internal edges route like the top.
static void rg_barycenter_relax(std::vector<int>* lcol, int maxCol,
	const std::vector<std::vector<int>>& lleft, const std::vector<std::vector<int>>& lright,
	const std::vector<float>& h, std::vector<float>& cy, float rowGap)
{
	int LN = (int)cy.size();
	static std::vector<int> lrank; lrank.assign(LN, 0);
	auto reix = [&](int c) { for (int r = 0; r < (int)lcol[c].size(); ++r) lrank[lcol[c][r]] = r; };
	for (int c = 0; c <= maxCol; ++c) reix(c);
	auto bnb = [&](int li, bool right) { const std::vector<int>& nb = right ? lright[li] : lleft[li]; float s = 0; int c = 0; for (int x : nb) { s += lrank[x]; ++c; } return c ? s / c : -1.0f; };
	auto lsweep = [&](bool backward) {
		int from = backward ? maxCol - 1 : 1, to = backward ? -1 : maxCol + 1, step = backward ? -1 : 1;
		for (int c = from; c != to; c += step) {
			std::vector<int>& m = lcol[c];
			std::vector<float> key(m.size());
			for (int j = 0; j < (int)m.size(); ++j) { float b = bnb(m[j], backward); key[j] = b < 0 ? (float)lrank[m[j]] : b; }
			for (int a = 1; a < (int)m.size(); ++a) { int mv = m[a]; float kv = key[a]; int b = a - 1; while (b >= 0 && key[b] > kv) { m[b + 1] = m[b]; key[b + 1] = key[b]; --b; } m[b + 1] = mv; key[b + 1] = kv; }
			reix(c);
		}
		};
	lsweep(true); lsweep(false); lsweep(true);
	for (int c = 0; c <= maxCol; ++c) { float y = 0; for (int li : lcol[c]) { cy[li] = y + h[li] * 0.5f; y += h[li] + rowGap; } }
	for (int it = 0; it < 8; ++it)
		for (int c = 0; c <= maxCol; ++c) {
			std::vector<int>& m = lcol[c];
			for (int r = 0; r < (int)m.size(); ++r) {
				int li = m[r]; float s = 0; int cnt = 0;
				for (int x : lleft[li])  { s += cy[x]; ++cnt; }
				for (int x : lright[li]) { s += cy[x]; ++cnt; }
				if (!cnt) continue;
				float d = s / cnt;
				float lo = r > 0                ? cy[m[r - 1]] + (h[m[r - 1]] + h[li]) * 0.5f + rowGap : -1e30f;
				float hi = r + 1 < (int)m.size() ? cy[m[r + 1]] - (h[li] + h[m[r + 1]]) * 0.5f - rowGap :  1e30f;
				cy[li] = d < lo ? lo : d > hi ? hi : d;
			}
		}
}

static void rg_draw_dag(RenderGraph* rg, RenderGraphStorage& s)
{
	float kBoxW = 190.0f, kColGap = 65.0f, kRowGap = 50.0f;
	float kHeaderH = 22.0f, kFooterH = 14.0f, kPinRowH = 18.0f, kMinBodyH = 12.0f;
	float kPinR = 5.0f, kPinHit = 8.0f;
	float kRegionPad = 14.0f, kInnerGap = 10.0f;   // region: inner padding + vertical gap between stacked members
	float kInnerColGap = 46.0f;                    // horizontal gap between inner dependency columns (chain spacing)


	// virtual nodes (frame-boundary endpoints: temporal read/write so far) -- toggled from the toolbar.
	// read before the layout pass so disabling them also drops their layout influence (no reserved columns).
	static bool showVirtual = true;
	// imported buffers (uniforms read by many passes): on = one source node fanning faint edges to every
	// reader; off = a node at each use site, like an imported texture. read here so it shapes the layout too.
	static bool fanBuffers = true;
	// collapse groups by default (a group-header click overrides per group). read here, before the layout,
	// so a collapsed group can reserve one compact slot the same frame the checkbox flips.
	static bool collapseDefault = false;
	// one storage object for all group collapse state, captured here in the OUTER window so the pre-layout
	// reads and the in-canvas-child click writes hit the same store (see rg_grp_key for the matching key).
	ImGuiStorage* grpStore = ImGui::GetStateStorage();

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

	// ---- collapsed-group slot reservation (pre-layout): a collapsed group must occupy ONE compact slot,
	// like a virtual node -- otherwise its members overflow their boxes or span columns full of unrelated
	// passes, and the group node collides. so BEFORE the slot packer runs, give each collapsed group's
	// anchor (first member) the compact node height and shrink the other members to zero.
	// ponytail: the interface walk mirrors the post-layout one below; duplicated so the height is known
	// pre-layout. fold into a shared lambda if it drifts.
	float effH[kRgDagMax];
	for (int i = 0; i < n; ++i) effH[i] = box[i].h;
	// expanded-group region width + each member's inner column (offset within the region). a non-grouped
	// pass is kBoxW wide / inner column 0; an expanded group's anchor carries the whole region's width.
	float effW[kRgDagMax]; int innerCol[kRgDagMax];
	for (int i = 0; i < n; ++i) { effW[i] = kBoxW; innerCol[i] = 0; }
	// expanded-group member's y offset within its region body, from the recursive interior layout (phase 2).
	float memRelY[kRgDagMax]; for (int i = 0; i < n; ++i) memRelY[i] = 0.0f;
	// subgroup collapse: a collapsed subgroup (same-name run) folds to its first member (the rep, shown as a
	// compact box with the count); the others hide and its inner columns compact. sgHidden = non-rep member
	// (skip everywhere); sgCount = member count on the rep (>0 => draw compact).
	bool sgHidden[kRgDagMax]; int sgCount[kRgDagMax];
	for (int i = 0; i < n; ++i) { sgHidden[i] = false; sgCount[i] = 0; }
	{
		for (int a = 0; a < n;) {
			WGPUStringView pre = group_prefix(box[a].p->name);
			int b = a + 1;
			while (b < n && sv_length(pre) && sv_eq(group_prefix(box[b].p->name), pre)) ++b;
			if (sv_length(pre) && b - a >= 2) {
					int st = grpStore->GetInt(rg_grp_key(pre), 0);
				if (st == 2 || (st != 1 && collapseDefault)) {
					uint32_t inId[kRgGPinMax], outId[kRgGPinMax]; int ni = 0, no = 0;
					for (int k = a; k < b; ++k) { PassNode* p = box[k].p;
						for (uint32_t ci = 0; ci < p->accessCount; ++ci) {
							if (!rg_access_reads(p->accesses[ci])) continue;
							uint32_t id = p->accesses[ci].handle.id;
							int pr = rg_producer_of(box, n, p, id);
							if (pr >= a && pr < b) continue;
							bool sn = false; for (int x = 0; x < ni; ++x) sn |= inId[x] == id;
							if (!sn && ni < kRgGPinMax) inId[ni++] = id;
						}
					}
					for (int k = a; k < b; ++k) { PassNode* p = box[k].p;
						for (uint32_t ci = 0; ci < p->accessCount; ++ci) {
							if (!access_is_write(p->accesses[ci].type)) continue;
							uint32_t id = p->accesses[ci].handle.id;
							ResourceNode* r = find_node(rg, { id });
							bool ext = r && (r->imported || r->persistent);   // temporal write leaves to next frame -> group output
							for (int j = 0; j < n && !ext; ++j) {
								if (j >= a && j < b) continue;
								if (rg_in_slot(box[j].p, id) < 0) continue;
								int pr = rg_producer_of(box, n, box[j].p, id);
								if (pr >= a && pr < b) ext = true;
							}
							if (!ext) continue;
							bool sn = false; for (int x = 0; x < no; ++x) sn |= outId[x] == id;
							if (!sn && no < kRgGPinMax) outId[no++] = id;
						}
					}
					int rows = ni > no ? ni : no;
					effH[a] = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
					for (int k = a + 1; k < b; ++k) effH[k] = 0.0f;
					// pull the whole collapsed group onto the anchor's column; the columns it vacates
					// compact away below, so a deep group (bloom's mip chain) stops widening the graph.
					int colA = colOf[a]; for (int k = a + 1; k < b; ++k) if (colOf[k] < colA) colA = colOf[k];
						for (int k = a; k < b; ++k) colOf[k] = colA;   // snap to LEFTMOST member (not first -- it may be a late-column sink, e.g. pt.accum, colliding the group with a downstream consumer)
				}
			}
			a = b;
		}
	}

	// expanded groups: reserve ONE exclusive column slot tall enough to stack the members in a bordered
	// region (header + members + padding) and snap members onto the anchor's column; members are positioned
	// inside the region post-layout (see the group draw). the freed columns compact away below, same as the
	// collapsed case. P2-lite: vertical stack; horizontal inner layout + subgroups come next.
	{
		for (int a = 0; a < n;) {
			WGPUStringView pre = group_prefix(box[a].p->name);
			int b = a + 1;
			while (b < n && sv_length(pre) && sv_eq(group_prefix(box[b].p->name), pre)) ++b;
			if (sv_length(pre) && b - a >= 2) {
					int st = grpStore->GetInt(rg_grp_key(pre), 0);
				if (!((st == 2) || (st != 1 && collapseDefault))) {   // expanded -> horizontal inner layout
					// members' pre-snap columns already encode internal dependency order, so inner column =
					// colOf - colMin. the region spans those inner columns (width) and the tallest inner column
					// (height); members sharing an inner column stack vertically inside.
					int colMin = colOf[a]; for (int k = a; k < b; ++k) if (colOf[k] < colMin) colMin = colOf[k];
					int rawCols = 1;
					for (int k = a; k < b; ++k) { innerCol[k] = colOf[k] - colMin; if (innerCol[k] + 1 > rawCols) rawCols = innerCol[k] + 1; }
					// fold collapsed subgroups (same-name runs): pull members onto the run's first inner column,
					// hide all but the first (the rep), and mark the vacated columns so the remap compacts them.
					bool absorbed[kRgDagMax]; for (int c = 0; c < rawCols; ++c) absorbed[c] = false;
					for (int sa = a; sa < b;) {
						int sb = sa + 1; while (sb < b && sv_eq(box[sb].p->name, box[sa].p->name)) ++sb;
						if (sb - sa >= 2 && !(sa == a && sb == b) && grpStore->GetInt(rg_grp_key(box[sa].p->name), 0) == 2) {
							int rmin = innerCol[sa], rmax = innerCol[sa];
							for (int k = sa; k < sb; ++k) { if (innerCol[k] < rmin) rmin = innerCol[k]; if (innerCol[k] > rmax) rmax = innerCol[k]; }
							for (int c = rmin + 1; c <= rmax; ++c) absorbed[c] = true;
							for (int k = sa; k < sb; ++k) innerCol[k] = rmin;
							sgCount[sa] = sb - sa; for (int k = sa + 1; k < sb; ++k) sgHidden[k] = true;
						}
						sa = sb;
					}
					int remapIC[kRgDagMax]; int innerCols = 0;
					for (int c = 0; c < rawCols; ++c) remapIC[c] = absorbed[c] ? innerCols - 1 : innerCols++;
					for (int k = a; k < b; ++k) innerCol[k] = remapIC[innerCol[k]];
					// lay out the interior with the shared layered relax (innerCol = layer) so members align to
					// their internal edges instead of a naive per-column stack; the region height then fits it.
					// (routing internal edges through dummy lanes is phase 2b -- here members just get good y's.)
					static std::vector<int> ilcol[kRgDagMax]; for (int c = 0; c < innerCols; ++c) ilcol[c].clear();
					static int inode[kRgDagMax]; static std::vector<int> imem; imem.clear(); static std::vector<float> ih; ih.clear();
					for (int k = a; k < b; ++k) { inode[k] = -1; if (sgHidden[k]) continue; inode[k] = (int)imem.size(); ilcol[innerCol[k]].push_back((int)imem.size()); imem.push_back(k); ih.push_back(box[k].h); }
					int iN = (int)imem.size();
					static std::vector<std::vector<int>> illeft, ilright; illeft.assign(iN, {}); ilright.assign(iN, {});
					for (int k = a; k < b; ++k) {
						if (sgHidden[k]) continue;
						PassNode* pk = box[k].p;
						for (uint32_t ci = 0; ci < pk->accessCount; ++ci) {
							if (!rg_access_reads(pk->accesses[ci])) continue;
							int prod = rg_producer_of(box, n, pk, pk->accesses[ci].handle.id);
							if (prod < a || prod >= b || sgHidden[prod] || innerCol[prod] >= innerCol[k]) continue;
							ilright[inode[prod]].push_back(inode[k]); illeft[inode[k]].push_back(inode[prod]);
						}
					}
					static std::vector<float> icy; icy.assign(iN, 0.0f);
					rg_barycenter_relax(ilcol, innerCols - 1, illeft, ilright, ih, icy, kInnerGap);
					float ymin = 1e30f, ymax = -1e30f;
					for (int ni = 0; ni < iN; ++ni) { float t = icy[ni] - ih[ni] * 0.5f, bt = icy[ni] + ih[ni] * 0.5f; if (t < ymin) ymin = t; if (bt > ymax) ymax = bt; }
					if (iN == 0) { ymin = ymax = 0; }
					for (int ni = 0; ni < iN; ++ni) memRelY[imem[ni]] = icy[ni] - ih[ni] * 0.5f - ymin;   // icy is a centre; pos is top-left
					effH[a] = kHeaderH + 2.0f * kRegionPad + (ymax - ymin);
					effW[a] = 2.0f * kRegionPad + innerCols * kBoxW + (innerCols - 1) * kInnerColGap;
					colOf[a] = colMin; for (int k = a + 1; k < b; ++k) { effH[k] = 0.0f; effW[k] = 0.0f; colOf[k] = colMin; }   // leftmost member, not first (see collapsed branch)
				}
			}
			a = b;
		}
	}

	// recompact columns after collapsed groups vacated theirs (mirror of the compaction above), so a deep
	// collapsed group leaves no empty columns / horizontal gap behind.
	for (int c = 0; c <= maxDist; ++c) remap[c] = -1;
	for (int i = 0; i < n; ++i) remap[colOf[i]] = 1;
	nextCol = 0; for (int c = 0; c <= maxDist; ++c) if (remap[c] == 1) remap[c] = nextCol++;
	for (int i = 0; i < n; ++i) colOf[i] = remap[colOf[i]];
	maxDist = nextCol ? nextCol - 1 : 0;

	// ---- virtual nodes: frame-boundary endpoints drawn as real DAG nodes so the column/barycenter/y code
	// below places them like any pass (no bespoke overlay). each attaches to ONE pass pin -- a read node one
	// column BEFORE its reader, a write node one column AFTER its writer -- so a widely-read texture gets a
	// small node at each use site (like repeated power symbols) instead of one node fanning edges everywhere.
	// three kinds, all gated by the toolbar toggle:
	//   * temporal: create_temporal_image/_buffer make two resource nodes sharing a name -- curr (temporalIndex 0,
	//     written THIS frame for next) and prev (temporalIndex 1, read this frame = LAST frame's curr; the
	//     pool rotates two physical textures/buffers). cross-frame, so no in-frame edge joins the pair.
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

	// barycenter crossing-reduction + y-relax, factored out so each expanded group's interior reuses it.
	// materialize node heights first: a box -> its reserved height, a temporal node -> its own, dummy -> a lane.
	const float kLane = 16.0f;
	static std::vector<float> lh; lh.assign(LN, 0.0f);
	for (int li = 0; li < LN; ++li) { int b = lnode[li].box; lh[li] = b >= 0 ? effH[b] : (lnode[li].tn >= 0 ? tnodes[lnode[li].tn].h : kLane); }
	static std::vector<float> cy; cy.assign(LN, 0.0f);
	rg_barycenter_relax(lcol, maxDist, lleft, lright, lh, cy, kRowGap);

	// write positions + the graph's bounding box (canvas-local), used to centre the view.
	float gxMin = 1e30f, gyMin = 1e30f, gxMax = -1e30f, gyMax = -1e30f;
	// column x is cumulative so a wide expanded-group column (effW) pushes later columns right; with no wide
	// group every colW == kBoxW and this is identical to the old uniform c*(kBoxW+kColGap) spacing.
	float colW[kRgDagMax], colX[kRgDagMax];
	for (int c = 0; c <= maxDist; ++c) colW[c] = kBoxW;
	for (int i = 0; i < n; ++i) if (effW[i] > colW[colOf[i]]) colW[colOf[i]] = effW[i];
	colX[0] = 0; for (int c = 1; c <= maxDist; ++c) colX[c] = colX[c - 1] + colW[c - 1] + kColGap;
	for (int c = 0; c <= maxDist; ++c) {
		float cx = colX[c];
		for (int li : lcol[c]) {
			if (lnode[li].box >= 0) {
				int b = lnode[li].box; box[b].tl = ImVec2(cx, cy[li] - effH[b] * 0.5f); box[b].layer = c;
				if (cx < gxMin) gxMin = cx; if (cx + effW[b] > gxMax) gxMax = cx + effW[b];
				if (box[b].tl.y < gyMin) gyMin = box[b].tl.y; if (box[b].tl.y + effH[b] > gyMax) gyMax = box[b].tl.y + effH[b];
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

	// ---- IR mirror: build RgNodes (pass nodes index-aligned to box[], then Virtual endpoints) + RgEdges.
	// the draw consumes this for the nodes + virtual links; the pass-graph (box[]/edge[]) still drives layout
	// and the solid pass-edge routing. layout() producing the IR directly (retiring box[]) is future work.
	static std::vector<RgNode> rgn; rgn.clear();
	static std::vector<RgEdge> rge; rge.clear();
	for (int i = 0; i < n; ++i) {
		RgNode nd{}; nd.kind = RgNode::Kind::Pass; nd.pass = box[i].p;
		nd.label = box[i].p->name; nd.pos = box[i].tl; nd.w = box[i].w; nd.h = box[i].h; nd.col = box[i].layer;
		rgn.push_back(nd);
	}
	for (REdge& e : edge) {
		RgEdge ge{}; ge.srcNode = e.src; ge.dstNode = e.dst;
		ge.srcPin = (uint16_t)e.sOut; ge.dstPin = (uint16_t)e.dIn; ge.resId = e.id; ge.kind = RgEdge::Kind::Raw;
		rge.push_back(ge);
	}
	// virtual endpoints mirror into the IR: one Virtual RgNode at its laid-out spot + an RgEdge per link to
	// the anchor pass pin (a fan source emits one edge per reader). lets the draw + cones treat them like any
	// node, retiring the parallel tnodes draw path (Phase C). additive here -- the draw still reads tnodes.
	for (TNode& t : tnodes) {
		int vi = (int)rgn.size();
		RgNode nd{}; nd.kind = RgNode::Kind::Virtual; nd.res = t.res;
		nd.label = WGPUStringView{ t.cap, t.cap ? strlen(t.cap) : 0 };
		nd.pos = ImVec2(lnode[t.li].x - t.w * 0.5f, lnode[t.li].y - t.h * 0.5f);
		nd.w = t.w; nd.h = t.h; nd.col = t.col; nd.tint = t.tint;
		rgn.push_back(nd);
		RgEdge::Kind ek = t.res->persistent ? RgEdge::Kind::Temporal : RgEdge::Kind::Raw;
		bool fan = fanBuffers && t.isRead && t.res->imported && t.res->kind == ResourceNode::Kind::Buffer;
		if (fan)
			for (int i = 0; i < n; ++i) { int sl = rg_in_slot(box[i].p, t.res->handle.id); if (sl < 0) continue;
				rge.push_back(RgEdge{ vi, i, 0, (uint16_t)sl, t.res->handle.id, RgEdge::Kind::Fanout }); }
		else if (t.isRead) rge.push_back(RgEdge{ vi, t.passBox, 0, (uint16_t)t.pin, t.res->handle.id, ek });
		else               rge.push_back(RgEdge{ t.passBox, vi, (uint16_t)t.pin, 0, t.res->handle.id, ek });
	}

	// toolbar: reset/centre the view + a name filter that dims non-matching passes.
	static char filter[64] = "";
	bool doReset = ImGui::Button("Reset view");
	ImGui::SameLine(); ImGui::SetNextItemWidth(180);
	ImGui::InputTextWithHint("##rgfilter", "filter passes...", filter, sizeof filter);
	ImGui::SameLine(); ImGui::Checkbox("virtual nodes", &showVirtual);
	ImGui::SameLine(); ImGui::Checkbox("fan-out buffers", &fanBuffers);   // off -> one virtual node per buffer use site
	ImGui::SameLine(); ImGui::Checkbox("collapse groups", &collapseDefault);   // default state; a group header click overrides
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

	// ---- group runs: a contiguous run of passes sharing a dotted-name prefix (shadow.cascade x3,
	// bloom.extract/down/up/composite) is one logical group. expanded -> a labelled frame behind the
	// member boxes (keyed on the prefix, so bloom.* reads as one); collapsed -> a single synthetic node
	// (drawn later, on top of the edges) carrying the group's external read/write pins. the collapse flag
	// is a tri-state in ImGuiStorage (0 follow-default, 1 force-open, 2 force-closed) so it survives the
	// per-frame rebuild and the toolbar default can flip every group at once.
	struct GView {
		int gi, gj; WGPUStringView prefix; bool collapsed; ImGuiID key;
		ImVec2 bb0, bb1, h0, h1;                                    // box rect + header (click) rect, screen
		int nIn, nOut; uint32_t inId[kRgGPinMax], outId[kRgGPinMax];
		ImVec2 inC[kRgGPinMax], outC[kRgGPinMax]; bool inDrawn[kRgGPinMax];
	};
	static std::vector<GView> groups; groups.clear();
	struct SgClick { ImVec2 h0, h1; ImGuiID key; };   // subgroup label click target (toggles its collapse)
	static std::vector<SgClick> sgClicks; sgClicks.clear();
	int collapsedBox[kRgDagMax], mapTo[kRgDagMax], groupOf[kRgDagMax];
	for (int i = 0; i < n; ++i) { collapsedBox[i] = 0; mapTo[i] = i; groupOf[i] = -1; }
	for (int gi = 0; gi < n;) {
		WGPUStringView pre = group_prefix(box[gi].p->name);
		int gj = gi + 1;
		while (gj < n && sv_length(pre) && sv_eq(group_prefix(box[gj].p->name), pre)) ++gj;
		if (!(sv_length(pre) && gj - gi >= 2)) { gi = gj; continue; }

		float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
		for (int k = gi; k < gj; ++k) {
			float ax = origin.x + box[k].tl.x, ay = origin.y + box[k].tl.y;
			if (ax < x0) x0 = ax; if (ay < y0) y0 = ay;
			if (ax + box[k].w > x1) x1 = ax + box[k].w; if (ay + box[k].h > y1) y1 = ay + box[k].h;
		}

		GView g{}; g.gi = gi; g.gj = gj; g.prefix = pre; g.key = rg_grp_key(pre);
		int st = grpStore->GetInt(g.key, 0);
		g.collapsed = st == 2 ? true : st == 1 ? false : collapseDefault;

		// external interface pins, computed for collapsed AND expanded groups: a member read whose producer
		// is outside the group -> in-pin; an external/imported/persistent member write -> out-pin. interior
		// resources (produced + consumed within the group, e.g. bloom's mips) get no pin.
		for (int k = gi; k < gj; ++k) {
			PassNode* p = box[k].p;
			for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
				if (!rg_access_reads(p->accesses[ai])) continue;
				uint32_t id = p->accesses[ai].handle.id;
				int prod = rg_producer_of(box, n, p, id);
				if (prod >= gi && prod < gj) continue;
				bool seen = false; for (int s = 0; s < g.nIn; ++s) seen |= g.inId[s] == id;
				if (!seen && g.nIn < kRgGPinMax) g.inId[g.nIn++] = id;
			}
		}
		for (int k = gi; k < gj; ++k) {
			PassNode* p = box[k].p;
			for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
				if (!access_is_write(p->accesses[ai].type)) continue;
				uint32_t id = p->accesses[ai].handle.id;
				ResourceNode* r = find_node(rg, { id });
				bool external = r && (r->imported || r->persistent);   // temporal write leaves to next frame -> group output
				for (int j = 0; j < n && !external; ++j) {
					if (j >= gi && j < gj) continue;
					if (rg_in_slot(box[j].p, id) < 0) continue;
					int pr = rg_producer_of(box, n, box[j].p, id);
					if (pr >= gi && pr < gj) external = true;
				}
				if (!external) continue;
				bool seen = false; for (int s = 0; s < g.nOut; ++s) seen |= g.outId[s] == id;
				if (!seen && g.nOut < kRgGPinMax) g.outId[g.nOut++] = id;
			}
		}

		if (!g.collapsed) {
			// exclusive bordered region: members stacked vertically inside it. the slot was reserved at region
			// height on the group's own column pre-layout, so nothing outside the group overlaps the border.
			ImU32 gcol = group_color(pre);
			float rx = rgn[gi].pos.x, ry = rgn[gi].pos.y, rh = effH[gi], rw = effW[gi];   // canvas-local region rect
			// members at their inner-column x; relaxed y from the recursive interior layout (memRelY).
			for (int k = gi; k < gj; ++k)
				rgn[k].pos = ImVec2(rx + kRegionPad + innerCol[k] * (kBoxW + kInnerColGap), ry + kHeaderH + kRegionPad + memRelY[k]);
			ImVec2 a(origin.x + rx, origin.y + ry), b(a.x + rw, a.y + rh);
			dl->AddRectFilled(a, b, rg_with_alpha(gcol, 22), 6.0f);
			dl->AddRect(a, b, gcol, 6.0f, 0, 2.0f);
			char lbl[64]; std::snprintf(lbl, sizeof lbl, "[-] %.*s  x%d", (int)sv_length(pre), pre.data, gj - gi);
			dl->AddText(ImVec2(a.x + 6, a.y + 3), gcol, lbl);
			// subgroups: a same-name run (>=2) of members inside the region gets its own nested border
			// (bloom.down, bloom.up). being chains, they occupy disjoint contiguous inner columns, so the
			// nested borders never overlap; the suffix colour (group_color of the full name) distinguishes them.
			for (int sa = gi; sa < gj;) {
				int sb = sa + 1;
				while (sb < gj && sv_eq(rgn[sb].pass->name, rgn[sa].pass->name)) ++sb;
				if (sb - sa >= 2 && !(sa == gi && sb == gj)) {   // skip a run that IS the whole group (no sub-structure)
					float sx0 = 1e30f, sy0 = 1e30f, sx1 = -1e30f, sy1 = -1e30f;
					for (int k = sa; k < (sgCount[sa] > 0 ? sa + 1 : sb); ++k) {   // collapsed -> frame only the rep
						float kx = origin.x + rgn[k].pos.x, ky = origin.y + rgn[k].pos.y;
						if (kx < sx0) sx0 = kx; if (ky < sy0) sy0 = ky;
						if (kx + rgn[k].w > sx1) sx1 = kx + rgn[k].w; if (ky + rgn[k].h > sy1) sy1 = ky + rgn[k].h;
					}
					dl->AddRect(ImVec2(sx0 - 6, sy0 - 6), ImVec2(sx1 + 6, sy1 + 6), group_color(rgn[sa].pass->name), 5.0f, 0, 1.5f);
					{   // clickable [-]/[+] badge above the subgroup that toggles its collapse
						WGPUStringView snm = rgn[sa].pass->name; size_t off = sv_length(pre) + 1;
						char slbl[40]; std::snprintf(slbl, sizeof slbl, "%s %.*s x%d", sgCount[sa] > 0 ? "[+]" : "[-]",
							off < sv_length(snm) ? (int)(sv_length(snm) - off) : (int)sv_length(snm),
							off < sv_length(snm) ? snm.data + off : snm.data, sb - sa);
						ImVec2 sts = ImGui::CalcTextSize(slbl), slp(sx0 - 6, sy0 - 6 - sts.y - 1);
						ImVec2 c0(slp.x - 2, slp.y - 1), c1(slp.x + sts.x + 2, slp.y + sts.y + 1);
						dl->AddRectFilled(c0, c1, IM_COL32(18, 20, 28, 230), 3.0f);
						dl->AddText(slp, group_color(snm), slbl);
						sgClicks.push_back({ c0, c1, rg_grp_key(snm) });
					}
				}
				sa = sb;
			}
			g.bb0 = a; g.bb1 = b; g.h0 = a; g.h1 = ImVec2(b.x, a.y + kHeaderH);
			// border interface pins: distribute evenly down the left (in) / right (out) region edges so the
			// dots always fit regardless of region height. ponytail: even spread, not aligned to the member.
			for (int s = 0; s < g.nIn; ++s)  g.inC[s]  = ImVec2(a.x, a.y + kHeaderH + (rh - kHeaderH) * (s + 0.5f) / (g.nIn  ? g.nIn  : 1));
			for (int s = 0; s < g.nOut; ++s) g.outC[s] = ImVec2(b.x, a.y + kHeaderH + (rh - kHeaderH) * (s + 0.5f) / (g.nOut ? g.nOut : 1));
		}
		else {
			int rows = g.nIn > g.nOut ? g.nIn : g.nOut;
			float needH = kHeaderH + (rows ? rows * kPinRowH : kMinBodyH) + kFooterH;
			// collapse to a single node over the FIRST member's slot (already an overlap-free reserved
			// position), not the member bbox -- a multi-column group's bbox would span and cover the nodes
			// laid out between its members. the other members' slots simply vacate. ponytail: no relayout,
			// so a group with more pins than the anchor pass's rows grows down and can still touch a
			// neighbour; promote collapsed groups to single layout nodes if that ever bites.
			ImVec2 a(origin.x + rgn[gi].pos.x, origin.y + rgn[gi].pos.y);
			ImVec2 b(a.x + kBoxW, a.y + needH);   // == effH[gi], the slot reserved before layout
			g.bb0 = a; g.bb1 = b; g.h0 = a; g.h1 = ImVec2(b.x, a.y + kHeaderH);
			for (int s = 0; s < g.nIn; ++s)  g.inC[s]  = ImVec2(a.x, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f);
			for (int s = 0; s < g.nOut; ++s) g.outC[s] = ImVec2(b.x, a.y + kHeaderH + s * kPinRowH + kPinRowH * 0.5f);
		}

		int idxG = (int)groups.size(); groups.push_back(g);
		for (int k = gi; k < gj; ++k) { groupOf[k] = idxG; if (g.collapsed) { collapsedBox[k] = 1; mapTo[k] = gi; } }
		gi = gj;
	}

	// pin centres, screen space. reads fill input slots (left) in encounter order; writes fill output
	// slots (right) the same way.
	auto inPin = [&](int b, int slot) { return ImVec2(origin.x + rgn[b].pos.x,
		origin.y + rgn[b].pos.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
	auto outPin = [&](int b, int slot) { return ImVec2(origin.x + rgn[b].pos.x + rgn[b].w,
		origin.y + rgn[b].pos.y + kHeaderH + slot * kPinRowH + kPinRowH * 0.5f); };
	// screen-space polyline of an edge (src out-pin, two points per dummy, dst in-pin); shared by hit-test + draw.
	auto edgePoints = [&](const REdge& e, ImVec2* pts) -> int {
		int np = 0;
		pts[np++] = outPin(e.src, e.sOut);
		for (int t = 0; t < e.chainN; ++t) { float lx = origin.x + lnode[e.chain[t]].x, ly = origin.y + lnode[e.chain[t]].y; pts[np++] = ImVec2(lx, ly); pts[np++] = ImVec2(lx + kBoxW, ly); }
		pts[np++] = inPin(e.dst, e.dIn);
		return np;
	};
	bool matchBox[kRgDagMax];
	for (int i = 0; i < n; ++i) matchBox[i] = rg_name_has(rgn[i].pass->name, filter);

	// ---- find the single hovered pin (manual rect test: pins are small + overlap the box button).
	int hovB = -1, hovSlot = -1; bool hovWrite = false; uint32_t hovId = 0, hovMip = 0, hovLayer = 0; AccessType hovType{};
	if (canvasHovered && !canvasActive) {
		for (int i = 0; i < n && hovB < 0; ++i) {
			if (collapsedBox[i] || sgHidden[i]) continue;   // hidden inside a collapsed group
			int inS = 0, outS = 0; PassNode* p = rgn[i].pass;
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
	// ---- group border pins (collapsed compact node AND expanded region edge): hover/select them like a
	// member's pin so cone + tooltip + lock work. map the border pin to a representative member that reads/
	// writes that resource, then reuse the normal hovered-pin path below. (expanded member pins are caught
	// directly by the per-pass test above; this adds the border pins, which sit on the region edge.)
	if (hovB < 0 && canvasHovered && !canvasActive)
		for (GView& g : groups) {
			if (hovB >= 0) continue;
			for (int s = 0; s < g.nIn && hovB < 0; ++s) {
				ImVec2 c = g.inC[s];
				if (!ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit))) continue;
				for (int k = g.gi; k < g.gj && hovB < 0; ++k) { PassNode* mp = rgn[k].pass;
					for (uint32_t ai = 0; ai < mp->accessCount; ++ai)
						if (mp->accesses[ai].handle.id == g.inId[s] && rg_access_reads(mp->accesses[ai]))
							{ hovB = k; hovWrite = false; hovId = g.inId[s]; hovSlot = rg_in_slot(mp, g.inId[s]); hovType = mp->accesses[ai].type; hovMip = mp->accesses[ai].baseMip; hovLayer = mp->accesses[ai].baseLayer; break; }
				}
			}
			for (int s = 0; s < g.nOut && hovB < 0; ++s) {
				ImVec2 c = g.outC[s];
				if (!ImGui::IsMouseHoveringRect(ImVec2(c.x - kPinHit, c.y - kPinHit), ImVec2(c.x + kPinHit, c.y + kPinHit))) continue;
				for (int k = g.gi; k < g.gj && hovB < 0; ++k) { PassNode* mp = rgn[k].pass;
					for (uint32_t ai = 0; ai < mp->accessCount; ++ai)
						if (mp->accesses[ai].handle.id == g.outId[s] && access_is_write(mp->accesses[ai].type))
							{ hovB = k; hovWrite = true; hovId = g.outId[s]; hovSlot = rg_out_slot(mp, g.outId[s]); hovType = mp->accesses[ai].type; hovMip = mp->accesses[ai].baseMip; hovLayer = mp->accesses[ai].baseLayer; break; }
				}
			}
		}

	// hovered box: only used for the fallback reads/writes tooltip when no pin caught the mouse.
	int hovBox = -1;
	if (hovB < 0 && canvasHovered && !canvasActive)
		for (int i = 0; i < n; ++i) {
			if (collapsedBox[i] || sgHidden[i]) continue;   // hidden inside a collapsed group
			ImVec2 tl(origin.x + rgn[i].pos.x, origin.y + rgn[i].pos.y);
			if (ImGui::IsMouseHoveringRect(tl, ImVec2(tl.x + rgn[i].w, tl.y + rgn[i].h))) { hovBox = i; break; }
		}

	// ---- click a pin to LOCK its cone (stays without holding the mouse); click empty canvas to clear.
	static int lockB = -1, lockSlot = -1; static bool lockWrite = false; static uint32_t lockId = 0;
	static bool pressed = false; static ImVec2 pressAt;
	if (canvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { pressed = true; pressAt = io.MousePos; }
	if (pressed && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
		pressed = false;
		float mdx = io.MousePos.x - pressAt.x, mdy = io.MousePos.y - pressAt.y;
		if (mdx * mdx + mdy * mdy < 16) {   // a click, not a pan
			int hitSg = -1;   // a subgroup badge click toggles just that subgroup's collapse
			for (int sI = 0; sI < (int)sgClicks.size(); ++sI)
				if (io.MousePos.x >= sgClicks[sI].h0.x && io.MousePos.x <= sgClicks[sI].h1.x &&
					io.MousePos.y >= sgClicks[sI].h0.y && io.MousePos.y <= sgClicks[sI].h1.y) hitSg = sI;
			int hitG = -1;   // a group header click toggles that group's collapse state, overriding the default
			for (int gI = 0; gI < (int)groups.size(); ++gI)
				if (io.MousePos.x >= groups[gI].h0.x && io.MousePos.x <= groups[gI].h1.x &&
					io.MousePos.y >= groups[gI].h0.y && io.MousePos.y <= groups[gI].h1.y) hitG = gI;
			if (hitSg >= 0) { int cur = grpStore->GetInt(sgClicks[hitSg].key, 0); grpStore->SetInt(sgClicks[hitSg].key, cur == 2 ? 0 : 2); }
			else if (hitG >= 0) grpStore->SetInt(groups[hitG].key, groups[hitG].collapsed ? 1 : 2);   // flip effective state
			else if (hovB >= 0) {
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
			if (collapsedBox[edge[ei].src] || collapsedBox[edge[ei].dst] || sgHidden[edge[ei].src] || sgHidden[edge[ei].dst]) continue;
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
	// slot of resource `id` among a collapsed group's interface pins (-1 = none), for edge rerouting.
	auto gpin_slot = [](const uint32_t* ids, int cnt, uint32_t id) { for (int i = 0; i < cnt; ++i) if (ids[i] == id) return i; return -1; };
	// collapsed-subgroup fold: rep of each member (rep maps to itself, -1 = not in a collapsed subgroup). a
	// boundary edge reroutes onto the rep box edge (right=producer, left=consumer); interior edges vanish.
	int sgRep[kRgDagMax]; for (int i = 0; i < n; ++i) sgRep[i] = -1;
	for (int i = 0; i < n; ++i) if (sgCount[i] > 0) for (int k = i; k < i + sgCount[i]; ++k) sgRep[k] = i;
	auto sgEdgePt = [&](int rep, bool out) { return ImVec2(origin.x + rgn[rep].pos.x + (out ? rgn[rep].w : 0.0f),
		origin.y + rgn[rep].pos.y + rgn[rep].h * 0.5f); };

	// ---- data edges, routed through their dummy waypoints so none hides behind a box. gold = upstream
	// cone, teal = downstream consumers, white = hovered, dim otherwise. an edge touching a collapsed
	// group reroutes onto that group's interface pin; an edge interior to one collapsed group vanishes.
	// a group in-pin can have several distinct producers (e.g. the 3 shadow cascades), so dedup the
	// reroute by (group,slot,src) -- not slot alone -- else only one producer's edge survives.
	static std::vector<int> inDrawnKeys; inDrawnKeys.clear();
	for (int ei = 0; ei < (int)edge.size(); ++ei) {
		REdge& e = edge[ei];
		if ((sgHidden[e.src] || sgHidden[e.dst]) && sgRep[e.src] == sgRep[e.dst]) continue;   // interior to one collapsed subgroup
		if ((collapsedBox[e.src] || collapsedBox[e.dst]) && mapTo[e.src] == mapTo[e.dst]) continue;   // interior
		bool eup = inUp(e.src) && inUp(e.dst), edn = inDn(e.src) && inDn(e.dst);
		ImU32 col; float th;
		if (ei == hovEdge)                     { col = IM_COL32(255, 255, 255, 255); th = 3.0f; }
		else if (eup)                          { col = IM_COL32(245, 222, 120, 235); th = 2.5f; }
		else if (edn)                          { col = IM_COL32(120, 222, 180, 235); th = 2.5f; }
		else if (isDim(e.src) || isDim(e.dst)) { col = IM_COL32(150, 150, 150, 34);  th = 2.0f; }
		else                                   { col = IM_COL32(170, 170, 170, 200); th = 2.0f; }
		// an edge crossing a group boundary reroutes onto that group's border pin -- the compact node when
		// collapsed, the region edge when expanded (same pins, computed for both). sExit/dExit = the endpoint
		// is grouped and the edge leaves its group; a hidden subgroup member with no border pin falls to its rep.
		bool sExit = groupOf[e.src] >= 0 && groupOf[e.src] != groupOf[e.dst];
		bool dExit = groupOf[e.dst] >= 0 && groupOf[e.src] != groupOf[e.dst];
		if (sExit || dExit || sgHidden[e.src] || sgHidden[e.dst]) {
			ImVec2 p, q;
			if (sExit) { GView& g = groups[groupOf[e.src]]; int sl = gpin_slot(g.outId, g.nOut, e.id); if (sl < 0) continue; p = g.outC[sl]; }
			else if (sgHidden[e.src]) p = sgEdgePt(sgRep[e.src], true);
			else p = outPin(e.src, e.sOut);
			if (dExit) { int gx = groupOf[e.dst]; GView& g = groups[gx]; int sl = gpin_slot(g.inId, g.nIn, e.id); if (sl < 0) continue;
				int key = (gx << 12) | (sl << 7) | e.src; bool dup = false; for (int kk : inDrawnKeys) if (kk == key) { dup = true; break; } if (dup) continue; inDrawnKeys.push_back(key); q = g.inC[sl]; }
			else if (sgHidden[e.dst]) q = sgEdgePt(sgRep[e.dst], false);
			else q = inPin(e.dst, e.dIn);
			float dx = (q.x - p.x) * 0.5f;
			dl->AddBezierCubic(p, ImVec2(p.x + dx, p.y), ImVec2(q.x - dx, q.y), q, col, th);
			continue;
		}
		ImVec2 pts[2 * kRgDagMax + 2]; int np = edgePoints(e, pts);
		for (int t = 0; t + 1 < np; ++t) {
			ImVec2 a2 = pts[t], b2 = pts[t + 1]; float dx = (b2.x - a2.x) * 0.5f;
			dl->AddBezierCubic(a2, ImVec2(a2.x + dx, a2.y), ImVec2(b2.x - dx, b2.y), b2, col, th);
		}
	}

	// ---- internal group edges: the routed pass above skips intra-group edges (an expanded group's members
	// share a column), so draw a group's member->member flow here, on the IR's stacked pin positions.
	for (GView& g : groups) {
		if (g.collapsed) continue;
		ImU32 ec = rg_with_alpha(group_color(g.prefix), 170);
		for (int k = g.gi; k < g.gj; ++k) {
			PassNode* p = rgn[k].pass;
			for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
				if (!rg_access_reads(p->accesses[ai])) continue;
				uint32_t id = p->accesses[ai].handle.id;
				int prod = rg_producer_of(box, n, p, id);
				if (prod < g.gi || prod >= g.gj) continue;   // intra-group only
				bool hk = sgHidden[k], hp = sgHidden[prod];
				if ((hk || hp) && sgRep[k] == sgRep[prod]) continue;   // interior to one collapsed subgroup
				ImVec2 q, pt;   // a collapsed subgroup's boundary edge reroutes onto the rep box edge
				if (hk) q = sgEdgePt(sgRep[k], false); else { int ds = rg_in_slot(rgn[k].pass, id); if (ds < 0) continue; q = inPin(k, ds); }
				if (hp) pt = sgEdgePt(sgRep[prod], true); else { int ss = rg_out_slot(rgn[prod].pass, id); if (ss < 0) continue; pt = outPin(prod, ss); }
				float dx = (q.x - pt.x) * 0.5f;
				dl->AddBezierCubic(pt, ImVec2(pt.x + dx, pt.y), ImVec2(q.x - dx, q.y), q, ec, 2.0f);
			}
		}
	}

	// expanded group border stubs: the short hop from a border pin into the region, to the member(s) that
	// read/write the external resource. the external segment (producer/consumer <-> border pin) is drawn by
	// the edge loop above. ponytail: direct hop, may cross a deep mid-group member (rare; entry/exit are short).
	for (GView& g : groups) {
		if (g.collapsed) continue;
		ImU32 sc = rg_with_alpha(group_color(g.prefix), 120);
		for (int k = g.gi; k < g.gj; ++k) {
			if (sgHidden[k]) continue;
			PassNode* p = rgn[k].pass;
			for (uint32_t ai = 0; ai < p->accessCount; ++ai) {
				uint32_t id = p->accesses[ai].handle.id;
				if (rg_access_reads(p->accesses[ai])) {
					int prod = rg_producer_of(box, n, p, id); if (prod >= g.gi && prod < g.gj) continue;   // internal producer: drawn above
					int sl = gpin_slot(g.inId, g.nIn, id); if (sl < 0) continue;
					int ds = rg_in_slot(rgn[k].pass, id); if (ds < 0) continue;
					ImVec2 pt = g.inC[sl], q = inPin(k, ds); float dx = (q.x - pt.x) * 0.5f;
					dl->AddBezierCubic(pt, ImVec2(pt.x + dx, pt.y), ImVec2(q.x - dx, q.y), q, sc, 1.5f);
				}
				else if (access_is_write(p->accesses[ai].type)) {
					int sl = gpin_slot(g.outId, g.nOut, id); if (sl < 0) continue;   // interior write: no border pin
					int ss = rg_out_slot(rgn[k].pass, id); if (ss < 0) continue;
					ImVec2 pt = outPin(k, ss), q = g.outC[sl]; float dx = (q.x - pt.x) * 0.5f;
					dl->AddBezierCubic(pt, ImVec2(pt.x + dx, pt.y), ImVec2(q.x - dx, q.y), q, sc, 1.5f);
				}
			}
		}
	}

	// ---- boxes.
	for (int i = 0; i < n; ++i) {
		if (collapsedBox[i] || sgHidden[i]) continue;   // hidden inside a collapsed group node (drawn separately below)
		ImVec2 tl(origin.x + rgn[i].pos.x, origin.y + rgn[i].pos.y), br(tl.x + rgn[i].w, tl.y + rgn[i].h);
		bool dim = isDim(i), up = inUp(i), dn = inDn(i);

		ImU32 fill = rg_kind_color(rgn[i].pass->kind);
		dl->AddRectFilled(tl, br, dim ? rg_with_alpha(fill, 55) : fill, 5.0f);
		ImU32 edgec = up ? IM_COL32(255, 255, 255, 255) : dn ? IM_COL32(120, 222, 180, 255) : dim ? IM_COL32(40, 40, 40, 120) : IM_COL32(20, 20, 20, 255);
		dl->AddRect(tl, br, edgec, 5.0f, 0, (up || dn) ? 2.5f : 1.0f);
		dl->AddLine(ImVec2(tl.x, tl.y + kHeaderH), ImVec2(br.x, tl.y + kHeaderH), IM_COL32(255, 255, 255, dim ? 18 : 40));

		// sinks: red halo. imported writers only -- temporal/history writers are sinks to the culler
		// (compile sets p->sink) but aren't graph outputs, so they don't get the halo.
		if (rg_pass_is_sink(rg, rgn[i].pass))
			for (int e = 3; e >= 1; --e)
				dl->AddRect(ImVec2(tl.x - e, tl.y - e), ImVec2(br.x + e, br.y + e), IM_COL32(255, 100, 0, dim ? 80 : 200), 6.0f, 0, 1.0f);

		WGPUStringView nm = rgn[i].pass->name;
		char head[96];
		std::snprintf(head, sizeof head, "P%d  %.*s", i, (int)nm.length, nm.data ? nm.data : "");
		dl->PushClipRect(tl, ImVec2(br.x - 4, br.y), true);
		dl->AddText(ImVec2(tl.x + 7, tl.y + 4), dim ? IM_COL32(190, 190, 190, 120) : IM_COL32(255, 255, 255, 255), head);
		dl->PopClipRect();

		const char* kn = rg_kind_name(rgn[i].pass->kind);
		ImVec2 ks = ImGui::CalcTextSize(kn);
		dl->AddText(ImVec2(tl.x + (rgn[i].w - ks.x) * 0.5f, br.y - kFooterH + 1),
			dim ? IM_COL32(200, 200, 200, 110) : IM_COL32(225, 225, 225, 220), kn);
	}

	// ---- pins + resource labels.
	for (int i = 0; i < n; ++i) {
		if (collapsedBox[i] || sgHidden[i]) continue;   // hidden inside a collapsed group node (drawn separately below)
		PassNode* p = rgn[i].pass; bool dim = isDim(i);
		const float mid = origin.x + rgn[i].pos.x + rgn[i].w * 0.5f;
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

	// ---- collapsed group nodes: one synthetic box per collapsed group, drawn on top of the edges like a
	// real pass, carrying the group's external read pins (left) and write pins (right).
	for (GView& g : groups) {
		if (!g.collapsed) continue;
		ImVec2 a = g.bb0, b = g.bb1;
		const float mid = (a.x + b.x) * 0.5f;
		dl->AddRectFilled(a, b, IM_COL32(70, 70, 96, 255), 5.0f);
		dl->AddRect(a, b, IM_COL32(20, 20, 20, 255), 5.0f, 0, 1.0f);
		dl->AddLine(ImVec2(a.x, a.y + kHeaderH), ImVec2(b.x, a.y + kHeaderH), IM_COL32(255, 255, 255, 40));
		char head[64]; std::snprintf(head, sizeof head, "[+] %.*s  x%d", (int)sv_length(g.prefix), g.prefix.data, g.gj - g.gi);
		dl->PushClipRect(a, ImVec2(b.x - 4, b.y), true);
		dl->AddText(ImVec2(a.x + 7, a.y + 4), IM_COL32(255, 255, 255, 255), head);
		dl->PopClipRect();
		for (int s = 0; s < g.nIn; ++s) {
			ResourceNode* r = find_node(rg, { g.inId[s] });
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			bool buf = r && r->kind == ResourceNode::Kind::Buffer;
			bool prod = false; for (int j = 0; j < n; ++j) if (rg_pass_writes(box[j].p, g.inId[s])) prod = true;
			rg_draw_pin(dl, g.inC[s], kPinR, kRGRead, prod, buf);   // hollow = external input (no in-graph producer)
			if (hovB >= g.gi && hovB < g.gj && !hovWrite && hovId == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
			if (lockB >= g.gi && lockB < g.gj && !lockWrite && lockId == g.inId[s]) dl->AddCircle(g.inC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
			char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
			ImVec2 ls = ImGui::CalcTextSize(lbl);
			dl->PushClipRect(ImVec2(g.inC[s].x + kPinR + 3, g.inC[s].y - kPinRowH * 0.5f), ImVec2(mid, g.inC[s].y + kPinRowH * 0.5f), true);
			dl->AddText(ImVec2(g.inC[s].x + kPinR + 3, g.inC[s].y - ls.y * 0.5f), IM_COL32(230, 230, 230, 255), lbl);
			dl->PopClipRect();
		}
		for (int s = 0; s < g.nOut; ++s) {
			ResourceNode* r = find_node(rg, { g.outId[s] });
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			bool buf = r && r->kind == ResourceNode::Kind::Buffer;
			rg_draw_pin(dl, g.outC[s], kPinR, kRGWrite, true, buf);
			if (hovB >= g.gi && hovB < g.gj && hovWrite && hovId == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
			if (lockB >= g.gi && lockB < g.gj && lockWrite && lockId == g.outId[s]) dl->AddCircle(g.outC[s], kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
			char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
			ImVec2 ls = ImGui::CalcTextSize(lbl);
			dl->PushClipRect(ImVec2(mid, g.outC[s].y - kPinRowH * 0.5f), ImVec2(g.outC[s].x - kPinR - 3, g.outC[s].y + kPinRowH * 0.5f), true);
			dl->AddText(ImVec2(g.outC[s].x - kPinR - 3 - ls.x, g.outC[s].y - ls.y * 0.5f), IM_COL32(230, 230, 230, 255), lbl);
			dl->PopClipRect();
		}
	}

	// expanded group border pins: the dots + hover/lock rings on the region edges (the region box + header
	// are drawn elsewhere). same dot/ring treatment as the collapsed compact node above, factored here.
	auto drawGPin = [&](GView& g, ImVec2 c, uint32_t id, bool write) {
		ResourceNode* r = find_node(rg, { id });
		bool buf = r && r->kind == ResourceNode::Kind::Buffer;
		bool prod = write; if (!write) for (int j = 0; j < n; ++j) if (rg_pass_writes(box[j].p, id)) { prod = true; break; }
		rg_draw_pin(dl, c, kPinR, write ? kRGWrite : kRGRead, prod, buf);
		if (hovB  >= g.gi && hovB  < g.gj && hovWrite  == write && hovId  == id) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
		if (lockB >= g.gi && lockB < g.gj && lockWrite == write && lockId == id) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(90, 200, 230, 255), 16, 2.0f);
	};
	for (GView& g : groups) {
		if (g.collapsed) continue;
		for (int s = 0; s < g.nIn;  ++s) drawGPin(g, g.inC[s],  g.inId[s],  false);
		for (int s = 0; s < g.nOut; ++s) drawGPin(g, g.outC[s], g.outId[s], true);
	}

	// ---- virtual endpoint nodes + dashed links, drawn from the IR mirror (rgn Virtual nodes + rge links).
	// a read/source node feeds a pass input pin from the left; a write/sink node is fed by an output pin; an
	// imported buffer fans one source to all readers. collapse-reroute + cone dimming reuse the pass helpers.
	// interface-pin position of resource `id` on a collapsed group, or the group's box edge at height `y` when
	// the resource crosses the boundary without a pin (e.g. a temporal write consumed only next frame).
	auto grpPin = [&](GView& g, uint32_t id, bool write, float y) -> ImVec2 {
		int sl = gpin_slot(write ? g.outId : g.inId, write ? g.nOut : g.nIn, id);
		if (sl >= 0) return write ? g.outC[sl] : g.inC[sl];
		return ImVec2(write ? g.bb1.x : g.bb0.x, y);
	};
	static std::vector<char> vLive; vLive.assign(rgn.size(), 0);
	for (RgEdge& ge : rge) {
		bool srcV = ge.srcNode >= n, dstV = ge.dstNode >= n;
		if (!srcV && !dstV) continue;                 // pass<->pass edge: drawn earlier
		int vi = srcV ? ge.srcNode : ge.dstNode, pe = srcV ? ge.dstNode : ge.srcNode;   // virtual node, anchor pass
		bool faint = ge.kind == RgEdge::Kind::Fanout;   // imported-buffer fan: one faint edge per reader
		if (fout(pe) || (faint && sgHidden[pe])) continue;
		RgNode& vn = rgn[vi];
		ImU32 tint = isDim(pe) ? rg_with_alpha(vn.tint, 40) : (faint ? rg_with_alpha(vn.tint, 90) : vn.tint);   // cone-dim with anchor
		float vcy = origin.y + vn.pos.y + vn.h * 0.5f;
		ImVec2 p, q;
		if (srcV) {   // read: node right edge -> reader input pin
			p = ImVec2(origin.x + vn.pos.x + vn.w, vcy);
			if (groupOf[pe] >= 0) {
				GView& g = groups[groupOf[pe]];
				if (faint) { int s = gpin_slot(g.inId, g.nIn, ge.resId); if (s < 0 || g.inDrawn[s]) continue; g.inDrawn[s] = true; q = g.inC[s]; }
				else q = grpPin(g, ge.resId, false, p.y);
			}
			else q = inPin(pe, ge.dstPin);
		}
		else {        // write: writer output pin -> node left edge
			q = ImVec2(origin.x + vn.pos.x, vcy);
			p = groupOf[pe] >= 0 ? grpPin(groups[groupOf[pe]], ge.resId, true, q.y) : outPin(pe, ge.srcPin);
		}
		vLive[vi] = 1;
		float dx = (q.x - p.x) * 0.5f;
		rg_dashed_cubic(dl, p, ImVec2(p.x + dx, p.y), ImVec2(q.x - dx, q.y), q, tint, faint ? 1.5f : 2.0f);
		rg_arrowhead(dl, ImVec2(q.x - 8, q.y), q, tint, faint ? 6.0f : 7.0f);
	}
	for (int vi = n; vi < (int)rgn.size(); ++vi) {
		if (!vLive[vi]) continue;   // node with no surviving link (all readers filtered/deduped) stays hidden
		RgNode& vn = rgn[vi];
		ImVec2 a(origin.x + vn.pos.x, origin.y + vn.pos.y), b(a.x + vn.w, a.y + vn.h);
		char nm[48]; std::snprintf(nm, sizeof nm, "%.*s", (int)vn.res->name.length, vn.res->name.data ? vn.res->name.data : "?");
		const char* cap = vn.label.data ? vn.label.data : "";
		dl->AddRectFilled(a, b, IM_COL32(32, 30, 40, 240), 4.0f);
		dl->AddRect(a, b, vn.tint, 4.0f, 0, 1.5f);
		float cx = (a.x + b.x) * 0.5f;
		ImVec2 ns = ImGui::CalcTextSize(nm), cs = ImGui::CalcTextSize(cap);
		dl->AddText(ImVec2(cx - ns.x * 0.5f, a.y + 3), IM_COL32(238, 236, 242, 255), nm);
		dl->AddText(ImVec2(cx - cs.x * 0.5f, b.y - cs.y - 3), rg_with_alpha(vn.tint, 220), cap);
	}

	// ---- tooltip: hovered pin wins; else fall back to the per-pass reads/writes list.
	if (hovB >= 0) {
		PassNode* p = rgn[hovB].pass; ResourceNode* r = find_node(rg, { hovId });
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
				WGPUStringView pn = rgn[prod].pass->name;
				ImGui::Text("produced by P%d %.*s", prod, (int)pn.length, pn.data ? pn.data : "");
			}
			else ImGui::TextDisabled(r && r->imported ? "imported (external input)" : "external input (no producer)");
		}
		ImGui::EndTooltip();
	}
	else if (hovBox >= 0) {
		PassNode* p = rgn[hovBox].pass; WGPUStringView nm = p->name;
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
		WGPUStringView sn = rgn[e.src].pass->name, dn = rgn[e.dst].pass->name;
		ImGui::BeginTooltip();
		ImGui::Text("%.*s", (int)rn.length, rn.data ? rn.data : "?");
		ImGui::TextDisabled("P%d %.*s  ->  P%d %.*s", e.src, (int)sn.length, sn.data ? sn.data : "",
			e.dst, (int)dn.length, dn.data ? dn.data : "");
		ImGui::EndTooltip();
	}

	ImGui::EndChild();
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

// distinct-ish colour per physical alias slot, so transients packed onto the same slot read as one hue
// (swatch in the gutter + bar outline). cycles a small palette; adjacent slot indices stay distinguishable.
static ImU32 rg_slot_color(uint32_t slot)
{
	static const ImU32 pal[] = {
		IM_COL32( 90, 170, 250, 255), IM_COL32(250, 170,  90, 255), IM_COL32(130, 220, 130, 255),
		IM_COL32(220, 130, 220, 255), IM_COL32(230, 210,  90, 255), IM_COL32(110, 210, 220, 255),
		IM_COL32(240, 140, 140, 255), IM_COL32(170, 160, 250, 255),
	};
	return pal[slot % (sizeof pal / sizeof pal[0])];
}

// short format tag for the physical-slot row labels in the lifetime view (display only).
static const char* rg_format_short(WGPUTextureFormat f)
{
	switch (f) {
	  case WGPUTextureFormat_RGBA8Unorm:   return "RGBA8";
	  case WGPUTextureFormat_BGRA8Unorm:   return "BGRA8";
	  case WGPUTextureFormat_RGBA16Float:  return "RGBA16F";
	  case WGPUTextureFormat_R32Float:     return "R32F";
	  case WGPUTextureFormat_Depth32Float: return "D32F";
	  case WGPUTextureFormat_Depth24Plus:  return "D24+";
	  case WGPUTextureFormat_R8Unorm:      return "R8";
	  default:                             return "tex";
	}
}

// Lifetime grid over the graph's own per-frame GPU textures. Top axis = passes in execution order. Each
// row is ONE dedicated allocation, never a bare logical handle:
//   * aliasing ON  -> a physical slot ("image N (fmt)"), its bar showing every occupant's writes (warm)
//     and reads (cool) with empty gaps where the texture is free between occupants;
//   * aliasing OFF -> each realized transient is its own texture (one row, named for the resource).
// Toggling 'alias transients' (Demos) collapses N transient rows into M shared textures -- that row-count
// drop, plus the reuse gaps, is the aliasing made visible. Imported (caller-owned), temporal/persistent
// (pool-backed) and dead/zero-usage resources have no dedicated per-frame allocation and are NOT shown.
// firstUse/lastUse come from compile() phase 3; the slot table from phase 4.
static void rg_draw_lifetimes(RenderGraph* rg, RenderGraphStorage& s)
{
	constexpr int   kMax = 128;
	constexpr float kLabelW = 150.0f, kColW = 88.0f, kRowH = 24.0f, kHeaderH = 24.0f;

	ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGWrite), "write");
	ImGui::SameLine(0, 4);
	ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kRGRead), "read");
	ImGui::SameLine();
	ImGui::TextDisabled("(rows = the graph's dedicated per-frame GPU objects: textures + buffers)");

	// header: rows below are physical slots (aliasing on) or per-transient objects (off). either way only
	// dedicated, graph-owned allocations -- imported/temporal/dead excluded. the row-count + saved MB are
	// the win; toggle 'alias transients' in Demos and watch the rows collapse.
	auto phBytes = [](const PhysicalResource& ph) -> uint64_t {
		return ph.kind == ResourceNode::Kind::Buffer ? ph.bufferSize : texture_bytes(ph.size, ph.format);
	};
	if (s.m_slotCount) {
		uint32_t logical = 0; uint64_t logicalBytes = 0, physicalBytes = 0;
		for (uint32_t i = 0; i < s.m_slotCount; ++i) physicalBytes += phBytes(s.m_slots[i]);
		for (ResourceNode* r = s.m_resouces; r; r = r->next)
			if (r->aliasSlot != ResourceNode::kNoSlot) { ++logical; logicalBytes += phBytes(s.m_slots[r->aliasSlot]); }
		ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(IM_COL32(150, 230, 150, 255)),
			"aliasing ON: %u transients packed onto %u GPU objects, saved %.2f MB",
			logical, s.m_slotCount, (double)(logicalBytes - physicalBytes) / (1024.0 * 1024.0));
	} else {
		uint32_t live = 0;
		for (ResourceNode* r = s.m_resouces; r; r = r->next)
			if (!r->is_external() && r->firstUse != ResourceNode::kNoPass) ++live;
		ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(IM_COL32(205, 205, 120, 255)),
			"aliasing OFF: %u dedicated transient objects (tick 'alias transients' in Demos to pack)", live);
	}

	PassNode* passAt[kMax];
	int nPass = 0;
	for (PassNode* p = s.m_passes; p && nPass < kMax; p = p->next) passAt[nPass++] = p;

	// rows = the graph's dedicated per-frame GPU textures, unified across aliasing on/off: every row is one
	// physical texture. a slot (aliasing on) hosts >=1 disjoint-lifetime transient; a non-aliased transient
	// is its own single-occupant texture. with aliasing off there are no slots, so each transient is its own
	// row -- same view, no sharing. imported/temporal/persistent (not graph-allocated this frame) and
	// dead/zero-usage transients (never realized) are excluded: only real GPU textures appear.
	struct Row { ResourceNode* r; int slot; uint32_t first, last; };
	Row row[kMax];
	int nRow = 0;
	for (uint32_t si = 0; si < s.m_slotCount && nRow < kMax; ++si) {
		uint32_t f = ResourceNode::kNoPass, l = 0;
		for (ResourceNode* o = s.m_resouces; o; o = o->next)
			if (o->aliasSlot == si) { if (o->firstUse < f) f = o->firstUse; if (o->lastUse > l) l = o->lastUse; }
		if (f != ResourceNode::kNoPass) row[nRow++] = { nullptr, (int)si, f, l };
	}
	for (ResourceNode* r = s.m_resouces; r && nRow < kMax; r = r->next)
		if (!r->is_external() && r->aliasSlot == ResourceNode::kNoSlot && r->firstUse != ResourceNode::kNoPass)
			row[nRow++] = { r, -1, r->firstUse, r->lastUse };

	// uniform per-row queries over the two kinds (a slot reads its m_slots entry + ORs its occupants; a solo
	// row is its one resource). keeps the draw + details code identical for aliased and non-aliased.
	auto rowFmt   = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].format   : rw.r->format;   };
	auto rowSize  = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].size     : rw.r->resolved; };
	auto rowUsage = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].texUsage : rw.r->texUsage; };
	auto rowKind  = [&](const Row& rw) { return rw.slot >= 0 ? s.m_slots[rw.slot].kind     : rw.r->kind;     };
	auto rowAccess = [&](const Row& rw, uint32_t c) -> int {
		if (rw.slot < 0) return rg_pass_access(passAt[c], rw.r->handle.id);
		int a = 0;
		for (ResourceNode* o = s.m_resouces; o; o = o->next)
			if (o->aliasSlot == (uint32_t)rw.slot) a |= rg_pass_access(passAt[c], o->handle.id);
		return a;
	};
	// live this pass? a solo row spans [first,last]; a slot row is live only while some occupant is, so the
	// gaps between successive occupants render empty -- the texture being freed and reused, on the timeline.
	auto rowLive = [&](const Row& rw, uint32_t c) -> bool {
		if (rw.slot < 0) return rw.first <= c && c <= rw.last;
		for (ResourceNode* o = s.m_resouces; o; o = o->next)
			if (o->aliasSlot == (uint32_t)rw.slot && o->firstUse <= c && c <= o->lastUse) return true;
		return false;
	};
	// occupant count, and (when exactly one) that sole resource -> drives the label: a texture backing one
	// resource is shown by its NAME (plain); a shared texture (>1 occupant) by the synthetic "image N".
	auto rowOccupants = [&](const Row& rw, ResourceNode*& sole) -> int {
		if (rw.slot < 0) { sole = rw.r; return 1; }
		int n = 0; sole = nullptr;
		for (ResourceNode* o = s.m_resouces; o; o = o->next)
			if (o->aliasSlot == (uint32_t)rw.slot) { if (!n) sole = o; ++n; }
		return n;
	};

	// selection persists across frames. the graph is rebuilt each frame (handle ids not stable), so key a
	// slot row by its slot index and a solo row by its resource NAME (stable for a given graph).
	static int  selSlot = -2;        // -2 = nothing; -1 = a solo row (selName); >= 0 = slot index
	static char selName[96] = {};
	auto rowSelected = [&](const Row& rw) -> bool {
		if (rw.slot >= 0) return rw.slot == selSlot;
		return selSlot == -1 && rw.r->name.data && std::strcmp(selName, rw.r->name.data) == 0;
	};

	// lifetime grid on top, details panel for the selected texture below (only when the tab is tall enough).
	const float availY = ImGui::GetContentRegionAvail().y;
	const float gridH  = availY > 240.0f ? availY * 0.62f : availY;

	ImGui::BeginChild("rg_life_grid", ImVec2(0, gridH), true, ImGuiWindowFlags_HorizontalScrollbar);
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

	// one row per physical texture: "image i (fmt WxH)" + swatch in the gutter, occupant write/read cells
	// across the timeline with empty gaps where the texture is free. click a row to inspect it below.
	for (int i = 0; i < nRow; ++i) {
		const float y    = gridT + i * kRowH;
		ResourceNode* sole = nullptr;
		const int   nocc   = rowOccupants(row[i], sole);
		const bool  shared = nocc > 1;
		// shared textures get the synthetic "image N" + a slot colour so they stand out; a single-occupant
		// texture keeps the resource name and reads as a plain row. format/size are in the details panel.
		const ImU32 col  = shared ? rg_slot_color((uint32_t)i) : IM_COL32(210, 210, 210, 255);
		const bool  seld = rowSelected(row[i]);
		if (seld)       dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(gridR, y + kRowH), IM_COL32(255, 255, 255, 30));
		else if (i & 1) dl->AddRectFilled(ImVec2(origin.x, y), ImVec2(gridR, y + kRowH), IM_COL32(255, 255, 255, 10));

		const bool isBuf = rowKind(row[i]) == ResourceNode::Kind::Buffer;
		char label[96];
		if (shared) std::snprintf(label, sizeof label, "%s %d (x%d)", isBuf ? "buffer" : "image", i, nocc);
		else        std::snprintf(label, sizeof label, "%.*s", (int)sole->name.length, sole->name.data ? sole->name.data : "");
		if (shared)   // gutter swatch only for shared (aliased) textures
			dl->AddRectFilled(ImVec2(origin.x + 1, y + 4), ImVec2(origin.x + 4, y + kRowH - 4), col);
		dl->PushClipRect(ImVec2(origin.x + 6, y), ImVec2(origin.x + kLabelW - 4, y + kRowH), true);
		dl->AddText(ImVec2(origin.x + 8, y + 4), col, label);
		dl->PopClipRect();

		// whole-row click target (gutter + grid) selects this texture for the details panel below.
		ImGui::SetCursorScreenPos(ImVec2(origin.x, y));
		ImGui::PushID(kMax + i);
		if (ImGui::InvisibleButton("row", ImVec2(kLabelW + nPass * kColW, kRowH))) {
			selSlot = row[i].slot;
			if (row[i].slot < 0) std::snprintf(selName, sizeof selName, "%.*s",
				(int)row[i].r->name.length, row[i].r->name.data ? row[i].r->name.data : "");
		}
		const bool hov = ImGui::IsItemHovered();
		ImGui::PopID();

		const float x0 = origin.x + kLabelW + row[i].first * kColW + 3.0f;
		const float x1 = origin.x + kLabelW + (row[i].last + 1) * kColW - 3.0f;
		const float ty = y + 3.0f, by = y + kRowH - 3.0f;

		// occupant write/read cells; gaps (texture free) left empty so reuse reads off the timeline.
		const ImU32 bandCol = rg_with_alpha(col, 45);
		for (uint32_t c = row[i].first; c <= row[i].last; ++c) {
			if (!rowLive(row[i], c)) continue;
			float sx0 = (c == row[i].first) ? x0 : origin.x + kLabelW + c * kColW;
			float sx1 = (c == row[i].last)  ? x1 : origin.x + kLabelW + (c + 1) * kColW;
			dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, by), bandCol, 0.0f);
			int acc = rowAccess(row[i], c);
			if (!acc) continue;
			if (acc == 3) {   // read + write -> split top (write) / bottom (read)
				float mid = (ty + by) * 0.5f;
				dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, mid), kRGWrite, 0.0f);
				dl->AddRectFilled(ImVec2(sx0, mid), ImVec2(sx1, by), kRGRead, 0.0f);
			} else
				dl->AddRectFilled(ImVec2(sx0, ty), ImVec2(sx1, by), acc == 2 ? kRGWrite : kRGRead, 0.0f);
		}

		// outline each contiguous occupancy run in the row colour (brighter when selected or hovered), so the
		// gaps stay open and each occupant reads as its own block.
		const ImU32 edge = (seld || hov) ? IM_COL32(255, 255, 255, 255) : col;
		const float wth  = (seld || hov) ? 2.0f : 1.0f;
		for (uint32_t c = row[i].first; c <= row[i].last; ) {
			if (!rowLive(row[i], c)) { ++c; continue; }
			uint32_t segStart = c;
			while (c <= row[i].last && rowLive(row[i], c)) ++c;
			float sx0 = origin.x + kLabelW + segStart * kColW + 3.0f;
			float sx1 = origin.x + kLabelW + c * kColW - 3.0f;
			dl->AddRect(ImVec2(sx0, ty), ImVec2(sx1, by), edge, 0.0f, 0, wth);
		}
	}
	ImGui::EndChild();

	// ---- details panel: the selected physical texture and the logical resource(s) it backs ----
	if (gridH < availY) {
		ImGui::BeginChild("rg_life_details", ImVec2(0, 0), true);
		int selIdx = -1;
		for (int i = 0; i < nRow; ++i) if (rowSelected(row[i])) { selIdx = i; break; }
		if (selIdx < 0) {
			ImGui::TextDisabled("click a texture above to inspect it");
		} else {
			const Row&    rw    = row[selIdx];
			const bool    isBuf = rowKind(rw) == ResourceNode::Kind::Buffer;
			ResourceNode* psole = nullptr;
			const int     pnocc = rowOccupants(rw, psole);
			const ImU32   hcol  = pnocc > 1 ? rg_slot_color((uint32_t)selIdx) : IM_COL32(225, 225, 225, 255);
			if (pnocc > 1) ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(hcol), "%s %d  (shared by %d)",
				isBuf ? "buffer" : "image", selIdx, pnocc);
			else           ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(hcol), "%.*s",
				(int)psole->name.length, psole->name.data ? psole->name.data : "");
			ImGui::SameLine();

			char ub[160]; ub[0] = '\0';
			if (isBuf) {
				uint64_t        bytes = rw.slot >= 0 ? s.m_slots[rw.slot].bufferSize : rw.r->bufferSize;
				WGPUBufferUsage u     = rw.slot >= 0 ? s.m_slots[rw.slot].bufUsage   : rw.r->bufUsage;
				ImGui::Text("- buffer  -  %.1f KB", bytes / 1024.0);
				auto addb = [&](WGPUBufferUsage bit, const char* nm) { if (u & bit) { if (ub[0]) std::strcat(ub, " | "); std::strcat(ub, nm); } };
				addb(WGPUBufferUsage_Storage,  "Storage");   addb(WGPUBufferUsage_Uniform,  "Uniform");
				addb(WGPUBufferUsage_Vertex,   "Vertex");    addb(WGPUBufferUsage_Index,    "Index");
				addb(WGPUBufferUsage_Indirect, "Indirect");  addb(WGPUBufferUsage_CopySrc,  "CopySrc");
				addb(WGPUBufferUsage_CopyDst,  "CopyDst");
			} else {
				WGPUExtent3D      sz  = rowSize(rw);
				WGPUTextureFormat fmt = rowFmt(rw);
				WGPUTextureUsage  u   = rowUsage(rw);
				ImGui::Text("- %s  %u x %u  -  %.1f KB", rg_format_short(fmt), sz.width, sz.height, texture_bytes(sz, fmt) / 1024.0);
				auto addu = [&](WGPUTextureUsage bit, const char* nm) { if (u & bit) { if (ub[0]) std::strcat(ub, " | "); std::strcat(ub, nm); } };
				addu(WGPUTextureUsage_RenderAttachment, "RenderAttachment");
				addu(WGPUTextureUsage_TextureBinding,   "TextureBinding");
				addu(WGPUTextureUsage_StorageBinding,   "StorageBinding");
				addu(WGPUTextureUsage_CopySrc,          "CopySrc");
				addu(WGPUTextureUsage_CopyDst,          "CopyDst");
			}
			ImGui::TextDisabled("usage: %s", ub[0] ? ub : "(none)");

			int occ = 0;
			if (rw.slot >= 0) { for (ResourceNode* o = s.m_resouces; o; o = o->next) if (o->aliasSlot == (uint32_t)rw.slot) ++occ; }
			else occ = 1;
			ImGui::Separator();
			if (occ > 1) ImGui::Text("%d logical resources share this %s (disjoint lifetimes):", occ, isBuf ? "buffer" : "texture");
			else         ImGui::Text("backs 1 logical resource:");

			auto detailOne = [&](ResourceNode* o) {
				WGPUStringView f = pass_name_at(s.m_passes, o->firstUse);
				WGPUStringView l = pass_name_at(s.m_passes, o->lastUse);
				ImGui::BulletText("%.*s   [%.*s .. %.*s]",
					(int)o->name.length, o->name.data ? o->name.data : "",
					(int)f.length, f.data ? f.data : "", (int)l.length, l.data ? l.data : "");
				ImGui::Indent();
				for (uint32_t c = o->firstUse; c <= o->lastUse; ++c) {
					int a = rg_pass_access(passAt[c], o->handle.id);
					if (!a) continue;
					WGPUStringView pn = passAt[c]->name;
					ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(a == 1 ? kRGRead : kRGWrite),
						"%s  %.*s", a == 3 ? "rw" : a == 2 ? " w" : " r", (int)pn.length, pn.data ? pn.data : "");
				}
				ImGui::Unindent();
			};
			if (rw.slot >= 0) { for (ResourceNode* o = s.m_resouces; o; o = o->next) if (o->aliasSlot == (uint32_t)rw.slot) detailOne(o); }
			else detailOne(rw.r);
		}
		ImGui::EndChild();
	}
}

// GPU-memory view across every pool the graph owns: the transient pool (one descriptor-keyed cache of
// textures + buffers, tagged by kind) and temporal/history textures + buffers (PersistentResourcePool
// ping-pong). Grand total at the top answers "how much VRAM does the graph cost"; the transient pool also
// keeps its create/evict log so steady-state reuse is still verifiable (0 created after warmup). Drawn after
// realize() and before release_resources()/end_frame(), so every count is this frame's live allocation.
static void rg_draw_transient_pool(RenderGraphStorage& s)
{
	TransientResourcePool&  tp   = s.m_allocator->transient;
	PersistentResourcePool& pool = s.m_allocator->pool;

	// transient pool: textures + buffers in one descriptor-keyed cache, tagged by Entry::isBuffer. one
	// physical object per entry; idle ones retained kRetain frames. tally each kind (held includes idle).
	int held = 0, inUse = 0, texHeld = 0, bufHeld = 0;
	uint64_t texBytes = 0, texInUseBytes = 0, bufBytes = 0;
	for (const TransientResourcePool::Entry& e : tp.entries) {
		++held; if (e.inUse) ++inUse;
		if (e.isBuffer) { ++bufHeld; bufBytes += e.bufferSize; }
		else {
			const uint64_t b = rg_entry_bytes(e);
			++texHeld; texBytes += b;
			if (e.inUse) texInUseBytes += b;
		}
	}

	// temporal/history resources: kLayers physical textures OR buffers per entry (current + previous),
	// ping-ponged. the buffer arm leaves size {} / format Undefined, so split on bufferSize to size each right.
	int tmpTexCount = 0, tmpBufCount = 0;
	uint64_t tmpTexBytes = 0, tmpBufBytes = 0;
	for (const PersistentResourcePool::Entry& e : pool.entries) {
		if (!e.created) continue;
		if (e.bufferSize) { ++tmpBufCount; tmpBufBytes += e.bufferSize * e.layers; }
		else              { ++tmpTexCount; tmpTexBytes += rg_texture_bytes(e.size, e.format, e.mipLevelCount) * e.layers; }
	}

	const uint64_t grand = texBytes + bufBytes + tmpTexBytes + tmpBufBytes;

	ImGui::Text("frame %llu  --  transient pool: %d held, %d in use", (unsigned long long)tp.frame, held, inUse);
	ImGui::SameLine();
	if (tp.createdThisFrame == 0)
		ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1), "  --  0 created (reused from pool)");
	else
		ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "  --  %u created this frame", tp.createdThisFrame);

	char gb[24]; rg_bytes_str(grand, gb, sizeof gb);
	ImGui::Text("VRAM %s total", gb);
	{
		char a[24], ib[24], idb[24], bb[24], cb[24], cbb[24];
		rg_bytes_str(texBytes, a, sizeof a);
		rg_bytes_str(texInUseBytes, ib, sizeof ib);
		rg_bytes_str(texBytes - texInUseBytes, idb, sizeof idb);   // in use is a subset sum -> no underflow
		rg_bytes_str(bufBytes, bb, sizeof bb);
		rg_bytes_str(tmpTexBytes, cb, sizeof cb);
		rg_bytes_str(tmpBufBytes, cbb, sizeof cbb);
		ImGui::BulletText("transient tex  %-9s  %d held (%s in use, %s idle)", a, texHeld, ib, idb);
		ImGui::BulletText("transient buf  %-9s  %d held", bb, bufHeld);
		ImGui::BulletText("temporal tex   %-9s  %d entries", cb, tmpTexCount);
		ImGui::BulletText("temporal buf   %-9s  %d entries", cbb, tmpBufCount);
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
			if (e.isBuffer) continue;   // buffers listed in their own table below
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

	// currently held physical buffers (the buffer arm of the same pool, tagged isBuffer).
	ImGui::Spacing();
	ImGui::TextDisabled("transient buffers");
	if (ImGui::BeginTable("tp_buf", 4, tf)) {
		ImGui::TableSetupColumn("size");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("mem");
		ImGui::TableSetupColumn("state");
		ImGui::TableHeadersRow();
		bool any = false;
		for (const TransientResourcePool::Entry& e : tp.entries) {
			if (!e.isBuffer) continue;
			any = true;
			char ub[12]; rg_buf_usage_str(e.bufUsage, ub, sizeof ub);
			char mb[24]; rg_bytes_str(e.bufferSize, mb, sizeof mb);
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%llu B", (unsigned long long)e.bufferSize);
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn(); ImGui::Text("%s", mb);
			ImGui::TableNextColumn();
			if (e.inUse) ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1), "in use");
			else         ImGui::TextDisabled("idle %lluf", (unsigned long long)(tp.frame - e.lastUsedFrame));
		}
		ImGui::EndTable();
		if (!any) ImGui::TextDisabled("(none)");
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
			if (!e.created || e.bufferSize) continue;   // buffer-arm entries listed in their own table below
			any = true;
			char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
			const uint64_t eb = rg_texture_bytes(e.size, e.format, e.mipLevelCount) * e.layers;
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%s", e.name.c_str());
			ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
			ImGui::TableNextColumn(); ImGui::Text("%u", e.mipLevelCount);
			ImGui::TableNextColumn(); ImGui::Text("%u x%u", e.size.depthOrArrayLayers, e.layers);
			ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn();
			if (eb) { char mb[24]; rg_bytes_str(eb, mb, sizeof mb); ImGui::Text("%s", mb); }
			else    ImGui::TextDisabled("?");
		}
		ImGui::EndTable();
		if (!any) ImGui::TextDisabled("(none)");
	}

	// temporal/history buffers (PersistentResourcePool buffer arm): one row per name, kLayers buffers each.
	ImGui::Spacing();
	ImGui::TextDisabled("temporal (history) buffers");
	if (ImGui::BeginTable("tp_tmpbuf", 3, tf)) {
		ImGui::TableSetupColumn("name");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("mem (x layers)");
		ImGui::TableHeadersRow();
		bool any = false;
		for (const PersistentResourcePool::Entry& e : pool.entries) {
			if (!e.created || !e.bufferSize) continue;
			any = true;
			char ub[12]; rg_buf_usage_str(e.bufUsage, ub, sizeof ub);
			char mb[24]; rg_bytes_str(e.bufferSize * e.layers, mb, sizeof mb);
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%s", e.name.c_str());
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
			ImGui::TableNextColumn(); ImGui::Text("%s", mb);
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
	ImGui::Text(" compile %.0f us  realize %.0f us  execute %.0f us",
	            s.timing_compile_us, s.timing_realize_us, s.timing_execute_us);
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
