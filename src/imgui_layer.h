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

// one pool entry's footprint: every mip level, times array layers. ponytail: layers stays constant
// across mips -- right for 2D / 2D-array (all this sample makes), over-counts a true 3D texture whose
// depth also halves each level.
static uint64_t rg_entry_bytes(const TransientResourcePool::Entry& e)
{
	const uint64_t bpp = rg_format_bytes(e.format);
	if (!bpp) return 0;
	const uint32_t layers = e.size.depthOrArrayLayers ? e.size.depthOrArrayLayers : 1;
	uint64_t total = 0;
	for (uint32_t m = 0; m < e.mipLevelCount; ++m) {
		const uint32_t w = (e.size.width  >> m) ? (e.size.width  >> m) : 1u;   // max(1, ..) without <algorithm>
		const uint32_t h = (e.size.height >> m) ? (e.size.height >> m) : 1u;
		total += (uint64_t)w * h * layers * bpp;
	}
	return total;
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

static void rg_draw_dag(RenderGraph* rg, RenderGraphStorage& s)
{
	constexpr float kBoxW = 190.0f, kColGap = 64.0f, kRowGap = 20.0f;
	constexpr float kHeaderH = 22.0f, kFooterH = 14.0f, kPinRowH = 18.0f, kMinBodyH = 12.0f;
	constexpr float kPinR = 5.0f, kPinHit = 8.0f;

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

	// runs of consecutive same-named passes (shadow.cascade x3, bloom.down/up xN) = a visual group; give
	// each run a unique id (singletons too) so crossing-min can keep a group's members contiguous.
	int groupId[kRgDagMax]; int runs = 0;
	for (int i = 0; i < n;) {
		int j = i + 1;
		while (j < n && rg_same_name(box[j].p->name, box[i].p->name)) ++j;
		for (int k = i; k < j; ++k) groupId[k] = runs;
		++runs; i = j;
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

	// drawn-edge (RAW pin) successors, for crossing reduction only: a read pin's producer is
	// rg_producer_of; WAW writers contribute none, matching what's actually drawn.
	static std::vector<int> rawSucc[kRgDagMax];
	for (int i = 0; i < n; ++i) rawSucc[i].clear();
	for (int v = 0; v < n; ++v) {
		PassNode* p = box[v].p;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (!rg_access_reads(p->accesses[k])) continue;
			int u = rg_producer_of(box, n, p, p->accesses[k].handle.id);
			if (u >= 0) rawSucc[u].push_back(v);
		}
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

	// columns, seeded in execution order (a stable starting permutation).
	static std::vector<int> column[kRgDagMax];
	for (int c = 0; c <= maxDist; ++c) column[c].clear();
	int rowOf[kRgDagMax] = {};
	for (int i = 0; i < n; ++i) { box[i].layer = colOf[i]; column[colOf[i]].push_back(i); }
	auto reindex = [&](int c) { for (int r = 0; r < (int)column[c].size(); ++r) rowOf[column[c][r]] = r; };
	for (int c = 0; c <= maxDist; ++c) reindex(c);

	// mean row of a node's RAW neighbours on one side; -1 = none -> leave it where it is.
	auto baryR = [&](int v) { float s = 0; int c = 0; for (int w : rawSucc[v]) { s += rowOf[w]; ++c; } return c ? s / c : -1.0f; };
	auto baryL = [&](int v) {
		float s = 0; int c = 0; PassNode* p = box[v].p;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (!rg_access_reads(p->accesses[k])) continue;
			int u = rg_producer_of(box, n, p, p->accesses[k].handle.id);
			if (u >= 0) { s += rowOf[u]; ++c; }
		}
		return c ? s / c : -1.0f;
		};
	auto sweep = [&](bool backward) {
		int from = backward ? maxDist - 1 : 1, to = backward ? -1 : maxDist + 1, step = backward ? -1 : 1;
		for (int c = from; c != to; c += step) {
			std::vector<int>& m = column[c];
			float key[kRgDagMax];
			for (int j = 0; j < (int)m.size(); ++j) { float b = backward ? baryR(m[j]) : baryL(m[j]); key[j] = b < 0 ? (float)rowOf[m[j]] : b; }
			for (int a = 1; a < (int)m.size(); ++a) {   // stable insertion sort; tied / neighbourless nodes stay put
				int mv = m[a]; float kv = key[a]; int b = a - 1;
				while (b >= 0 && key[b] > kv) { m[b + 1] = m[b]; key[b + 1] = key[b]; --b; }
				m[b + 1] = mv; key[b + 1] = kv;
			}
			reindex(c);
		}
		};
	sweep(true); sweep(false); sweep(true);   // from the sinks back, then settle

	// keep each group's members contiguous within their column so a cluster reads as one block (no foreign
	// node splitting the cascades). order groups + singletons by their post-sweep mean rank.
	for (int c = 0; c <= maxDist; ++c) {
		std::vector<int>& m = column[c];
		if ((int)m.size() < 2) continue;
		float gkey[kRgDagMax]; int gcnt[kRgDagMax] = {};
		for (int g = 0; g < runs; ++g) gkey[g] = 0;
		for (int r = 0; r < (int)m.size(); ++r) { gkey[groupId[m[r]]] += r; gcnt[groupId[m[r]]]++; }
		for (int g = 0; g < runs; ++g) if (gcnt[g]) gkey[g] /= gcnt[g];
		int ord[kRgDagMax], no = 0; bool seen[kRgDagMax] = {};
		for (int r = 0; r < (int)m.size(); ++r) { int g = groupId[m[r]]; if (!seen[g]) { seen[g] = true; ord[no++] = g; } }
		for (int a = 1; a < no; ++a) { int gv = ord[a]; float kv = gkey[gv]; int b = a - 1; while (b >= 0 && gkey[ord[b]] > kv) { ord[b + 1] = ord[b]; --b; } ord[b + 1] = gv; }
		std::vector<int> out; out.reserve(m.size());
		for (int oi = 0; oi < no; ++oi) for (int r = 0; r < (int)m.size(); ++r) if (groupId[m[r]] == ord[oi]) out.push_back(m[r]);
		m.swap(out);
		reindex(c);
	}

	// pixel positions: x by column, y by stacking each column top-to-bottom in its settled row order.
	for (int c = 0; c <= maxDist; ++c) {
		float y = 0;
		for (int idx : column[c]) { box[idx].tl = ImVec2(c * (kBoxW + kColGap), y); y += box[idx].h + kRowGap; }
	}

	// pannable canvas: a static scroll offset (drag left/middle to pan) + a grid, after the imgui node-
	// graph example. no scrollbar -- navigation is panning, so a big graph isn't boxed in.
	static ImVec2 scrolling(0, 0);
	ImGui::BeginChild("rg_canvas", ImVec2(0, 0), true,
		ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
	ImGuiIO& io = ImGui::GetIO();
	const ImVec2 winPos = ImGui::GetCursorScreenPos();
	const ImVec2 winSize = ImGui::GetContentRegionAvail();
	ImGui::InvisibleButton("canvas", ImVec2(winSize.x > 0 ? winSize.x : 1, winSize.y > 0 ? winSize.y : 1));
	const bool canvasHovered = ImGui::IsItemHovered();
	const bool canvasActive = ImGui::IsItemActive();
	if (canvasActive && (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f))) {
		scrolling.x += io.MouseDelta.x; scrolling.y += io.MouseDelta.y;
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
	// output slot index of resource `id` on box b, or -1 if it doesn't write it.
	auto outSlotOf = [&](int b, uint32_t id) -> int {
		int slot = 0; PassNode* p = box[b].p;
		for (uint32_t k = 0; k < p->accessCount; ++k)
			if (access_is_write(p->accesses[k].type)) { if (p->accesses[k].handle.id == id) return slot; ++slot; }
		return -1;
		};

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

	// ---- the upstream cone of the hovered pin. write pin: the pass itself; read pin: its producer.
	bool inCone[kRgDagMax] = {}; bool coneActive = false;
	if (hovB >= 0) {
		int seed = hovWrite ? hovB : rg_producer_of(box, n, box[hovB].p, hovId);
		if (seed >= 0) { rg_mark_cone(box, n, seed, inCone); coneActive = true; }
	}

	// ---- data edges (RAW pin-to-pin) so boxes paint over them. a read is fed by its producer AND any
	// parallel-writer siblings of that producer that also write the resource (CSM cascades all feed the
	// reader); the usual single producer is just one edge.
	for (int i = 0; i < n; ++i) {
		int inS = 0; PassNode* p = box[i].p;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			if (!rg_access_reads(p->accesses[k])) continue;
			uint32_t id = p->accesses[k].handle.id; int slot = inS++;
			int prod = rg_producer_of(box, n, p, id);
			if (prod < 0) continue;
			for (int w = 0; w < n; ++w) {
				if (groupRep[w] != groupRep[prod]) continue;
				int os = outSlotOf(w, id);
				if (os < 0) continue;
				ImVec2 src = outPin(w, os), dst = inPin(i, slot);
				float dx = (dst.x - src.x) * 0.5f;
				bool lit = coneActive && inCone[i] && inCone[w];
				ImU32 col = !coneActive ? IM_COL32(170, 170, 170, 200)
					: lit ? IM_COL32(245, 222, 120, 235)
					: IM_COL32(150, 150, 150, 38);
				dl->AddBezierCubic(src, ImVec2(src.x + dx, src.y), ImVec2(dst.x - dx, dst.y), dst, col, lit ? 2.5f : 2.0f);
			}
		}
	}

	// ---- boxes.
	for (int i = 0; i < n; ++i) {
		ImVec2 tl(origin.x + box[i].tl.x, origin.y + box[i].tl.y), br(tl.x + box[i].w, tl.y + box[i].h);
		bool dim = coneActive && !inCone[i], lit = coneActive && inCone[i];

		ImU32 fill = rg_kind_color(box[i].p->kind);
		dl->AddRectFilled(tl, br, dim ? rg_with_alpha(fill, 55) : fill, 5.0f);
		dl->AddRect(tl, br, lit ? IM_COL32(255, 255, 255, 255) : dim ? IM_COL32(40, 40, 40, 120) : IM_COL32(20, 20, 20, 255),
			5.0f, 0, lit ? 2.5f : 1.0f);
		dl->AddLine(ImVec2(tl.x, tl.y + kHeaderH), ImVec2(br.x, tl.y + kHeaderH), IM_COL32(255, 255, 255, dim ? 18 : 40));

		// sinks: red halo.
		if (box[i].p->sink)
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
		PassNode* p = box[i].p; bool dim = coneActive && !inCone[i];
		const float mid = origin.x + box[i].tl.x + box[i].w * 0.5f;
		int inS = 0, outS = 0;
		for (uint32_t k = 0; k < p->accessCount; ++k) {
			const ResourceAccess& acc = p->accesses[k];
			ResourceNode* r = rg->node(acc.handle);
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			char lbl[48]; std::snprintf(lbl, sizeof lbl, "%.*s", (int)rn.length, rn.data ? rn.data : "?");
			ImVec2 ls = ImGui::CalcTextSize(lbl);
			const ImU32 lc = dim ? IM_COL32(190, 190, 190, 110) : IM_COL32(230, 230, 230, 255);

			if (rg_access_reads(acc)) {   // input pin (left); hollow if no in-graph producer (external input)
				int slot = inS++; ImVec2 c = inPin(i, slot);
				ImU32 base = dim ? rg_with_alpha(kRGRead, 70) : kRGRead;
				if (rg_producer_of(box, n, p, acc.handle.id) < 0) dl->AddCircle(c, kPinR, base, 12, 2.0f);
				else                                              dl->AddCircleFilled(c, kPinR, base, 12);
				if (i == hovB && !hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
				dl->PushClipRect(ImVec2(c.x + kPinR + 3, c.y - kPinRowH * 0.5f), ImVec2(mid, c.y + kPinRowH * 0.5f), true);
				dl->AddText(ImVec2(c.x + kPinR + 3, c.y - ls.y * 0.5f), lc, lbl);
				dl->PopClipRect();
			}
			if (access_is_write(acc.type)) {   // output pin (right)
				int slot = outS++; ImVec2 c = outPin(i, slot);
				ImU32 base = dim ? rg_with_alpha(kRGWrite, 70) : kRGWrite;
				dl->AddCircleFilled(c, kPinR, base, 12);
				if (i == hovB && hovWrite && slot == hovSlot) dl->AddCircle(c, kPinR + 3.0f, IM_COL32(255, 255, 255, 255), 16, 2.0f);
				dl->PushClipRect(ImVec2(mid, c.y - kPinRowH * 0.5f), ImVec2(c.x - kPinR - 3, c.y + kPinRowH * 0.5f), true);
				dl->AddText(ImVec2(c.x - kPinR - 3 - ls.x, c.y - ls.y * 0.5f), lc, lbl);
				dl->PopClipRect();
			}
		}
	}

	// ---- tooltip: hovered pin wins; else fall back to the per-pass reads/writes list.
	if (hovB >= 0) {
		PassNode* p = box[hovB].p; ResourceNode* r = rg->node({ hovId });
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
			ResourceNode* r = rg->node(acc.handle);
			WGPUStringView rn = r ? r->name : WGPUStringView{};
			ImGui::Text("[%s] %.*s  (%s)%s", access_is_write(acc.type) ? "W" : "R",
				(int)rn.length, rn.data ? rn.data : "", rg_access_name(acc.type),
				r ? (r->imported ? "  [imported]" : r->persistent ? "  [temporal]" : "") : "");
		}
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

// TransientResourcePool view: the live cache contents + a create/evict event log. The point is to
// verify the pool reuses textures across frames -- after warmup the "created this frame" count should
// sit at 0 and the log should stop growing, proving realize() hands back cached textures instead of
// recreating them. Drawn after realize() (claims set, createdThisFrame still this frame's misses) and
// before end_frame() releases the claims, so "in use" reflects what this frame actually held.
static void rg_draw_transient_pool(RenderGraphStorage& s)
{
	TransientResourcePool& tp = s.m_allocator->transient;

	int held = (int)tp.entries.size(), inUse = 0;
	uint64_t totalBytes = 0, inUseBytes = 0;
	for (const TransientResourcePool::Entry& e : tp.entries) {
		const uint64_t b = rg_entry_bytes(e);
		totalBytes += b;
		if (e.inUse) { ++inUse; inUseBytes += b; }
	}

	ImGui::Text("frame %llu  --  %d held, %d in use", (unsigned long long)tp.frame, held, inUse);
	ImGui::SameLine();
	if (tp.createdThisFrame == 0)
		ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1), "  --  0 created (reused from pool)");
	else
		ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "  --  %u created this frame", tp.createdThisFrame);

	char tb[24], ib[24], idb[24];
	rg_bytes_str(totalBytes, tb, sizeof tb);
	rg_bytes_str(inUseBytes, ib, sizeof ib);
	rg_bytes_str(totalBytes - inUseBytes, idb, sizeof idb);   // in use is a subset sum -> no underflow
	ImGui::Text("VRAM %s total  --  %s in use, %s idle", tb, ib, idb);

	ImGui::TextDisabled("usage A=attach T=sampled S=storage r=copy-src w=copy-dst   |   evict after %llu idle frames",
		(unsigned long long)TransientResourcePool::kRetain);
	ImGui::Separator();

	const ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

	// memory by format -- which texture types the pool spends its bytes on.
	ImGui::TextDisabled("by format");
	{
		struct FmtAgg { WGPUTextureFormat fmt; int count; uint64_t bytes; };
		FmtAgg agg[16]; int nAgg = 0;
		for (const TransientResourcePool::Entry& e : tp.entries) {
			int j = 0;
			for (; j < nAgg; ++j) if (agg[j].fmt == e.format) break;
			if (j == nAgg) { if (nAgg == 16) continue; agg[nAgg++] = { e.format, 0, 0 }; }   // 17th+ format folded away
			agg[j].count++;
			agg[j].bytes += rg_entry_bytes(e);
		}
		if (ImGui::BeginTable("tp_fmt", 3, tf)) {
			ImGui::TableSetupColumn("format");
			ImGui::TableSetupColumn("count");
			ImGui::TableSetupColumn("bytes");
			ImGui::TableHeadersRow();
			for (int j = 0; j < nAgg; ++j) {
				char bb[24]; rg_bytes_str(agg[j].bytes, bb, sizeof bb);
				ImGui::TableNextRow();
				ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(agg[j].fmt));
				ImGui::TableNextColumn(); ImGui::Text("%d", agg[j].count);
				ImGui::TableNextColumn(); ImGui::Text("%s", bb);
			}
			ImGui::EndTable();
		}
	}
	ImGui::Separator();

	// currently held physical textures.
	if (ImGui::BeginTable("tp_live", 8, tf)) {
		ImGui::TableSetupColumn("#");
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
			ImGui::TableNextColumn(); ImGui::Text("%d", idx++);
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

	ImGui::Spacing();
	ImGui::Text("events (newest first)");
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
		if (ImGui::BeginTabItem("Pool"))      { rg_draw_transient_pool(s);     ImGui::EndTabItem(); }
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
