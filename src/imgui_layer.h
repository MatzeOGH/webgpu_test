#pragma once
// Dear ImGui bring-up for the RenderGraph sample: SDL3 platform + WebGPU(Dawn) renderer backends,
// plus a debug widget that draws the compiled graph. #included once into the single TU
// (RenderGraph_main.cpp), after RenderGraph.h, so imgui_layer_draw_graph can read the RG:: internals.
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_wgpu.h"
#include <cstdio>   // snprintf for node labels

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

// DAG view: a box per pass, bezier lines for the dependency edges, laid out left-to-right by
// dependency depth. Reads the .cpp-internal node structs directly. Assumes a compiled, valid graph
// (realize() has run) -- no hedging, like the other debug dumps.
static void rg_draw_dag(RenderGraph* rg, RenderGraphStorage& s)
{
	// layout: column = longest dependency chain to a root, row = stacking slot within the column.
	// m_passes is topo-sorted, so every predecessor's layer is known before we reach the pass.
	constexpr int   kMaxNodes = 128;
	constexpr float kBoxW = 132.0f, kBoxH = 44.0f, kColGap = 56.0f, kRowGap = 22.0f;
	const float colStep = kBoxW + kColGap, rowStep = kBoxH + kRowGap;

	int layer[kMaxNodes] = {};
	int rowsInLayer[kMaxNodes] = {};
	ImVec2 pos[kMaxNodes];
	PassNode* node[kMaxNodes];
	int n = 0, maxLayer = 0, maxRow = 0;
	for (PassNode* p = s.m_passes; p && n < kMaxNodes; p = p->next, ++n) {
		int lay = 0;
		for (NodeAdjacency* a = p->adjacency; a; a = a->next) {
			int pi = (int)pass_index(s.m_passes, a->pass);
			if (pi < n && layer[pi] + 1 > lay) lay = layer[pi] + 1;
		}
		int row = rowsInLayer[lay]++;
		layer[n] = lay;
		node[n] = p;
		pos[n] = ImVec2(lay * colStep, row * rowStep);
		if (lay > maxLayer) maxLayer = lay;
		if (row > maxRow)   maxRow = row;
	}

	ImVec2 canvas((maxLayer + 1) * colStep, (maxRow + 1) * rowStep);
	ImGui::BeginChild("rg_canvas", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
	const ImVec2 origin = ImGui::GetCursorScreenPos();   // content top-left (already scroll-adjusted)
	ImGui::Dummy(canvas);                                // reserve the scroll region
	ImDrawList* dl = ImGui::GetWindowDrawList();

	// edges first so the boxes paint over them; line runs dep -> pass (data/order flow).
	for (int i = 0; i < n; ++i) {
		ImVec2 dst(origin.x + pos[i].x, origin.y + pos[i].y + kBoxH * 0.5f);   // pass: left-center
		for (NodeAdjacency* a = node[i]->adjacency; a; a = a->next) {
			int pi = (int)pass_index(s.m_passes, a->pass);
			if (pi >= n) continue;
			ImVec2 src(origin.x + pos[pi].x + kBoxW, origin.y + pos[pi].y + kBoxH * 0.5f); // dep: right-center
			float dx = (dst.x - src.x) * 0.5f;
			dl->AddBezierCubic(src, ImVec2(src.x + dx, src.y), ImVec2(dst.x - dx, dst.y), dst,
				IM_COL32(170, 170, 170, 200), 2.0f);
		}
	}

	// nodes: an InvisibleButton per box gives a hover item; we paint the box on top of it.
	for (int i = 0; i < n; ++i) {
		ImVec2 tl(origin.x + pos[i].x, origin.y + pos[i].y);
		ImVec2 br(tl.x + kBoxW, tl.y + kBoxH);
		ImGui::SetCursorScreenPos(tl);
		ImGui::PushID(i);
		ImGui::InvisibleButton("n", ImVec2(kBoxW, kBoxH));
		bool hov = ImGui::IsItemHovered();

		dl->AddRectFilled(tl, br, rg_kind_color(node[i]->kind), 5.0f);
		dl->AddRect(tl, br, hov ? IM_COL32(255, 255, 255, 255) : IM_COL32(20, 20, 20, 255),
			5.0f, 0, hov ? 2.0f : 1.0f);

		// mark sinks with a red outline
		if (node[i]->sink)
		{
			for (int i = 3; i >= 1; --i)
			{
				ImVec2 pad(i, i);
				dl->AddRect(ImVec2(tl.x - pad.x, tl.y - pad.y), ImVec2(br.x + pad.x, br.y + pad.y), IM_COL32(255, 100, 0, 200), 6.0f, 0, 1.0f);
			}
		}

		WGPUStringView nm = node[i]->name;
		char head[96];
		std::snprintf(head, sizeof head, "P%d  %.*s", i, (int)nm.length, nm.data ? nm.data : "");
		ImVec2 hs = ImGui::CalcTextSize(head);
		dl->AddText(ImVec2(tl.x + (kBoxW - hs.x) * 0.5f, tl.y + 6.0f), IM_COL32(255, 255, 255, 255), head);
		const char* kn = rg_kind_name(node[i]->kind);
		ImVec2 ks = ImGui::CalcTextSize(kn);
		dl->AddText(ImVec2(tl.x + (kBoxW - ks.x) * 0.5f, br.y - 6.0f - ks.y),
			IM_COL32(225, 225, 225, 220), kn);

		if (hov) {   // detail the box doesn't have room for: this pass's reads/writes.
			ImGui::BeginTooltip();
			ImGui::Text("%.*s  [%s]", (int)nm.length, nm.data ? nm.data : "", kn);
			ImGui::Separator();
			for (uint32_t k = 0; k < node[i]->accessCount; ++k) {
				const ResourceAccess& acc = node[i]->accesses[k];
				ResourceNode* r = rg->node(acc.handle); // TODO: can r ever be null?
				WGPUStringView rn = r ? r->name : WGPUStringView{};
				ImGui::Text("[%s] %.*s  (%s)%s",
					access_is_write(acc.type) ? "W" : "R",
					(int)rn.length, rn.data ? rn.data : "",
					rg_access_name(acc.type),
					[](ResourceNode* r) {
						if (r->imported) return "[imported]";
						if (r->persistent) return "[temporal]";
						return "";
					}(r));
			}
			ImGui::EndTooltip();
		}
		ImGui::PopID();
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

static ImU32 rg_with_alpha(ImU32 c, ImU32 a) { return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT); }

// per-pass access tint inside a lifetime bar: write warm, read cool -- so produce vs consume reads at a glance.
static constexpr ImU32 kRGWrite = IM_COL32(232, 145, 64, 255);
static constexpr ImU32 kRGRead  = IM_COL32(74, 158, 206, 255);

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
			if (r->imported)        ImGui::TextDisabled("imported -- caller-owned, not aliased");
			else if (r->persistent) ImGui::TextDisabled("temporal layer %u -- pool-owned, not aliased", r->temporalIndex);
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
	for (const TransientResourcePool::Entry& e : tp.entries) if (e.inUse) ++inUse;

	ImGui::Text("frame %llu  --  %d held, %d in use", (unsigned long long)tp.frame, held, inUse);
	ImGui::SameLine();
	if (tp.createdThisFrame == 0)
		ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1), "  --  0 created (reused from pool)");
	else
		ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "  --  %u created this frame", tp.createdThisFrame);
	ImGui::TextDisabled("usage A=attach T=sampled S=storage r=copy-src w=copy-dst   |   evict after %llu idle frames",
		(unsigned long long)TransientResourcePool::kRetain);
	ImGui::Separator();

	// currently held physical textures.
	const ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
	if (ImGui::BeginTable("tp_live", 5, tf)) {
		ImGui::TableSetupColumn("#");
		ImGui::TableSetupColumn("size");
		ImGui::TableSetupColumn("format");
		ImGui::TableSetupColumn("usage");
		ImGui::TableSetupColumn("state");
		ImGui::TableHeadersRow();
		int idx = 0;
		for (const TransientResourcePool::Entry& e : tp.entries) {
			char ub[8]; rg_usage_str(e.usage, ub, sizeof ub);
			ImGui::TableNextRow();
			ImGui::TableNextColumn(); ImGui::Text("%d", idx++);
			ImGui::TableNextColumn(); ImGui::Text("%ux%u", e.size.width, e.size.height);
			ImGui::TableNextColumn(); ImGui::Text("%s", rg_format_name(e.format));
			ImGui::TableNextColumn(); ImGui::Text("%s", ub);
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

// The RenderGraph debug window: FPS readout + a tab bar over the dependency DAG and the resource-
// lifetime grid. Built after compile()+realize(), so both tabs read a finished graph.
static void imgui_layer_draw_graph(RenderGraph* rg)
{
	RenderGraphStorage& s = *storage(rg);

	ImGui::Begin("RenderGraph");
	ImGui::Text(" %.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
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
