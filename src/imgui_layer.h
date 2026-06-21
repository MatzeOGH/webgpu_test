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
/*

// will become the memory debug view
struct Task {
	const char* name;
	float start;
	float end;
	ImU32 color;
};

static Task tasks[] = {
	{ "Design",  0.0f, 3.0f, IM_COL32(120,180,255,255) },
	{ "Coding",  2.0f, 7.0f, IM_COL32(120,255,140,255) },
	{ "Testing", 6.0f, 9.0f, IM_COL32(255,200,120,255) },
};

void DrawGantt()
{
	ImDrawList* draw = ImGui::GetWindowDrawList();

	float rowHeight = 28.0f;
	float scale = 80.0f; // pixels per time unit

	ImVec2 canvasPos = ImGui::GetCursorScreenPos();
	ImVec2 canvasSize(900, 3 * rowHeight);

	draw->AddRectFilled(
		canvasPos,
		ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
		IM_COL32(40, 40, 40, 255));

	for (int i = 0; i < IM_ARRAYSIZE(tasks); i++)
	{
		const Task& t = tasks[i];

		float y = canvasPos.y + i * rowHeight;

		float x0 = canvasPos.x + t.start * scale;
		float x1 = canvasPos.x + t.end * scale;

		ImVec2 p0(x0, y + 4);
		ImVec2 p1(x1, y + rowHeight - 4);

		draw->AddRectFilled(p0, p1, t.color, 4.0f);
		draw->AddText(
			ImVec2(canvasPos.x + 5, y + 6),
			IM_COL32_WHITE,
			t.name);
	}

	ImGui::Dummy(canvasSize);
}
/**/

// Draw the compiled graph: a box per pass, bezier lines for the dependency edges, laid out
// left-to-right by dependency depth. Reads the .cpp-internal node structs directly. Assumes a
// compiled, valid graph (realize() has run) -- no hedging, like the other debug dumps.
static void imgui_layer_draw_graph(RenderGraph* rg)
{
	RenderGraphStorage& s = *storage(rg);

	ImGui::Begin("RenderGraph");
	ImGui::Text(" %.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
	ImGui::Separator();


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
	ImGui::End();
}

static void imgui_layer_shutdown()
{
	ImGui_ImplWGPU_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
}
